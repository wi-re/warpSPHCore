"""CRK preparation against independent differentiated Torch moment equations."""
from dataclasses import replace
import sys

import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture, normalization
from _support.kernels import shape


def full_csr(q, r, index_dtype=torch.int32, offset_dtype=torch.int64):
    n, m = len(q.positions), len(r.positions)
    ids = torch.arange(m - 1, -1, -1, device=q.positions.device, dtype=index_dtype).repeat(n)
    counts = torch.full((n,), m, device=q.positions.device, dtype=torch.int32)
    # A nonzero prefix and reversed rows exercise independent CSR addressing.
    return sph.AdjacencyList(None, torch.cat((ids.new_zeros(3), ids)), counts,
        (torch.arange(n, device=q.positions.device) * m + 3).to(offset_dtype), n, m,
        numEdges=ids.numel())


def equations(q, r, domain, properties, volumes):
    # Each pair gets a virtual displacement. Differentiate geometric moments,
    # rather than copying the production derivative formula into this oracle.
    x, y = q.positions.double(), r.positions.double()
    delta = torch.zeros((len(x), len(y), domain.dim), device=x.device,
                        dtype=torch.float64, requires_grad=True)
    dx = x[:, None] + delta - y[None]
    length = (domain.max - domain.min).double()
    dx = torch.where(domain.periodic, (dx + length / 2).remainder(length) - length / 2, dx)
    radius = dx.norm(dim=-1)
    hi, hj = q.supports.double()[:, None], r.supports.double()[None]
    mode = properties.supportMode.value
    h = {11: hi, 12: hj, 13: (hi + hj) / 2,
         14: torch.maximum(hi, hj), 15: torch.maximum(hi, hj)}[mode]

    def kernel(h):
        ratio = radius / h
        active = (h > 0) & (ratio < 1)
        if properties.kernel == sph.KernelFunctions.AdhesionKernel:
            active = active & (ratio > .5)
            ratio = torch.where(active, ratio, .75)
        return torch.where(active, normalization(properties.kernel, domain.dim) *
                           shape(ratio, properties.kernel, domain.dim) / h**domain.dim, 0.)

    w = (kernel(hi) + kernel(hj)) / 2 if mode in (14, 15) else kernel(h)
    allowed = torch.ones_like(w, dtype=torch.bool)
    if properties.operationMode == sph.OperationDirection.FluidToBoundary:
        allowed = (q.kinds == 1)[:, None] & (r.kinds == 0)[None]
    elif properties.operationMode == sph.OperationDirection.AllToAll:
        allowed = allowed & (r.kinds != 2)[None]
    w = w * allowed
    weighted = w * volumes.double()
    m0 = weighted.sum(1)
    m1 = (weighted[..., None] * dx).sum(1)
    m2 = torch.einsum('ij,ija,ijb->iab', weighted, dx, dx)
    def derivative(value):
        return torch.autograd.grad(value.sum(), delta, retain_graph=True)[0].sum(1)
    dm0 = derivative(m0)
    dm1 = torch.stack([derivative(m1[:, a]) for a in range(domain.dim)], 1)
    dm2 = torch.stack([derivative(m2[:, a, b]) for a in range(domain.dim)
                      for b in range(domain.dim)], -1).reshape(-1, domain.dim, domain.dim, domain.dim)
    return w.detach(), dx.detach(), tuple(v.detach() for v in (m0, m1, m2, dm0, dm1, dm2)), allowed.sum(1).int()


def check_equations(device, dimension, dtype, kernel, support, index_dtype, offset_dtype, filtered):
    q, r, domain, _, _ = fixture(device, dimension, 0)
    def cast(state):
        return replace(state, **{key: getattr(state, key).to(dtype) for key in
            ('positions', 'supports', 'masses', 'densities')})
    q, r = cast(q), cast(r)
    # Reciprocal tests need resolvable positive kernel sums: Gaussian tails
    # underflow in FP32 and viscosity values cancel close to the cutoff.
    # Kernel boundary behavior is tested separately, including underflow below.
    if kernel in (sph.KernelFunctions.Gaussian, sph.KernelFunctions.ViscosityKernel):
        q.supports *= 2
        r.supports *= 2
    # Unwrapped periodic coordinates must still produce minimum-image moments.
    q.positions[:, 0] += 6
    r.positions[:, 0] -= 4
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    properties = sph.OperationProperties(kernel, supportMode=support,
        operationMode=sph.OperationDirection.FluidToBoundary if filtered else sph.OperationDirection.TrueAllToToAll)
    graph = full_csr(q, r, index_dtype, offset_dtype)
    volumes = torch.linspace(.2, .7, len(r.positions), device=device, dtype=dtype)
    w, dx, expected, count = equations(q, r, domain, properties, volumes)
    tolerance = 8e-4 if dtype == torch.float32 else 4e-9
    actual = sph.computeCRKMoments(q, properties, domain, None, volumes, graph, r)
    for a, e in zip(actual[:6], expected):
        torch.testing.assert_close(a.double(), e, rtol=tolerance, atol=tolerance)
        assert a.is_contiguous()
    torch.testing.assert_close(actual[-1], count)
    volume = sph.computeCRKVolume(q, properties, domain, graph, r)
    total = w.sum(1)
    torch.testing.assert_close(volume.double(), torch.where(total == 0, 0., total.reciprocal()),
                               rtol=tolerance, atol=tolerance)
    state = sph.CRKState(q.positions.new_full((len(q.positions),), 1.2),
                         q.positions.new_full(q.positions.shape, -.3))
    # The consistency diagnostic always uses Scatter, even when properties
    # select another support rule. Its numerator and denominator are different.
    density = sph.computeCRKDensity(q, properties, domain, state, None, volumes, graph, r)
    w, dx, _, _ = equations(q, r, domain, replace(properties, supportMode=sph.SupportScheme.Scatter), volumes)
    corrected = w * state.A.double()[:, None] * (1 + (state.B.double()[:, None] * dx).sum(-1))
    numerator = (corrected * r.masses.double() * volumes.double()).sum(1)
    denominator = (corrected * volumes.double().square()).sum(1)
    expected_density = torch.where(denominator == 0, 0., numerator / denominator)
    torch.testing.assert_close(density.double(), expected_density, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('index_dtype,offset_dtype', [(torch.int32, torch.int32), (torch.int32, torch.int64),
                                                    (torch.int64, torch.int32), (torch.int64, torch.int64)])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:5])
def test_preparation_layouts(device, dimension, dtype, index_dtype, offset_dtype, support):
    check_equations(device, dimension, dtype, sph.KernelFunctions.Wendland2, support,
                    index_dtype, offset_dtype, bool((dimension + support.value) % 2))


@pytest.mark.extended
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:5])
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_preparation_kernels(device, dimension, dtype, support, kernel):
    check_equations(device, dimension, dtype, kernel, support, torch.int32, torch.int64,
                    bool((dimension + kernel.value) % 2))


def cloud(device, dimension, dtype):
    axis = torch.tensor([-.4, 0., .4], device=device, dtype=dtype)
    x = torch.cartesian_prod(*([axis] * dimension)).reshape(-1, dimension)
    def particles(x):
        return sph.ParticleState(x, x.new_full((len(x),), 1.3), x.new_full((len(x),), .2),
            torch.zeros(len(x), dtype=torch.int32, device=device))
    reference = particles(x)
    query = particles(x.new_tensor([[-.1] * dimension, [.12] * dimension]))
    domain = sph.DomainDescription(x.new_full((dimension,), -1), x.new_ones(dimension),
        torch.zeros(dimension, device=device, dtype=torch.bool), dimension)
    return query, reference, domain


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('automatic', [False, True])
def test_complete_linear_reproduction(device, dimension, dtype, automatic):
    q, r, domain = cloud(device, dimension, dtype)
    graph = None if automatic else full_csr(q, r)
    reference_graph = None if automatic else full_csr(r, r, torch.int64, torch.int32)
    result = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll, graph, r, referenceAdjacency=reference_graph)
    assert result.queryVolumes.shape == (len(q.positions),)
    assert result.referenceVolumes.shape == (len(r.positions),)
    assert result.queryVolumes.data_ptr() != result.referenceVolumes.data_ptr()
    tolerance = 5e-5 if dtype == torch.float32 else 4e-12
    # For constant reference masses the diagnostic must equal mass divided by
    # the apparent reference volume under its corrected ratio, not mass alone.
    assert torch.isfinite(result.consistencyDensity).all()
    slope = torch.arange(1, dimension + 1, device=device, dtype=dtype)
    fi, fj = 1 + q.positions @ slope, 1 + r.positions @ slope
    options = dict(referenceParticles=r, adjacency=graph, referenceVolumes=result.referenceVolumes,
                   crkState=result.crkState)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        supportMode=sph.SupportScheme.Scatter, operationMode=sph.OperationDirection.TrueAllToToAll)
    interpolation = sph.warpOperation(q, properties, domain, fi, fj, **options)
    gradient = sph.warpOperation(q, replace(properties, operation=sph.WarpOperation.Gradient),
                                 domain, fi, fj, **options)
    torch.testing.assert_close(interpolation, fi, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(gradient, slope.expand_as(gradient), rtol=tolerance, atol=tolerance)
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)


def test_reference_roles_and_compatibility(device):
    q, r, domain = cloud(device, 2, torch.float64)
    q.supports[:] = .7
    r.supports[:] = torch.linspace(.8, 1.4, len(r.positions), device=device)
    r.kinds[::2] = 2
    mode = sph.OperationDirection.AllToAll
    result = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2, mode,
        referenceState=r, referenceVolumeMode=sph.OperationDirection.TrueAllToToAll)
    expected_query = sph.computeCRKVolume(q, sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operationMode=mode), domain, referenceParticles=r)
    expected_reference = sph.computeCRKVolume(r, sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operationMode=sph.OperationDirection.TrueAllToToAll), domain)
    torch.testing.assert_close(result.queryVolumes, expected_query)
    torch.testing.assert_close(result.referenceVolumes, expected_reference)
    legacy = sph.computeCRKFactors(q, domain, sph.KernelFunctions.Wendland2, mode,
        referenceState=r, referenceVolumeMode=sph.OperationDirection.TrueAllToToAll)
    for a, b in zip(legacy[:2], (result.queryVolumes, result.consistencyDensity)):
        torch.testing.assert_close(a, b)
    for name in ('A', 'B', 'gradA', 'gradB'):
        torch.testing.assert_close(getattr(legacy[2], name), getattr(result.crkState, name))
    aliased = sph.prepareCRK(r, domain, sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll)
    assert aliased.queryVolumes is aliased.referenceVolumes
    overridden = sph.prepareCRK(r, domain, sph.KernelFunctions.Wendland2, mode,
        referenceVolumeMode=sph.OperationDirection.TrueAllToToAll)
    torch.testing.assert_close(overridden.referenceVolumes, expected_reference)
    assert overridden.queryVolumes is not overridden.referenceVolumes


@pytest.mark.parametrize('population', ['empty', 'one', 'line'])
def test_degenerate_systems(device, population):
    q, r, domain = cloud(device, 2, torch.float64)
    n = {'empty': 0, 'one': 1, 'line': 3}[population]
    r = replace(r, positions=r.positions[:n].clone(), supports=r.supports[:n],
                masses=r.masses[:n], kinds=r.kinds[:n])
    if n: r.positions[:, 1] = 0
    result = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2,
                            sph.OperationDirection.TrueAllToToAll, referenceState=r)
    for name in ('A', 'B', 'gradA', 'gradB'):
        value = getattr(result.crkState, name)
        torch.testing.assert_close(value, torch.ones_like(value) if name == 'A' else torch.zeros_like(value))
    assert torch.isfinite(result.consistencyDensity).all()
    if n == 0:
        assert torch.count_nonzero(result.queryVolumes) == 0
        assert torch.count_nonzero(result.consistencyDensity) == 0


def test_empty_queries_and_unused_fields(device):
    q, r, domain = cloud(device, 3, torch.float64)
    q = replace(q, positions=q.positions[:0], supports=q.supports[:0], kinds=q.kinds[:0], masses=None)
    result = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2,
                            sph.OperationDirection.TrueAllToToAll, referenceState=r)
    assert result.crkState.B.shape == (0, 3)
    assert result.crkState.gradB.shape == (0, 3, 3)
    # Volume and moments never consume masses or densities.
    r = replace(r, masses=None)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2)
    volume = sph.computeCRKVolume(r, properties, domain)
    moments = sph.computeCRKMoments(r, properties, domain, None, volume)
    assert torch.isfinite(moments[0]).all()


def test_preparation_ad_guards(device):
    q, r, domain = cloud(device, 2, torch.float64)
    q.positions.requires_grad_()
    with pytest.raises(NotImplementedError, match='forward-only'):
        sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2, referenceState=r)
    with torch.no_grad():
        result = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2, referenceState=r)
    assert not result.crkState.A.requires_grad
    q.positions = q.positions.detach()
    volumes = torch.ones_like(r.supports, requires_grad=True)
    with pytest.raises(NotImplementedError, match='forward-only'):
        sph.computeCRKMoments(q, sph.OperationProperties(sph.KernelFunctions.Wendland2),
                              domain, None, volumes, referenceParticles=r)
    with torch.autograd.forward_ad.dual_level():
        q.supports = torch.autograd.forward_ad.make_dual(q.supports, torch.ones_like(q.supports))
        with pytest.raises(NotImplementedError, match='forward-mode'):
            sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2, referenceState=r)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gaussian_volume_underflow(device, dtype):
    q = sph.ParticleState(torch.zeros((1, 1), device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype), None,
        torch.zeros(1, device=device, dtype=torch.int32))
    r = replace(q, positions=q.positions.new_full((1, 1), .99))
    domain = sph.DomainDescription(q.positions.new_tensor([-2]), q.positions.new_tensor([2]),
                                   torch.zeros(1, device=device, dtype=torch.bool), 1)
    properties = sph.OperationProperties(sph.KernelFunctions.Gaussian,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    actual = sph.computeCRKVolume(q, properties, domain, referenceParticles=r)
    weight = normalization(properties.kernel, 1) * torch.exp(-128 * r.positions[:, 0]**2)
    expected = torch.where(weight == 0, 0., weight.reciprocal())
    torch.testing.assert_close(actual, expected)


def test_permutation_and_supplied_topology(device, monkeypatch):
    import importlib
    search = importlib.import_module('warpSPHCore.radiusSearch.native')
    q, r, domain = cloud(device, 2, torch.float64)
    original = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll, full_csr(q, r), r,
        referenceAdjacency=full_csr(r, r))
    qi = torch.tensor([1, 0], device=device)
    ri = torch.tensor([5, 1, 8, 0, 3, 6, 2, 7, 4], device=device)
    def reorder(state, permutation):
        return replace(state, **{key: getattr(state, key)[permutation] for key in
                                ('positions', 'supports', 'masses', 'kinds')})
    shuffled_q, shuffled_r = reorder(q, qi), reorder(r, ri)
    def forbidden(*args, **kwargs):
        raise AssertionError('Supplied topology must not trigger a hidden radius search')
    monkeypatch.setattr(search, 'radiusSearch', forbidden)
    shuffled = sph.prepareCRK(shuffled_q, domain, sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll, full_csr(shuffled_q, shuffled_r), shuffled_r,
        referenceAdjacency=full_csr(shuffled_r, shuffled_r, torch.int64, torch.int32))
    torch.testing.assert_close(shuffled.queryVolumes, original.queryVolumes[qi])
    torch.testing.assert_close(shuffled.referenceVolumes, original.referenceVolumes[ri])
    torch.testing.assert_close(shuffled.consistencyDensity, original.consistencyDensity[qi])
    for name in ('A', 'B', 'gradA', 'gradB'):
        torch.testing.assert_close(getattr(shuffled.crkState, name), getattr(original.crkState, name)[qi])
