"""Production MFM/MFV contracts; independent rank and conservation checks."""

import pytest
import torch
import warp as wp

from warpSPHCore import DomainDescription, KernelFunctions, ParticleState
from warpSPHCore.mfm import MeshlessGeometry, MeshlessWarp, mfmRates
from warpSPHCore.mfm.geometry import invertMoments
from warpSPHCore.type_config import scalar_t


KERNEL = KernelFunctions.Wendland2


def _cloud(device, dim=2, shape='regular'):
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D SPH qualification runs on CUDA')
    dtype = wp.dtype_to_torch(scalar_t)
    if shape == 'line':
        x = torch.zeros(9, dim, dtype=dtype, device=device)
        x[:, 0] = torch.linspace(.2, .8, 9, dtype=dtype, device=device)
    elif shape in ('coincident', 'isolated', 'empty'):
        count = {'coincident': 6, 'isolated': 1, 'empty': 0}[shape]
        x = torch.full((count, dim), .5, dtype=dtype, device=device)
    else:
        ax = (torch.arange(6, dtype=dtype, device=device)+.5)/6
        x = torch.stack(torch.meshgrid(*([ax]*dim), indexing='ij'), -1).reshape(-1, dim)
        x = x + .008 * torch.sin(torch.arange(x.numel(), dtype=dtype, device=device).reshape_as(x)*2.3)
    dom = DomainDescription(x.new_full((dim,), -1), x.new_full((dim,), 2),
                            torch.zeros(dim, dtype=torch.bool, device=device), dim)
    n = len(x)
    p = ParticleState(positions=x, supports=x.new_full((n,), .55), masses=x.new_ones(n),
                      densities=x.new_ones(n), kinds=torch.zeros(n, dtype=torch.int32, device=device))
    return p, dom


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_selected_rank_is_not_the_condition_of_the_retained_subspace(dim, dtype):
    diag = torch.ones(3, dim, dtype=dtype)
    diag[0] = 0
    diag[1, -1] = 0
    diag[2, -1] = 1e-14
    matrix = torch.diag_embed(diag)
    inverse, condition = invertMoments(matrix)
    assert torch.isfinite(inverse).all()
    # In 1D a nonzero scalar has condition one; no relative eigenvalue is lost.
    expected = torch.tensor([True, True, dim > 1])
    assert torch.equal(torch.isinf(condition), expected)
    retained = (diag > (1e-12 if dtype == torch.float64 else 50*torch.finfo(dtype).eps)
                * diag.abs().amax(-1, keepdim=True))
    projector = matrix @ inverse
    torch.testing.assert_close(projector, torch.diag_embed(retained.to(dtype)))


def test_inactive_singular_lu_does_not_poison_inverse_reverse_ad():
    matrix = torch.tensor([[[1., 0.], [0., 0.]], [[2., .1], [.1, 1.]]],
                          dtype=torch.float64, requires_grad=True)
    inverse, condition = invertMoments(matrix)
    inverse.square().sum().backward()
    assert torch.isfinite(matrix.grad).all()
    assert torch.isinf(condition[0]) and torch.isfinite(condition[1])


@pytest.mark.parametrize('shape', ['line', 'coincident', 'isolated', 'empty'])
@pytest.mark.parametrize('closure', ['none', 'project'])
def test_degenerate_geometry_is_flagged_with_finite_rates(device, shape, closure):
    p, dom = _cloud(device, shape=shape)
    g = MeshlessGeometry.build(p, dom, KERNEL, closure=closure)
    w = MeshlessWarp.build(p.positions, p.supports, dom, KERNEL, closure=closure)
    assert g.deficient.all() and w.deficient.all()
    assert torch.isinf(g.cond).all() and torch.isinf(w.cond).all()
    assert torch.isfinite(g.Einv).all() and torch.isfinite(w.Einv).all()
    assert torch.isfinite(g.volume).all() and torch.isfinite(w.volume).all()
    field = 1+p.positions[:, 0]
    assert torch.equal(g.gradient(field), torch.zeros_like(p.positions))
    rho, vel, pressure = field, p.positions*.1, field*.8
    for mode in ('MFM', 'MFV'):
        a, _ = mfmRates(g, rho, vel, pressure, 1.4, mode=mode, dt=.001)
        b = w.rates(rho, vel, pressure, 1.4, mode=mode, dt=.001)
        assert a.shape == b.shape == (len(field), 4)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        tol = 5e-5 if field.dtype == torch.float32 else 1e-10
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize('mode', ['MFM', 'MFV'])
@pytest.mark.parametrize('order,dt', [(1, 0.), (2, 0.), (2, .001)])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_resolved_rates_conserve_and_match_backends(device, dim, order, dt, mode):
    p, dom = _cloud(device, dim)
    # Open geometry is sufficient for global pairwise conservation. Uniform
    # equilibrium is a separate periodic-closure contract, not assumed here.
    g = MeshlessGeometry.build(p, dom, KERNEL, closure='none')
    w = MeshlessWarp.build(p.positions, p.supports, dom, KERNEL, closure='none')
    assert not g.deficient.any() and not w.deficient.any()
    x = p.positions
    rho, vel, pressure = 1+.1*x[:, 0].sin(), .2*x.cos(), 1+.15*x[:, -1].cos()
    a, _ = mfmRates(g, rho, vel, pressure, 1.4, mode=mode, order=order, dt=dt)
    b = w.rates(rho, vel, pressure, 1.4, mode=mode, order=order, dt=dt)
    tol = 3e-4 if x.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)
    for rate in (a, b):
        scale = rate.abs().sum(0).clamp_min(1)
        assert bool((rate.sum(0).abs()/scale < (2e-6 if x.dtype == torch.float32 else 2e-13)).all())
        if mode == 'MFM':
            assert torch.equal(rate[:, 0], torch.zeros_like(rho))
    # An affine field has a known matrix gradient, independent of the other backend.
    affine = 1 + 2*x.sum(-1)
    torch.testing.assert_close(g.gradient(affine), torch.full_like(x, 2), rtol=tol, atol=tol)


def test_unknown_mode_is_rejected_by_both_backends(device):
    p, dom = _cloud(device, dim=1)
    g = MeshlessGeometry.build(p, dom, KERNEL, closure='none')
    w = MeshlessWarp.build(p.positions, p.supports, dom, KERNEL, closure='none')
    field = p.supports.new_ones(len(p.supports))
    for mode in ('', 'MFF', 'FiniteMass'):
        with pytest.raises(ValueError, match='mode'):
            mfmRates(g, field, p.positions, field, 1.4, mode=mode)
        with pytest.raises(ValueError, match='mode'):
            w.rates(field, p.positions, field, 1.4, mode=mode)


@pytest.mark.parametrize('name', ['positions', 'supports'])
def test_warp_geometry_ad_is_rejected_before_build(device, name):
    import torch.autograd.forward_ad as fwAD
    p, dom = _cloud(device, dim=1)
    tensor = getattr(p, name)
    values = dict(positions=p.positions, supports=p.supports)
    values[name] = tensor.clone().requires_grad_()
    with pytest.raises(NotImplementedError, match='forward-only'):
        MeshlessWarp.build(**values, domain=dom, kernel=KERNEL)
    # requires_grad metadata is harmless during an explicitly no-grad forward.
    with torch.no_grad():
        MeshlessWarp.build(**values, domain=dom, kernel=KERNEL, closure='none')
    with fwAD.dual_level():
        values[name] = fwAD.make_dual(tensor, torch.ones_like(tensor))
        with pytest.raises(NotImplementedError, match='forward-only'):
            MeshlessWarp.build(**values, domain=dom, kernel=KERNEL)


@pytest.mark.parametrize('method', ['gradients', 'rates', 'timestep'])
def test_warp_field_ad_is_rejected_instead_of_returning_partial_derivatives(device, method):
    import torch.autograd.forward_ad as fwAD
    p, dom = _cloud(device, dim=1)
    w = MeshlessWarp.build(p.positions, p.supports, dom, KERNEL, closure='none')
    field = torch.ones_like(p.supports)
    kwargs = {} if method == 'gradients' else dict(gamma=1.4)
    fn = getattr(w, method)
    with pytest.raises(NotImplementedError, match='forward-only'):
        fn(field.requires_grad_(), p.positions, field, **kwargs)
    with torch.no_grad():
        fn(field, p.positions, field, **kwargs)
    with fwAD.dual_level():
        dual = fwAD.make_dual(field.detach(), torch.ones_like(field))
        with pytest.raises(NotImplementedError, match='forward-only'):
            fn(dual, p.positions, field.detach(), **kwargs)


@pytest.mark.parametrize('mode', ['MFM', 'MFV'])
def test_torch_field_reverse_and_dual_match_finite_differences(device, mode):
    import torch.autograd.forward_ad as fwAD
    p, dom = _cloud(device, dim=2)
    g = MeshlessGeometry.build(p, dom, KERNEL, closure='none')
    x = p.positions
    field = torch.cat([(1+.1*x[:, :1].sin()), .2*x.cos(), (1+.15*x[:, -1:].cos())], 1)
    # Separate near-tied extrema/limiter switches before checking a smooth
    # branch derivative; AD at a switch is not a central finite difference.
    field = field + .03*torch.sin(torch.arange(field.numel(), dtype=x.dtype, device=device).reshape_as(field)*3.17)
    tangent = .1*torch.sin(torch.arange(field.numel(), dtype=x.dtype, device=device).reshape_as(field)*1.7)

    def rates(f):
        return mfmRates(g, f[:, 0], f[:, 1:3], f[:, 3], 1.4, mode=mode, dt=.001)[0]

    primal = field.clone().requires_grad_()
    value = rates(primal)
    cotangent = torch.cos(torch.arange(value.numel(), dtype=x.dtype, device=device).reshape_as(value)*.7)
    reverse, = torch.autograd.grad((value*cotangent).sum(), primal)
    with fwAD.dual_level():
        dual = rates(fwAD.make_dual(field, tangent))
        forward = fwAD.unpack_dual(dual).tangent
    assert torch.isfinite(reverse).all() and torch.isfinite(forward).all()
    tol = 2e-3 if x.dtype == torch.float32 else 2e-7
    torch.testing.assert_close((reverse*tangent).sum(), (forward*cotangent).sum(), rtol=tol, atol=tol*1e-2)
    # Uniform primitive scaling preserves the ordering of limiter extrema.
    # An arbitrary FP32 perturbation can cross their much narrower gaps;
    # reverse/dual adjoint consistency above still uses the arbitrary tangent.
    tangent = .1*field
    with fwAD.dual_level():
        forward = fwAD.unpack_dual(rates(fwAD.make_dual(field, tangent))).tangent
    eps = .02 if x.dtype == torch.float32 else 1e-5
    fd = (rates(field+eps*tangent)-rates(field-eps*tangent))/(2*eps)
    torch.testing.assert_close(forward, fd, rtol=tol, atol=tol*1e-2)


@pytest.mark.parametrize('method', ['gradients', 'rates', 'timestep', 'close'])
def test_warp_rechecks_geometry_ad_metadata_after_build(device, method):
    p, dom = _cloud(device, dim=1)
    w = MeshlessWarp.build(p.positions, p.supports, dom, KERNEL, closure='none')
    p.positions.requires_grad_()
    field = torch.ones_like(p.supports)
    fn = getattr(w, method)
    args = () if method == 'close' else (field, torch.zeros_like(p.positions), field)
    kwargs = {} if method in ('close', 'gradients') else dict(gamma=1.4)
    with pytest.raises(NotImplementedError, match='forward-only'):
        fn(*args, **kwargs)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('count', [0, 1, 17])
@pytest.mark.parametrize('storeI', [False, True])
def test_row_chunked_geometry_moments_preserve_self_identity(device, dim, count, storeI):
    from warpSPHCore import OperationProperties, SupportScheme, buildVerletList
    from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
    from warpSPHCore.mfm.wp_mfm import mfmMomentsWarp
    dtype = wp.dtype_to_torch(scalar_t)
    generator = torch.Generator().manual_seed(413)
    positions = (.8*torch.rand(count, dim, generator=generator, dtype=dtype)-.4).to(device)
    p = ParticleState(positions=positions, supports=positions.new_full((count,), .37),
        masses=positions.new_ones(count), densities=positions.new_ones(count),
        kinds=torch.zeros(count, dtype=torch.int32, device=device))
    domain = DomainDescription(positions.new_full((dim,), -.5), positions.new_full((dim,), .5),
        torch.ones(dim, dtype=torch.bool, device=device), dim)
    flat = buildVerletList(p, domain, supportMode=SupportScheme.SuperSymmetric, storeI=storeI, queryOrder='input')
    chunks = buildChunkedVerletList(p, domain,
        policy=RowChunkPolicy(edgeBudget=128, maxRows=3, bytesPerEdge=8),
        supportMode=SupportScheme.SuperSymmetric, storeI=storeI)
    props = OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather)
    expected = mfmMomentsWarp(p, props, domain, flat)
    for a, b in zip(mfmMomentsWarp(p, props, domain, chunks), expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    control = MeshlessWarp.build(positions, p.supports, domain, KERNEL, adjacency=flat, closure='none')
    chunked = MeshlessWarp.build(positions, p.supports, domain, KERNEL, adjacency=chunks, closure='none')
    for name in ('omega', 'num_nbrs', 'Einv', 'cond', 'deficient', 'volume', 'condBad'):
        torch.testing.assert_close(getattr(chunked, name), getattr(control, name), rtol=0, atol=0, msg=name)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('closure,solver,cap', [('none', 'auto', 0.), ('project', 'cg', 0.),
    ('project', 'mg', 0.), ('project', 'cg', .02)])
def test_row_chunked_meshless_native_closure_and_flux(device, dim, closure, solver, cap):
    from warpSPHCore import SupportScheme, buildVerletList
    from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
    dtype = wp.dtype_to_torch(scalar_t)
    axis = torch.linspace(-.4, .4, 4, dtype=dtype, device=device)
    positions = torch.stack(torch.meshgrid(*([axis]*dim), indexing='ij'), -1).reshape(-1, dim)
    positions += .012*torch.sin(torch.arange(positions.numel(), device=device).reshape_as(positions)*1.7)
    p = ParticleState(positions=positions, supports=positions.new_full((len(positions),), .41),
        masses=positions.new_ones(len(positions)), densities=positions.new_ones(len(positions)),
        kinds=torch.zeros(len(positions), dtype=torch.int32, device=device))
    domain = DomainDescription(positions.new_full((dim,), -.5), positions.new_full((dim,), .5),
        torch.ones(dim, dtype=torch.bool, device=device), dim)
    flat = buildVerletList(p, domain, supportMode=SupportScheme.SuperSymmetric, storeI=False, queryOrder='input')
    chunks = buildChunkedVerletList(p, domain, policy=RowChunkPolicy(edgeBudget=4096, maxRows=3,
        bytesPerEdge=4+positions.element_size()), supportMode=SupportScheme.SuperSymmetric, storeI=False)
    options = dict(closure=closure, closure_solver=solver, closure_cap=cap, mg_max_coarse=16)
    a = MeshlessWarp.build(positions, p.supports, domain, KERNEL, adjacency=flat, **options)
    b = MeshlessWarp.build(positions, p.supports, domain, KERNEL, adjacency=chunks, **options)
    tol = 3e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(a.lam, b.lam, rtol=tol, atol=tol)
    rho, vel, pressure = 1+.1*positions[:, 0].sin(), .2*positions.cos(), 1+.15*positions[:, -1].cos()
    for mode in ('MFM', 'MFV'):
        ra = a.rates(rho, vel, pressure, 1.4, mode=mode, dt=.001)
        rb = b.rates(rho, vel, pressure, 1.4, mode=mode, dt=.001)
        torch.testing.assert_close(ra, rb, rtol=tol, atol=tol)
        assert bool((rb.sum(0).abs() / rb.abs().sum(0).clamp_min(1) < tol).all())
    torch.testing.assert_close(a.timestep(rho, vel, pressure, 1.4), b.timestep(rho, vel, pressure, 1.4), rtol=tol, atol=tol)
