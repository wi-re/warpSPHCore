"""Remaining corrected primals: equations, layout and integration contracts."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture
from _support.corrected import remaining_oracle
from test_corrected_primal import correction_fixture
from test_crk_preparation import full_csr, cloud

OPERATIONS = [sph.WarpOperation.Divergence, sph.WarpOperation.Curl,
              sph.WarpOperation.Laplacian, sph.WarpOperation.Covariance]
MODES = ['volume', 'crk', 'omega', 'renorm', 'all']
WIDTHS = [(torch.int32, torch.int32), (torch.int32, torch.int64),
          (torch.int64, torch.int32), (torch.int64, torch.int64)]


def check_case(device, dimension, dtype, rank, operation, mode, codec, kernel, support):
    if operation == sph.WarpOperation.Covariance and rank != 0: pytest.skip('geometric operator')
    if operation in (sph.WarpOperation.Divergence, sph.WarpOperation.Curl) and rank == 0 and dimension > 1:
        pytest.skip('vector or matrix field required')
    q, r, domain, fi, fj = fixture(device, dimension, rank)
    def cast(s):
        return replace(s, **{key: getattr(s, key).to(dtype) for key in
            ('positions', 'supports', 'masses', 'densities')})
    q, r = cast(q), cast(r)
    q.positions[:, 0] += 6; r.positions[:, 0] -= 4
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    fi, fj = fi.to(dtype), fj.to(dtype)
    state, (qv, rv), omega, R = correction_fixture(q, r)
    case = dimension + rank + codec + MODES.index(mode) + operation.value
    scheme = list(sph.GradientScheme)[case % 4]
    lap = list(sph.LaplacianScheme)[case % 4]
    if rank == 0 and dimension > 1 and lap == sph.LaplacianScheme.Dot:
        lap = sph.LaplacianScheme.Brookshaw
    prop = sph.OperationProperties(kernel, operation=operation, supportMode=support,
        gradientMode=scheme, laplacianMode=lap, divergenceDotMode=bool(case % 2),
        operationMode=sph.OperationDirection.FluidToBoundary if case % 2 else sph.OperationDirection.TrueAllToToAll)
    kwargs = {}
    if mode in ('crk', 'all'): kwargs['crkState'] = state
    if mode in ('volume', 'all'): kwargs['referenceVolumes'] = rv
    if mode in ('omega', 'all'): kwargs['gradHState'] = omega
    if mode in ('renorm', 'all'): kwargs['renormalizationState'] = R
    if operation == sph.WarpOperation.Divergence: kwargs['consistentDivergence'] = bool((case // 2) % 2)
    if operation == sph.WarpOperation.Covariance: kwargs['covarianceReturnNumNeighbors'] = True
    actual = sph.warpOperation(q, prop, domain, fi, fj, adjacency=full_csr(q, r, *WIDTHS[codec]),
                               referenceParticles=r, **kwargs)
    expected = remaining_oracle(q, r, domain, prop, fi, fj, **kwargs)
    actual, expected = (actual, expected) if isinstance(actual, tuple) else ((actual,), (expected,))
    tolerance = 8e-4 if dtype == torch.float32 else 4e-9
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.double(), b.double(), rtol=tolerance, atol=tolerance)
        assert a.is_contiguous()


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('codec', range(4))
def test_remaining_layouts(device, operation, mode, rank, dimension, dtype, codec):
    check_case(device, dimension, dtype, rank, operation, mode, codec, sph.KernelFunctions.Wendland2,
               list(sph.SupportScheme)[:5][(dimension + rank + codec + MODES.index(mode)) % 5])


@pytest.mark.extended
@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:5])
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_remaining_kernels(device, operation, dimension, dtype, support, kernel):
    rank = 0 if operation == sph.WarpOperation.Covariance else 1 + (dimension + kernel.value) % 2
    check_case(device, dimension, dtype, rank, operation, 'all', kernel.value % 4, kernel, support)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_native_crk_chain(device, dimension, dtype):
    q, r, domain = cloud(device, dimension, dtype)
    graph = full_csr(q, r)
    prepared = sph.prepareCRK(q, domain, sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll, graph, r, referenceAdjacency=full_csr(r, r))
    coefficients = torch.arange(1, dimension**2 + 1, device=device, dtype=dtype).reshape(dimension, dimension)
    fi = q.positions @ coefficients.T
    fj = r.positions @ coefficients.T
    options = dict(adjacency=graph, referenceParticles=r, referenceVolumes=prepared.referenceVolumes,
                   crkState=prepared.crkState)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operationMode=sph.OperationDirection.TrueAllToToAll, supportMode=sph.SupportScheme.Scatter,
        gradientMode=sph.GradientScheme.Naive)
    tolerance = 8e-5 if dtype == torch.float32 else 8e-12
    divergence = sph.warpOperation(q, replace(properties, operation=sph.WarpOperation.Divergence), domain, fi, fj, **options)
    torch.testing.assert_close(divergence, coefficients.trace().expand_as(divergence), rtol=tolerance, atol=tolerance)
    curl = sph.warpOperation(q, replace(properties, operation=sph.WarpOperation.Curl), domain, fi, fj, **options)
    expected = coefficients.new_zeros(1) if dimension == 1 else (coefficients[1, 0] - coefficients[0, 1]).reshape(1) if dimension == 2 else torch.stack((coefficients[2, 1] - coefficients[1, 2], coefficients[0, 2] - coefficients[2, 0], coefficients[1, 0] - coefficients[0, 1]))
    torch.testing.assert_close(curl, expected.expand_as(curl), rtol=tolerance, atol=tolerance)
    covariance = sph.warpOperation(q, replace(properties, operation=sph.WarpOperation.Covariance), domain, **options)
    torch.testing.assert_close(covariance, torch.eye(dimension, device=device, dtype=dtype).expand_as(covariance), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('gradient_scheme', list(sph.GradientScheme))
@pytest.mark.parametrize('laplacian_scheme', list(sph.LaplacianScheme))
def test_constant_laplacian_and_self_pair(device, dimension, gradient_scheme, laplacian_scheme):
    q, r, domain = cloud(device, dimension, torch.float64)
    state, (_, volumes), _, R = correction_fixture(q, r)
    # A matrix field also qualifies for Dot in every spatial dimension.
    fi = q.positions.new_full((len(q.positions), dimension, dimension), 2.3)
    fj = r.positions.new_full((len(r.positions), dimension, dimension), 2.3)
    q = replace(q, densities=q.supports.new_full(q.supports.shape, .7))
    r = replace(r, densities=r.supports.new_full(r.supports.shape, 1.9))
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Laplacian, gradientMode=gradient_scheme,
        laplacianMode=laplacian_scheme)
    actual = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
        referenceVolumes=volumes, crkState=state, renormalizationState=R)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    # Deliberately nonzero fi-fj at an exact self pair. Corrected gradients
    # need not vanish there; each Laplacian contribution must still be zero.
    q = replace(q, positions=q.positions[:1], supports=q.supports[:1], masses=q.masses[:1],
                kinds=q.kinds[:1], densities=q.densities[:1])
    state = sph.CRKState(*(getattr(state, name)[:1] for name in ('A', 'B', 'gradA', 'gradB')))
    actual = sph.warpOperation(q, properties, domain, fi[:1], fi[:1] + 1,
        referenceVolumes=volumes[:1], crkState=state)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('population', ['query', 'reference'])
def test_remaining_empty_populations(device, operation, population):
    q, r, domain, fi, fj = fixture(device, 3, 1)
    if population == 'query':
        q = replace(q, **{name: getattr(q, name)[:0] for name in ('positions', 'supports', 'masses', 'kinds', 'densities')})
        fi = fi[:0]
    else:
        r = replace(r, **{name: getattr(r, name)[:0] for name in ('positions', 'supports', 'masses', 'kinds', 'densities')})
        fj = fj[:0]
    state, (_, volumes), omega, R = correction_fixture(q, r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation)
    output = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
        referenceVolumes=volumes, crkState=state, gradHState=omega, renormalizationState=R,
        covarianceReturnNumNeighbors=operation == sph.WarpOperation.Covariance)
    outputs = output if isinstance(output, tuple) else (output,)
    for value in outputs:
        assert value.shape[0] == len(q.positions)
        assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('scheme', list(sph.GradientScheme))
def test_remaining_nullable_inputs(device, operation, scheme):
    q, r, domain, fi, fj = fixture(device, 2, 1)
    state, (_, volume), _, _ = correction_fixture(q, r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation, gradientMode=scheme)
    expected = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
                                 referenceVolumes=volume, crkState=state)
    if scheme != sph.GradientScheme.Symmetric or operation == sph.WarpOperation.Covariance:
        q, r = replace(q, masses=None, densities=None), replace(r, masses=None, densities=None)
    if operation == sph.WarpOperation.Covariance: fi = fj = None
    actual = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
                               referenceVolumes=volume, crkState=state)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('slot', ['positions', 'field', 'A', 'volume', 'renormalization'])
def test_remaining_ad_guards(device, operation, slot):
    q, r, domain, fi, fj = fixture(device, 2, 1)
    state, (_, volume), _, R = correction_fixture(q, r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation)
    if slot == 'positions': q.positions.requires_grad_()
    if slot == 'field': fj.requires_grad_()
    if slot == 'A': state.A.requires_grad_()
    if slot == 'volume': volume.requires_grad_()
    if slot == 'renormalization': R.renormalizationMatrices.requires_grad_()
    options = dict(referenceParticles=r, referenceVolumes=volume, crkState=state, renormalizationState=R)
    if slot == 'field' and operation == sph.WarpOperation.Covariance:
        sph.warpOperation(q, properties, domain, fi, fj, **options)  # field is not consumed
    else:
        with pytest.raises(NotImplementedError, match='autograd'):
            sph.warpOperation(q, properties, domain, fi, fj, **options)
    with torch.no_grad():
        assert not sph.warpOperation(q, properties, domain, fi, fj, **options).requires_grad


@pytest.mark.parametrize('operation', OPERATIONS)
def test_remaining_forward_ad_guard(device, operation):
    q, r, domain, fi, fj = fixture(device, 2, 1)
    state, (_, volume), _, _ = correction_fixture(q, r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation)
    with torch.autograd.forward_ad.dual_level():
        volume = torch.autograd.forward_ad.make_dual(volume, torch.ones_like(volume))
        with pytest.raises(NotImplementedError, match='forward-mode'):
            sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
                               referenceVolumes=volume, crkState=state)


def test_unused_naive_laplacian_and_covariance_inputs(device):
    q, r, domain, fi, fj = fixture(device, 2, 1)
    state, (_, volume), omega, R = correction_fixture(q, r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Laplacian, laplacianMode=sph.LaplacianScheme.Naive)
    expected = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r, referenceVolumes=volume)
    # These coefficients do not enter the raw Naive Laplacian equation.
    state.A.requires_grad_(); R.renormalizationMatrices.requires_grad_()
    actual = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
        referenceVolumes=volume, crkState=sph.CRKState(state.A, state.B), renormalizationState=R)
    torch.testing.assert_close(actual, expected)
    omega.referenceOmegas.requires_grad_()
    actual = sph.warpOperation(q, replace(properties, operation=sph.WarpOperation.Covariance), domain,
        referenceParticles=r, referenceVolumes=volume, gradHState=omega)
    assert not actual.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_covariance_positive_weight_counts(device, dtype):
    q, r, domain = cloud(device, 1, dtype)
    q = replace(q, positions=q.positions[:1], supports=q.supports[:1], masses=q.masses[:1], kinds=q.kinds[:1])
    r = replace(r, positions=q.positions + .99, supports=q.supports.new_ones(1), masses=q.masses, kinds=q.kinds)
    q.supports[:] = 1
    state = sph.CRKState(q.supports.new_ones(1), q.positions.new_zeros((1, 1)),
                         q.positions.new_zeros((1, 1)), q.positions.new_zeros((1, 1, 1)))
    properties = sph.OperationProperties(sph.KernelFunctions.Gaussian, operation=sph.WarpOperation.Covariance,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    _, count = sph.warpOperation(q, properties, domain, referenceParticles=r,
        referenceVolumes=r.supports, crkState=state, covarianceReturnNumNeighbors=True)
    assert count.item() == (0 if dtype == torch.float32 else 1)
    state.A[:] = -1
    _, count = sph.warpOperation(q, properties, domain, referenceParticles=r,
        referenceVolumes=r.supports, crkState=state, covarianceReturnNumNeighbors=True)
    assert count.item() == 0


def test_consistent_divergence_with_explicit_volumes(device):
    q, r, domain, fi, fj = fixture(device, 2, 2)
    state, (_, volume), _, _ = correction_fixture(q, r)
    q, r = replace(q, masses=None), replace(r, masses=None)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Divergence, gradientMode=sph.GradientScheme.Difference)
    actual = sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
        referenceVolumes=volume, crkState=state, consistentDivergence=True)
    expected = remaining_oracle(q, r, domain, properties, fi, fj,
        referenceVolumes=volume, crkState=state, consistentDivergence=True)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match='densities'):
        sph.warpOperation(replace(q, densities=None), properties, domain, fi, fj,
            referenceParticles=r, referenceVolumes=volume, crkState=state, consistentDivergence=True)



def test_density_ignores_common_correction_bundle(device):
    q, r, domain, _, _ = fixture(device, 2, 0)
    state, (_, volume), omega, R = correction_fixture(q, r)
    state.A.requires_grad_(); omega.referenceOmegas.requires_grad_()
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    expected = sph.warpOperation(q, properties, domain, referenceParticles=r)
    actual = sph.warpOperation(q, properties, domain, referenceParticles=r,
        referenceVolumes=volume.requires_grad_(), crkState=state, gradHState=omega, renormalizationState=R)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_query_omega_ad_policy_and_scalar_divergence(device):
    q, r, domain, fi, fj = fixture(device, 1, 0)
    state, (_, volume), omega, _ = correction_fixture(q, r)
    omega.queryOmegas.requires_grad_()
    for operation in (sph.WarpOperation.Divergence, sph.WarpOperation.Curl, sph.WarpOperation.Laplacian):
        properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
                                             gradientMode=sph.GradientScheme.Difference)
        with pytest.raises(NotImplementedError, match='autograd'):
            sph.warpOperation(q, properties, domain, fi, fj, referenceParticles=r,
                referenceVolumes=volume, crkState=state, gradHState=omega)
    # Scalar 1D divergence is the ordinary scalar derivative; its existing
    # uncorrected reverse path should match the scalar gradient's pullback.
    properties = replace(properties, operation=sph.WarpOperation.Divergence)
    qpositions = q.positions.detach().requires_grad_()
    query = replace(q, positions=qpositions)
    field = fj.detach().requires_grad_()
    actual = sph.warpOperation(query, properties, domain, fi, field, referenceParticles=r)
    expected = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Gradient),
        domain, fi, field, referenceParticles=r).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), (qpositions, field), retain_graph=True),
                    torch.autograd.grad(expected.sum(), (qpositions, field))):
        torch.testing.assert_close(a, b)
