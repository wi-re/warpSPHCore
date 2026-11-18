"""Six-output raw moments and their explicit product reverse against Warp."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.crk_moments_contract import case, flat


@pytest.fixture(scope='module', autouse=True)
def reference_runtime(): wp.init()


def compare(values, functions, explicit, tolerance, value_bound=None, reverse_bound=None):
    inputs = [x for x in values if x.requires_grad]
    for weighted in (False, True):
        actual, expected = functions('native')(*values), functions('warp')(*values)
        for k, (a, b) in enumerate(zip(actual[:6], expected[:6])):
            if value_bound is None:
                torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance, equal_nan=True)
            else:
                finite = a.isfinite() & b.isfinite()
                torch.testing.assert_close(a[~finite], b[~finite], rtol=0, atol=0, equal_nan=True)
                assert bool(((a-b).abs()[finite] <= (value_bound[k] + tolerance*(1+b.abs()))[finite]).all())
        if not explicit:
            torch.testing.assert_close(actual[-1], expected[-1]); assert not actual[-1].requires_grad
        seeds = tuple(torch.linspace(-.3, .7, x.numel(), device=x.device, dtype=x.dtype).reshape_as(x)
                      if weighted else torch.ones_like(x) for x in actual[:6])
        bars = torch.autograd.grad(actual[:6], inputs, seeds, allow_unused=True)
        wanted = torch.autograd.grad(expected[:6], inputs, seeds, allow_unused=True)
        budgets = {} if reverse_bound is None else reverse_bound(seeds)
        for k, (x, a, b) in enumerate(zip(inputs, bars, wanted)):
            slot = next(i for i, value in enumerate(values) if value is x)
            a, b = torch.zeros_like(x) if a is None else a, torch.zeros_like(x) if b is None else b
            if slot not in budgets:
                torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance, equal_nan=True, msg=lambda msg: f'moment role {k}: {msg}')
            else:
                finite = a.isfinite() & b.isfinite()
                torch.testing.assert_close(a[~finite], b[~finite], rtol=0, atol=0, equal_nan=True)
                assert bool(((a-b).abs()[finite] <= (budgets[slot] + tolerance*(1+b.abs()))[finite]).all()), f'moment role {k} exceeds expanded path bound'


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('explicit', [False, True])
def test_all_kernel_values_scalar_and_weighted_reverse(device, kernel, dimension, explicit):
    *_, values, states, functions = case(device, dimension, get_torch_precision(), kernel=kernel, explicit=explicit)
    compare(values, functions, explicit, 9e-3 if values[0].dtype == torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('explicit', [False, True])
def test_warp_all_kernel_actual_gradcheck(device, kernel, explicit):
    if get_torch_precision() != torch.float64: pytest.skip('actual finite differences require FP64')
    i = list(sph.KernelFunctions).index(kernel)
    *_, values, states, functions = case(device, 1 + i % 3, kernel=kernel, explicit=explicit)
    assert torch.autograd.gradcheck(lambda *x: functions('warp')(*x)[:6], values, fast_mode=True, atol=8e-5, rtol=2e-3,
                                   nondet_tol=1e-12 if device == 'cuda' else 0.)


@pytest.mark.parametrize('support', list(sph.SupportScheme))
@pytest.mark.parametrize('direction', list(sph.OperationDirection))
def test_ordinary_support_and_direction_contracts(device, support, direction):
    q, r, d, p, g, values, states, functions = case(device, dtype=get_torch_precision(), support=support)
    p = replace(p, operationMode=direction)
    compare(values, lambda backend: functions(backend, p), False, 9e-3 if values[0].dtype == torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('separation', ['self', 'ultratiny', 'clamped', 'epsilon', 'near', 'cutoff', 'outside', 'interior'])
def test_raw_boundary_values_and_reverse(device, kernel, dimension, explicit, separation):
    q, r, d, p, g, values, states, functions = case(device, dimension, get_torch_precision(), kernel=kernel, explicit=explicit)
    eps = 1e-7 if values[0].dtype == torch.float32 else 1e-15
    distance = {'self': 0., 'ultratiny': eps*1e-9, 'clamped': eps*.85, 'epsilon': eps,
                'near': eps*10, 'cutoff': .8, 'outside': .81, 'interior': .56}[separation]
    d.periodic.zero_(); q.kinds.zero_(); r.kinds.zero_()
    g.paddedJ = torch.arange(2, device=device, dtype=sph.get_torch_index_dtype())
    g.edgeOffsets = torch.arange(2, device=device, dtype=sph.get_torch_offset_dtype())
    g.numNeighbors = torch.ones(2, device=device, dtype=torch.int32)
    values = list(values); values[0] = torch.zeros_like(values[0]); values[0][:, 0] = distance
    values[1] = torch.zeros_like(values[1]); values[2] = torch.full_like(values[2], .8); values[3] = torch.full_like(values[3], .8)
    values = tuple(x.detach().requires_grad_(k not in (5, 6) or not explicit) for k, x in enumerate(values))
    bound = reverse_bound = None
    near_pole = separation in ('epsilon', 'near') or (separation == 'clamped' and not explicit)
    if kernel == sph.KernelFunctions.ViscosityKernel and near_pole:
        from _support.crk_moments_roundoff import viscosity_path_bounds, viscosity_decimal_values
        bound = viscosity_path_bounds(values, kernel, explicit=explicit)
        reverse_bound = None if explicit else lambda seeds: viscosity_path_bounds(values, kernel, seeds)
        decimal = viscosity_decimal_values(values, kernel, explicit)
        for backend in ('native', 'warp'):
            for a, b, budget in zip(functions(backend)(*values)[:6], decimal, bound):
                assert bool(((a-b).abs() <= budget).all()), 'moment value exceeds 90-digit oracle path bound'
    compare(values, functions, explicit, 2e-2 if values[0].dtype == torch.float32 else 3e-6, bound, reverse_bound)


@pytest.mark.parametrize('mode', ['query', 'none'])
def test_query_volume_fallback_and_absent_volume_counts(device, mode):
    *_, values, states, functions = case(device, dtype=get_torch_precision(), volume_mode=mode)
    compare(values, functions, False, 9e-3 if values[0].dtype == torch.float32 else 3e-7)


@pytest.mark.parametrize('positive', [False, True])
def test_calibration_and_unused_filter_flag(device, positive):
    *_, p, g, values, states, functions = case(device, dtype=get_torch_precision())
    p = replace(p, calibrateNormalization=True, n_h=1.8, positiveDivergence=positive)
    compare(values, lambda backend: functions(backend, p), False, 9e-3 if values[0].dtype == torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_full_rank_factor_products_scalar_and_weighted_reverse(device, kernel, dimension):
    from _support.crk_factors_contract import case as factor_case
    values, functions = factor_case(device, dimension, get_torch_precision(), kernel)
    tolerance = 2e-2 if values[0].dtype == torch.float32 else 2e-6
    for weighted in (False, True):
        actual, expected = functions('native')(*values), functions('warp')(*values)
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance, equal_nan=True)
        seeds = tuple(torch.linspace(-.3, .7, x.numel(), device=device, dtype=x.dtype).reshape_as(x)
                      if weighted else torch.ones_like(x) for x in actual)
        bars = torch.autograd.grad(actual, values, seeds, allow_unused=True)
        wanted = torch.autograd.grad(expected, values, seeds, allow_unused=True)
        for a, b in zip(bars, wanted): torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance, equal_nan=True)


def test_grid_and_ordinary_reference_ad_scope(device):
    from warpSPHCore.radiusSearch import buildCompactHashMap
    q, r, d, p, g, v, states, functions = case(device, dtype=get_torch_precision())
    q, r, d = states(v[:10])
    grid = buildCompactHashMap(q.positions, r.positions, q.supports, r.supports, d.periodic, d, p.supportMode)
    compare(v, lambda backend: functions(backend, adjacency=grid), False, 9e-3 if v[0].dtype == torch.float32 else 3e-7)
    for backend in ('native', 'warp'):
        outputs = functions(backend)(*v)[:6]
        bars = torch.autograd.grad(outputs, v, tuple(torch.ones_like(x) for x in outputs), create_graph=True, allow_unused=True)
        assert not any(x is not None and x.requires_grad for x in bars)
    with pytest.raises(NotImplementedError, match='no JVPSpec'):
        with torch.autograd.forward_ad.dual_level():
            dual = [x.detach() for x in v]
            dual[0] = torch.autograd.forward_ad.make_dual(dual[0], torch.ones_like(dual[0]))
            functions('warp')(*dual)


@pytest.mark.parametrize('role', [0, 1, 10, 12, 16])
def test_explicit_reference_second_forward_boundary(device, role):
    *_, v, states, functions = case(device, dtype=get_torch_precision(), explicit=True)
    for backend in ('native', 'warp'):
        with torch.autograd.forward_ad.dual_level():
            dual = [x.detach() for x in v]
            dual[role] = torch.autograd.forward_ad.make_dual(dual[role], torch.ones_like(dual[role]))
            with pytest.raises((NotImplementedError, RuntimeError)): functions(backend)(*dual)


@pytest.mark.parametrize('role', [5, 6])
def test_explicit_frozen_bounds_and_first_reverse_boundary(device, role):
    *_, v, states, functions = case(device, dtype=get_torch_precision(), explicit=True)
    for backend in ('native', 'warp'):
        only = [x.detach().requires_grad_(k == role) for k,x in enumerate(v)]
        assert not any(x.requires_grad for x in functions(backend)(*only))
        outputs = functions(backend)(*v)
        bars = torch.autograd.grad(outputs, [x for x in v if x.requires_grad], tuple(torch.ones_like(x) for x in outputs), create_graph=True, allow_unused=True)
        assert not any(x is not None and x.requires_grad for x in bars)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_reference_factor_pipeline_actual_gradcheck(device, kernel):
    if get_torch_precision() != torch.float64: pytest.skip('actual finite differences require FP64')
    from _support.crk_factors_contract import case as factor_case
    dimension = 1 + list(sph.KernelFunctions).index(kernel)%3
    values, functions = factor_case(device, dimension, kernel=kernel)
    assert torch.autograd.gradcheck(functions('warp'), values, fast_mode=True, atol=2e-4, rtol=3e-3,
                                   nondet_tol=1e-12 if device == 'cuda' else 0.)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_volume_only_cutoff_matches_reference_linear_moment_map(device, kernel):
    # With absent geometry directions the product is exactly the ordinary
    # moment map applied to dV. A literal geometry helper carrying zeros can
    # instead form unused 0*Inf terms at a singular cutoff.
    dimension = 1 + list(sph.KernelFunctions).index(kernel)%3
    q, r, d, p, g, v, states, functions = case(device, dimension, get_torch_precision(), kernel=kernel, explicit=True, geometry=False)
    d.periodic.zero_(); g.paddedJ = torch.arange(2, device=device, dtype=sph.get_torch_index_dtype())
    g.edgeOffsets = torch.arange(2, device=device, dtype=sph.get_torch_offset_dtype())
    g.numNeighbors = torch.ones(2, device=device, dtype=torch.int32)
    v = list(v); v[0] = torch.zeros_like(v[0]); v[0][:,0] = .8
    v[1] = torch.zeros_like(v[1]); v[2] = torch.full_like(v[2], .8); v[3] = torch.full_like(v[3], .8)
    v = tuple(x.detach().requires_grad_(k not in (5,6)) for k,x in enumerate(v))
    def linear(backend):
        if backend == 'native': return functions('native')
        from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
        def run(*x):
            qi, rj, domain = states(x[:10])
            return _computeCRKMoments_stateBackend(qi,p,domain,None,x[16],g,rj)[:6]
        return run
    compare(v, linear, True, 2e-2 if v[0].dtype == torch.float32 else 3e-6)
