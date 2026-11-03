"""Independent step/search references, finite best iterates and original residuals."""

import pytest
import torch

from warpSPHCore.linalg import LinearOperator, Status, richardson


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('fixed', [False, True])
def test_both_definiteness_signs_and_dense_reference(device, dtype, sign, fixed):
    A = sign * torch.diag(torch.tensor([1., 2., 3.], device=device, dtype=dtype))
    b = torch.tensor([1., -2., 3.], device=device, dtype=dtype)
    tol = 3e-5 if dtype == torch.float32 else 1e-11
    result = richardson(lambda v: A@v, b, omega=.4*sign if fixed else None,
                        rtol=tol, maxiter=200)
    assert result.statusInt() >= 0 and result.trueResidual <= tol*b.norm()
    actual = (b-A@result.x).norm()
    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
    torch.testing.assert_close(result.x, torch.linalg.solve(A, b), rtol=tol*5, atol=tol*5)
    assert result.iterations <= 200 and result.loopIterations == result.iterations


def test_first_adaptive_step_is_the_independent_residual_minimizer(device, dtype):
    A = torch.tensor([[2., .3], [-.2, 4.]], device=device, dtype=dtype)
    b = torch.tensor([1., -2.], device=device, dtype=dtype)
    x0 = torch.tensor([.2, .4], device=device, dtype=dtype)
    r = b-A@x0
    ar = A@r
    alpha = torch.dot(r.double(), ar.double()) / torch.dot(ar.double(), ar.double())
    expected = x0.double() + alpha*r.double()
    result = richardson(lambda v: A@v, b, x0, rtol=0, maxiter=1)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(result.x, expected.to(dtype), rtol=tolerance, atol=tolerance)
    assert result.statusInt() == Status.BUDGET and result.iterations == 1
    assert result.trialSteps == 1 and result.matvecs == 4 and result.verifications == 3


@pytest.mark.parametrize('scale', [1e-30, 1e30])
def test_adaptive_step_normalization_handles_small_and_large_actions(device, dtype, scale):
    b = torch.tensor([1., 0.], device=device, dtype=dtype)
    result = richardson(lambda v: scale*v, b, rtol=1e-5, maxiter=4)
    assert result.statusInt() >= 0 and result.trueResidual <= 1e-5
    torch.testing.assert_close(result.x, b/scale, rtol=2e-5, atol=0)


def test_backtracking_restarts_from_same_accepted_iterate_and_obeys_budget(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    result = richardson(lambda v: 2*v, b, omega=4, rtol=0, maxiter=1, maxBacktracks=3)
    assert result.statusInt() >= 0 and result.trueResidual == 0
    torch.testing.assert_close(result.x, b/2, rtol=0, atol=0)
    assert result.trialSteps == 4 and result.backtracks == 3
    assert result.iterations == 1 and result.stepSize == .5
    assert result.matvecs == result.verifications == 6
    failed = richardson(lambda v: 2*v, b, omega=4, rtol=0, maxiter=20, maxBacktracks=2)
    assert failed.statusInt() == Status.RICHARDSON_STAGNATION
    assert failed.iterations == 0 and failed.trialSteps == 3 and failed.backtracks == 2
    assert failed.trueResidual == b.norm() and failed.x.count_nonzero() == 0


def test_fixed_divergence_and_stagnation_return_best_finite_iterate(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    result = richardson(lambda v: 2*v, b, omega=2, backtracking=False,
                        stagnationWindow=0, maxiter=3, rtol=0)
    assert result.statusInt() == Status.BUDGET and result.iterations == 3
    assert result.bestIteration == 0 and result.x.count_nonzero() == 0
    assert result.trueResidual == b.norm()
    torch.testing.assert_close(result.norms/b.norm(), b.new_tensor([1., 3., 9., 27.]))
    stalled = richardson(lambda v: 2*v, b, omega=2, backtracking=False,
                         stagnationWindow=2, maxiter=20)
    assert stalled.statusInt() == Status.RICHARDSON_STAGNATION and stalled.iterations == 2
    assert stalled.x.count_nonzero() == 0


def test_incompatible_and_skew_systems_stagnate_without_false_success(device, dtype):
    b = torch.tensor([1., 1.], device=device, dtype=dtype)
    for A in (torch.zeros((2, 2), device=device, dtype=dtype),
              torch.tensor([[0., -1.], [1., 0.]], device=device, dtype=dtype),
              torch.diag(torch.tensor([-1., 1.], device=device, dtype=dtype))):
        result = richardson(lambda v: A@v, b, maxiter=20)
        assert result.statusInt() == Status.RICHARDSON_STAGNATION
        assert result.x.isfinite().all() and result.trueResidual == b.norm()


def test_empty_zero_warm_start_and_zero_budget(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    zero = richardson(torch.clone, b*0, b, omega=1, rtol=0, maxiter=4)
    assert zero.statusInt() >= 0 and zero.trueResidual == 0
    A = torch.diag(torch.tensor([1., 0.], device=device, dtype=dtype))
    null = torch.tensor([0., 3.], device=device, dtype=dtype)
    warm = richardson(lambda v: A@v, b*0, null)
    assert warm.statusInt() == 0 and warm.iterations == 0 and warm.matvecs == 1
    torch.testing.assert_close(warm.x, null, rtol=0, atol=0)
    assert warm.x.data_ptr() != null.data_ptr()
    budget = richardson(torch.clone, b, maxiter=0)
    assert budget.statusInt() == Status.BUDGET and budget.matvecs == 1 and budget.trialSteps == 0
    empty = richardson(torch.clone, b[:0])
    assert empty.statusInt() == 0 and empty.iterations == 0


def test_thresholds_apply_to_initial_and_accepted_candidates(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    for x0 in (None, b/2):
        result = richardson(lambda v: 2*v, b, x0, threshold=.1, dim=2)
        assert result.statusInt() == Status.THRESHOLD and result.x.norm() > .1
    # Trial x=4b exceeds the bound but is rejected; the accepted .5b is valid.
    safe = richardson(lambda v: 2*v, b, omega=4, threshold=1., dim=2, maxiter=1)
    assert safe.statusInt() >= 0 and safe.trueResidual == 0


def test_explicit_step_bounds_and_nonfinite_search(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    clipped = richardson(lambda v: 2*v, b, maxStep=.1, maxiter=1, rtol=0)
    assert clipped.stepSize == b.new_tensor(.1)
    torch.testing.assert_close(clipped.x, b*.1)
    small = richardson(lambda v: 2*v, b, minStep=1, maxiter=4)
    assert small.statusInt() == Status.RICHARDSON_STAGNATION and small.trialSteps == 0
    for bad in (float('inf'), float('nan')):
        result = richardson(lambda v: v*bad, b, maxiter=5)
        assert result.statusInt() < 0 and result.x.isfinite().all()
    # Finite initial evaluation, overflow on every available finite candidate.
    def extreme(v):
        return v * torch.finfo(dtype).max
    overflow = richardson(extreme, b, omega=8, maxBacktracks=1, maxiter=5)
    assert overflow.statusInt() == Status.DENOMINATOR_BREAKDOWN
    assert overflow.x.count_nonzero() == 0 and overflow.trialSteps == 2


def test_history_ownership_and_stamped_failure_residual_for_fallback_selection(device, dtype):
    b = torch.ones(2, device=device, dtype=dtype)
    result = richardson(lambda v: 2*v, b, history=False)
    assert result.norms.numel() == result.valid.numel() == 0
    assert result.convergence()[-1] == result.trueResidual
    saved = result.x.clone()
    richardson(lambda v: 2*v, b*3)
    torch.testing.assert_close(result.x, saved, rtol=0, atol=0)
    failed = richardson(lambda v: 2*v, b, omega=2, backtracking=False, maxiter=3)
    assert failed.convergence()[-1] == (b-2*failed.x).norm()
    assert failed.trueResidual == b.norm()  # owning frontend can retain its better Krylov iterate
    warm = b*.2
    saved = warm.clone()
    polish = richardson(lambda v: 2*v, b, warm, omega=2, backtracking=False, maxiter=3)
    torch.testing.assert_close(polish.x, saved, rtol=0, atol=0)
    torch.testing.assert_close(warm, saved, rtol=0, atol=0)
    assert polish.x.data_ptr() != warm.data_ptr()
    assert polish.stampedResidual == (b-2*warm).norm()


@pytest.mark.parametrize('kw', [
    {'omega': 0}, {'omega': float('nan')}, {'maxBacktracks': -1}, {'minStep': -1},
    {'maxStep': 0}, {'minStep': 2, 'maxStep': 1}, {'omega': 2, 'maxStep': 1},
    {'backtracking': 1}, {'history': 1}, {'stagnationWindow': -1}, {'rtol': -1},
])
def test_invalid_options(kw):
    with pytest.raises(ValueError):
        richardson(torch.clone, torch.ones(2), **kw)


def test_operator_metadata_and_callback_validation():
    b = torch.ones(2)
    op = LinearOperator((2, 3), b.dtype, b.device, lambda v: v[:2])
    with pytest.raises(ValueError, match='operator shape'):
        richardson(op, b)
    with pytest.raises(ValueError, match='matvec output'):
        richardson(lambda v: v.double(), b)


def test_cuda_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        b = torch.ones(3, device='cuda', dtype=torch.float64)
        result = richardson(lambda v: 2*v, b, omega=.5, rtol=0)
        assert result.statusInt() >= 0 and result.trueResidual == 0
