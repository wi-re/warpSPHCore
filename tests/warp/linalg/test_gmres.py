"""GMRES accuracy/failure gates, independent of its residual estimates."""

import pytest
import torch

from warpSPHCore.linalg import LinearOperator, Status, gmres


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def system(device, dtype, n=24):
    g = torch.Generator().manual_seed(31)
    A = torch.randn(n, n, generator=g, dtype=torch.float64) * .08 + torch.diag(torch.linspace(1., 3., n))
    b = torch.randn(n, generator=g, dtype=torch.float64)
    return A.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def certificate(result, A, b, rtol):
    actual = (b - A @ result.x).norm()
    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
    assert result.x.isfinite().all()
    assert result.statusInt() >= 0
    assert actual <= rtol * b.norm()
    reference = torch.linalg.solve(A.double(), b.double()).to(b.dtype)
    torch.testing.assert_close(result.x, reference, rtol=rtol * 5, atol=rtol * 5)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('restart', [4, 40])
@pytest.mark.parametrize('checkEvery', [1, 8])
def test_nonsymmetric_reference(device, dtype, side, restart, checkEvery):
    A, b = system(device, dtype)
    rtol = 2e-5 if dtype == torch.float32 else 1e-11
    # Nonsymmetric inverse action; no SPD/transpose assumption is permissible.
    P = torch.diag(1 / A.diag())
    P[0, 1] = .03
    x0 = b * .03
    saved = x0.clone()
    result = gmres(lambda v: A @ v, b, x0, precond=lambda v: P @ v,
                   preconditionerSide=side, restart=restart, checkEvery=checkEvery,
                   rtol=rtol, maxiter=200)
    certificate(result, A, b, rtol)
    assert result.loopIterations <= 200
    assert result.restarts > 0 if restart == 4 else result.restarts == 0
    torch.testing.assert_close(x0, saved, rtol=0, atol=0)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_scaled_preconditioner_cannot_certify_false_convergence(device, dtype, side, scale):
    A, b = system(device, dtype)
    rtol = 2e-5 if dtype == torch.float32 else 1e-11
    result = gmres(lambda v: A @ v, b, precond=torch.full_like(b, scale),
                   preconditionerSide=side, rtol=rtol, restart=12, maxiter=200)
    certificate(result, A, b, rtol)


def test_true_qr_residual_not_arnoldi_subdiagonal(device, dtype):
    # Arnoldi breaks down in one dimension for every scalar matrix, even if a
    # zero operator cannot possibly solve b != 0. That norm is not a residual.
    b = torch.ones(1, device=device, dtype=dtype)
    result = gmres(lambda v: 2 * v, b, rtol=0, maxiter=1)
    assert result.statusInt() >= 0 and result.trueResidual == 0
    assert result.x.item() == .5
    failed = gmres(torch.zeros_like, b, maxiter=7)
    assert failed.statusInt() == Status.DENOMINATOR_BREAKDOWN
    assert failed.trueResidual == 1 and failed.x == 0


def test_compatible_and_incompatible_singular_systems(device, dtype):
    A = torch.diag(torch.tensor([1., 2., 0.], device=device, dtype=dtype))
    b = torch.tensor([1., 1., 0.], device=device, dtype=dtype)
    x0 = torch.tensor([0., 0., 7.], device=device, dtype=dtype)
    result = gmres(lambda v: A @ v, b, x0, maxiter=12)
    assert result.statusInt() >= 0
    torch.testing.assert_close(result.x, torch.tensor([1., .5, 7.], device=device, dtype=dtype))
    impossible = gmres(lambda v: A @ v, torch.ones_like(b), maxiter=12)
    assert impossible.statusInt() < 0 and impossible.trueResidual >= 1
    assert impossible.x.isfinite().all()


def test_warm_start_zero_rhs_and_empty_system(device, dtype):
    b = torch.zeros(3, device=device, dtype=dtype)
    result = gmres(lambda v: v.clone(), b, torch.ones_like(b), atol=1e-6, maxiter=12)
    assert result.statusInt() >= 0 and result.trueResidual <= 1e-6
    empty = gmres(lambda v: v.clone(), b[:0])
    assert empty.statusInt() == 0 and empty.iterations == 0 and empty.matvecs == 1
    x0 = torch.tensor([0., 0., 7.], device=device, dtype=dtype)
    A = torch.diag(torch.tensor([1., 2., 0.], device=device, dtype=dtype))
    null = gmres(lambda v: A @ v, b, x0)
    assert null.statusInt() == 0 and null.iterations == 0
    assert torch.equal(null.x, x0) and null.x.data_ptr() != x0.data_ptr()


@pytest.mark.parametrize('warm', [False, True])
def test_threshold_precedes_convergence_at_each_candidate(device, dtype, warm):
    b = torch.ones(2, device=device, dtype=dtype)
    result = gmres(lambda v: 2 * v, b, b / 2 if warm else None,
                   threshold=.1, dim=2, checkEvery=8, maxiter=8)
    assert result.statusInt() == Status.THRESHOLD
    assert int(result.iterations) == (0 if warm else 1)
    assert result.x.norm() > .1


def test_strict_budget_and_actual_callback_counts(device, dtype):
    A, b = system(device, dtype)
    calls = dict(A=0, P=0)

    def action(v):
        calls['A'] += 1
        return A @ v

    def precond(v):
        calls['P'] += 1
        return v.clone()

    result = gmres(action, b, precond=precond, restart=4, maxiter=7, rtol=0, checkEvery=8)
    assert result.statusInt() == Status.BUDGET
    assert result.loopIterations == 7 and result.iterations == 7
    assert result.matvecs == calls['A'] == 10
    assert result.preconditionerApplications == calls['P'] == 7
    assert result.verifications == 3 and result.restarts == 1
    initial = gmres(action, b, maxiter=0)
    assert initial.statusInt() == Status.BUDGET and initial.matvecs == 1
    assert initial.loopIterations == 0 and initial.trueResidual == b.norm()


def test_frozen_columns_do_not_change_the_answer(device, dtype):
    b = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.], device=device, dtype=dtype)
    result = gmres(lambda v: 2 * v, b, rtol=0, checkEvery=8, maxiter=8)
    assert result.statusInt() >= 0 and result.trueResidual == 0
    assert result.iterations == 1 and result.loopIterations == 8
    assert result.valid.sum() == 2  # initial norm and one real Arnoldi step


@pytest.mark.parametrize('side', ['left', 'right'])
def test_degenerate_preconditioner_and_nonfinite_actions(device, dtype, side):
    b = torch.ones(3, device=device, dtype=dtype)
    zero = gmres(lambda v: v.clone(), b, precond=torch.zeros_like(b), preconditionerSide=side)
    assert zero.statusInt() < 0 and zero.trueResidual > 0 and zero.x.isfinite().all()
    bad = gmres(lambda v: v * float('nan'), b, preconditionerSide=side)
    assert bad.statusInt() < 0 and bad.x.isfinite().all()
    overflow = gmres(lambda v: v * 1e-38, b, precond=torch.full_like(b, 1e-38),
                     preconditionerSide=side, maxiter=8)
    assert overflow.statusInt() < 0 or overflow.trueResidual <= 1e-5 * b.norm()
    assert overflow.x.isfinite().all()


def test_stagnation_is_explicit_and_can_be_disabled(device, dtype):
    # GMRES(1) cannot progress on a skew-symmetric rotation: r is orthogonal Ar.
    A = torch.tensor([[0., -1.], [1., 0.]], device=device, dtype=dtype)
    b = torch.tensor([1., 0.], device=device, dtype=dtype)
    stopped = gmres(lambda v: A @ v, b, restart=1, maxiter=5)
    assert stopped.statusInt() == Status.STAGNATION and stopped.loopIterations == 1
    budget = gmres(lambda v: A @ v, b, restart=1, maxiter=5, stagnationTol=0)
    assert budget.statusInt() == Status.BUDGET and budget.loopIterations == 5
    certificate(gmres(lambda v: A @ v, b, restart=2, maxiter=2), A, b, 1e-5)


def test_ill_conditioned_success_is_always_verified(device):
    g = torch.Generator().manual_seed(41)
    U, _ = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64, generator=g))
    V, _ = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64, generator=g))
    A = ((U * torch.logspace(0, 5, 24, dtype=torch.float64)) @ V.T).float().to(device)
    b = torch.randn(24, generator=g).to(device)
    for accumulation in (torch.float32, torch.float64):
        for repairs in (0, 2):
            result = gmres(lambda v: A @ v, b, rtol=1e-6, maxiter=120,
                           accumulationDtype=accumulation, maxRestarts=repairs, stagnationTol=0)
            actual = (b - A @ result.x).norm()
            assert result.statusInt() < 0 or actual <= 1e-6 * b.norm()
            torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
            assert result.loopIterations <= 120 and result.x.isfinite().all()


def test_left_estimate_mismatch_has_bounded_repair(device):
    A = torch.diag(torch.tensor([1., 2.], device=device, dtype=torch.float64))
    b = torch.ones(2, device=device, dtype=A.dtype)
    P = torch.tensor([1e-12, 1.], device=device, dtype=A.dtype)
    kw = dict(precond=P, preconditionerSide='left', checkEvery=1, rtol=1e-8, maxiter=12)
    first = gmres(lambda v: A @ v, b, maxRestarts=0, **kw)
    assert first.statusInt() == Status.RESIDUAL_MISMATCH
    assert first.trueResidual > .9 and first.restarts == 0
    result = gmres(lambda v: A @ v, b, maxRestarts=2, **kw)
    certificate(result, A, b, 1e-8)
    assert result.restarts == 1 and result.verifications == 3


def test_nonnormal_system_requires_the_full_krylov_space(device, dtype):
    n = 24
    A = torch.eye(n, dtype=dtype, device=device) + torch.diag(torch.full((n - 1,), .99, dtype=dtype, device=device), 1)
    b = torch.zeros(n, dtype=dtype, device=device)
    b[-1] = 1
    rtol = 1e-5 if dtype == torch.float32 else 1e-11
    result = gmres(lambda v: A @ v, b, restart=n, maxiter=n, rtol=rtol)
    certificate(result, A, b, rtol)
    assert result.iterations == n


def test_mixed_precision_and_history_ownership(device):
    A, b = system(device, torch.float32)
    result = gmres(lambda v: A @ v, b, accumulationDtype=torch.float64, history=False)
    certificate(result, A, b, 1e-5)
    assert result.norms.numel() == result.valid.numel() == 0
    assert len(result.convergence()) == 1
    saved = result.x.clone()
    gmres(lambda v: A @ v, b * 2)
    torch.testing.assert_close(result.x, saved, rtol=0, atol=0)


@pytest.mark.parametrize('kw', [
    {'restart': 0}, {'restart': 1.5}, {'restart': True}, {'checkEvery': 0},
    {'maxiter': -1}, {'maxRestarts': -1}, {'rtol': -1}, {'atol': float('nan')},
    {'stagnationTol': -1}, {'stagnationTol': 1}, {'stagnationTol': float('inf')},
    {'preconditionerSide': 'both'}, {'dim': 3}, {'threshold': -1},
    {'accumulationDtype': torch.float16}, {'history': 1}, {'precond': torch.ones(2, 1)},
])
def test_invalid_options(kw):
    with pytest.raises(ValueError):
        gmres(lambda v: v.clone(), torch.ones(2), **kw)


def test_callbacks_and_operator_metadata_are_checked():
    b = torch.ones(2)
    with pytest.raises(ValueError, match='matvec output'):
        gmres(lambda v: v.double(), b)
    with pytest.raises(ValueError, match='precond output'):
        gmres(lambda v: v.clone(), b, precond=lambda v: v[:, None])
    op = LinearOperator((2, 3), b.dtype, b.device, lambda v: v[:2].clone())
    with pytest.raises(ValueError, match='operator shape'):
        gmres(op, b)


def test_supplied_output_operator_and_adjoint(device, dtype):
    A, b = system(device, dtype)

    def applyOut(v, out):
        torch.mv(A, v, out=out)

    op = LinearOperator(A.shape, dtype, device, matvecOut=applyOut, adjoint=lambda v: A.T @ v)
    certificate(gmres(op, b), A, b, 1e-5)
    out = torch.empty_like(b)
    assert op.apply(b, out) is out
    torch.testing.assert_close(out, A @ b)
    torch.testing.assert_close(torch.dot(op(b), b), torch.dot(b, op.applyAdjoint(b)))
    with pytest.raises(ValueError, match='overlap'):
        op.apply(b, b)


def test_operator_out_callback_contract():
    b = torch.ones(2)
    wrong = LinearOperator((2, 2), b.dtype, b.device, matvecOut=lambda v, out: v.clone())
    with pytest.raises(ValueError, match='supplied output'):
        wrong(b)

    def replace(v, out):
        out.set_(torch.ones(3))

    changed = LinearOperator((2, 2), b.dtype, b.device, matvecOut=replace)
    with pytest.raises(ValueError, match='metadata'):
        changed(b)
    plain = LinearOperator((2, 2), b.dtype, b.device, lambda v: 2 * v)
    out = torch.empty_like(b)
    torch.testing.assert_close(plain.apply(b, out), 2 * b)
    with pytest.raises(ValueError, match='no adjoint'):
        plain.applyAdjoint(b)


def test_cuda_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        A, b = system('cuda', torch.float64)
        result = gmres(lambda v: A @ v, b, rtol=1e-10)
        certificate(result, A, b, 1e-10)
