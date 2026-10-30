"""Independent dense/Krylov references and MINRES contract gates."""

import pytest
import torch

from warpSPHCore.linalg import (
    BlockJacobiPreconditioner, DiagonalPreconditioner, LinearOperator,
    NullspaceSystem, Status, minres,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def system(device, dtype, n=12):
    g = torch.Generator().manual_seed(173)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    eigen = torch.cat((-torch.linspace(.3, 3., n//2), torch.linspace(.5, 4., n-n//2))).double()
    A = (Q * eigen) @ Q.T
    A = (A + A.T) / 2
    b = torch.randn(n, generator=g, dtype=torch.float64)
    return A.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def certificate(result, A, b, tol):
    actual = (b - A @ result.x).norm()
    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
    assert result.statusInt() >= 0 and actual <= tol * b.norm()
    assert result.x.isfinite().all()
    torch.testing.assert_close(result.x, torch.linalg.solve(A.double(), b.double()).to(b.dtype), rtol=tol*10, atol=tol*10)


@pytest.mark.parametrize('preconditioning', ['none', 'diagonal', 'block'])
def test_indefinite_dense_systems(device, dtype, preconditioning):
    A, b = system(device, dtype)
    precond = None
    if preconditioning == 'diagonal':
        precond = DiagonalPreconditioner(torch.arange(1., b.numel()+1, device=device, dtype=dtype), requireSPD=True)
    elif preconditioning == 'block':
        blocks = torch.tensor([[[2., .2], [.2, 1.]]] * 6, device=device, dtype=dtype)
        precond = BlockJacobiPreconditioner(blocks, requireSPD=True)
    tol = 5e-5 if dtype == torch.float32 else 2e-11
    result = minres(lambda v: A @ v, b, symmetric=True, precond=precond, rtol=tol, maxiter=120)
    certificate(result, A, b, tol)
    assert result.loopIterations <= 120 and result.adjointMatvecs == 0


@pytest.mark.parametrize('k', [1, 2, 3, 5])
def test_prefix_iterates_minimize_over_an_independent_krylov_space(device, k):
    A, b = system(device, torch.float64, 8)
    columns, v = [], b
    for _ in range(k):
        columns.append(v)
        v = A @ v
    K = torch.stack(columns, dim=1)
    # Dense least squares is an independent oracle, never part of MINRES.
    AK = (A @ K).cpu()
    coefficients = torch.linalg.lstsq(AK, b.cpu()).solution.to(device)
    expected = K @ coefficients
    result = minres(lambda v: A @ v, b, symmetric=True, rtol=0, maxiter=k, checkEvery=1, maxRestarts=0)
    torch.testing.assert_close(result.x, expected, rtol=2e-10, atol=2e-10)
    assert result.iterations == k and result.loopIterations == k


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_preconditioner_scale_cannot_change_success_criterion(device, dtype, scale):
    A = torch.diag(torch.tensor([-2., 1., 3.], device=device, dtype=dtype))
    b = torch.ones(3, device=device, dtype=dtype)
    tol = 4e-5 if dtype == torch.float32 else 1e-11
    result = minres(lambda v: A @ v, b, symmetric=True, precond=torch.full_like(b, scale), rtol=tol, maxiter=48)
    certificate(result, A, b, tol)


def test_compatible_singular_subspace_and_projected_spd_inverse(device, dtype):
    A = torch.diag(torch.tensor([-3., 0., 1., 2.], device=device, dtype=dtype))
    Q = torch.tensor([[0.], [1.], [0.], [0.]], device=device, dtype=dtype)
    op = LinearOperator(A.shape, dtype, A.device, lambda v: A @ v, adjoint=lambda v: A.T @ v)
    helper = NullspaceSystem(op, Q, Q)
    b = torch.tensor([1., 0., -2., 3.], device=device, dtype=dtype)
    prepared = helper.prepareRhs(b)
    inverse = DiagonalPreconditioner(torch.tensor([3., 1., 1., 2.], device=device, dtype=dtype), requireSPD=True)
    projected = helper.projectPreconditioner(inverse.asLinearOperator())
    tol = 4e-5 if dtype == torch.float32 else 1e-11
    result = minres(helper, prepared.rhs, symmetric=True, precond=projected,
                    preconditionerSPD=True, rtol=tol, maxiter=48)
    assert result.statusInt() >= 0
    assert helper.verify(helper.gauge(result.x), prepared, rtol=tol*2).requestedConverged
    torch.testing.assert_close(result.x, (torch.linalg.pinv(A.double()) @ b.double()).to(dtype), rtol=tol*5, atol=tol*5)
    bad = b + Q[:, 0]
    failed = minres(op, bad, symmetric=True, rtol=tol, maxiter=32)
    assert failed.statusInt() < 0 and failed.trueResidual >= 1 and failed.x.isfinite().all()


def test_zero_rhs_exact_warm_starts_empty_and_last_lanczos_step(device, dtype):
    b = torch.tensor([1., 0., 0.], device=device, dtype=dtype)
    for value in (-2., 1.):
        result = minres(lambda v: value*v, b, symmetric=True, rtol=0, maxiter=1)
        assert result.statusInt() >= 0 and result.iterations == 1 and result.trueResidual == 0
        torch.testing.assert_close(result.x, b/value, rtol=0, atol=0)
    zero = minres(lambda v: v.clone(), b*0, b, symmetric=True, rtol=0, maxiter=16)
    assert zero.statusInt() >= 0 and zero.trueResidual == 0
    A = torch.diag(b)
    null = torch.tensor([0., 4., -2.], device=device, dtype=dtype)
    saved = null.clone()
    warm = minres(lambda v: A@v, b*0, null, symmetric=True)
    assert warm.statusInt() == 0 and warm.iterations == 0
    torch.testing.assert_close(warm.x, saved, rtol=0, atol=0)
    assert warm.x.data_ptr() != null.data_ptr()
    empty = minres(torch.clone, b[:0], symmetric=True)
    assert empty.statusInt() == 0 and empty.matvecs == 1
    exhausted = minres(torch.clone, b, symmetric=True, maxiter=0)
    assert exhausted.statusInt() == Status.BUDGET and exhausted.iterations == 0


def test_budget_counts_freeze_history_and_threshold_precedence(device, dtype):
    b = torch.tensor([1., 0., 0., 0.], device=device, dtype=dtype)
    calls = {'A': 0, 'P': 0}

    def action(v):
        calls['A'] += 1
        return 2*v

    def inverse(v):
        calls['P'] += 1
        return v.clone()

    result = minres(action, b, symmetric=True, precond=inverse, preconditionerSPD=True,
                    rtol=0, maxiter=5, checkEvery=8, history=False)
    assert result.statusInt() >= 0 and result.iterations == 1
    assert result.loopIterations == 5 and result.verifications == 2
    assert result.matvecs == calls['A'] == 7
    assert result.preconditionerApplications == calls['P'] == 6
    assert result.norms.numel() == result.valid.numel() == 0
    saved = result.x.clone()
    minres(action, b*2, symmetric=True)
    torch.testing.assert_close(result.x, saved, rtol=0, atol=0)
    for x0 in (None, b/2):
        bounded = minres(action, b, x0, symmetric=True, threshold=.1, dim=2, maxiter=8)
        assert bounded.statusInt() == Status.THRESHOLD
        assert bounded.x.norm() > .1


def test_nonfinite_and_invalid_inverse_preserve_finite_iterates(device, dtype):
    b = torch.ones(3, device=device, dtype=dtype)
    for value in (float('nan'), float('inf')):
        result = minres(lambda v: v*value, b, symmetric=True, maxiter=8)
        assert result.statusInt() < 0 and result.x.isfinite().all()
    for inverse in (lambda v: -v, torch.zeros_like, lambda v: v*float('nan')):
        result = minres(torch.clone, b, symmetric=True, precond=inverse, preconditionerSPD=True)
        assert result.statusInt() == Status.INDEFINITE and result.iterations == 0


def test_assumptions_metadata_and_options_are_explicit():
    b = torch.ones(2)
    for kw in ({}, {'symmetric': False}, {'symmetric': 1}, {'symmetric': True, 'history': 1},
               {'symmetric': True, 'precond': torch.clone}, {'symmetric': True, 'preconditionerSPD': 1},
               {'symmetric': True, 'checkEvery': 0}, {'symmetric': True, 'maxiter': -1},
               {'symmetric': True, 'rtol': float('nan')}):
        with pytest.raises(ValueError):
            minres(torch.clone, b, **kw)
    for diagonal in (torch.zeros_like(b), -b, b*float('inf')):
        with pytest.raises(ValueError, match='inverse diagonal'):
            minres(torch.clone, b, symmetric=True, precond=diagonal)
    rectangular = LinearOperator((2, 3), b.dtype, b.device, lambda v: v[:2])
    with pytest.raises(ValueError, match='operator shape'):
        minres(rectangular, b, symmetric=True)
    with pytest.raises(ValueError, match='precond output'):
        minres(torch.clone, b, symmetric=True, precond=lambda v: v[:, None], preconditionerSPD=True)


def test_residual_drift_cannot_certify_an_ill_conditioned_solution(device):
    A, b = system(device, torch.float32, 20)
    Q, _ = torch.linalg.qr(A.double())
    values = torch.logspace(0, 5, b.numel(), device=device, dtype=torch.float64)
    values[::2] *= -1
    A = ((Q * values) @ Q.T).float()
    result = minres(lambda v: A @ v, b, symmetric=True, rtol=1e-6, maxiter=300, maxRestarts=0)
    actual = (b-A@result.x).norm()
    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
    assert result.statusInt() < 0 or actual <= 1e-6*b.norm()


def test_cuda_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        A, b = system('cuda', torch.float64)
        result = minres(lambda v: A@v, b, symmetric=True, rtol=1e-11, maxiter=80)
        certificate(result, A, b, 1e-11)
