"""Independent residual, edge-case and diagnostics contracts for shared solvers."""

import pytest
import torch

from warpSPHCore.linalg import BiCGStabWorkspace, CGWorkspace, Status, bicg, bicgstab, cg


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return request.param


@pytest.fixture(params=['cg', 'cg_native', 'bicg', 'bicgstab', 'bicgstab_native'])
def solve(request):
    if request.param == 'bicgstab_native':
        def native(A, b, x0=None, **kw):
            return bicgstab(lambda v: A @ v, b, x0,
                           workspace=BiCGStabWorkspace(b, kw.get('maxiter', b.numel()*10)), **kw)
        return native
    if request.param == 'cg_native':
        def native(A, b, x0=None, **kw):
            capacity = kw.get('maxiter', b.numel() * 10)
            workspace = CGWorkspace(b, capacity)
            return cg(lambda v: A @ v, b, x0, workspace=workspace, **kw)
        return native
    if request.param == 'bicg':
        return lambda A, b, x0=None, **kw: bicg(lambda v: A @ v, lambda v: A.T @ v, b, x0, **kw)
    fn = {'cg': cg, 'bicgstab': bicgstab}[request.param]
    return lambda A, b, x0=None, **kw: fn(lambda v: A @ v, b, x0, **kw)


def test_zero_rhs_checks_warm_start(solve, device):
    A = torch.eye(3, device=device)
    b, x0 = torch.zeros(3, device=device), torch.ones(3, device=device)
    result = solve(A, b, x0)
    assert result.statusInt() >= 0
    assert result.trueResidual == 0
    assert torch.equal(result.x, b)
    assert torch.equal(x0, torch.ones_like(x0))
    assert int(result.iterations) == 1


def test_valid_nullspace_guess_is_preserved(solve, device):
    A = torch.diag(torch.tensor([1., 0.], device=device))
    x0 = torch.tensor([0., 5.], device=device)
    result = solve(A, torch.zeros_like(x0), x0)
    assert result.statusInt() == 0 and int(result.iterations) == 0
    assert torch.equal(result.x, x0)
    assert result.x.data_ptr() != x0.data_ptr()


def test_exact_answer_at_zero_tolerance(solve, device):
    b = torch.ones(3, device=device)
    result = solve(torch.eye(3, device=device), b, rtol=0, atol=0, maxiter=8)
    assert result.statusInt() >= 0
    assert torch.equal(result.x, b)
    assert result.trueResidual == 0
    assert int(result.iterations) == 1


@pytest.mark.parametrize('scale', [1e-12, 1., 1e6])
def test_preconditioner_scaling_does_not_accept_initial_guess(solve, device, scale):
    b = torch.ones(2, device=device)
    result = solve(torch.eye(2, device=device), b, precond=torch.full_like(b, scale))
    assert result.statusInt() >= 0
    torch.testing.assert_close(result.x, b)
    assert result.trueResidual <= 1e-5 * b.norm()


def test_tiny_rhs_is_not_absolute_rho_breakdown(solve, device):
    b = torch.full((2,), 1e-12, device=device)
    result = solve(torch.eye(2, device=device), b)
    assert result.statusInt() >= 0
    assert result.trueResidual <= 1e-5 * b.norm()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_operator_preserves_a_finite_iterate(solve, device, bad):
    A = torch.eye(2, device=device) * bad
    b = torch.ones(2, device=device)
    result = solve(A, b, maxiter=8)
    assert result.statusInt() < 0
    assert result.x.isfinite().all()


@pytest.mark.parametrize('kw', [
    {'checkEvery': 0}, {'checkEvery': -1}, {'maxiter': -1},
    {'maxiter': 1.5}, {'maxRestarts': -1}, {'dim': 0}, {'dim': 3},
    {'rtol': -1}, {'atol': float('nan')}, {'tol': float('inf')},
    {'threshold': -1}, {'precond': torch.ones(2, 1)},
])
def test_argument_validation(solve, kw):
    with pytest.raises(ValueError):
        solve(torch.eye(2), torch.ones(2), **kw)


def test_callable_outputs_are_validated(solve):
    with pytest.raises(ValueError, match='precond output'):
        solve(torch.eye(2), torch.ones(2), precond=lambda v: v[:, None])
    with pytest.raises(ValueError, match='x0'):
        solve(torch.eye(2), torch.ones(2), torch.ones(2, dtype=torch.float64))


def test_cg_recurrence_success_is_verified(device):
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(80, 80, dtype=torch.float64, generator=g))
    A = ((Q * torch.logspace(0, 5, 80, dtype=torch.float64)) @ Q.T).float().to(device)
    b = torch.randn(80, generator=g).to(device)
    # This used to report success with a true relative residual around 3.6e-3.
    result = cg(lambda v: A @ v, b, rtol=1e-5, maxiter=2000, maxRestarts=0, stamp=False)
    true = (b - A @ result.x).norm()
    torch.testing.assert_close(result.trueResidual, true, rtol=0, atol=0)
    assert result.stampedResidual is None
    assert result.statusInt() < 0 or true <= 1e-5 * b.norm()
    if result.statusInt() == Status.RESIDUAL_MISMATCH:
        assert true > 1e-5 * b.norm()


def test_work_counts_include_frozen_iterations():
    calls = {'A': 0, 'AT': 0, 'P': 0}
    A = torch.diag(torch.tensor([2., 3.], dtype=torch.float64))
    b = torch.ones(2, dtype=torch.float64)

    def action(key, matrix):
        def apply(v):
            calls[key] += 1
            return matrix @ v
        return apply

    result = bicg(action('A', A), action('AT', A.T), b,
                  precond=action('P', torch.eye(2, dtype=b.dtype)), checkEvery=8)
    assert result.statusInt() >= 0
    assert int(result.iterations) == 2
    assert result.loopIterations == 8
    assert (result.matvecs, result.adjointMatvecs, result.preconditionerApplications) == tuple(calls.values())
    assert result.verifications == 2 and result.restarts == 0


def test_bicg_nonsymmetric_preconditioner_has_independent_adjoint():
    A = torch.tensor([[3., 1., 0.], [-1., 4., 1.], [0., -1., 2.]], dtype=torch.float64)
    P = torch.tensor([[.4, .1, 0.], [0., .3, -.05], [0., 0., .6]], dtype=torch.float64)
    b = torch.tensor([1., 2., 3.], dtype=A.dtype)
    result = bicg(lambda v: A @ v, lambda v: A.T @ v, b,
                  precond=lambda v: P @ v, precondT=lambda v: P.T @ v, rtol=1e-10, maxiter=24)
    assert result.statusInt() >= 0
    torch.testing.assert_close(result.x, torch.linalg.solve(A, b), rtol=1e-10, atol=1e-12)
    assert result.trueResidual <= 1e-10 * b.norm()


def test_budget_zero_checks_only_the_initial_guess(solve):
    b = torch.ones(2)
    result = solve(torch.eye(2), b, maxiter=0)
    assert result.statusInt() == Status.BUDGET
    assert int(result.iterations) == 0 and result.loopIterations == 0
    assert result.matvecs == result.verifications == 1
    torch.testing.assert_close(result.trueResidual, b.norm())


@pytest.mark.parametrize('warm', [False, True])
def test_threshold_precedes_convergence_including_half_step(solve, device, warm):
    b = torch.ones(2, device=device)
    result = solve(torch.eye(2, device=device), b, b.clone() if warm else None, threshold=.1)
    assert result.statusInt() == Status.THRESHOLD
    assert int(result.iterations) == (0 if warm else 1)


def test_cg_bounded_residual_replacement_recovers_convergence():
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64, generator=g))
    A = ((Q * torch.logspace(0, 2, 24, dtype=torch.float64)) @ Q.T).float()
    b = torch.randn(24, generator=g)
    kw = dict(rtol=1.5e-6, maxiter=200)
    first = cg(lambda v: A @ v, b, maxRestarts=0, **kw)
    result = cg(lambda v: A @ v, b, maxRestarts=2, **kw)
    assert first.statusInt() == Status.RESIDUAL_MISMATCH
    assert result.statusInt() >= 0 and 1 <= result.restarts <= 2
    assert result.trueResidual <= kw['rtol'] * b.norm()
    assert result.loopIterations <= kw['maxiter']
    assert result.verifications == result.restarts + 2
