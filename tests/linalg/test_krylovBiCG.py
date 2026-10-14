"""`linalg.krylov.bicg`: textbook BiCG (coincides with CG for a symmetric operator), converges on nonsymmetric systems, and ends every other way
with the legacy statuses. (The bitwise reference of the old loop does not apply: it had a wrong `beta` and a wrongly-sided adjoint, see the module doc.)"""

import pytest
import torch

from warpSPHCore.linalg import Status, bicg, cg


@pytest.fixture(params=['cpu', 'cuda'])
def dev(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)


def _nonsym(n, dev, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.eye(n, dtype=dtype) * 2.0 + torch.randn(n, n, generator=g, dtype=dtype) * 0.04
    b = torch.randn(n, generator=g, dtype=dtype)
    return A.to(dev), b.to(dev)


def _spd(n, dev, dtype=torch.float32, seed=3):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, n, generator=g, dtype=dtype) * 0.03
    return (R @ R.T + torch.eye(n, dtype=dtype) * 0.5).to(dev), torch.randn(n, generator=g, dtype=dtype).to(dev)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_symmetricOperatorCoincidesWithCG(dev, dtype):
    A, b = _spd(200, dev, dtype)
    mv = lambda v: A @ v
    kw = dict(rtol=1e-5 if dtype == torch.float32 else 1e-10, maxiter=100)
    a, c = bicg(mv, mv, b, None, **kw), cg(mv, b, None, **kw)
    assert a.statusInt() == c.statusInt() >= 0
    tol = 2e-3 if dtype == torch.float32 else 1e-8
    ha, hc = a.convergence(), c.convergence()
    assert len(ha) == len(hc)
    assert torch.allclose(torch.stack(ha), torch.stack(hc), rtol=tol, atol=1e-5 * float(hc[0]) if dtype == torch.float32 else 1e-12)
    assert torch.allclose(a.x, c.x, rtol=tol, atol=tol * float(c.x.abs().max()))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('checkEvery', [1, 5, 8])
def test_convergesOnNonsymmetric(dev, dtype, checkEvery):
    A, b = _nonsym(250, dev, dtype)
    res = bicg(lambda v: A @ v, lambda v: A.T @ v, b, None, rtol=1e-4 if dtype == torch.float32 else 1e-9, maxiter=200, precond=1.0 / torch.diagonal(A),
               checkEvery=checkEvery)
    assert res.statusInt() >= 0
    true = float((A @ res.x - b).norm() / b.norm())
    assert true < (5e-4 if dtype == torch.float32 else 1e-8)


def test_checkIntervalDoesNotChangeTheResult(dev):
    A, b = _nonsym(150, dev)
    mv, mvT = (lambda v: A @ v), (lambda v: A.T @ v)
    ref = bicg(mv, mvT, b, None, rtol=1e-5, maxiter=100, checkEvery=1).legacy()
    for every in (3, 8, 50):
        out = bicg(mv, mvT, b, None, rtol=1e-5, maxiter=100, checkEvery=every).legacy()
        assert out[1] == ref[1] and torch.equal(out[0], ref[0]) and len(out[2]) == len(ref[2])


def test_preconditionerIsOnTheRightSideOfTheAdjoint(dev):
    A, b = _spd(120, dev, torch.float64)
    d = 1.0 / (torch.diagonal(A) * (1 + 0.5 * torch.rand(120, dtype=torch.float64, device=dev)))           # a non-uniform diagonal
    mv = lambda v: A @ v
    res = bicg(mv, mv, b, None, rtol=1e-10, maxiter=60, precond=d)
    assert res.statusInt() >= 0 and res.statusInt() < 40


def test_skewOperatorBreaksDown(dev):
    g = torch.Generator().manual_seed(1)
    S = torch.randn(60, 60, generator=g)
    A = (S - S.T).to(dev)                                         # r . A r == 0 at the first step
    b = torch.randn(60, generator=g).to(dev)
    new = bicg(lambda v: A @ v, lambda v: A.T @ v, b, None, rtol=1e-9, maxiter=50)
    assert new.statusInt() in (int(Status.DENOMINATOR_BREAKDOWN), int(Status.RHO_BREAKDOWN))
    assert new.stampedResidual is not None


def test_budgetThresholdTrivialConverged(dev):
    A, b = _nonsym(200, dev)
    mv, mvT = (lambda v: A @ v), (lambda v: A.T @ v)
    assert bicg(mv, mvT, b, None, rtol=1e-12, maxiter=2).statusInt() == int(Status.BUDGET)
    assert bicg(mv, mvT, b, None, rtol=1e-9, maxiter=100, threshold=1e-3, dim=2).statusInt() == int(Status.THRESHOLD)
    z = bicg(mv, mvT, torch.zeros_like(b), None)
    assert z.statusInt() == 0 and z.convergence() == []
    xs = torch.linalg.solve(A.double(), b.double()).to(b.dtype)
    assert bicg(mv, mvT, b, xs, rtol=1e-2).statusInt() == 0


def test_nonFiniteAdjointIsABreakdown(dev):
    A, b = _nonsym(100, dev)
    state = {'n': 0}

    def mvT(v):
        state['n'] += 1
        out = A.T @ v
        return out * float('inf') if state['n'] == 2 else out
    new = bicg(lambda v: A @ v, mvT, b, None, rtol=1e-9, maxiter=30)
    assert new.statusInt() < 0
