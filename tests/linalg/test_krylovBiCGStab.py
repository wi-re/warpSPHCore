"""`linalg.krylov.bicgstab`: bitwise the host-branching loop on every system where the recurrence residual is honest; a converged status always
means the verified true residual passed the tolerance (restart when the recurrence lied); and every other way a solve can end."""

import pytest
import torch

from warpSPHCore.linalg import Status, bicgstab
from _legacyKrylov import bicgstabSolveLegacy


@pytest.fixture(params=['cpu', 'cuda'])
def dev(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)


def _nonsym(n, dev, dtype=torch.float32, seed=0, spread=0.04):
    g = torch.Generator().manual_seed(seed)
    A = torch.eye(n, dtype=dtype) * 2.0 + torch.randn(n, n, generator=g, dtype=dtype) * spread
    return A.to(dev), torch.randn(n, generator=g, dtype=dtype).to(dev)


def _same(new, old):
    x, status, conv = new.legacy()
    xo, so, co = old
    assert status == so
    assert torch.equal(x, xo)
    assert len(conv) == len(co)
    assert all(torch.equal(a, b) for a, b in zip(conv, co))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('checkEvery', [1, 4, 8])
@pytest.mark.parametrize('precond', [False, True])
def test_matchesLegacyWhenTheRecurrenceIsHonest(dev, dtype, checkEvery, precond):
    A, b = _nonsym(250, dev, dtype)
    mv = lambda v: A @ v
    kw = dict(rtol=1e-5 if dtype == torch.float32 else 1e-10, maxiter=200, precond=(1.0 / torch.diagonal(A)) if precond else None)
    old = bicgstabSolveLegacy(mv, b, None, **kw)
    assert old[1] >= 0
    new = bicgstab(mv, b, None, checkEvery=checkEvery, **kw)
    if not precond:
        _same(new, old)
        return
    # with a preconditioner the old half-step candidate used the raw direction (wrong), failed its own verification and iterated on: the corrected
    # candidate may converge one step earlier, never later, and the verified residual passes either way
    assert 0 <= new.statusInt() <= old[1]
    assert float(new.stampedResidual) < kw['rtol'] * float(b.norm()) * 1.0001
    if new.statusInt() == old[1]:
        _same(new, old)


@pytest.mark.parametrize('seed', [0, 6, 24])
def test_warmStartAndCallablePrecond(dev, seed):
    A, b = _nonsym(180, dev)
    mv = lambda v: A @ v
    d = 1.0 / torch.diagonal(A)
    generator = torch.Generator(device=dev).manual_seed(seed)
    x0 = torch.randn(180, device=dev, generator=generator) * 0.1
    kw = dict(rtol=1e-5, maxiter=100, precond=lambda v: d * v)
    new, old = bicgstab(mv, b, x0, **kw), bicgstabSolveLegacy(mv, b, x0, **kw)
    # As above, the legacy half-step uses pk rather than preconditioned phat.
    # CUDA seed 24 takes an unnecessary full step; seed 6 later breaks down.
    # The corrected half-step already satisfies the original system in both.
    assert new.statusInt() >= 0
    assert new.trueResidual <= kw['rtol'] * b.norm()
    assert old[1] < 0 or new.statusInt() <= old[1]
    reference = torch.linalg.solve(A.double(), b.double()).to(b.dtype)
    assert (new.x - reference).norm() <= 1e-4 * reference.norm()
    if new.statusInt() == old[1]:
        _same(new, old)


def test_convergedAlwaysMeansTheTrueResidualPassed(dev):
    """Ill-conditioned float32 systems at a tight tolerance: the recurrence residual can under-report; whatever the status, a converged one is verified."""
    for seed in range(12):
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(120, 120, generator=g, dtype=torch.float64))
        s = torch.logspace(0, 4.5, 120, dtype=torch.float64)
        A = (Q @ torch.diag(s) @ Q.T).to(torch.float32).to(dev)
        b = torch.randn(120, generator=g).to(dev)
        res = bicgstab(lambda v: A @ v, b, None, rtol=1e-6, maxiter=400)
        if res.statusInt() >= 0:
            assert float(res.stampedResidual) < 1e-6 * float(b.norm()) * 1.0001, seed


def test_breakdownBudgetThresholdTrivialConverged(dev):
    A, b = _nonsym(200, dev)
    mv = lambda v: A @ v
    g = torch.Generator().manual_seed(1)
    S = torch.randn(60, 60, generator=g)
    Sk = (S - S.T).to(dev)
    bs = torch.randn(60, generator=g).to(dev)
    old = bicgstabSolveLegacy(lambda v: Sk @ v, bs, None, rtol=1e-9, maxiter=50)
    assert old[1] < 0
    _same(bicgstab(lambda v: Sk @ v, bs, None, rtol=1e-9, maxiter=50), old)
    old = bicgstabSolveLegacy(mv, b, None, rtol=1e-12, maxiter=2)
    assert old[1] == int(Status.BUDGET)
    _same(bicgstab(mv, b, None, rtol=1e-12, maxiter=2), old)
    kw = dict(rtol=1e-9, maxiter=100, threshold=1e-3, dim=2)
    old = bicgstabSolveLegacy(mv, b, None, **kw)
    assert old[1] == int(Status.THRESHOLD)
    _same(bicgstab(mv, b, None, **kw), old)
    z = torch.zeros_like(b)
    _same(bicgstab(mv, z, None), bicgstabSolveLegacy(mv, z, None))
    xs = torch.linalg.solve(A.double(), b.double()).to(b.dtype)
    new = bicgstab(mv, b, xs, rtol=1e-2)
    assert new.statusInt() == 0 and float(new.stampedResidual) < 1e-2 * float(b.norm())


def test_hostReadsDrop(dev, monkeypatch):
    A, b = _nonsym(300, dev)
    mv = lambda v: A @ v
    kw = dict(rtol=1e-6, maxiter=200)
    reads = {'n': 0}
    for name in ('__bool__', '__float__', 'item'):
        real = getattr(torch.Tensor, name)

        def counted(self, _real=real):
            reads['n'] += 1
            return _real(self)
        monkeypatch.setattr(torch.Tensor, name, counted)
    bicgstabSolveLegacy(mv, b, None, **kw)
    legacyReads = reads['n']
    reads['n'] = 0
    bicgstab(mv, b, None, checkEvery=8, **kw)
    assert reads['n'] * 3 < legacyReads, (reads['n'], legacyReads)
