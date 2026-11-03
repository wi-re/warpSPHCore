"""`linalg.krylov.cg`: the device-predicated solver returns the iterate, status and history of the host-branching loop it replaces (bitwise),
for every way a CG solve can end, at any host-check interval, with a fraction of the host reads."""

import pytest
import torch

from warpSPHCore.linalg import Status, cg
from _legacyKrylov import cgSolveLegacy


@pytest.fixture(params=['cpu', 'cuda'])
def dev(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)


def _spd(n, dev, dtype=torch.float32, seed=0, spread=0.02):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, n, generator=g, dtype=dtype) * spread
    A = (R @ R.T + torch.eye(n, dtype=dtype) * 0.5).to(dev)
    b = torch.randn(n, generator=g, dtype=dtype).to(dev)
    return A, b


def _same(new, old):
    x, status, conv = new.legacy()
    xo, so, co = old
    assert status == so
    assert torch.equal(x, xo)
    assert len(conv) == len(co)
    assert all(torch.equal(a, b) for a, b in zip(conv, co))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('checkEvery', [1, 3, 8, 64])
def test_matchesLegacyOnSPD(dev, dtype, checkEvery):
    A, b = _spd(300, dev, dtype)
    mv = lambda v: A @ v
    pre = 1.0 / torch.diagonal(A)
    kw = dict(rtol=1e-6 if dtype == torch.float32 else 1e-10, maxiter=500, precond=pre)
    old = cgSolveLegacy(mv, b, None, **kw)
    assert old[1] >= 0
    _same(cg(mv, b, None, checkEvery=checkEvery, **kw), old)


def test_callablePrecondAndWarmStart(dev):
    A, b = _spd(200, dev)
    mv = lambda v: A @ v
    d = 1.0 / torch.diagonal(A)
    x0 = torch.randn(200, device=dev) * 0.1
    kw = dict(rtol=1e-5, maxiter=400, precond=lambda v: d * v)
    _same(cg(mv, b, x0, **kw), cgSolveLegacy(mv, b, x0, **kw))


def test_indefiniteIsReportedAndStamped(dev):
    A, b = _spd(120, dev)
    A = A - 3.0 * torch.eye(120, device=dev)                   # indefinite
    mv = lambda v: A @ v
    old = cgSolveLegacy(mv, b, None, rtol=1e-7, maxiter=300)
    assert old[1] == int(Status.INDEFINITE) == -16
    new = cg(mv, b, None, rtol=1e-7, maxiter=300)
    _same(new, old)
    assert new.stampedResidual is not None


def test_budgetExhausted(dev):
    A, b = _spd(300, dev)
    mv = lambda v: A @ v
    old = cgSolveLegacy(mv, b, None, rtol=1e-12, maxiter=3)
    assert old[1] == int(Status.BUDGET)
    _same(cg(mv, b, None, rtol=1e-12, maxiter=3), old)


def test_thresholdBailout(dev):
    A, b = _spd(240, dev)
    mv = lambda v: A @ v
    kw = dict(rtol=1e-9, maxiter=300, threshold=1e-3, dim=2)
    old = cgSolveLegacy(mv, b, None, **kw)
    assert old[1] == int(Status.THRESHOLD)
    _same(cg(mv, b, None, **kw), old)


def test_zeroRhsAndAlreadyConverged(dev):
    A, b = _spd(100, dev)
    mv = lambda v: A @ v
    z = torch.zeros_like(b)
    _same(cg(mv, z, None), cgSolveLegacy(mv, z, None))
    xs = torch.linalg.solve(A.double(), b.double()).to(b.dtype)
    _same(cg(mv, b, xs, rtol=1e-2), cgSolveLegacy(mv, b, xs, rtol=1e-2))


def test_nonFiniteOperatorIsIndefinite(dev):
    A, b = _spd(80, dev)
    calls = {'n': 0}

    def mv(v):
        calls['n'] += 1
        out = A @ v
        if calls['n'] > 3:
            out = out * float('nan')
        return out
    new = cg(mv, b, None, rtol=1e-9, maxiter=50)
    calls['n'] = 0
    old = cgSolveLegacy(mv, b, None, rtol=1e-9, maxiter=50)
    assert new.statusInt() == old[1] == int(Status.INDEFINITE)
    assert torch.equal(new.x, old[0])


def test_hostReadsDropWithCheckEvery(dev, monkeypatch):
    A, b = _spd(300, dev)
    mv = lambda v: A @ v
    kw = dict(rtol=1e-6, maxiter=500, precond=1.0 / torch.diagonal(A))
    reads = {'n': 0}
    for name in ('__bool__', '__float__', 'item'):
        real = getattr(torch.Tensor, name)

        def counted(self, _real=real):
            reads['n'] += 1
            return _real(self)
        monkeypatch.setattr(torch.Tensor, name, counted)
    cgSolveLegacy(mv, b, None, **kw)
    legacyReads = reads['n']
    reads['n'] = 0
    cg(mv, b, None, checkEvery=8, **kw)
    assert reads['n'] * 4 < legacyReads, (reads['n'], legacyReads)
