"""`runStationary`: the device-side stopping test returns the iterate of a loop that reads the host every iteration, the adaptive schedule
reads the host less and never stops earlier, the histories come back in one transfer, and the statistics match their definitions."""

import pytest
import torch

from warpSPHCore.linalg import CheckSchedule, residualStatistic, runStationary


class Store:
    pass


def _system(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float32) * 0.02
    A = A @ A.T + torch.eye(n, dtype=torch.float32) * 2.0           # SPD, diagonally dominant enough for Jacobi
    b = torch.randn(n, generator=g, dtype=torch.float32)
    return A, b


def _jacobiStep(A, b, omega=0.7):
    diag = torch.diagonal(A)

    def step(p, i):
        residual = b - A @ p
        return p + omega * residual / diag, torch.mean(torch.abs(residual)), None
    return step


def _hostLoop(A, b, tol, maxIters, minIters, omega=0.7):
    diag = torch.diagonal(A)
    p = torch.zeros_like(b)
    errors = []
    for i in range(maxIters):
        residual = b - A @ p
        p = p + omega * residual / diag
        errors.append(float(torch.mean(torch.abs(residual))))
        if i >= minIters and errors[-1] < tol:
            break
    return p, errors


def test_everyScheduleMatchesTheHostLoopBitwise():
    A, b = _system()
    tol, maxIters, minIters = 1e-6, 200, 2
    ref, errors = _hostLoop(A, b, tol, maxIters, minIters)
    sched = CheckSchedule(Store(), 'k', 'every', minIters, maxIters)
    out = runStationary(_jacobiStep(A, b), torch.zeros_like(b), maxIters=maxIters, tolerance=tol, schedule=sched)
    assert out.iterations == len(errors)
    assert torch.equal(out.x, ref)
    assert out.stats == errors
    assert out.converged


def test_adaptiveScheduleNeverStopsEarlierAndReadsLess(monkeypatch):
    A, b = _system()
    tol, maxIters, minIters = 1e-6, 200, 2
    store = Store()
    first = runStationary(_jacobiStep(A, b), torch.zeros_like(b), maxIters=maxIters, tolerance=tol,
                          schedule=CheckSchedule(store, 'k', 'adaptive', minIters, maxIters))
    assert store._jacobiIterationHistory['k'] == first.iterations      # the first solve checks every iteration
    reads = []
    real = torch.Tensor.__bool__
    monkeypatch.setattr(torch.Tensor, '__bool__', lambda self: (reads.append(1), real(self))[1])
    second = runStationary(_jacobiStep(A, b), torch.zeros_like(b), maxIters=maxIters, tolerance=tol,
                           schedule=CheckSchedule(store, 'k', 'adaptive', minIters, maxIters))
    assert second.iterations >= first.iterations
    assert len(reads) < first.iterations                               # checkpoints, not every iteration
    assert second.stats[:first.iterations] == first.stats


def test_relativeDisjunct():
    A, b = _system()
    diag = torch.diagonal(A)

    def step(p, i):
        residual = b - A @ p
        r = torch.mean(torch.abs(residual))
        return p + 0.7 * residual / diag, r, r
    sched = CheckSchedule(Store(), 'k', 'every', 0, 300)
    track = lambda p: torch.stack([p.min(), p.max(), p.mean()])
    out = runStationary(step, torch.zeros_like(b), maxIters=300, tolerance=0.0, schedule=sched, relTarget=1e-5, track=track)
    assert out.converged and out.iterations < 300
    assert len(out.tracked) == out.iterations and len(out.tracked[0]) == 3
    assert out.stats[-1] <= 1e-5


@pytest.mark.parametrize('kind', ['flooredOneSided', 'oneSided', 'meanAbsolute'])
def test_residualStatistics(kind):
    r = torch.tensor([-0.5, 0.25, -2.0, 1.0, 0.0])
    idx = torch.tensor([0, 1, 2, 3])
    stat, rn = residualStatistic(r, idx, kind, floor=1.0, relative=True)
    rr = r[idx]
    expect = {'flooredOneSided': torch.clamp(-rr, min=-1.0).mean(), 'oneSided': (-rr).mean(), 'meanAbsolute': rr.abs().mean()}[kind]
    assert torch.equal(stat, expect)
    assert torch.equal(rn, rr.abs().mean())
    assert residualStatistic(r, None, kind, 1.0, False)[1] is None


def test_unknownStatisticRaises():
    with pytest.raises(ValueError):
        residualStatistic(torch.zeros(3), None, 'nope', 1.0, False)
