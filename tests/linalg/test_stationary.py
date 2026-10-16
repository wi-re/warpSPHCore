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


def test_nonmonotone_adaptive_statistic_is_not_a_convergence_certificate():
    store = Store()
    store._jacobiIterationHistory = {'k': 8}
    schedule = CheckSchedule(store, 'k', 'adaptive', 0, 4)

    def step(x, i):
        return x + 1, torch.tensor(0. if i == 0 else 1.), None

    out = runStationary(step, torch.zeros(1), maxIters=4, tolerance=.5, schedule=schedule)
    assert out.iterations == 4 and not out.converged
    assert out.termination == 'budget' and out.firstConvergedIteration == 1
    assert store._jacobiIterationHistory['k'] == 1
    assert torch.equal(out.x, torch.tensor([4.]))


@pytest.mark.parametrize('where', ['iterate', 'stat', 'relative'])
def test_nonfinite_step_freezes_last_finite_iterate(where):
    store = Store()
    store._jacobiIterationHistory = {'k': 8}
    schedule = CheckSchedule(store, 'k', 'adaptive', 0, 6)

    def step(x, i):
        xn, stat, norm = x + 1, torch.tensor(1.), torch.tensor(1.)
        if i == 1:
            if where == 'iterate':
                xn *= float('nan')
            elif where == 'stat':
                stat *= float('nan')
            else:
                norm *= float('nan')
        return xn, stat, norm

    out = runStationary(step, torch.zeros(1), maxIters=6, tolerance=0, relTarget=.1, schedule=schedule)
    assert out.iterations == 4 and not out.converged and out.termination == 'nonfinite'
    assert torch.equal(out.x, torch.ones(1))


@pytest.mark.parametrize('kind', ['flooredOneSided', 'oneSided', 'meanAbsolute'])
def test_empty_selection_has_no_active_equations(kind):
    residual = torch.ones(3)
    stat, norm = residualStatistic(residual, torch.empty(0, dtype=torch.int64), kind, floor=1, relative=True)
    assert stat == norm == 0


def test_missing_relative_statistic_is_a_clear_error():
    schedule = CheckSchedule(Store(), 'k', 'every', 0, 2)
    with pytest.raises(ValueError, match='rNorm'):
        runStationary(lambda x, i: (x, torch.tensor(1.), None), torch.zeros(1),
                      maxIters=2, tolerance=0, relTarget=.1, schedule=schedule)


def test_tuple_pressure_and_residual_state_is_preserved():
    # The downstream optimal pressure solve carries precisely this tuple.
    pressure, residual = torch.zeros(2), torch.ones(2)
    schedule = CheckSchedule(Store(), 'optimal', 'every', 0, 4)

    def step(state, i):
        p, r = state
        rn = r / 2
        return (p + r, rn), rn.abs().mean(), None

    out = runStationary(step, (pressure, residual), maxIters=4, tolerance=.2,
                        schedule=schedule, track=lambda state: state[0].mean()[None])
    assert out.converged and out.iterations == 3
    torch.testing.assert_close(out.x[0], torch.full((2,), 1.75))
    torch.testing.assert_close(out.x[1], torch.full((2,), .125))
    assert out.tracked[-1] == (1.75,)
    assert torch.equal(pressure, torch.zeros(2))
    assert torch.equal(residual, torch.ones(2))
