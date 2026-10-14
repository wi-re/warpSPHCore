"""Stationary (relaxed-Jacobi) iterations with a device-resident stopping test.

The pressure solves of the incompressible schemes iterate `p <- project(update(p, residual(p)))` (relaxed Jacobi, optimal-step Richardson, ...)
until a residual statistic passes a tolerance. Each used to spell its own loop, and the loops read the host after every iteration (1-4 syncs per
iteration: `dfsphReference` 310 syncs per step). `runStationary` owns the loop and the stopping; the scheme supplies one `step`:

    def step(p, i):
        residual = source - A(p)                        # the scheme's own operator
        pNext = project(p + omega * residual / diag)    # the scheme's own update and projection
        stat, rNorm = residualStatistic(residual, ...)  # 0-d device tensors
        return pNext, stat, rNorm

Per iteration the driver keeps `stat`, the convergence flag (`stat < tolerance`, or `rNorm <= relTarget`) and an optional tracked vector (the
`(min, max, mean)` of the iterate) on the device, and reads the flag back only where the `CheckSchedule` says so (every iteration, or the adaptive
marks learned from the previous solve of the same kind). The histories come back in one transfer at the end. With `schedule='every'` the iterate
returned is bitwise the one a loop reading the host after every iteration returned; `adaptive` may run past the first converged iteration by up to
the gap to the next checkpoint (more converged, never less).
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

__all__ = ['CheckSchedule', 'residualStatistic', 'runStationary', 'StationaryResult', 'RESIDUAL_STATISTICS']

#: the residual statistics a stationary iteration can stop on (all means over the rows of `index`)
RESIDUAL_STATISTICS = ('flooredOneSided', 'oneSided', 'meanAbsolute')


class CheckSchedule:
    """At which iterations a stationary loop reads its convergence test back to the host (one device sync per read).

    `mode == 'every'`: after every iteration (the historical behaviour). `'adaptive'`: assume this solve needs about as many iterations `P` as
    the previous one did to *first* meet the test, and check at ceil(0.50 P), ceil(0.75 P), ceil(0.85 P), then every other iteration. The previous
    count is the first-converged iteration recovered from the device-side history, not the (possibly overshot) iteration the loop stopped at, so
    overshoot does not feed back into the next solve's schedule. The first solve of a kind, and verbose runs, check every iteration.

    `key` names the solve (a step may run several differently-behaved solves); the history lives on `store` (typically the scheme config) under
    `_jacobiIterationHistory`.
    """

    def __init__(self, store, key: str, mode: str, minIters: int, maxIters: int, verbose: bool = False):
        self.store, self.key = store, key
        self.minIters, self.maxIters = minIters, maxIters
        history = getattr(store, '_jacobiIterationHistory', None)
        if history is None:
            history = {}
            try:
                setattr(store, '_jacobiIterationHistory', history)
            except Exception:  # noqa: BLE001 -- a frozen/slotted config: no memory, check every iteration
                pass
        self.history = history
        previous = history.get(key)
        self.every = mode != 'adaptive' or verbose or previous is None
        if not self.every:
            self.marks = sorted({max(1, math.ceil(f * previous)) for f in (0.50, 0.75, 0.85)})
            self.dense = self.marks[-1]

    def check(self, i: int) -> bool:
        """Read the test back after iteration index `i` (`i + 1` done)?"""
        if i < self.minIters:
            return False            # the loop may not stop here anyway
        if self.every:
            return True
        n = i + 1
        return n in self.marks or (n > self.dense and (n - self.dense) % 2 == 0)

    def record(self, convergedFlags) -> None:
        """Remember the first iteration (count) that met the test, from the host copy of the per-iteration flags; `maxIters` if none did."""
        first = next((k + 1 for k, c in enumerate(convergedFlags) if c and k >= self.minIters), self.maxIters)
        self.history[self.key] = first


def residualStatistic(residual: torch.Tensor, index: Optional[torch.Tensor], kind: str, floor: float,
                      relative: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """`(stat, rNorm)` of one iteration's residual as 0-d device tensors, over the rows `index` (None: all rows).

    `kind`: `flooredOneSided` -- `mean(clamp(-r, min=-floor))` (one-sided and floored: an under-dense particle contributes at most `-floor`, so it
    cannot cancel an over-dense one); `oneSided` -- `mean(-r)`; `meanAbsolute` -- `mean(|r|)`, the only one that is a norm. `rNorm` (None unless
    `relative`) is `mean|r|`, for the relative disjunct of the stopping test.
    """
    r = residual if index is None else residual.index_select(0, index)
    if kind == 'flooredOneSided':
        stat = torch.mean(torch.clamp(-r, min=-floor))
    elif kind == 'oneSided':
        stat = torch.mean(-r)
    elif kind == 'meanAbsolute':
        stat = torch.mean(torch.abs(r))
    else:
        raise ValueError(f'unknown residual statistic {kind!r}; one of {RESIDUAL_STATISTICS}')
    if not relative:
        return stat, None
    if kind == 'meanAbsolute':
        return stat, stat
    return stat, torch.mean(torch.abs(r))


@dataclass
class StationaryResult:
    x: torch.Tensor                                     # the last iterate
    iterations: int                                     # iterations run (the check schedule may run past the first converged one)
    stats: List[float] = field(default_factory=list)    # the statistic of every iteration
    tracked: List[tuple] = field(default_factory=list)  # `track(x)` of every iteration (e.g. (min, max, mean) of the pressure)
    converged: bool = False                             # the last iteration met the test


def runStationary(step: Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
                  x0: torch.Tensor, *, maxIters: int, tolerance: float, schedule: CheckSchedule,
                  relTarget: Optional[float] = None, inclusive: bool = False, track: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                  onIteration: Optional[Callable] = None) -> StationaryResult:
    """Iterate `x, stat, rNorm = step(x, i)` until the statistic passes, reading the host only where `schedule.check(i)` says so.

    Converged: `stat < tolerance` (`stat <= tolerance` with `inclusive`), or (when `relTarget` is given) `rNorm <= relTarget`. `track(x)` returns a small vector logged per iteration
    (on the device, read once at the end). `onIteration(i, x, stat)` is a host-side debug hook (verbose runs; it may read the host).
    `schedule.record` is called with the per-iteration flags so the next solve of the same kind checks where this one first converged.
    """
    x = x0
    stats: List[torch.Tensor] = []
    flags: List[torch.Tensor] = []
    tracked: List[torch.Tensor] = []
    converged = None
    for i in range(maxIters):
        x, stat, rNorm = step(x, i)
        converged = (stat <= tolerance) if inclusive else (stat < tolerance)
        if relTarget is not None:
            converged = converged | (rNorm <= relTarget)
        stats.append(stat)
        flags.append(converged)
        if track is not None:
            tracked.append(track(x))
        if onIteration is not None:
            onIteration(i, x, stat)
        if schedule.check(i) and bool(converged):
            break

    result = StationaryResult(x=x, iterations=len(stats))
    if stats:
        s = torch.stack(stats)
        columns = [s.view(-1, 1), torch.stack(flags).to(s.dtype).view(-1, 1)]
        if tracked:
            columns.append(torch.stack(tracked).to(s.dtype))
        host = torch.cat(columns, dim=1).cpu()                  # the one device -> host transfer of the solve
        result.stats = host[:, 0].tolist()
        flagsHost = [bool(f) for f in host[:, 1].tolist()]
        schedule.record(flagsHost)
        result.converged = flagsHost[-1]
        if tracked:
            result.tracked = [tuple(row) for row in host[:, 2:].tolist()]
    return result
