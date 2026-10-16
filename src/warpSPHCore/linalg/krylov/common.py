"""Shared pieces of the Krylov solvers: the status codes, the device-resident result, and the helpers every solver builds on.

CG/BiCG/BiCGStab/MINRES recurrence loops are *device-predicated*: the stopping and breakdown tests are evaluated on the device each iteration, the iterate is frozen
(`torch.where`) from the iteration a test fires, and the host reads the "done" flag only every `checkEvery` iterations. The returned iterate is
therefore unaffected by frozen work (at most `checkEvery - 1` iterations per
pass). A shared outer driver verifies the original residual and bounds fresh
passes when a recurrence under-reports it. GMRES has its own restarted driver
and verifies at cycle boundaries. Richardson uses eager bounded trial acceptance.
Checkpoints, trial decisions and final verification
read the host; these are not yet fully capturable solves.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from numbers import Integral, Real
from typing import Callable, List, Optional, Union

import torch

__all__ = ['Status', 'SolveResult', 'makePsolve', 'thresholdExceeded', 'absoluteTolerance']


class Status(IntEnum):
    """Solver outcome. A non-negative status is the iteration at which the solve converged; the negative ones are failures."""
    RHO_BREAKDOWN = -10          # BiCG family: |rho| below the relative guard
    DENOMINATOR_BREAKDOWN = -11  # alpha / omega / beta denominator
    THRESHOLD = -12              # a per-entry |x| (norm over `dim` components) exceeded `threshold`
    STAGNATION = -13             # GMRES: insufficient full-cycle true-residual decrease
    BUDGET = -14                 # maxiter exhausted
    RICHARDSON_STAGNATION = -15
    INDEFINITE = -16             # CG curvature / CG or MINRES nonpositive or nonfinite active inverse quadratic form
    RESIDUAL_MISMATCH = -17      # recurrence converged, but bounded verification/restarts did not


@dataclass
class SolveResult:
    """Device solution, status, verified residual, update count and residual history, plus host work counts.

    `norms[i]` contributes to history when `valid[i]` is true. `trueResidual`
    certifies the returned iterate independently of history stamping. Completed
    updates are in `iterations`, while `loopIterations` counts scheduled/frozen
    work; operator counts include initial/final verification and frozen work.
    Solving materializes status for verification; legacy history is copied only
    when `convergence()`/`legacy()` is requested.
    """
    x: torch.Tensor
    status: torch.Tensor
    norms: torch.Tensor
    valid: torch.Tensor
    stampedResidual: Optional[torch.Tensor] = None
    trueResidual: Optional[torch.Tensor] = None
    iterations: Optional[torch.Tensor] = None  # completed updates/GMRES Arnoldi steps, including accepted half steps; device int
    loopIterations: int = 0                   # scheduled iterations, including frozen work
    matvecs: int = 0                          # actual operator calls, including verification and frozen work
    adjointMatvecs: int = 0
    preconditionerApplications: int = 0
    verifications: int = 0                    # direct original-system residual evaluations (includes initial)
    restarts: int = 0
    setupIterations: int = 0                  # executed capture warmup updates, excluded from numerical iteration budget
    _host: dict = field(default_factory=dict, repr=False)
    trialSteps: int = 0                       # Richardson attempted candidates, including rejected/nonfinite trials
    backtracks: int = 0                       # Richardson step reductions actually performed
    bestIteration: int = 0                    # Richardson update index of the returned best iterate (unless threshold)
    stepSize: Optional[torch.Tensor] = None   # Richardson last accepted signed step, or None

    def statusInt(self) -> int:
        """The status as a Python int (one device read, cached)."""
        if 'status' not in self._host:
            self._host['status'] = int(self.status)
        return self._host['status']

    def convergence(self) -> List[torch.Tensor]:
        """The legacy `convergence` list: the 0-d residual norms of the iterations that ran, then the stamped true residual (if any)."""
        keep = self.valid.cpu().tolist()
        out = [n for n, v in zip(self.norms.unbind(0), keep) if v]
        if self.stampedResidual is not None:
            out.append(self.stampedResidual)
        return out

    def legacy(self):
        """`(x, status:int, convergence:list)`, the return convention of `modules/shifting` before the move."""
        return self.x, self.statusInt(), self.convergence()


def makePsolve(precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None]) -> Callable[[torch.Tensor], torch.Tensor]:
    """`v -> M v`: identity, the elementwise product with a flat vector, or the callable itself."""
    if precond is None:
        return lambda v: v
    if callable(precond):
        return precond
    return lambda v: precond * v


def thresholdExceeded(x: torch.Tensor, threshold: Optional[float], dim: int) -> Optional[torch.Tensor]:
    """0-d bool: some entry (the norm over `dim` consecutive components) of `x` is above `threshold`; None without a threshold."""
    if threshold is None:
        return None
    return torch.any(torch.linalg.norm(x.reshape(-1, dim), dim=-1) > threshold)


def absoluteTolerance(bnrm2: torch.Tensor, tol: float, rtol: float, atol: float) -> torch.Tensor:
    """`max(atol, tol, rtol * ||b||)` as a 0-d device tensor of `bnrm2`'s dtype: evaluated in float64 like the host expression
    `max(float(atol), float(tol), float(rtol) * float(bnrm2))` and rounded to the residual's dtype once, which is what comparing a float32 residual
    with a Python float does."""
    floor = torch.full((), max(float(atol), float(tol)), dtype=torch.float64, device=bnrm2.device)
    return torch.maximum(floor, float(rtol) * bnrm2.double()).to(bnrm2.dtype)


def converged(norm, tolerance):
    """Exact convergence is valid even when every tolerance is zero."""
    return torch.isfinite(norm) & (norm <= tolerance)


def safeDivide(numerator, denominator):
    """Avoid invalid divisions in predicated, inactive updates; callers still test breakdown."""
    usable = torch.isfinite(denominator) & (denominator != 0)
    return numerator / torch.where(usable, denominator, torch.ones_like(denominator))


def nonnegativeInt(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < int(positive):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return int(value)


def finiteScalar(name, value, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be a finite {"nonnegative " if nonnegative else ""}number')


def validateVector(value, reference, name):
    if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
        raise ValueError(f'{name} must be a tensor of shape {tuple(reference.shape)}')
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f'{name} must have dtype {reference.dtype} and device {reference.device}')
    return value


def validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts):
    if not isinstance(b, torch.Tensor) or b.ndim != 1 or not b.is_floating_point():
        raise ValueError('b must be a flat real floating-point tensor')
    if x0 is not None:
        validateVector(x0, b, 'x0')
    for name, value in (('tol', tol), ('rtol', rtol), ('atol', atol)):
        finiteScalar(name, value)
    if threshold is not None:
        finiteScalar('threshold', threshold)
    nonnegativeInt('dim', dim, positive=True)
    if b.numel() % dim:
        raise ValueError('b length must be divisible by dim')
    nonnegativeInt('checkEvery', checkEvery, positive=True)
    nonnegativeInt('maxRestarts', maxRestarts)
    return b.numel() * 10 if maxiter is None else nonnegativeInt('maxiter', maxiter)


def checkedAction(action, b, name, counts, counter):
    if not callable(action):
        raise ValueError(f'{name} must be callable')

    def apply(v):
        counts[counter] += 1
        return validateVector(action(v), b, name + ' output')
    return apply


def checkedPreconditioner(precond, b, counts, name='precond'):
    if precond is None:
        return lambda v: v
    if not callable(precond):
        validateVector(precond, b, name)
    return checkedAction(makePsolve(precond), b, name, counts, 'preconditionerApplications')


@dataclass
class PassResult:
    x: torch.Tensor
    status: torch.Tensor
    norms: list
    valid: list
    iterations: torch.Tensor
    loopIterations: int


def runVerified(run, matvec, b, x0, *, tol, rtol, atol, maxiter, maxRestarts, stamp, counts):
    """Bounded passes with a true-residual certificate; no successful return relies on a recurrence.

    `maxiter` bounds scheduled iterations across all passes, including frozen work.
    `iterations` counts accepted updates; the legacy status retains each method's
    indexing, offset by completed updates in preceding passes. Histories keep
    their legacy layout, with another initial entry on CG/BiCG restarts.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    bnrm = torch.linalg.vector_norm(b)
    target = absoluteTolerance(bnrm, tol, rtol, atol)
    residual = b - matvec(x)
    true = torch.linalg.vector_norm(residual)
    verifications = 1
    norms, valid = [], []
    completed = torch.zeros((), dtype=torch.int32, device=b.device)
    loops, restarts = 0, 0
    for attempt in range(maxRestarts + 1):
        part = run(x, residual, target, maxiter - loops, bnrm == 0)
        norms.extend(part.norms)
        valid.extend(part.valid)
        status = torch.where(part.status >= 0, part.status + completed, part.status)
        completed = completed + part.iterations
        loops += part.loopIterations
        x = part.x
        if part.loopIterations:
            residual = b - matvec(x)
            true = torch.linalg.vector_norm(residual)
            verifications += 1
        # A pass that did not iterate tested the supplied true initial residual.
        statusInt = int(status)
        if statusInt < 0 or bool(converged(true, target)):
            break
        if attempt == maxRestarts or loops >= maxiter:
            status = torch.full_like(status, int(Status.RESIDUAL_MISMATCH))
            statusInt = int(Status.RESIDUAL_MISMATCH)
            break
        restarts += 1
    result = SolveResult(
        x=x, status=status,
        norms=torch.stack(norms) if norms else b.new_empty(0),
        valid=torch.stack(valid) if valid else torch.empty(0, dtype=torch.bool, device=b.device),
        trueResidual=true, iterations=completed, loopIterations=loops,
        verifications=verifications, restarts=restarts, _host={'status': statusInt}, **counts,
    )
    # `always` is BiCGStab's legacy history convention, except a trivial zero solve.
    if (stamp == 'always' and (loops or not bool(bnrm == 0))) or (stamp is True and statusInt < 0):
        result.stampedResidual = true
    return result
