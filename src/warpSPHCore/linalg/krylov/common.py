"""Shared pieces of the Krylov solvers: the status codes, the device-resident result, and the helpers every solver builds on.

Every solver here is *device-predicated*: the stopping and breakdown tests are evaluated on the device each iteration, the iterate is frozen
(`torch.where`) from the iteration a test fires, and the host reads the "done" flag only every `checkEvery` iterations. The returned iterate is
therefore exactly the one a loop that tested on the host and returned at once would give (the frozen iterations' work is discarded), at the cost of
at most `checkEvery - 1` wasted iterations and with 1/`checkEvery` of the host reads.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Union

import torch

__all__ = ['Status', 'SolveResult', 'makePsolve', 'thresholdExceeded', 'absoluteTolerance']


class Status(IntEnum):
    """Solver outcome. A non-negative status is the iteration at which the solve converged; the negative ones are failures."""
    RHO_BREAKDOWN = -10          # BiCG family: |rho| below the relative guard
    DENOMINATOR_BREAKDOWN = -11  # alpha / omega / beta denominator
    THRESHOLD = -12              # a per-entry |x| (norm over `dim` components) exceeded `threshold`
    STAGNATION = -13             # GMRES: a full cycle reduced the residual by < 0.1 %
    BUDGET = -14                 # maxiter exhausted
    RICHARDSON_STAGNATION = -15
    INDEFINITE = -16             # CG: non-positive curvature / non-finite `r^T M r` or `p^T A p`


@dataclass
class SolveResult:
    """`x`, the 0-d integer `status` tensor and the residual-norm history, all on the device until asked.

    `norms[i]` is the norm of the residual after `valid[i]` is true (frozen iterations contribute no entry). `stampedResidual` is the verified true
    residual `||b - A x||` of the returned iterate, present when the solve failed (as the legacy convention: every failing return is stamped).
    """
    x: torch.Tensor
    status: torch.Tensor
    norms: torch.Tensor
    valid: torch.Tensor
    stampedResidual: Optional[torch.Tensor] = None
    _host: dict = field(default_factory=dict, repr=False)

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
    return torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold)


def absoluteTolerance(bnrm2: torch.Tensor, tol: float, rtol: float, atol: float) -> torch.Tensor:
    """`max(atol, tol, rtol * ||b||)` as a 0-d device tensor of `bnrm2`'s dtype: evaluated in float64 like the host expression
    `max(float(atol), float(tol), float(rtol) * float(bnrm2))` and rounded to the residual's dtype once, which is what comparing a float32 residual
    with a Python float does."""
    floor = torch.full((), max(float(atol), float(tol)), dtype=torch.float64, device=bnrm2.device)
    return torch.maximum(floor, float(rtol) * bnrm2.double()).to(bnrm2.dtype)
