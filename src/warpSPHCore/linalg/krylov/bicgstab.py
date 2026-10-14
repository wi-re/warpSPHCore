"""Right-preconditioned BiCGStab, device-predicated (see `common.py`).

The loop it replaces (`warpSPH.modules.shifting.bicgstabSolve`) branched on the host at two places: when the recurrence residual passed the
tolerance it spent an extra matvec on the *true* residual and kept iterating if the recurrence had under-reported it, and the same for the
half-step residual `||s||`. A device formulation cannot do work only when a test fires, so the tests here are made on the recurrence residuals and
the verification moves behind the loop: one matvec for the true residual of the returned iterate, and if a solve that reported convergence fails
it, the solve restarts from that iterate (fresh shadow residual, up to `maxRestarts` times; the BiCGStab restart is standard practice and cures the
under-reporting the old loop kept iterating through). When the recurrence is honest (every well-conditioned system) the iterate, status and history
are bitwise the old loop's, which `tests/linalg/_legacyKrylov.py` checks.

Two intended differences: the half-step candidate `x + alpha * phat` uses the preconditioned direction (the old verification used the raw
direction `p`, which is wrong for a preconditioned solve), and a recurrence that lied now ends in a restart instead of a silent continuation.

Status: `>= 0` converged (the half step reports its iteration, a full step the number of completed iterations); `RHO_BREAKDOWN`,
`DENOMINATOR_BREAKDOWN` (the curvature or `omega`), `THRESHOLD`, `BUDGET`. The history is the residual norm of every completed iteration followed by
the verified true residual of the returned iterate.
"""

from typing import Callable, Optional, Union

import torch

from .common import SolveResult, Status, absoluteTolerance, makePsolve, thresholdExceeded

__all__ = ['bicgstab']

#: relative breakdown margin: ~8x float32 eps, it only fires once the corresponding update (alpha / omega) is already ~1e6-scaled
_REL_BREAKDOWN_EPS = 1e-6


def _run(matvec, psolve, b, xk, rk, atolT, rhotol, omegatol, budget, threshold, dim, checkEvery):
    """One BiCGStab pass of at most `budget` iterations from `xk` with residual `rk`; returns `(x, status, norms, valid, iterations)`."""
    dev = b.device
    i32 = lambda v: torch.full((), int(v), dtype=torch.int32, device=dev)
    ZERO, RHO, DEN, THR, BUDGET = i32(0), i32(Status.RHO_BREAKDOWN), i32(Status.DENOMINATOR_BREAKDOWN), i32(Status.THRESHOLD), i32(Status.BUDGET)
    one = torch.ones((), dtype=torch.int32, device=dev)

    r0 = rk.clone()
    pk = rk.clone()
    r0norm = torch.linalg.norm(r0)
    rkNorm = torch.linalg.norm(rk)
    conv0 = rkNorm < atolT
    live = ~conv0
    status = torch.where(live, BUDGET, ZERO)
    norms, valid = [], []
    it = ZERO
    used = 0

    for k in range(budget if bool(live) else 0):
        used = k + 1
        rho = torch.dot(rk, r0)
        badRho = (rho.abs() < rhotol) | (rho.abs() < _REL_BREAKDOWN_EPS * r0norm * rkNorm)
        phat = psolve(pk)
        apk = matvec(phat)
        rv = torch.dot(apk, r0)
        badRv = rv.abs() < _REL_BREAKDOWN_EPS * rho.abs()
        stopRho = live & badRho
        stopRv = live & ~badRho & badRv
        go = live & ~badRho & ~badRv

        alpha = rho / rv
        sk = rk - alpha * apk
        convS = go & (torch.linalg.norm(sk) < atolT)                   # the half step already meets the tolerance
        xHalf = xk + alpha * phat

        shat = psolve(sk)
        ask = matvec(shat)
        omega = torch.dot(ask, sk) / torch.dot(ask, ask)
        badOmega = ~torch.isfinite(omega) | (omega.abs() < omegatol)
        stopOmega = go & ~convS & badOmega
        full = go & ~convS & ~badOmega

        xNew = xk + alpha * phat + omega * shat
        rNew = sk - omega * ask
        beta = (torch.dot(rNew, r0) / rho) * (alpha / omega)
        pNew = rNew + beta * (pk - omega * apk)
        rNewNorm = torch.linalg.norm(rNew)
        thr = thresholdExceeded(xNew, threshold, dim)
        thr = torch.zeros_like(full) if thr is None else (full & thr)
        convR = full & ~thr & (rNewNorm < atolT)
        cont = full & ~thr & ~convR

        xk = torch.where(convS, xHalf, torch.where(full, xNew, xk))
        norms.append(rNewNorm)
        valid.append(full)
        status = torch.where(stopRho, RHO, torch.where(stopRv | stopOmega, DEN, torch.where(convS, it,
                             torch.where(thr, THR, torch.where(convR, it + one, status)))))
        pk = torch.where(cont, pNew, pk)
        rk = torch.where(cont, rNew, rk)
        rkNorm = torch.where(cont, rNewNorm, rkNorm)
        live = cont
        it = it + one

        if (k + 1) % checkEvery == 0 and not bool(live):
            break

    return xk, status, norms, valid, used


def bicgstab(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x0: Optional[torch.Tensor] = None, *, tol: float = 0.0,
             rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
             precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None, threshold: Optional[float] = None,
             dim: int = 1, checkEvery: int = 8, maxRestarts: int = 2) -> SolveResult:
    """Solve `matvec(x) == b`. `tol` / `atol` absolute floors, `rtol * ||b||` the relative one; the residual is the true one (right
    preconditioning). `maxRestarts`: how often a solve that reported convergence but fails the final true-residual check is restarted."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)
    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10
    psolve = makePsolve(precond)

    bnrm2 = torch.linalg.norm(b)
    atolT = absoluteTolerance(bnrm2, tol, rtol, atol)
    if bool(bnrm2 == 0):                                                # b == 0: the answer is zero, empty history (as the loop it replaces)
        dev = b.device
        return SolveResult(x=b, status=torch.zeros((), dtype=torch.int32, device=dev), norms=torch.zeros(0, dtype=b.dtype, device=dev),
                           valid=torch.zeros(0, dtype=torch.bool, device=dev))
    eps = torch.finfo(xk.dtype).eps
    rhotol = eps ** 2
    omegatol = rhotol

    rk = b.clone() if x0 is None else b - matvec(xk)                    # A @ 0 == 0 exactly, so the cold start skips the matvec
    allNorms, allValid = [], []
    budget = maxiter
    for attempt in range(maxRestarts + 1):
        xk, status, norms, valid, used = _run(matvec, psolve, b, xk, rk, atolT, rhotol, omegatol, budget, threshold, dim, checkEvery)
        allNorms += norms
        allValid += valid
        true = torch.linalg.norm(matvec(xk) - b)                         # the verified true residual of the returned iterate
        statusInt = int(status)
        if statusInt >= 0 and not bool(true < atolT) and attempt < maxRestarts and budget - used > 0:
            budget -= used                                               # the recurrence under-reported: restart from the iterate
            rk = b - matvec(xk)
            continue
        break
    if statusInt >= 0 and not bool(true < atolT):
        status = torch.full((), int(Status.BUDGET), dtype=torch.int32, device=b.device)
    norms = torch.stack(allNorms) if allNorms else torch.zeros(0, dtype=b.dtype, device=b.device)
    valid = torch.stack(allValid) if allValid else torch.zeros(0, dtype=torch.bool, device=b.device)
    return SolveResult(x=xk, status=status, norms=norms, valid=valid, stampedResidual=true)
