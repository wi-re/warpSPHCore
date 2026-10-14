"""Preconditioned conjugate gradients, device-predicated (see `common.py`).

Same algorithm, same arithmetic and same status codes as the loop it replaces (`warpSPH.modules.shifting.cgSolve`): the iterate it returns is
bitwise the one that loop returned. The tests, the breakdown guards and the history append all happen on the device; the host reads one flag per
`checkEvery` iterations and the status once at the end.

CG is only well-posed for a symmetric positive definite operator. A non-positive or non-finite `r^T M r` / `p^T A p` is reported as
`Status.INDEFINITE`; the caller flips the sign of a negative definite operator, right-hand side and preconditioner (the IISPH pressure operator).
"""

from typing import Callable, Optional, Union

import torch

from .common import SolveResult, Status, absoluteTolerance, makePsolve, thresholdExceeded

__all__ = ['cg']


def cg(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x0: Optional[torch.Tensor] = None, *,
       tol: float = 0.0, rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
       precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
       threshold: Optional[float] = None, dim: int = 1, checkEvery: int = 8, stamp: bool = True) -> SolveResult:
    """Solve `matvec(x) == b`. Converged: `||r|| < max(atol, tol, rtol ||b||)`; `threshold`: bail out when some `dim`-component entry of `x`
    exceeds it. `stamp`: after a failing solve, append the verified true residual (one matvec) to the history, as the legacy convention did."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)
    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10
    psolve = makePsolve(precond)

    bnrm2 = torch.linalg.norm(b)
    atolT = absoluteTolerance(bnrm2, tol, rtol, atol)
    trivial = bnrm2 == 0                                               # b == 0: x0 is the answer, empty history

    dev = b.device
    INDEFINITE = torch.full((), int(Status.INDEFINITE), dtype=torch.int32, device=dev)
    THRESHOLD = torch.full((), int(Status.THRESHOLD), dtype=torch.int32, device=dev)
    ZERO = torch.zeros((), dtype=torch.int32, device=dev)
    one = torch.ones((), dtype=torch.int32, device=dev)

    r = b - matvec(xk)
    rnrm = torch.linalg.norm(r)
    conv0 = rnrm < atolT
    thr0 = thresholdExceeded(xk, threshold, dim)
    thr0 = torch.zeros_like(conv0) if thr0 is None else (~conv0 & thr0)
    live = ~(trivial | conv0 | thr0)                                     # still iterating
    status = torch.where(thr0, THRESHOLD, torch.where(live, torch.full((), int(Status.BUDGET), dtype=torch.int32, device=dev), ZERO))
    norms, valid = [rnrm], [~trivial]

    z = psolve(r)
    p = z.clone()
    rz = torch.dot(r, z)
    it = ZERO                                                            # iteration index on the device (the status of a converged solve)

    for k in range(maxiter if bool(live) else 0):                        # one read: a converged / trivial / out-of-range start skips the loop
        ok1 = torch.isfinite(rz) & (rz > 0)
        ap = matvec(p)
        pap = torch.dot(p, ap)
        ok = ok1 & torch.isfinite(pap) & (pap > 0)
        go = live & ok
        stop = live & ~ok

        alpha = rz / pap
        xn = xk + alpha * p
        rn = r - alpha * ap
        rnrm = torch.linalg.norm(rn)
        conv = go & (rnrm < atolT)
        thr = thresholdExceeded(xn, threshold, dim)
        thr = None if thr is None else (go & ~conv & thr)
        cont = go & ~conv if thr is None else (go & ~conv & ~thr)

        xk = torch.where(go, xn, xk)
        r = torch.where(go, rn, r)
        norms.append(rnrm)
        valid.append(go)
        status = torch.where(stop, INDEFINITE, torch.where(conv, it, status if thr is None else torch.where(thr, THRESHOLD, status)))

        zn = psolve(rn)
        rzn = torch.dot(rn, zn)
        pn = zn + (rzn / rz) * p
        p = torch.where(cont, pn, p)
        rz = torch.where(cont, rzn, rz)
        live = cont
        it = it + one

        if (k + 1) % checkEvery == 0 and not bool(live):
            break

    result = SolveResult(x=xk, status=status, norms=torch.stack(norms), valid=torch.stack(valid))
    if stamp and result.statusInt() < 0:
        result.stampedResidual = torch.linalg.norm(matvec(xk) - b)
    return result
