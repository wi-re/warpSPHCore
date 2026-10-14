"""Left-preconditioned bi-conjugate gradients, device-predicated (see `common.py`).

Needs the adjoint operator `matvecT` (the dual residual lives in the `A^T`-Krylov space). The incompressible pressure dispatch supplies the
placeholder `A^T := A`, exact for a symmetric operator. The solver iterates on `M^{-1} A x = M^{-1} b`, whose adjoint is `A^T M^{-T}`; the
preconditioner is a symmetric (diagonal) one, so the adjoint matvec is `matvecT(M^{-1} q)`.

This is a correction of the loop it replaces (`warpSPH.modules.shifting.bicgSolve`, kept as `tests/linalg/_legacyKrylov.py`'s reference of the
*stopping logic*), which had three defects: `beta` carried the BiCGStab factor `alpha / alpha_prev` and was lagged by one iteration (textbook BiCG:
`beta = rho_{k+1} / rho_k` with `rho = rhat . r` of the *updated* residuals), and the adjoint applied the preconditioner on the wrong side (`M^{-1} A^T q` instead of `A^T M^{-1} q`). With `A^T = A` and no preconditioner the iterates
now coincide with CG's, which the tests check; the old loop needed 14 iterations where CG needs 9 on a 200-unknown SPD system. The `|denom| < eps |rho|`
guard of the old loop was a double-precision comparison and stays one.
"""

from typing import Callable, Optional, Union

import torch

from .common import SolveResult, Status, absoluteTolerance, makePsolve, thresholdExceeded

__all__ = ['bicg']

#: relative breakdown margin (scale-free; fires only once the corresponding scalar is already ~1e6-scaled)
_REL_BREAKDOWN_EPS = 1e-6


def bicg(matvec: Callable[[torch.Tensor], torch.Tensor], matvecT: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
         x0: Optional[torch.Tensor] = None, *, tol: float = 0.0, rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
         precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None, threshold: Optional[float] = None,
         dim: int = 1, checkEvery: int = 8, stamp: bool = True) -> SolveResult:
    """Solve `matvec(x) == b`; `matvecT` is the adjoint operator. Status codes and arguments as `cg`; breakdowns: `RHO_BREAKDOWN` (the
    dual-residual inner product vanished), `DENOMINATOR_BREAKDOWN` (the curvature or `beta` is unusable)."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)
    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    psolve = makePsolve(precond)
    if precond is None:
        Apre, ApreT, bpre = matvec, matvecT, b
    else:
        Apre = lambda p: psolve(matvec(p))
        ApreT = lambda q: matvecT(psolve(q))                           # (M^{-1} A)^T = A^T M^{-T}, M diagonal
        bpre = psolve(b)

    bnrm2 = torch.linalg.norm(b)
    atolT = absoluteTolerance(bnrm2, tol, rtol, atol)
    trivial = bnrm2 == 0

    dev = b.device
    i32 = lambda v: torch.full((), int(v), dtype=torch.int32, device=dev)
    ZERO, RHO, DEN, THR, BUDGET = i32(0), i32(Status.RHO_BREAKDOWN), i32(Status.DENOMINATOR_BREAKDOWN), i32(Status.THRESHOLD), i32(Status.BUDGET)
    one = torch.ones((), dtype=torch.int32, device=dev)

    r = bpre - Apre(xk)
    rnrm = torch.linalg.norm(r)
    conv0 = rnrm < atolT
    thr0 = thresholdExceeded(xk, threshold, dim)
    thr0 = torch.zeros_like(conv0) if thr0 is None else (~conv0 & thr0)
    live = ~(trivial | conv0 | thr0)
    status = torch.where(thr0, THR, torch.where(live, BUDGET, ZERO))
    norms, valid = [rnrm], [~trivial]

    rhat = r.clone()
    p = r.clone()
    pT = rhat.clone()
    it = ZERO

    for k in range(maxiter if bool(live) else 0):
        rho = torch.dot(rhat, r)
        badRho = ~torch.isfinite(rho) | (rho.abs() < _REL_BREAKDOWN_EPS * torch.linalg.norm(rhat) * torch.linalg.norm(r))
        ap = Apre(p)
        denom = torch.dot(rhat, ap)
        badDen = ~torch.isfinite(denom) | (denom.abs().double() < _REL_BREAKDOWN_EPS * rho.abs().double())
        stopRho = live & badRho
        stopDen = live & ~badRho & badDen
        go = live & ~badRho & ~badDen

        alpha = rho / denom
        xn = xk + alpha * p
        rn = r - alpha * ap
        rnrm = torch.linalg.norm(rn)
        conv = go & (rnrm < atolT)
        thr = thresholdExceeded(xn, threshold, dim)
        thr = torch.zeros_like(conv) if thr is None else (go & ~conv & thr)
        flowing = go & ~conv & ~thr

        rhatn = rhat - alpha * ApreT(pT)                                 # the dual residual of the updated iterate: one adjoint matvec per iterate
        beta = torch.dot(rhatn, rn) / rho                                # rho_{k+1} / rho_k
        badBeta = flowing & ~torch.isfinite(beta)
        cont = flowing & ~badBeta

        xk = torch.where(go, xn, xk)
        r = torch.where(go, rn, r)
        norms.append(rnrm)
        valid.append(go)
        status = torch.where(stopRho, RHO, torch.where(stopDen | badBeta, DEN, torch.where(conv, it, torch.where(thr, THR, status))))

        pn = r + beta * p                                                # r is the updated residual here
        pTn = rhatn + beta * pT
        p = torch.where(cont, pn, p)
        rhat = torch.where(cont, rhatn, rhat)
        pT = torch.where(cont, pTn, pT)
        live = cont
        it = it + one

        if (k + 1) % checkEvery == 0 and not bool(live):
            break

    result = SolveResult(x=xk, status=status, norms=torch.stack(norms), valid=torch.stack(valid))
    if stamp and result.statusInt() < 0:
        result.stampedResidual = torch.linalg.norm(matvec(xk) - b)
    return result
