"""Left-preconditioned BiCG with a verified original-system residual.

For B = P A, B.T = A.T P.T: `precond` applies P (the inverse
preconditioner), and `precondT` applies its transpose. Omitting precondT
asserts that P is symmetric, as for a diagonal preconditioner. An independent
unpreconditioned residual recurrence provides the stopping statistic; every
successful candidate is verified against b - A x before return.
"""

from typing import Callable, Optional, Union

import torch

from .common import (
    PassResult, SolveResult, Status, checkedAction, checkedPreconditioner,
    converged, runVerified, safeDivide, thresholdExceeded, validateSolve,
)

__all__ = ['bicg']
_REL_BREAKDOWN_EPS = 1e-6


def _run(matvec, matvecT, psolve, psolveT, xk, residual, target, budget, zeroRhs, threshold, dim, checkEvery):
    dev = residual.device
    i32 = lambda v: torch.full((), int(v), dtype=torch.int32, device=dev)
    zero, rhoStatus, denStatus, thrStatus, budgetStatus = (i32(v) for v in (0, Status.RHO_BREAKDOWN, Status.DENOMINATOR_BREAKDOWN, Status.THRESHOLD, Status.BUDGET))
    rnrm = torch.linalg.norm(residual)
    conv0 = converged(rnrm, target)
    thr0 = thresholdExceeded(xk, threshold, dim)
    thr0 = torch.zeros_like(conv0) if thr0 is None else thr0
    live = ~(conv0 | thr0)
    status = torch.where(thr0, thrStatus, torch.where(live, budgetStatus, zero))
    norms, valid = [rnrm], [~(zeroRhs & conv0)]
    r = psolve(residual)
    rhat, p, pT = r.clone(), r.clone(), r.clone()
    it, completed, used = zero, zero, 0

    for k in range(budget if bool(live) else 0):
        used = k + 1
        rho = torch.dot(rhat, r)
        badRho = ~torch.isfinite(rho) | (rho == 0) | (rho.abs() < _REL_BREAKDOWN_EPS * torch.linalg.norm(rhat) * torch.linalg.norm(r))
        apOriginal = matvec(p)
        ap = psolve(apOriginal)
        denom = torch.dot(rhat, ap)
        badDen = ~torch.isfinite(denom) | (denom == 0) | (denom.abs() < _REL_BREAKDOWN_EPS * torch.linalg.norm(rhat) * torch.linalg.norm(ap))
        alpha = safeDivide(rho, denom)
        xn = xk + alpha * p
        rn = r - alpha * ap
        residualNew = residual - alpha * apOriginal
        rnrm = torch.linalg.norm(residualNew)
        badDen = badDen | ~torch.isfinite(rnrm) | ~torch.isfinite(xn).all()
        stopRho, stopDen = live & badRho, live & ~badRho & badDen
        go = live & ~badRho & ~badDen
        thr = thresholdExceeded(xn, threshold, dim)
        thr = torch.zeros_like(go) if thr is None else (go & thr)
        conv = go & ~thr & converged(rnrm, target)
        flowing = go & ~conv & ~thr
        rhatn = rhat - alpha * matvecT(psolveT(pT))
        beta = safeDivide(torch.dot(rhatn, rn), rho)
        badBeta = flowing & ~torch.isfinite(beta)
        cont = flowing & ~badBeta

        xk = torch.where(go, xn, xk)
        r = torch.where(go, rn, r)
        residual = torch.where(go, residualNew, residual)
        norms.append(rnrm)
        valid.append(go)
        status = torch.where(stopRho, rhoStatus, torch.where(stopDen | badBeta, denStatus, torch.where(conv, it, torch.where(thr, thrStatus, status))))
        completed = completed + go.to(torch.int32)
        p = torch.where(cont, r + beta * p, p)
        rhat = torch.where(cont, rhatn, rhat)
        pT = torch.where(cont, rhatn + beta * pT, pT)
        live = cont
        it = it + 1
        if (k + 1) % checkEvery == 0 and not bool(live):
            break
    return PassResult(xk, status, norms, valid, completed, used)


def bicg(matvec: Callable[[torch.Tensor], torch.Tensor], matvecT: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
         x0: Optional[torch.Tensor] = None, *, tol: float = 0.0, rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
         precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None, threshold: Optional[float] = None,
         dim: int = 1, checkEvery: int = 8, stamp: bool = True, maxRestarts: int = 2,
         precondT: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None) -> SolveResult:
    """Solve A x = b using A.T. Tolerances/budgets/verification are as for cg.

    History norms describe the original system, independent of preconditioner
    scaling. A nonsymmetric preconditioner requires its transpose in precondT;
    omitting it declares a symmetric inverse preconditioner.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts)
    if precond is None and precondT is not None:
        raise ValueError('precondT requires precond')
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    applyT = checkedAction(matvecT, b, 'matvecT', counts, 'adjointMatvecs')
    psolve = checkedPreconditioner(precond, b, counts)
    psolveT = psolve if precondT is None else checkedPreconditioner(precondT, b, counts, 'precondT')
    run = lambda x, r, target, budget, zero: _run(apply, applyT, psolve, psolveT, x, r, target, budget, zero, threshold, dim, checkEvery)
    return runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter,
                       maxRestarts=maxRestarts, stamp=stamp, counts=counts)
