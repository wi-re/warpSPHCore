"""Right-preconditioned device-predicated BiCGStab.

Full and half-step recurrence convergence is verified against b - A x.
Bounded restarts replace a drifted recurrence; failure retains the last
accepted finite iterate. See SolveResult for completed updates and actual work.
"""

from typing import Callable, Optional, Union

import torch

from .common import (
    PassResult, SolveResult, Status, checkedAction, checkedPreconditioner,
    converged, runVerified, safeDivide, thresholdExceeded, validateSolve,
)

__all__ = ['bicgstab']
_REL_BREAKDOWN_EPS = 1e-6


def _run(matvec, psolve, xk, rk, target, budget, zeroRhs, threshold, dim, checkEvery):
    dev = rk.device
    i32 = lambda v: torch.full((), int(v), dtype=torch.int32, device=dev)
    zero, rhoStatus, denStatus, thrStatus, budgetStatus = (i32(v) for v in (0, Status.RHO_BREAKDOWN, Status.DENOMINATOR_BREAKDOWN, Status.THRESHOLD, Status.BUDGET))
    r0, pk = rk.clone(), rk.clone()
    r0norm, rkNorm = torch.linalg.norm(r0), torch.linalg.norm(rk)
    conv0 = converged(rkNorm, target)
    thr0 = thresholdExceeded(xk, threshold, dim)
    thr0 = torch.zeros_like(conv0) if thr0 is None else thr0
    live = ~(conv0 | thr0)
    status = torch.where(thr0, thrStatus, torch.where(live, budgetStatus, zero))
    norms, valid = [], []
    it, completed, used = zero, zero, 0
    for k in range(budget if bool(live) else 0):
        used = k + 1
        rho = torch.dot(rk, r0)
        badRho = ~torch.isfinite(rho) | (rho == 0) | (rho.abs() < _REL_BREAKDOWN_EPS * r0norm * rkNorm)
        phat = psolve(pk)
        apk = matvec(phat)
        rv = torch.dot(apk, r0)
        badRv = ~torch.isfinite(rv) | (rv == 0) | (rv.abs() < _REL_BREAKDOWN_EPS * r0norm * torch.linalg.norm(apk))
        alpha = safeDivide(rho, rv)
        sk = rk - alpha * apk
        xHalf = xk + alpha * phat
        badRv = badRv | ~torch.isfinite(xHalf).all()
        stopRho, stopRv = live & badRho, live & ~badRho & badRv
        go = live & ~badRho & ~badRv
        convS = go & converged(torch.linalg.norm(sk), target)
        thrHalf = thresholdExceeded(xHalf, threshold, dim)
        thrHalf = torch.zeros_like(go) if thrHalf is None else (convS & thrHalf)
        convS = convS & ~thrHalf

        shat = psolve(sk)
        ask = matvec(shat)
        ask2 = torch.dot(ask, ask)
        omega = safeDivide(torch.dot(ask, sk), ask2)
        badOmega = ~torch.isfinite(omega) | (omega == 0) | ~torch.isfinite(ask2) | (ask2 == 0)
        xNew = xk + alpha * phat + omega * shat
        rNew = sk - omega * ask
        beta = safeDivide(torch.dot(rNew, r0), rho) * safeDivide(alpha, omega)
        pNew = rNew + beta * (pk - omega * apk)
        rNewNorm = torch.linalg.norm(rNew)
        badOmega = badOmega | ~torch.isfinite(xNew).all() | ~torch.isfinite(rNewNorm)
        stopOmega = go & ~convS & ~thrHalf & badOmega
        full = go & ~convS & ~thrHalf & ~badOmega
        thr = thresholdExceeded(xNew, threshold, dim)
        thr = torch.zeros_like(full) if thr is None else (full & thr)
        convR = full & ~thr & converged(rNewNorm, target)
        flowing = full & ~thr & ~convR
        badBeta = flowing & (~torch.isfinite(beta) | ~torch.isfinite(pNew).all())
        cont = flowing & ~badBeta

        xk = torch.where(convS | thrHalf, xHalf, torch.where(full, xNew, xk))
        norms.append(rNewNorm)
        valid.append(full)
        status = torch.where(stopRho, rhoStatus, torch.where(stopRv | stopOmega | badBeta, denStatus, torch.where(convS, it,
                             torch.where(thr | thrHalf, thrStatus, torch.where(convR, it + 1, status)))))
        completed = completed + (convS | thrHalf | full).to(torch.int32)
        pk = torch.where(cont, pNew, pk)
        rk = torch.where(cont, rNew, rk)
        rkNorm = torch.where(cont, rNewNorm, rkNorm)
        live = cont
        it = it + 1
        if (k + 1) % checkEvery == 0 and not bool(live):
            break
    return PassResult(xk, status, norms, valid, completed, used)


def bicgstab(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x0: Optional[torch.Tensor] = None, *, tol: float = 0.0,
             rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
             precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None, threshold: Optional[float] = None,
             dim: int = 1, checkEvery: int = 8, maxRestarts: int = 2,
             workspace=None, matvecOut=None, precondOut=None) -> SolveResult:
    """Solve A x = b with right preconditioning and a verified true residual.

    Success: ||b - A x|| <= max(tol, atol, rtol ||b||), including exact zero.
    `maxiter` bounds scheduled iterations across all residual-replacement
    passes. Legacy histories contain full-step norms and the verified final
    residual, except an already-solved zero RHS has an empty history.
    A BiCGStabWorkspace opts into native inference-only vector updates and
    optional cached guarded chunks. Supplied-output callbacks must implement
    the same actions and preserve the supplied destination's storage/metadata.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts)
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    psolve = checkedPreconditioner(precond, b, counts)
    if workspace is not None:
        from .bicgstabWorkspace import BiCGStabWorkspace
        if not isinstance(workspace, BiCGStabWorkspace):
            raise ValueError('workspace must be a BiCGStabWorkspace')
        workspace._check(b, maxiter)
        for value in (b, x0, precond):
            if isinstance(value, torch.Tensor):
                workspace._inert(value)
        baseApply = apply
        apply = lambda v: workspace._inert(baseApply(v))

        def checkedOut(action, name, counter):
            if action is None:
                return None
            if not callable(action):
                raise ValueError(f'{name} must be callable')

            def into(v, out):
                counts[counter] += 1
                # A view protects workspace buffer metadata if a bad callback
                # rebinds its destination. The backing address stays owned.
                destination = out.view_as(out)
                metadata = (destination.data_ptr(), destination.shape, destination.stride(), destination.dtype, destination.device)
                try:
                    returned = action(v, destination)
                    if returned is not None and returned is not destination:
                        raise ValueError(f'{name} must return None or its supplied output')
                    if metadata != (destination.data_ptr(), destination.shape, destination.stride(), destination.dtype, destination.device):
                        raise ValueError(f'{name} must preserve output storage and metadata')
                    workspace._inert(destination)
                except Exception:
                    out.detach_()
                    raise
            return into

        if precondOut is not None and precond is None:
            raise ValueError('precondOut requires a preconditioner')
        outAction = checkedOut(matvecOut, 'matvecOut', 'matvecs')
        outPrecond = checkedOut(precondOut, 'precondOut', 'preconditionerApplications')
        workspace._bindCapture(matvec, matvecOut, precond, threshold, dim, checkEvery, precondOut)
        run = lambda x, r, target, budget, zero: workspace._run(
            apply, psolve, precond, x, r, target, budget, zero, threshold, dim, checkEvery,
            counts, outAction, outPrecond)
        workspace._busy = True
        try:
            result = runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter,
                                 maxRestarts=maxRestarts, stamp='always', counts=counts)
            result.x = result.x.clone()
            return result
        finally:
            workspace._busy = False
    if matvecOut is not None or precondOut is not None:
        raise ValueError('supplied-output callbacks require a BiCGStabWorkspace')
    run = lambda x, r, target, budget, zero: _run(apply, psolve, x, r, target, budget, zero, threshold, dim, checkEvery)
    return runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter,
                       maxRestarts=maxRestarts, stamp='always', counts=counts)
