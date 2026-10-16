"""Device-predicated preconditioned CG with a verified original residual.

The operator and inverse preconditioner must be symmetric positive definite
(on the solved subspace). Non-positive/nonfinite curvature is INDEFINITE.
Recurrence convergence is verified and can trigger bounded fresh-residual
restarts. Legacy arithmetic/history is retained on ordinary honest passes.
"""

from typing import Callable, Optional, Union

import torch
from ...autograd.compileGlue import compileGlue

from .common import (
    PassResult, SolveResult, Status, checkedAction, checkedPreconditioner,
    converged, runVerified, safeDivide, thresholdExceeded, validateSolve,
)

__all__ = ['cg']


@compileGlue
def _candidate(xk, r, p, ap, rz, pap, live, it, status, completed, target, indefinite, thrStatus, threshold, dim):
    """Pure Torch chain; opt-in compilation fuses guards, reductions and updates."""
    ok = torch.isfinite(rz) & (rz > 0) & torch.isfinite(pap) & (pap > 0)
    alpha = safeDivide(rz, pap)
    xn = xk + alpha * p
    rn = r - alpha * ap
    rnrm = torch.linalg.norm(rn)
    ok = ok & torch.isfinite(rnrm) & torch.isfinite(xn).all()
    go, stop = live & ok, live & ~ok
    thr = thresholdExceeded(xn, threshold, dim)
    thr = None if thr is None else (go & thr)
    conv = go & converged(rnrm, target)
    if thr is not None:
        conv = conv & ~thr
    cont = go & ~conv if thr is None else (go & ~conv & ~thr)
    status = torch.where(stop, indefinite, torch.where(conv, it, status if thr is None else torch.where(thr, thrStatus, status)))
    return (torch.where(go, xn, xk), torch.where(go, rn, r), rn, rnrm, go, cont,
            status, completed + go.to(torch.int32))


@compileGlue
def _direction(zn, p, rz, rzn, cont):
    pn = zn + safeDivide(rzn, rz) * p
    return torch.where(cont, pn, p), torch.where(cont, rzn, rz)


def _run(matvec, psolve, xk, r, target, budget, zeroRhs, threshold, dim, checkEvery):
    dev = r.device
    i32 = lambda v: torch.full((), int(v), dtype=torch.int32, device=dev)
    indefinite, thrStatus, zero, budgetStatus = (i32(v) for v in (Status.INDEFINITE, Status.THRESHOLD, 0, Status.BUDGET))
    rnrm = torch.linalg.norm(r)
    conv0 = converged(rnrm, target)
    thr0 = thresholdExceeded(xk, threshold, dim)
    thr0 = torch.zeros_like(conv0) if thr0 is None else thr0
    live = ~(conv0 | thr0)
    status = torch.where(thr0, thrStatus, torch.where(live, budgetStatus, zero))
    norms, valid = [rnrm], [~(zeroRhs & conv0)]
    z = psolve(r)
    p = z.clone()
    rz = torch.dot(r, z)
    # Separate scalar objects avoid compiler alias specializations between
    # the first iteration and later ones (both counters initially equal 0).
    it, completed, used = i32(0), i32(0), 0

    for k in range(budget if bool(live) else 0):
        used = k + 1
        ap = matvec(p)
        if ap is p:
            ap = ap.clone()  # identity operators must not add a distinct alias specialization
        # Keep all Krylov inner products on the same reduction implementation:
        # fusing only p.Ap can round it differently from r.z even for A = I,
        # manufacturing a non-unit alpha and defeating exact convergence.
        pap = torch.dot(p, ap)
        xk, r, rn, rnrm, go, cont, status, completed = _candidate(
            xk, r, p, ap, rz, pap, live, it, status, completed, target, indefinite, thrStatus, threshold, dim)
        norms.append(rnrm)
        valid.append(go)
        zn = psolve(rn)
        rzn = torch.dot(rn, zn)
        p, rz = _direction(zn, p, rz, rzn, cont)
        live = cont
        it = it + 1
        if (k + 1) % checkEvery == 0 and not bool(live):
            break
    return PassResult(xk, status, norms, valid, completed, used)


def cg(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x0: Optional[torch.Tensor] = None, *,
       tol: float = 0.0, rtol: float = 1e-5, atol: float = 0.0, maxiter: Optional[int] = None,
       precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
       threshold: Optional[float] = None, dim: int = 1, checkEvery: int = 8, stamp: bool = True,
       maxRestarts: int = 2, workspace=None, matvecOut=None) -> SolveResult:
    """Solve A x = b. Success certifies ||b - A x|| <= max(tol, atol, rtol ||b||).

    `checkEvery` must be positive. `maxiter` bounds all scheduled iterations,
    including frozen work and up to `maxRestarts` residual-replacement passes.
    `stamp=False` suppresses failure history stamping, never verification.
    See SolveResult for actual work and completed-update counts.
    Pass a CGWorkspace for opt-in native inference-only updates. Optional
    `matvecOut(v, out)` writes a workspace-owned destination and returns None
    or that destination; ordinary `matvec` still verifies the original system.
    A workspace constructed with `capture=True` caches guarded CUDA chunks;
    use stable graph-safe callbacks and invalidate hidden storage replacements.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts)
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    psolve = checkedPreconditioner(precond, b, counts)
    if workspace is not None:
        from .cgWorkspace import CGWorkspace
        if not isinstance(workspace, CGWorkspace):
            raise ValueError('workspace must be a CGWorkspace')
        workspace._check(b, maxiter)
        for value in (b, x0, precond):
            if isinstance(value, torch.Tensor):
                workspace._inert(value)
        baseApply = apply
        apply = lambda v: workspace._inert(baseApply(v))
        if matvecOut is not None and not callable(matvecOut):
            raise ValueError('matvecOut must be callable')
        workspace._bindCapture(matvec, matvecOut, precond, threshold, dim, checkEvery)

        def applyOut(v, out):
            counts['matvecs'] += 1
            result = matvecOut(v, out)
            if result is not None and result is not out:
                raise ValueError('matvecOut must return None or its supplied output')
            workspace._inert(out)

        run = lambda x, r, target, budget, zero: workspace._run(
            apply, psolve, precond, x, r, target, budget, zero, threshold, dim, checkEvery,
            counts, applyOut if matvecOut is not None else None)
        workspace._busy = True
        try:
            result = runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter,
                                 maxRestarts=maxRestarts, stamp=stamp, counts=counts)
            result.x = result.x.clone()
            return result
        finally:
            workspace._busy = False
    if matvecOut is not None:
        raise ValueError('matvecOut requires a CGWorkspace')
    run = lambda x, r, target, budget, zero: _run(apply, psolve, x, r, target, budget, zero, threshold, dim, checkEvery)
    return runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter,
                       maxRestarts=maxRestarts, stamp=stamp, counts=counts)
