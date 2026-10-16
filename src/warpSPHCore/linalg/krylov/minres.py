"""Short-recurrence MINRES with explicit assumptions and original residuals.

The Lanczos/Givens recurrence follows Paige and Saunders (1975):
https://web.stanford.edu/group/SOL/software/minres/
An accompanying A-times-direction recurrence tracks the original residual;
the shared driver verifies it directly before any successful return.
"""

import torch

from ..operator import LinearOperator
from .common import (
    PassResult, SolveResult, Status, checkedAction, checkedPreconditioner,
    converged, runVerified, safeDivide, thresholdExceeded, validateSolve,
)

__all__ = ['minres']


def _run(apply, psolve, x, residual, target, budget, threshold, dim, checkEvery, history):
    scalar = lambda v: torch.tensor(v, dtype=torch.int32, device=x.device)
    completed, used = scalar(0), 0
    norm = residual.norm()
    initialThreshold = thresholdExceeded(x, threshold, dim)
    initialThreshold = torch.zeros((), dtype=torch.bool, device=x.device) if initialThreshold is None else initialThreshold
    live = ~initialThreshold & ~converged(norm, target) & torch.isfinite(norm)
    status = torch.where(initialThreshold, scalar(Status.THRESHOLD),
                         torch.where(~torch.isfinite(norm), scalar(Status.DENOMINATOR_BREAKDOWN),
                                     torch.where(live, scalar(Status.BUDGET), scalar(0))))
    norms = [norm] if history else []
    valid = [torch.ones((), dtype=torch.bool, device=x.device)] if history else []
    if budget == 0 or not bool(live):
        return PassResult(x, status, norms, valid, completed, used)

    rPrevious, rCurrent = torch.zeros_like(residual), residual.clone()
    y = psolve(rCurrent)
    betaSquared = torch.dot(rCurrent, y)
    positive = torch.isfinite(betaSquared) & (betaSquared > 0) & torch.isfinite(y).all()
    status = torch.where(live & ~positive, scalar(Status.INDEFINITE), status)
    live = live & positive
    if not bool(live):
        return PassResult(x, status, norms, valid, completed, used)
    beta = betaSquared.clamp_min(0).sqrt()
    oldBeta = beta.new_ones(())
    cosine, sine = beta.new_tensor(-1.), beta.new_zeros(())
    dbar, epsilon, phibar = beta.new_zeros(()), beta.new_zeros(()), beta.clone()
    wOld, wPrevious = torch.zeros_like(x), torch.zeros_like(x)
    awOld, awPrevious = torch.zeros_like(x), torch.zeros_like(x)
    eps = torch.finfo(x.dtype).eps

    for k in range(budget):
        used = k + 1
        v = torch.where(live, safeDivide(y, beta), torch.zeros_like(y))
        av = apply(v)
        lanczos = av - safeDivide(beta, oldBeta) * rPrevious
        alpha = torch.dot(v, lanczos)
        rNext = lanczos - safeDivide(alpha, beta) * rCurrent
        yNext = psolve(rNext)
        betaNextSquared = torch.dot(rNext, yNext)
        betaNext = betaNextSquared.clamp_min(0).sqrt()

        delta = cosine * dbar + sine * alpha
        gbar = sine * dbar - cosine * alpha
        gamma = torch.hypot(gbar, betaNext)
        nextCosine, nextSine = safeDivide(gbar, gamma), safeDivide(betaNext, gamma)
        phi = nextCosine * phibar
        w = safeDivide(v - epsilon * wOld - delta * wPrevious, gamma)
        aw = safeDivide(av - epsilon * awOld - delta * awPrevious, gamma)
        candidate = x + phi * w
        physical = residual - phi * aw
        norm = physical.norm()

        invalidInverse = ~torch.isfinite(betaNextSquared) | (betaNextSquared < 0) | ~torch.isfinite(yNext).all()
        finite = (torch.isfinite(alpha) & torch.isfinite(gamma) & (gamma > 0)
                  & torch.isfinite(candidate).all() & torch.isfinite(w).all()
                  & torch.isfinite(aw).all() & torch.isfinite(norm))
        go = live & ~invalidInverse & finite
        exceeded = thresholdExceeded(candidate, threshold, dim)
        exceeded = torch.zeros_like(go) if exceeded is None else go & exceeded
        convergence = go & ~exceeded & converged(norm, target)
        # A zero inverse on a nonzero active residual violates solved-subspace
        # positivity. At exact convergence, no further inverse direction is used.
        killedResidual = go & ~convergence & ~exceeded & (betaNextSquared == 0) & (rNext.norm() > 0)
        lanczosScale = alpha.abs() if k == 0 else alpha.abs() + beta
        happy = go & (betaNext <= eps * lanczosScale)
        cont = go & ~exceeded & ~convergence & ~happy & ~killedResidual
        status = torch.where(live & (invalidInverse | killedResidual), scalar(Status.INDEFINITE),
                             torch.where(live & ~invalidInverse & ~finite, scalar(Status.DENOMINATOR_BREAKDOWN),
                                         torch.where(exceeded, scalar(Status.THRESHOLD),
                                                     torch.where(convergence | (happy & ~exceeded), scalar(k + 1), status))))
        x = torch.where(go, candidate, x)
        residual = torch.where(go, physical, residual)
        completed = completed + go.to(torch.int32)
        if history:
            norms.append(norm)
            valid.append(go)
        wOld, wPrevious = torch.where(cont, wPrevious, wOld), torch.where(cont, w, wPrevious)
        awOld, awPrevious = torch.where(cont, awPrevious, awOld), torch.where(cont, aw, awPrevious)
        rPrevious, rCurrent = torch.where(cont, rCurrent, rPrevious), torch.where(cont, rNext, rCurrent)
        y = torch.where(cont, yNext, y)
        oldBeta, beta = torch.where(cont, beta, oldBeta), torch.where(cont, betaNext, beta)
        epsilon, dbar = torch.where(cont, sine * betaNext, epsilon), torch.where(cont, -cosine * betaNext, dbar)
        cosine, sine = torch.where(cont, nextCosine, cosine), torch.where(cont, nextSine, sine)
        phibar = torch.where(cont, nextSine * phibar, phibar)
        live = cont
        if (k + 1) % checkEvery == 0 and not bool(live):
            break
    return PassResult(x, status, norms, valid, completed, used)


def minres(matvec, b, x0=None, *, symmetric=False, precond=None,
           preconditionerSPD=False, tol=0.0, rtol=1e-5, atol=0.0,
           maxiter=None, threshold=None, dim=1, checkEvery=8, maxRestarts=2,
           history=True) -> SolveResult:
    """Solve a declared symmetric, possibly indefinite or singular system.

    ``symmetric=True`` is required; no callback probing infers self-adjointness.
    An inverse diagonal is checked positive/finite. A general inverse action
    needs ``preconditionerSPD=True`` or a helper with ``spdVerified=True``.
    The assertion may concern the solved subspace (e.g. projected inverses),
    and must hold throughout the solve. Active nonpositive inverse quadratic
    forms fail with INDEFINITE; this is not a global symmetry/SPD proof.

    Success certifies the original residual using shared tolerances/budgets and
    bounded residual-replacement passes. Incompatible singular requests fail;
    least-squares or MINRES-QLP minimum-length success is not supplied.
    Without an autograd trace, recurrence storage is O(n); optional history is
    O(maxiter). No growing basis/triangular solve, workspace or capture is used.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts)
    if not isinstance(symmetric, bool) or not symmetric:
        raise ValueError('minres requires an explicit symmetric=True declaration')
    if not isinstance(preconditionerSPD, bool) or not isinstance(history, bool):
        raise ValueError('preconditionerSPD and history must be bool')
    if b.dtype not in (torch.float32, torch.float64) or b.device.type not in ('cpu', 'cuda'):
        raise ValueError('minres supports CPU/CUDA float32 and float64 vectors')
    if isinstance(matvec, LinearOperator):
        if matvec.shape != (b.numel(), b.numel()) or matvec.dtype != b.dtype or matvec.device != b.device:
            raise ValueError('operator shape, dtype and device must match the square system')
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    psolve = checkedPreconditioner(precond, b, counts)
    if isinstance(precond, torch.Tensor):
        if not bool((torch.isfinite(precond) & (precond > 0)).all()):
            raise ValueError('inverse diagonal must be finite and strictly positive')
    elif precond is not None and not preconditionerSPD and getattr(precond, 'spdVerified', False) is not True:
        raise ValueError('general inverse action requires preconditionerSPD=True or an explicit SPD certificate')
    run = lambda x, r, target, budget, zero: _run(apply, psolve, x, r, target, budget,
                                                threshold, dim, checkEvery, history)
    return runVerified(run, apply, b, x0, tol=tol, rtol=rtol, atol=atol,
                       maxiter=maxiter, maxRestarts=maxRestarts, stamp='always', counts=counts)
