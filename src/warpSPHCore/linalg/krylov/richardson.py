"""Bounded Richardson updates with explicit step search and best-iterate return."""

import torch

from ..operator import LinearOperator
from .common import (
    SolveResult, Status, absoluteTolerance, checkedAction, converged,
    finiteScalar, nonnegativeInt, thresholdExceeded, validateSolve,
)

__all__ = ['richardson']


def richardson(matvec, b, x0=None, *, omega=None, backtracking=True,
               maxBacktracks=8, minStep=0.0, maxStep=None, stagnationWindow=8,
               tol=0.0, rtol=1e-5, atol=0.0, maxiter=None, threshold=None,
               dim=1, history=True) -> SolveResult:
    """Apply x <- x + step * (b - A x), with no inverse preconditioner.

    ``omega`` selects a fixed signed step. None selects the residual-minimizing
    scalar along r: (r.T A r) / ||A r||^2, evaluated with a normalized A r.
    The automatic sign supports either definiteness; arbitrary indefinite or
    nonsymmetric systems can stagnate. No global convergence is promised.

    Backtracking accepts only a strict residual decrease or verified convergence,
    retries from the same accepted iterate, and halves at most maxBacktracks
    times. Disabling it gives literal fixed/adaptive Richardson updates, including
    possible increases. minStep/maxStep bound magnitudes when supplied. A finite
    best iterate is retained on budget, search, stagnation or nonfinite failure.
    Threshold failure returns the accepted violating candidate, as in the other
    solvers. maxiter bounds outer update/search attempts; iterations counts
    accepted updates and rejected trials count separately.

    Every candidate and the returned iterate use a direct original residual.
    Trial acceptance reads the host; this is an eager numerical baseline, not
    a capturable/checkpointed loop. Without an AD trace, vector storage is O(n)
    and optional accepted-step history is O(maxiter). No implicit solve AD.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, 1, 0)
    if b.dtype not in (torch.float32, torch.float64) or b.device.type not in ('cpu', 'cuda'):
        raise ValueError('richardson supports CPU/CUDA float32 and float64 vectors')
    if not isinstance(backtracking, bool) or not isinstance(history, bool):
        raise ValueError('backtracking and history must be bool')
    maxBacktracks = nonnegativeInt('maxBacktracks', maxBacktracks)
    stagnationWindow = nonnegativeInt('stagnationWindow', stagnationWindow)
    finiteScalar('minStep', minStep)
    if maxStep is not None:
        finiteScalar('maxStep', maxStep)
        if maxStep <= 0 or maxStep < minStep:
            raise ValueError('maxStep must be positive and at least minStep')
    if omega is not None:
        finiteScalar('omega', omega, nonnegative=False)
        if omega == 0 or abs(omega) < minStep or (maxStep is not None and abs(omega) > maxStep):
            raise ValueError('omega magnitude must be nonzero and within step bounds')
    if isinstance(matvec, LinearOperator):
        if matvec.shape != (b.numel(), b.numel()) or matvec.dtype != b.dtype or matvec.device != b.device:
            raise ValueError('operator shape, dtype and device must match the square system')
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    residual = b - apply(x)
    norm = residual.norm()
    target = absoluteTolerance(b.norm(), tol, rtol, atol)
    best, bestNorm, bestIteration = x.clone(), norm.clone(), 0
    completed, loops, trials, shrinks, stagnant, verifications = 0, 0, 0, 0, 0, 1
    stepSize = None
    norms = [norm.clone()] if history else []
    exceeded = thresholdExceeded(x, threshold, dim)
    if exceeded is not None and bool(exceeded):
        status = int(Status.THRESHOLD)
    elif bool(converged(norm, target)):
        status = 0
    elif not bool(torch.isfinite(norm)):
        status = int(Status.DENOMINATOR_BREAKDOWN)
    else:
        status = int(Status.BUDGET)
        for _ in range(maxiter):
            loops += 1
            if omega is None:
                ar = apply(residual)
                scale = ar.abs().amax()
                if not bool(torch.isfinite(scale)):
                    status = int(Status.DENOMINATOR_BREAKDOWN)
                    break
                if not bool(scale > 0):
                    status = int(Status.RICHARDSON_STAGNATION)
                    break
                scaled = ar / scale
                length = scaled.norm()
                step = (torch.dot(residual, scaled / length) / length) / scale
                if maxStep is not None:
                    step = step.clamp(-maxStep, maxStep)
            else:
                step = b.new_tensor(omega)
            if not bool(torch.isfinite(step)):
                status = int(Status.DENOMINATOR_BREAKDOWN)
                break
            if bool(step == 0) or bool(step.abs() < minStep):
                status = int(Status.RICHARDSON_STAGNATION)
                break
            accepted, anyFinite = False, False
            for attempt in range(maxBacktracks + 1 if backtracking else 1):
                candidate = x + step * residual
                trials += 1
                if bool(torch.isfinite(candidate).all()):
                    candidateResidual = b - apply(candidate)
                    verifications += 1
                    candidateNorm = candidateResidual.norm()
                    finite = bool(torch.isfinite(candidateNorm))
                else:
                    finite = False
                anyFinite |= finite
                if finite and (not backtracking or bool(candidateNorm < norm) or bool(converged(candidateNorm, target))):
                    accepted = True
                    break
                if attempt < maxBacktracks and backtracking:
                    reduced = step * .5
                    if bool(reduced == 0) or bool(reduced.abs() < minStep):
                        break
                    step = reduced
                    shrinks += 1
            if not accepted:
                status = int(Status.RICHARDSON_STAGNATION if anyFinite else Status.DENOMINATOR_BREAKDOWN)
                break
            x, residual, norm = candidate, candidateResidual, candidateNorm
            completed += 1
            stepSize = step.clone()
            if history:
                norms.append(norm.clone())
            if bool(norm < bestNorm):
                best, bestNorm, bestIteration, stagnant = x.clone(), norm.clone(), completed, 0
            else:
                stagnant += 1
            exceeded = thresholdExceeded(x, threshold, dim)
            if exceeded is not None and bool(exceeded):
                status = int(Status.THRESHOLD)
                break
            if bool(converged(norm, target)):
                status = completed
                break
            if stagnationWindow and stagnant >= stagnationWindow:
                status = int(Status.RICHARDSON_STAGNATION)
                break
    if status < 0 and status != Status.THRESHOLD:
        x = best
    if trials:
        norm = (b - apply(x)).norm()
        verifications += 1
        if status >= 0 and not bool(converged(norm, target)):
            status = int(Status.RESIDUAL_MISMATCH)
    # Unrequested history never forces per-step lists or host history transfers.
    stored = torch.stack(norms) if norms else b.new_empty(0)
    return SolveResult(
        x=x, status=torch.tensor(status, dtype=torch.int32, device=b.device),
        norms=stored, valid=torch.ones(stored.numel(), dtype=torch.bool, device=b.device),
        stampedResidual=norm.clone(), trueResidual=norm, iterations=torch.tensor(completed, dtype=torch.int32, device=b.device),
        loopIterations=loops, verifications=verifications, trialSteps=trials,
        backtracks=shrinks, bestIteration=bestIteration, stepSize=stepSize,
        _host={'status': status}, **counts)
