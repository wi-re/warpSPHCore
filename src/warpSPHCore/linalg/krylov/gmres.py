"""Restarted real GMRES with twice-orthogonalized Arnoldi and Givens QR.

This is an eager, checkpointed numerical baseline, not a captured workspace.
No normal equations or dense least-squares host-status reads are used.
"""

import torch

from ..operator import LinearOperator
from .common import (
    SolveResult, Status, absoluteTolerance, checkedAction, checkedPreconditioner,
    converged, finiteScalar, nonnegativeInt, safeDivide, thresholdExceeded,
    validateSolve,
)

__all__ = ['gmres']


def _cycle(apply, psolve, x, residual, true, target, budget, *, side, threshold,
           dim, checkEvery, accumulationDtype, history):
    """One Arnoldi cycle; inactive columns are harmless identity extensions."""
    work = accumulationDtype
    scalar = lambda v: torch.tensor(v, dtype=torch.int32, device=x.device)
    zero, breakdown = scalar(0), scalar(Status.DENOMINATOR_BREAKDOWN)
    status = scalar(Status.BUDGET)
    completed = zero
    r = psolve(residual) if side == 'left' else residual
    beta = torch.linalg.vector_norm(r.to(work))
    if not bool(torch.isfinite(beta) & (beta > 0)):
        return x, breakdown, completed, 0, [], []
    basis = [(r.to(work) / beta).to(x.dtype)]
    directions, columns, cosines, sines = [], [], [], []
    rhs = [beta]
    norms, valid = [], []
    live = torch.ones((), dtype=torch.bool, device=x.device)
    accepted = x
    eps = torch.finfo(x.dtype).eps

    def candidate():
        k = len(columns)
        triangular = torch.stack([torch.cat((c, c.new_zeros(k - c.numel()))) for c in columns], dim=1)
        coefficients = torch.linalg.solve_triangular(triangular, torch.stack(rhs[:k])[:, None], upper=True)[:, 0]
        correction = torch.stack(directions, dim=1).to(work) @ coefficients
        return (x.to(work) + correction).to(x.dtype)

    for j in range(budget):
        direction = psolve(basis[j]) if side == 'right' else basis[j]
        directions.append(direction)
        w = apply(direction)
        w = psolve(w) if side == 'left' else w
        w = w.to(work)
        originalNorm = torch.linalg.vector_norm(w)
        Q = torch.stack(basis, dim=1).to(work)
        # CGS2: repeat the projection to control loss of orthogonality.
        h = Q.T @ w
        w = w - Q @ h
        repair = Q.T @ w
        w = w - Q @ repair
        h = h + repair
        nextNorm = torch.linalg.vector_norm(w)
        values = list(h.unbind()) + [nextNorm]
        for i in range(j):
            upper, lower = values[i], values[i + 1]
            values[i] = cosines[i] * upper + sines[i] * lower
            values[i + 1] = -sines[i] * upper + cosines[i] * lower
        rho = torch.hypot(values[j], values[j + 1])
        good = (torch.isfinite(originalNorm) & torch.isfinite(nextNorm)
                & torch.isfinite(h).all() & torch.isfinite(rho) & (rho != 0))
        go = live & good
        c = torch.where(go, safeDivide(values[j], rho), torch.ones_like(rho))
        s = torch.where(go, safeDivide(values[j + 1], rho), torch.zeros_like(rho))
        cosines.append(c)
        sines.append(s)
        column = torch.stack(values[:j] + [rho])
        identity = torch.cat((rho.new_zeros(j), rho.new_ones(1)))
        columns.append(torch.where(go, column, identity))
        oldRhs = rhs[j]
        rhs[j] = torch.where(go, c * oldRhs, torch.zeros_like(beta))
        rhs.append(torch.where(go, -s * oldRhs, torch.zeros_like(beta)))
        # Left-preconditioned estimates are rescaled to the cycle's initial
        # original norm. They are triggers only, never convergence certificates.
        estimate = rhs[-1].abs()
        if side == 'left':
            estimate = safeDivide(estimate, beta) * true.to(work)
        estimateConverged = converged(estimate, target.to(work))
        happy = nextNorm <= eps * originalNorm
        # Happy breakdown is an optimistic exit too: roundoff can leave a
        # nonzero original residual, in which case bounded repair is allowed.
        status = torch.where(live & ~good, breakdown,
                             torch.where(go & (estimateConverged | happy), zero, status))
        completed = completed + go.to(torch.int32)
        if history:
            norms.append(estimate.to(x.dtype))
            valid.append(go)
        live = go & ~estimateConverged & ~happy
        # Thresholds apply to each candidate, including a convergence candidate.
        # Otherwise materialize x only at a checkpoint or the cycle's end.
        checkpoint = (j + 1) % checkEvery == 0 or j + 1 == budget
        if threshold is not None or checkpoint:
            proposed = candidate()
            finite = torch.isfinite(proposed).all()
            accepted = torch.where(finite, proposed, accepted)
            status = torch.where(~finite, breakdown, status)
            live = live & finite
            exceeded = thresholdExceeded(accepted, threshold, dim)
            if exceeded is not None:
                status = torch.where(exceeded, scalar(Status.THRESHOLD), status)
                live = live & ~exceeded
        if checkpoint and not bool(live):
            break
        basis.append(torch.where(live, safeDivide(w, nextNorm), torch.zeros_like(w)).to(x.dtype))
    return accepted, status, completed, j + 1, norms, valid


def gmres(matvec, b, x0=None, *, tol=0.0, rtol=1e-5, atol=0.0,
          maxiter=None, precond=None, preconditionerSide='right', restart=30,
          threshold=None, dim=1, checkEvery=8, maxRestarts=2,
          stagnationTol=1e-3, accumulationDtype=None, history=True) -> SolveResult:
    """Solve A x = b using restarted GMRES and a verified original residual.

    ``maxiter`` bounds all scheduled Arnoldi steps, including frozen work.
    ``restart`` bounds a basis cycle, capped at the vector length. Ordinary
    cycles are budget-limited; ``maxRestarts`` bounds additional repairs after
    an optimistic residual estimate fails verification. ``stagnationTol`` is
    the required fractional true-residual decrease in a full cycle; zero
    disables that guard. Preconditioners are fixed linear inverse actions.

    Vectors retain b's float32/float64 dtype; ``accumulationDtype=float64``
    optionally promotes orthogonalization, Givens QR and the coefficient solve.
    This does not promote operator/preconditioner callbacks. Storage is bounded
    by O(n * min(restart, n, maxiter)) plus the small triangular system.
    """
    maxiter = validateSolve(b, x0, tol, rtol, atol, maxiter, threshold, dim, checkEvery, maxRestarts)
    restart = nonnegativeInt('restart', restart, positive=True)
    finiteScalar('stagnationTol', stagnationTol)
    if stagnationTol >= 1:
        raise ValueError('stagnationTol must be less than one')
    if preconditionerSide not in ('left', 'right'):
        raise ValueError("preconditionerSide must be 'left' or 'right'")
    if b.dtype not in (torch.float32, torch.float64):
        raise ValueError('gmres supports float32 and float64 vectors')
    work = b.dtype if accumulationDtype is None else accumulationDtype
    if work not in (torch.float32, torch.float64) or (b.dtype == torch.float64 and work != b.dtype):
        raise ValueError('accumulationDtype must be the vector dtype or float64')
    if not isinstance(history, bool):
        raise ValueError('history must be bool')
    if isinstance(matvec, LinearOperator):
        if matvec.shape != (b.numel(), b.numel()) or matvec.dtype != b.dtype or matvec.device != b.device:
            raise ValueError('operator shape, dtype and device must match the square system')
    counts = dict(matvecs=0, adjointMatvecs=0, preconditionerApplications=0)
    apply = checkedAction(matvec, b, 'matvec', counts, 'matvecs')
    psolve = checkedPreconditioner(precond, b, counts)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    target = absoluteTolerance(torch.linalg.vector_norm(b), tol, rtol, atol)
    residual = b - apply(x)
    true = torch.linalg.vector_norm(residual)
    verifications, loops, restarts, repairs = 1, 0, 0, 0
    completed = torch.zeros((), dtype=torch.int32, device=b.device)
    norms = [true.clone()] if history else []
    valid = [torch.ones((), dtype=torch.bool, device=b.device)] if history else []
    initialThreshold = thresholdExceeded(x, threshold, dim)
    if initialThreshold is not None and bool(initialThreshold):
        status = int(Status.THRESHOLD)
    elif bool(converged(true, target)):
        status = 0
    elif not bool(torch.isfinite(true)):
        status = int(Status.DENOMINATOR_BREAKDOWN)
    else:
        status = int(Status.BUDGET)
        while loops < maxiter:
            if loops:
                restarts += 1
            startTrue = true
            cycleBudget = min(restart, b.numel(), maxiter - loops)
            x, cycleStatus, steps, used, ns, vs = _cycle(
                apply, psolve, x, residual, true, target, cycleBudget,
                side=preconditionerSide, threshold=threshold, dim=dim,
                checkEvery=checkEvery, accumulationDtype=work, history=history)
            loops += used
            completed = completed + steps
            norms.extend(ns)
            valid.extend(vs)
            if used:
                residual = b - apply(x)
                true = torch.linalg.vector_norm(residual)
                verifications += 1
            status = int(cycleStatus)
            if status == Status.THRESHOLD:
                break
            if bool(converged(true, target)):
                status = int(completed)
                break
            if status == 0:
                if repairs >= maxRestarts or loops >= maxiter:
                    status = int(Status.RESIDUAL_MISMATCH)
                    break
                repairs += 1
            elif status != Status.BUDGET:
                break
            elif loops < maxiter and stagnationTol and bool(true >= (1 - stagnationTol) * startTrue):
                status = int(Status.STAGNATION)
                break
    statusTensor = torch.tensor(status, dtype=torch.int32, device=b.device)
    return SolveResult(
        x=x, status=statusTensor, trueResidual=true, stampedResidual=true,
        norms=torch.stack(norms) if norms else b.new_empty(0),
        valid=torch.stack(valid) if valid else torch.empty(0, dtype=torch.bool, device=b.device),
        iterations=completed, loopIterations=loops, verifications=verifications,
        restarts=restarts, _host={'status': status}, **counts)
