"""The pre-2026-10-14 host-branching Krylov solvers, verbatim, as the bitwise reference of the device-predicated ones."""
from typing import Callable, List, Optional, Tuple, Union
import math
import torch




def cgSolveLegacy(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solve ``matvec(x) == b`` matrix-free with (left-)preconditioned conjugate
    gradient. The operator must be symmetric positive definite (see the module
    docstring for how the IISPH operator's negative diagonal is sign-flipped by
    the dispatch before calling here)."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'PCG: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return xk, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10
    eps = torch.finfo(xk.dtype).eps

    if precond is None:
        psolve = (lambda v: v)
    elif callable(precond):
        psolve = precond
    else:
        psolve = (lambda v: precond * v)

    def thresholdExceeded(x: torch.Tensor) -> bool:
        if threshold is None:
            return False
        return bool(torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold))

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual, so
        # callers always know the quality of what they received
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    # --- initial residual ----------------------------------------------------
    r = b - matvec(xk)
    rnrm = torch.linalg.norm(r)
    convergence.append(rnrm)
    if rnrm < atol:
        if verbose:
            print(f'\t[  0] already converged, |r| = {rnrm}')
        return xk, 0, convergence
    if thresholdExceeded(xk):
        return finish(xk, -12)

    # --- Krylov recurrence ---------------------------------------------------
    z = psolve(r)
    p = z.clone()
    rz = torch.dot(r, z)

    for k in range(maxiter):
        if (not bool(torch.isfinite(rz))) or rz <= 0:
            if verbose:
                print(f'\t[{k:3d}] breakdown: r^T M r = {rz} (non-finite or non-positive)')
            return finish(xk, -16)
        ap = matvec(p)
        pap = torch.dot(p, ap)
        if (not bool(torch.isfinite(pap))) or pap <= 0:
            # the operator is not positive definite in the search direction:
            # CG's curvature is non-positive, so the method is not well-posed
            # here (this is the indefinite case the operator probe quantifies)
            if verbose:
                print(f'\t[{k:3d}] indefinite: p^T A p = {pap} <= 0 (operator not SPD)')
            return finish(xk, -16)
        alpha = rz / pap
        xk = xk + alpha * p
        r = r - alpha * ap
        rnrm = torch.linalg.norm(r)
        convergence.append(rnrm)
        if rnrm < atol:
            if verbose:
                print(f'\t[{k:3d}] converged, |r| = {rnrm}')
            return xk, k, convergence
        if thresholdExceeded(xk):
            return finish(xk, -12)
        z = psolve(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return finish(xk, -14)


# ---- BiCGStab ----------------------------------------------------------------------------------------------------




# relative breakdown margin (see module docstring): ~8x float32 eps, i.e. it
# only fires once the corresponding update (alpha/omega) is already
# ~1e6-scaled and the iterate is diverging, so it never triggers on
# well-behaved solves
_REL_BREAKDOWN_EPS = 1e-6


def bicgstabSolveLegacy(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solves `matvec(x) == b` matrix-free. `tol` is an absolute residual
    floor (0.0 = relative tolerance only), `rtol*||b||` the relative one,
    `atol` an additional explicit floor. See the module docstring for the
    status-code convention and the `convergence` list semantics."""
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'BiCGStab: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return b, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    eps = torch.finfo(xk.dtype).eps
    rhotol = eps ** 2
    omegatol = rhotol

    if precond is None:
        psolve = (lambda x: x)
    elif callable(precond):
        psolve = precond
    else:
        psolve = (lambda x: precond * x)

    # zero x0 is the common production case: A @ 0 == 0, skip the matvec
    if x0 is None or not x0.any():
        rk = b.clone()
    else:
        rk = b - matvec(xk)
    r0 = rk.clone()
    pk = rk.clone()
    r0norm = torch.linalg.norm(r0)

    def finish(x: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual, so
        # callers always know the quality of what they received
        convergence.append(torch.linalg.norm(matvec(x) - b))
        return x, status, convergence

    for iteration in range(maxiter):
        rkNorm = torch.linalg.norm(rk)
        if rkNorm < atol:
            # the recurrence residual can under-report near breakdown: verify
            # with the true residual before declaring convergence (one extra
            # matvec, only on the would-be final iterate); if the recurrence
            # is lying, fall through and keep iterating (the guards below
            # catch the near-singular case)
            trueResid = torch.linalg.norm(matvec(xk) - b)
            if trueResid < atol:
                if verbose:
                    print(f'\t[{iteration:3d}] converged, |r| = {trueResid}')
                convergence.append(trueResid)
                return xk, iteration, convergence

        rho = torch.dot(rk, r0)
        if torch.abs(rho) < rhotol or torch.abs(rho) < _REL_BREAKDOWN_EPS * r0norm * rkNorm:
            if verbose:
                print(f'\t[{iteration:3d}] rho breakdown {rho} | {rhotol}')
            return finish(xk, -10)

        phat = psolve(pk)
        apk = matvec(phat)
        rv = torch.dot(apk, r0)
        if torch.abs(rv) < _REL_BREAKDOWN_EPS * torch.abs(rho):
            if verbose:
                print(f'\t[{iteration:3d}] rv breakdown {rv}')
            return finish(xk, -11)
        alpha = rho / rv
        sk = rk - alpha * apk

        if torch.linalg.norm(sk) < atol:
            # verify before committing: in fp32 `sk = rk - alpha*apk` is a
            # cancellation of O(1) terms, so the computed ||sk|| can
            # under-report the true residual of the updated iterate
            x_cand = xk + alpha * pk
            trueResid = torch.linalg.norm(matvec(x_cand) - b)
            if trueResid < atol:
                xk = x_cand
                if verbose:
                    print(f'\t[{iteration:3d}] converged, |s| = {trueResid}')
                convergence.append(trueResid)
                return xk, iteration, convergence
            # otherwise the omega part of the step is not negligible: fall
            # through and complete the full iteration

        shat = psolve(sk)
        ask = matvec(shat)
        omega = torch.dot(ask, sk) / torch.dot(ask, ask)
        if (not bool(torch.isfinite(omega))) or torch.abs(omega) < omegatol:
            if verbose:
                print(f'\t[{iteration:3d}] omega breakdown {omega} | {omegatol}')
            return finish(xk, -11)

        xk = xk + alpha * phat + omega * shat
        rho_prev = rho
        rk = sk - omega * ask
        beta = (torch.dot(rk, r0) / rho_prev) * (alpha / omega)
        pk = rk + beta * (pk - omega * apk)

        convergence.append(torch.linalg.norm(rk))
        if verbose:
            print(f'\t[{iteration:3d}] residual: {convergence[-1]}')

        if threshold is not None:
            dist = torch.linalg.norm(xk.view(-1, dim), dim=-1)
            if torch.any(dist > threshold):
                if verbose:
                    print(f'\t[{iteration:3d}] xk breakdown: max |dx| = {dist.max()} > {threshold}')
                return finish(xk, -12)

    return finish(xk, -14)
