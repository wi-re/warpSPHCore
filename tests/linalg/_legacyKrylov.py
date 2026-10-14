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
