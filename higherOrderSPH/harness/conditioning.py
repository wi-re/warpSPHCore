#!/usr/bin/env python3
"""Correction-matrix condition numbers for the convergence harness.

The renorm mode's `computeRenormalizationMatrices` returns the
per-particle covariance (moment) matrix C and its eigenvalues; the
condition number is max|lambda| / min|lambda|. Tracked as a function of
particle disorder (jitter) and neighbor count per the parent plan, plus
the open-boundary degradation.

Rows whose C is *exactly* the identity are the shipped low-neighbor
fallback (`num_nbrs < dim + 2` replaces C with I before the
pseudo-inverse -- see `src/warpSPHCore/renorm.py`); they would otherwise
read as perfectly conditioned, so they are flagged separately and
excluded from the condition-number statistics.

Pure torch -- no warp, no warpSPHCore -- unit-testable on CPU.
"""

from __future__ import annotations

import torch

__all__ = ["condition_numbers", "matrix_condition_numbers", "fallback_mask",
           "cond_summary"]


def fallback_mask(C: torch.Tensor) -> torch.Tensor:
    """True for particles whose C was replaced by the identity fallback
    (the low-neighbor-count guard in the shipped renormalization)."""
    dim = C.shape[1]
    identity = torch.eye(dim, dtype=C.dtype, device=C.device)
    return (C - identity).abs().flatten(1).max(dim=1).values == 0


def condition_numbers(eigvals: torch.Tensor) -> torch.Tensor:
    """Per-particle condition number max|lambda| / min|lambda|.

    Returns +inf where the smallest magnitude eigenvalue is exactly zero
    (a truly singular moment matrix); the identity-fallback rows come out
    as 1 and are meant to be excluded via `fallback_mask` first.
    """
    mag = eigvals.abs()
    lam_max = mag.max(dim=1).values
    lam_min = mag.min(dim=1).values
    # torch has no errstate: the division below may produce inf (zero
    # smallest eigenvalue), which is exactly what we want to keep and
    # flag via the where.
    cond = torch.where(lam_min == 0, torch.full_like(lam_max, float("inf")),
                       lam_max / lam_min)
    return cond


def matrix_condition_numbers(M: torch.Tensor,
                             equilibrate: bool = False) -> torch.Tensor:
    """Per-particle 2-norm condition number sigma_max / sigma_min of a batch
    of n x n matrices ``(N, n, n)`` -- the general path for the order-p
    moment matrices of Phase 4/5 (n = 3..20; not necessarily symmetric), where
    `computeRenormalizationMatrices`' d x d eigenvalues do not exist. For a
    symmetric matrix this equals `condition_numbers` of its eigenvalues.

    `equilibrate=True` first applies the symmetric Jacobi scaling
    ``D^-1/2 M D^-1/2`` with ``D = |diag M|``: an unscaled monomial basis
    mixes powers of h (a p=2 moment matrix has entries from O(1) to O(h^4)),
    so its raw kappa mostly measures that scaling, not the particle geometry.
    Report the equilibrated value for geometry studies (or build the matrix
    in the h-scaled basis x_ij / h). Returns +inf for a zero smallest
    singular value, and for equilibration of a row with a zero diagonal.
    """
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError("M must be a (N, n, n) batch of square matrices")
    A = M
    if equilibrate:
        d = M.diagonal(dim1=1, dim2=2).abs()
        bad = (d == 0).any(dim=1)
        s_ = torch.where(d == 0, torch.ones_like(d), d).rsqrt()
        A = s_[:, :, None] * M * s_[:, None, :]
    sv = torch.linalg.svdvals(A)                     # (N, n), descending
    smax, smin = sv[:, 0], sv[:, -1]
    cond = torch.where(smin == 0, torch.full_like(smax, float("inf")),
                       smax / smin)
    if equilibrate:
        cond = torch.where(bad, torch.full_like(cond, float("inf")), cond)
    return cond


def cond_summary(cond: torch.Tensor,
                 mask: torch.Tensor | None = None) -> dict[str, float]:
    """Summary statistics of condition numbers over a particle mask.

    Returns n (count), fallback_frac (share of identity-fallback rows),
    mean/p50/p95/max of the finite condition numbers (fallback and inf
    rows excluded from the quantiles).
    """
    if mask is not None:
        cond = cond[mask]
    n = int(cond.numel())
    if n == 0:
        return {"n": 0, "fallback_frac": float("nan"), "mean": float("nan"),
                "p50": float("nan"), "p95": float("nan"),
                "max": float("nan")}
    finite = cond[torch.isfinite(cond)]
    if finite.numel() == 0:
        return {"n": n, "fallback_frac": float("nan"), "mean": float("nan"),
                "p50": float("nan"), "p95": float("nan"),
                "max": float("nan")}
    sorted_c, _ = torch.sort(finite)
    k95 = max(1, int(round(0.95 * (sorted_c.numel() - 1))))
    return {
        "n": n,
        "fallback_frac": float((~torch.isfinite(cond)).float().mean()),
        "mean": float(finite.mean()),
        "p50": float(sorted_c[sorted_c.numel() // 2]),
        "p95": float(sorted_c[k95]),
        "max": float(finite.max()),
    }
