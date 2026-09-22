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

__all__ = ["condition_numbers", "fallback_mask", "cond_summary"]


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
