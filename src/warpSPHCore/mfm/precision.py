"""Floating-point-aware defaults for the thresholds of the MFM backend.

GIZMO's thresholds (``N_cond > 1e6`` for the face fallback, ``rtol`` 1e-12 in the pseudo-inverse, ...)
are tuned for double precision. How far a condition number can be trusted is a statement about
``eps``, not about the physics: the moment matrix ``E`` is formed by sums of ``O(30)`` terms in the
working precision, and for an anisotropy that is *not* aligned with the coordinate axes its small
eigenvalue comes out of a cancellation. Measured (``higher_order.md`` Phase 8, strips of particles
squeezed along a rotated line, float32 against float64): the relative error of both ``N_cond`` and
the matrix gradient is ``~0.4 eps N_cond`` -- 2e-4 at ``N_cond`` = 5e3, 2e-3 at 4e4, 2e-2 at 5e5, 15 %
at 4e6 -- whereas for an axis-aligned anisotropy float32 stays accurate to ``N_cond`` = 4e6 (the
equilibrated matrix is diagonal-dominant), so the rotated case is the one to budget for.
"""

from __future__ import annotations

import torch

__all__ = ["faceCondMax", "pinvAbove", "pinvRtol", "tiny", "COND_ERROR_PER_EPS"]

#: relative error of N_cond / gradient per unit eps * N_cond (measured, rotated anisotropy)
COND_ERROR_PER_EPS = 0.4


def eps(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).eps


def tiny(dtype: torch.dtype) -> float:
    """A safe divide-by-zero floor: ``1e-30`` (representable in float32 and negligible in float64);
    ``1e-300`` underflows to 0 in float32."""
    return 1.0e-30


def faceCondMax(dtype: torch.dtype, accuracy: float = 1.0e-3, physical: float = 1.0e6) -> float:
    """The largest ``N_cond`` whose face / gradient is trusted: the smaller of GIZMO's ``physical``
    cutoff (degenerate neighbour geometry, 1e6) and the value at which the working precision still
    gives a relative error of ``accuracy`` (``accuracy / (0.4 eps)``: 1e14 in float64, 2.1e4 in float32)."""
    return min(physical, accuracy / (COND_ERROR_PER_EPS * eps(dtype)))


def pinvAbove(dtype: torch.dtype) -> float:
    """Condition number above which a batched LU inverse is replaced by the pseudo-inverse."""
    return min(1.0e8, 0.1 / eps(dtype))


def pinvRtol(dtype: torch.dtype) -> float:
    """Relative singular-value cutoff of the pseudo-inverse: 1e-12 in float64, ``50 eps`` (6e-6) in float32."""
    return max(1.0e-12, 50.0 * eps(dtype))
