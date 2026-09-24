#!/usr/bin/env python3
"""Particle-distribution anisotropy measures (pure torch, CPU-testable).

The SPH approximation error depends critically on the *particle distribution*,
not just the resolution: a disordered (anisotropic) neighbourhood breaks the
kernel's moment assumptions and degrades the order of the discrete operators
(Vacondio et al. 2021, SPH grand challenges, GC1: "convergence depends
critically on particle distributions ... convergence flat-lining, even
diverging, once particles become sufficiently disordered").

This module quantifies the local distribution anisotropy of a particle set
(positions + masses + support) with two per-particle, *dimensionless* measures
that are properties of the distribution alone (no flow state, no solution
densities), so they can characterise saved distributions as test data for the
operator-consistency probes:

* **first-moment residual** -- ``f_i = sum_j m_j W_ij (x_j - x_i)``. For a
  locally symmetric neighbourhood this vanishes (odd integrand over a
  symmetric cloud); its norm, scaled by ``h C_i`` (``C_i`` the kernel sum),
  is a dimensionless measure of the local asymmetry.
* **second-moment (shape) anisotropy** -- ``M_i = sum_j m_j W_ij
  (x_j - x_i)(x_j - x_i)^T``. For an isotropic cloud ``M_i`` is proportional
  to the identity; the measure is the eigenvalue spread
  ``(lambda_max - lambda_min) / lambda_max`` of ``M_i``, which lies in [0, 1]
  (0 = isotropic, 1 = rank-1, all the mass on a line). Being an eigenvalue
  ratio it is rotationally invariant and well defined in any dimension.

The kernel *normalisation* is irrelevant to both measures (they are ratios),
so the unnormalised Wendland shapes (``kernel_shape``) are used; the shapes
match ``warpSPHCore``'s ``kernelFunctions/wendland{2,4,6}.py``.

Blockwise over query particles so the O(N^2) pair work stays in a few hundred
MB at 2D benchmark sizes (N ~ 25k).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["kernel_shape", "DistributionMoments", "distribution_moments",
           "anisotropy_summary"]


# ---------------------------------------------------------------------------
# kernel shapes (unnormalised; normalisation cancels in the ratio measures)
# ---------------------------------------------------------------------------

def kernel_shape(q: torch.Tensor, kernel: str = "wendland4",
                 dim: int = 2) -> torch.Tensor:
    """Unnormalised compact-support kernel shape W(q) for q = r/h in [0, 1].

    Matches the unnormalised polynomials in `warpSPHCore`'s
    `kernelFunctions/wendland{2,4,6}.py` (the 2D/3D branches share one
    formula there). Zero for q >= 1.
    """
    if kernel == "wendland2":
        if dim == 1:
            poly = 1.0 + 3.0 * q
            return (1.0 - q).clamp(min=0.0) ** 3 * poly
        poly = 1.0 + 4.0 * q
        return (1.0 - q).clamp(min=0.0) ** 4 * poly
    if kernel == "wendland4":
        if dim == 1:
            poly = 1.0 + 5.0 * q + 8.0 * q ** 2
            return (1.0 - q).clamp(min=0.0) ** 5 * poly
        poly = 1.0 + 6.0 * q + (35.0 / 3.0) * q ** 2
        return (1.0 - q).clamp(min=0.0) ** 6 * poly
    if kernel == "wendland6":
        if dim == 1:
            poly = 1.0 + 7.0 * q + 19.0 * q ** 2 + 21.0 * q ** 3
            return (1.0 - q).clamp(min=0.0) ** 7 * poly
        poly = 1.0 + 8.0 * q + 25.0 * q ** 2 + 32.0 * q ** 3
        return (1.0 - q).clamp(min=0.0) ** 8 * poly
    raise ValueError(f"unknown kernel '{kernel}' "
                     f"(want wendland2/wendland4/wendland6)")


# ---------------------------------------------------------------------------
# per-particle moments
# ---------------------------------------------------------------------------

@dataclass
class DistributionMoments:
    """Per-particle distribution moments and anisotropy measures.

    All tensors are (N,) or (N, dim)/(N, dim, dim), host or device as given.
    """
    N: int
    dim: int
    kernel_sum: torch.Tensor              # C_i = sum_j m_j W_ij        (N,)
    first_moment: torch.Tensor            # f_i = sum_j m_j W_ij dx_ij   (N, dim)
    second_moment: torch.Tensor           # M_i = sum_j m_j W_ij dx dx^T (N, dim, dim)
    first_anisotropy: torch.Tensor        # |f_i| / (h C_i)              (N,)
    second_anisotropy: torch.Tensor       # (lam_max - lam_min)/lam_max of M_i  (N,) in [0, 1]


def _minimum_image(dx: torch.Tensor, L: float | torch.Tensor) -> torch.Tensor:
    if L is None:
        return dx
    return dx - L * torch.round(dx / L)


def distribution_moments(
    positions: torch.Tensor,
    masses: torch.Tensor,
    h: float,
    L: float | None,
    dim: int,
    kernel: str = "wendland4",
    periodic: bool = True,
    block: int = 256,
) -> DistributionMoments:
    """Per-particle distribution moments of a particle set.

    `positions` (N, dim), `masses` (N,), `h` the support, `L` the box side
    (periodic minimum-image) or None (open), `periodic` whether to wrap.
    Self-pairs contribute W(0) to `kernel_sum` and 0 to the moments, exactly
    as in the SPH sums.
    """
    N = positions.shape[0]
    dev = positions.device
    x = positions.detach().double()
    m = masses.detach().double()

    C = torch.zeros(N, dtype=torch.float64, device=dev)
    F = torch.zeros(N, dim, dtype=torch.float64, device=dev)
    M = torch.zeros(N, dim, dim, dtype=torch.float64, device=dev)

    for lo in range(0, N, block):
        hi = min(lo + block, N)
        dx = x[None, :, :] - x[lo:hi, None, :]          # (b, N, dim): x_j - x_i
        if periodic:
            dx = _minimum_image(dx, L)
        r2 = (dx ** 2).sum(dim=-1)                       # (b, N)
        q = torch.sqrt(r2) / h
        W = kernel_shape(q, kernel, dim) * (r2 < h * h)  # (b, N), 0 outside support
        mw = W * m[None, :]                              # (b, N)
        C[lo:hi] = mw.sum(dim=1)
        F[lo:hi] = (mw[:, :, None] * dx).sum(dim=1)
        M[lo:hi] = (mw[:, :, None, None] * dx[:, :, :, None] * dx[:, :, None, :]).sum(dim=1)

    # dimensionless anisotropy measures (kernel normalisation cancels)
    tiny = torch.finfo(torch.float64).tiny
    c_safe = C.clamp(min=tiny)
    first_aniso = F.norm(dim=1) / (h * c_safe)
    # shape anisotropy: eigenvalue spread of the PSD tensor M_i (rotationally
    # invariant, in [0, 1]; 1 = rank-1, all mass on a line).
    lam = torch.linalg.eigvalsh(M)                     # (N, dim), ascending
    second_aniso = (lam[:, -1] - lam[:, 0].clamp(min=0.0)) / lam[:, -1].clamp(min=tiny)

    return DistributionMoments(
        N=N, dim=dim, kernel_sum=C, first_moment=F, second_moment=M,
        first_anisotropy=first_aniso, second_anisotropy=second_aniso)


def anisotropy_summary(mom: DistributionMoments) -> dict:
    """Scalar summary (rms/mean/max) of the anisotropy measures, ready for a
    log line or a CSV row."""
    f, s = mom.first_anisotropy.detach(), mom.second_anisotropy.detach()
    return {
        "first_anisotropy_rms": float(f.pow(2).mean().sqrt()),
        "first_anisotropy_mean": float(f.mean()),
        "first_anisotropy_max": float(f.max()),
        "second_anisotropy_rms": float(s.pow(2).mean().sqrt()),
        "second_anisotropy_mean": float(s.mean()),
        "second_anisotropy_max": float(s.max()),
        "kernel_sum_min": float(mom.kernel_sum.min()),
        "kernel_sum_max": float(mom.kernel_sum.max()),
    }
