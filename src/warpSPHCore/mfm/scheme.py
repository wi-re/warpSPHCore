"""MFM / MFV face-flux rates: slope-limited, time-centred reconstruction to the
effective faces, a Riemann solve per pair and the conservative assembly
(Hopkins 2015, Sec. 2.4 and App. A, B).

For every pair ``(i, j)`` of the :class:`~warpSPHCore.mfm.geometry.MeshlessGeometry`:

1. gradients of the primitives ``(rho, v, P)`` (matrix least squares), limited
   per particle (:func:`~warpSPHCore.mfm.limiters.slopeLimiter`);
2. reconstruction to the quadrature point ``x_ij = x_i + h_i/(h_i + h_j) d_ij`` (Eq. 20)
   from both sides, then the pairwise limiter (Eq. B4);
3. boost to the face frame ``v_frame = v_i + (v_j - v_i) h_i/(h_i + h_j)`` (Eq. 21),
   half-step MUSCL-Hancock prediction with the primitive Euler equations
   (Eq. A2-A4) using the particle's own limited gradients;
4. HLLC Riemann problem along the face normal ``A_ij / |A_ij|``
   (:mod:`~warpSPHCore.mfm.riemann`), MFM or MFV flux;
5. ``dQ_i/dt = - sum_j |A_ij| F_ij`` with ``Q = (m, m v, m e_tot)`` per particle
   (Eq. 19/23), antisymmetric per pair.

With ``dt = 0`` the prediction is skipped and the states are the spatially
reconstructed ones (first order in time). ``order = 1`` uses the particle
values as the states (piecewise-constant). Ideal gas only.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .geometry import MeshlessGeometry
from .limiters import conditionBeta, pairLimit, slopeLimiter
from .riemann import faceFlux

__all__ = ["mfmRates", "primitives", "conserved", "signalTimestep"]


def conserved(mass, rho, vel, P, gamma):
    """``Q = (m, m v, m e_tot)`` of shape ``(N, 2 + dim)``."""
    e_tot = P / ((gamma - 1.0) * rho) + 0.5 * (vel * vel).sum(-1)
    return torch.cat([mass[:, None], mass[:, None] * vel, (mass * e_tot)[:, None]], dim=-1)


def primitives(Q, volume, gamma, floor: float = 1e-30):
    """``(rho, v, P)`` from the conserved ``Q`` and the effective volumes."""
    m = Q[:, 0]
    rho = m / volume
    vel = Q[:, 1:-1] / m[:, None]
    e = Q[:, -1] / m - 0.5 * (vel * vel).sum(-1)
    P = ((gamma - 1.0) * rho * e).clamp_min(floor)
    return rho, vel, P


def _timeDerivative(rho, vel, P, grad, gamma):
    """Primitive-variable time derivatives (Eq. A4) from the (limited) gradients
    ``grad`` ``(N, 2 + dim, dim)`` of ``(rho, v, P)`` and the (boosted) velocity ``vel``."""
    dim = vel.shape[-1]
    g_rho, g_v, g_P = grad[:, 0], grad[:, 1:1 + dim], grad[:, -1]          # (N,d), (N,d,d), (N,d)
    div_v = torch.diagonal(g_v, dim1=-2, dim2=-1).sum(-1)
    adv = lambda g: (vel * g).sum(-1)
    d_rho = -(adv(g_rho) + rho * div_v)
    d_v = -(torch.einsum("nb,nab->na", vel, g_v) + g_P / rho[:, None])
    d_P = -(adv(g_P) + gamma * P * div_v)
    return d_rho, d_v, d_P


def mfmRates(geom: MeshlessGeometry, rho: torch.Tensor, vel: torch.Tensor, P: torch.Tensor,
             gamma: float, dt: float = 0.0, mode: str = "MFM", order: int = 2,
             beta_min: float = 1.0, beta_max: float = 2.0, cond_crit: float = 100.0,
             psi1: float = 0.5, psi2: float = 0.25, conservativeLimiter: bool = False,
             timeCentredFrame: bool = True, starFn=None):
    """Rates ``dQ/dt`` of shape ``(N, 2 + dim)`` and a dict of diagnostics
    (``flux``, ``Sstar``, ``Pstar``, ``alpha``)."""
    N, dim = vel.shape
    i, j, d = geom.i, geom.j, geom.d
    frac = geom.faceFraction()

    W = torch.cat([rho[:, None], vel, P[:, None]], dim=-1)                  # (N, K)
    if order >= 2:
        grad = geom.gradient(W)                                              # (N, K, dim)
        beta = conditionBeta(geom.cond, beta_min, beta_max, cond_crit)
        alpha = slopeLimiter(geom, W, grad, beta, conservative=conservativeLimiter)
        glim = alpha[:, :, None] * grad
        # lab-frame reconstruction to the face from both sides, then the pairwise limiter
        W0_L = W[i] + torch.einsum("pkd,pd->pk", glim[i], frac[:, None] * d)
        W0_R = W[j] + torch.einsum("pkd,pd->pk", glim[j], -(1.0 - frac)[:, None] * d)
        WL = pairLimit(W[i], W[j], W0_L, frac[:, None], psi1, psi2)
        WR = pairLimit(W[j], W[i], W0_R, (1.0 - frac)[:, None], psi1, psi2)
    else:
        WL, WR = W[i], W[j]
        alpha = torch.ones_like(W)
        glim = torch.zeros(N, W.shape[1], dim, dtype=W.dtype, device=W.device)

    # particle velocities at the half step (the face moves with the time-centred velocities)
    vel_h = vel
    if order >= 2 and dt > 0.0 and timeCentredFrame:
        _, dv_p, _ = _timeDerivative(rho, vel, P, glim, gamma)
        vel_h = vel + 0.5 * dt * dv_p
    vframe = vel_h[i] + frac[:, None] * (vel_h[j] - vel_h[i])                # Eq. 21
    if order >= 2 and dt > 0.0:
        # half-step prediction with each particle's own gradients, in the boosted frame
        for side, idx, Wface in (("L", i, WL), ("R", j, WR)):
            v_b = vel[idx] - vframe
            dr, dv, dp = _timeDerivative_pair(rho[idx], v_b, P[idx], glim[idx], gamma)
            pred = torch.cat([dr[:, None], dv, dp[:, None]], dim=-1) * (0.5 * dt)
            Wp = Wface + pred
            ok = (Wp[:, 0] > 0) & (Wp[:, -1] > 0)                           # keep the unpredicted state otherwise
            Wface = torch.where(ok[:, None], Wp, Wface)
            if side == "L":
                WL = Wface
            else:
                WR = Wface

    n = geom.A / torch.linalg.norm(geom.A, dim=-1, keepdim=True).clamp_min(torch.finfo(geom.A.dtype).tiny)   # pairs with A = 0 (kernel edge) carry no flux
    rL, rR = WL[:, 0].clamp_min(1e-30), WR[:, 0].clamp_min(1e-30)
    pL, pR = WL[:, -1].clamp_min(1e-30), WR[:, -1].clamp_min(1e-30)
    vL, vR = WL[:, 1:1 + dim] - vframe, WR[:, 1:1 + dim] - vframe
    uL, uR = (vL * n).sum(-1), (vR * n).sum(-1)
    vtL, vtR = vL - uL[:, None] * n, vR - uR[:, None] * n
    flux, Ss, Ps = faceFlux(rL, uL, vtL, pL, rR, uR, vtR, pR, gamma, mode, n, vframe, starFn)
    rates = geom.divergence(flux)
    return rates, dict(flux=flux, Sstar=Ss, Pstar=Ps, alpha=alpha)


def _timeDerivative_pair(rho, v_b, P, glim, gamma):
    """Eq. A4 for per-pair copies of a particle's data (``glim`` is its limited gradient)."""
    return _timeDerivative(rho, v_b, P, glim, gamma)


def signalTimestep(geom: MeshlessGeometry, rho, vel, P, gamma: float, cfl: float = 0.2) -> torch.Tensor:
    """Per-particle CFL time step ``2 C h_i / v_sig,i`` (Eq. 24-25); take the
    minimum for a global step."""
    i, j, d = geom.i, geom.j, geom.d
    c = torch.sqrt(gamma * P / rho)
    r = torch.linalg.norm(d, dim=-1).clamp_min(1e-300)
    # (v_i - v_j) . (x_i - x_j) / |x_ij| = (v_j - v_i) . d / |d|
    approach = torch.clamp(((vel[j] - vel[i]) * d).sum(-1) / r, max=0.0)
    vsig = c[i] + c[j] - approach
    own = c * 2.0
    out = own.scatter_reduce(0, i, vsig, "amax").scatter_reduce(0, j, vsig, "amax")
    return 2.0 * cfl * geom.supports / out
