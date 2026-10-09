"""Slope limiting for meshless Riemann reconstruction (Hopkins 2015, App. B).

Two stages, both operating on the primitive variables ``phi`` (any number of
components, all independent):

* :func:`slopeLimiter` -- a per-particle factor ``alpha_i`` in ``[0, 1]`` on the
  matrix gradient (Eq. B1-B3, Barth & Jespersen-type): the linearly
  reconstructed values at the faces of ``i`` must not leave the range spanned
  by ``phi_i`` and its neighbours. ``beta_i`` relaxes the bound for
  well-conditioned gradient matrices (``beta in [beta_min, beta_max]``,
  Eq. B3; ``beta > 0.5`` keeps second order, Balsara 2004).
* :func:`pairLimit` -- the pairwise limiter of Eq. B4 applied to the
  reconstructed face value of each pair, with the two free parameters
  ``psi_1 = 1/2``, ``psi_2 = 1/4`` the paper recommends (nearly TVD; it also
  prevents a sign change of a positive field).

All functions are plain torch (differentiable).
"""

from __future__ import annotations

import torch

__all__ = ["conditionBeta", "slopeLimiter", "pairLimit"]


def conditionBeta(cond: torch.Tensor, beta_min: float = 1.0, beta_max: float = 2.0,
                  cond_crit: float = 100.0) -> torch.Tensor:
    """``beta_i = max(beta_min, beta_max min(1, N_crit / N_cond,i))`` (Eq. B3)."""
    return torch.clamp(beta_max * torch.clamp(cond_crit / cond, max=1.0), min=beta_min)


def _extrema(n: int, i: torch.Tensor, j: torch.Tensor, vi: torch.Tensor, vj: torch.Tensor, own: torch.Tensor):
    """Per-particle max/min over the pair values seen from each side, starting
    from the particle's own value ``own``. ``vi`` is what pair ``p`` contributes
    to particle ``i[p]``, ``vj`` what it contributes to ``j[p]``."""
    idx_i = i.view(-1, *([1] * (vi.dim() - 1))).expand_as(vi)
    idx_j = j.view(-1, *([1] * (vj.dim() - 1))).expand_as(vj)
    hi, lo = own.clone(), own.clone()
    hi = hi.scatter_reduce(0, idx_i, vi, "amax").scatter_reduce(0, idx_j, vj, "amax")
    lo = lo.scatter_reduce(0, idx_i, vi, "amin").scatter_reduce(0, idx_j, vj, "amin")
    return hi, lo


def slopeLimiter(geom, phi: torch.Tensor, grad: torch.Tensor, beta: torch.Tensor,
                 conservative: bool = False) -> torch.Tensor:
    """``alpha`` of shape ``(N, K)`` for fields ``phi`` ``(N, K)`` with matrix
    gradients ``grad`` ``(N, K, dim)`` (Eq. B1-B2). With ``conservative`` the
    reconstructed extrema are replaced by the largest value the gradient could
    produce, ``|grad phi| h_i / 2`` (the paper's cheaper, slightly more
    diffusive variant that needs no second neighbour loop)."""
    N = phi.shape[0]
    i, j = geom.i, geom.j
    frac = geom.faceFraction()
    ph_i, ph_j = phi[i], phi[j]
    # neighbour extrema (the particle's own value included)
    ngb_hi, ngb_lo = _extrema(N, i, j, ph_j, ph_i, phi)
    tiny = torch.finfo(phi.dtype).tiny
    if conservative:
        gmag = torch.linalg.norm(grad, dim=-1)
        reach = gmag * (0.5 * geom.supports)[:, None]
        up, dn = reach, reach
    else:
        # value of i's linear reconstruction at the face of each pair (and j's)
        mid_i = ph_i + torch.einsum("pkd,pd->pk", grad[i], frac[:, None] * geom.d)
        mid_j = ph_j + torch.einsum("pkd,pd->pk", grad[j], -(1.0 - frac)[:, None] * geom.d)
        mid_hi, mid_lo = _extrema(N, i, j, mid_i, mid_j, phi)
        up, dn = mid_hi - phi, phi - mid_lo
    big = torch.full_like(phi, float("inf"))
    r_up = torch.where(up > tiny, (ngb_hi - phi) / up.clamp_min(tiny), big)
    r_dn = torch.where(dn > tiny, (phi - ngb_lo) / dn.clamp_min(tiny), big)
    return torch.clamp(beta[:, None] * torch.minimum(r_up, r_dn), max=1.0)


def pairLimit(phi_i: torch.Tensor, phi_j: torch.Tensor, phi0: torch.Tensor, frac: torch.Tensor,
              psi1: float = 0.5, psi2: float = 0.25) -> torch.Tensor:
    """Eq. B4: the limited face value seen from ``i`` given the unlimited
    reconstruction ``phi0`` there. ``frac = |x_face - x_i| / |x_j - x_i|``
    (broadcastable against the fields)."""
    dphi = (phi_i - phi_j).abs()
    d1, d2 = psi1 * dphi, psi2 * dphi
    bar = phi_i + frac * (phi_j - phi_i)
    pmin, pmax = torch.minimum(phi_i, phi_j), torch.maximum(phi_i, phi_j)
    tiny = torch.finfo(phi_i.dtype).tiny

    lo = pmin - d1
    lo = torch.where(torch.sign(lo) == torch.sign(pmin), lo, pmin / (1.0 + d1 / pmin.abs().clamp_min(tiny)))
    hi = pmax + d1
    hi = torch.where(torch.sign(hi) == torch.sign(pmax), hi, pmax / (1.0 + d1 / pmax.abs().clamp_min(tiny)))

    up = torch.maximum(lo, torch.minimum(bar + d2, phi0))          # phi_i < phi_j
    dn = torch.minimum(hi, torch.maximum(bar - d2, phi0))          # phi_i > phi_j
    return torch.where(phi_i < phi_j, up, torch.where(phi_i > phi_j, dn, phi_i))
