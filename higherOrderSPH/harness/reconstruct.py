#!/usr/bin/env python3
"""Non-oscillatory interface reconstruction for Riemann-SPH (Phases 6 and 7).

Per particle, nine candidate polynomial fits (one central, eight 45-degree
sector "one-sided" stencils in 2-D; one central + left / right in 1-D),
built on the order-p moment systems of `rkpm.py`, combined by a nonlinear
weighting into one reconstruction polynomial whose value at a pair midpoint
is the left / right state a Riemann solver consumes. Two weightings share the
same candidates:

* **TENO** -- Gao, Liang & Fu (2023, JCP 489:112270), "MLS-TENO-SPH":
  kernel-weighted MLS fits (central degree 3/4/5 = their "O4/O5/O6" at
  radius 2.5/3.2/4.0 sqrt(V); sector fits of degree 2 at radius 4.5 sqrt(V));
  smoothness ``beta_s`` = exact integral of the squared derivatives of the
  stencil polynomial over the square of edge 2 (Eq. 26-27); strong scale
  separation ``gamma_s = 1/(beta_s + 1e-12)^6`` (Eq. 28), ``chi_0 =
  gamma_0 / sum_s gamma_s`` (Eq. 29); the central stencil is used alone when
  ``chi_0 >= C_T`` (1e-5 / 1e-6 / 1e-7 for O4 / O5 / O6), otherwise the sector
  stencils are blended with ``omega_s = gamma_s / sum_{s>=1} gamma_s``
  (Eq. 30-33).
* **WENO** -- Avesani, Dumbser & Bertaux (2014, JCP 270:278), "MLS-WENO-SPH":
  all nine fits are the *same* degree M, unweighted least squares on
  ``Q_j - Q_i`` (Eq. 34, no constant), central stencil ``r <= h_mls``,
  sector stencils ``r <= 2 h_mls`` with ``h_mls = sigma_mls sqrt(V)``;
  smoothness ``sigma_s = sum_m w_m^2`` (Eq. 38, coefficients in the common
  normalised coordinates), ``omega_s ~ lambda_s / (1e-14 + sigma_s)^4`` with
  ``lambda_0 = 1e5``, ``lambda_s = 1`` (Eq. 36-37).

Assumptions where the papers are not explicit (flagged for review):
(a) the TENO ``beta`` sums *all* partial derivatives of order 1..p with unit
weight, on the square [-sqrt(V), sqrt(V)]^d common to every stencil (the
paper nondimensionalises by the stencil scale, which would make stencils of
different radius incomparable); (b) TENO stencils are selected with a hard
radius around ``sqrt(V)`` (uniform-volume form of ``r_ij < c max(sqrt V_i,
sqrt V_j)``); (c) the interface point is the arithmetic midpoint (equal
supports; the paper's Eq. 25 is the h-weighted midpoint); (d) a stencil with
fewer particles than basis functions or ``cond > cond_max`` is deactivated
(weight 0), as in Avesani; if every stencil is deactivated the central fit is
used regardless.

Pure torch, scalar fields, uniform particle volume (the harness lattices);
2-D and 1-D. The Riemann coupling is NOT here -- these are reconstruction
states only (see `higher_order.md`, Phase 6).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

try:
    from .rkpm import RKPMOperator, RKPMSystem, build_system
except ImportError:  # pragma: no cover
    from rkpm import RKPMOperator, RKPMSystem, build_system

__all__ = ["Reconstructor", "teno_reconstructor", "weno_reconstructor",
           "smoothness_gram"]

TENO_ORDERS = {  # name -> (central degree, central radius / sqrt V, C_T)
    "O4": (3, 2.5, 1e-5),
    "O5": (4, 3.2, 1e-6),
    "O6": (5, 4.0, 1e-7),
}


# ---------------------------------------------------------------------------
# smoothness Gram matrix (exact, monomial basis on [-1, 1]^d)
# ---------------------------------------------------------------------------

def smoothness_gram(exps: torch.Tensor) -> torch.Tensor:
    """``G[a, b] = sum_{1 <= |alpha| <= p} int_{[-1,1]^d} D^alpha m_a D^alpha m_b``
    for the monomials ``m = xi^e`` (the quadratic form of Gao et al. Eq. 27,
    evaluated exactly instead of by Gauss quadrature)."""
    E = [tuple(e) for e in exps.tolist()]
    dim = len(E[0])
    pmax = max(sum(e) for e in E)
    alphas = [a for a in itertools.product(range(pmax + 1), repeat=dim)
              if 1 <= sum(a) <= pmax]
    G = torch.zeros(len(E), len(E), dtype=torch.float64)

    def deriv(e, a):                    # coefficient and exponent of D^a xi^e
        if any(ek < ak for ek, ak in zip(e, a)):
            return 0.0, None
        coef = math.prod(math.factorial(ek) // math.factorial(ek - ak)
                         for ek, ak in zip(e, a))
        return float(coef), tuple(ek - ak for ek, ak in zip(e, a))

    for a in alphas:
        d = [deriv(e, a) for e in E]
        for i, (ci, ui) in enumerate(d):
            if ui is None:
                continue
            for j, (cj, uj) in enumerate(d):
                if uj is None:
                    continue
                u = tuple(x + y for x, y in zip(ui, uj))
                if any(k % 2 for k in u):
                    continue            # odd integrand over [-1, 1]
                G[i, j] += ci * cj * math.prod(2.0 / (k + 1) for k in u)
    return G


# ---------------------------------------------------------------------------
# reconstructor
# ---------------------------------------------------------------------------

def _sector(dim: int, s: int):
    """Pair selector of sector ``s`` (0..7 in 2-D, 0..1 in 1-D)."""
    if dim == 1:
        return (lambda i, j, dx: dx[:, 0] < 0) if s == 0 \
            else (lambda i, j, dx: dx[:, 0] > 0)
    if dim != 2:
        raise NotImplementedError("sector stencils are 1-D / 2-D")

    def sel(i, j, dx):
        th = torch.atan2(dx[:, 1], dx[:, 0]) % (2.0 * math.pi)
        return (th >= s * math.pi / 4) & (th < (s + 1) * math.pi / 4)
    return sel


@dataclass
class _Candidate:
    op: RKPMOperator
    h: float
    degree: int
    constant: bool


class Reconstructor:
    """Nine-candidate nonlinear reconstruction. Build with
    `teno_reconstructor` / `weno_reconstructor`."""

    def __init__(self, kind, candidates, ell, sqrtV, params):
        self.kind = kind
        self.c = candidates
        self.ell = ell                  # common length for the TENO beta
        self.sqrtV = sqrtV
        self.p = params
        self.dim = candidates[0].op.s.dim
        self.n_cand = len(candidates)
        self._gram = {}                 # per-candidate Gram (TENO)

    # -- per-particle quantities ------------------------------------------
    def _coeffs(self, f):
        return [c.op.coefficients(f) for c in self.c]

    def _active(self):
        return torch.stack([~c.op.s.deficient for c in self.c], dim=1)  # (N, K)

    def smoothness(self, coeffs):
        """``beta`` (N, K) per candidate; deficient stencils get +inf."""
        cols = []
        for k, (c, cf) in enumerate(zip(self.c, coeffs)):
            s = c.op.s
            deg = torch.tensor([sum(e) for e in s.exps.tolist()],
                               dtype=cf.dtype, device=cf.device)
            # coefficients in the common coordinates x / ell (every stencil
            # fits in its own x / h_s; the indicators must compare like with like)
            ct = cf * (self.ell / c.h) ** deg             # (n,)
            if self.kind == "weno":
                beta = (ct ** 2).sum(-1)    # constant=False: all columns |m| >= 1
            else:
                if k not in self._gram:
                    self._gram[k] = smoothness_gram(s.exps).to(cf)
                G = self._gram[k]
                beta = torch.einsum("na,ab,nb->n", ct, G, ct)
            cols.append(beta)
        beta = torch.stack(cols, dim=1)
        return torch.where(self._active(), beta, torch.full_like(beta, float("inf")))

    def weights(self, beta):
        """Nonlinear weights ``omega`` (N, K) (rows sum to 1) and, for TENO,
        the per-particle boolean 'central stencil used'."""
        act = torch.isfinite(beta)
        if self.kind == "weno":
            lam = torch.ones_like(beta)
            lam[:, 0] = self.p["lambda0"]
            w = lam / (self.p["eps"] + beta) ** self.p["r"]
            w = torch.where(act, w, torch.zeros_like(w))
        else:
            gam = 1.0 / (beta + self.p["eps"]) ** 6
            gam = torch.where(act, gam, torch.zeros_like(gam))
            chi0 = gam[:, 0] / gam.sum(1).clamp_min(1e-300)
            central = (chi0 >= self.p["C_T"]) & act[:, 0]
            side = gam[:, 1:]
            side_w = side / side.sum(1, keepdim=True).clamp_min(1e-300)
            w = torch.zeros_like(beta)
            w[:, 1:] = side_w
            w[central] = 0.0
            w[central, 0] = 1.0
            # nothing usable among the sectors: fall back to the central fit
            dead = (~central) & (side.sum(1) == 0)
            w[dead] = 0.0
            w[dead, 0] = 1.0
            self.last_central = central
        s = w.sum(1, keepdim=True)
        dead = s.squeeze(1) == 0
        w = w / s.clamp_min(1e-300)
        w[dead, 0] = 1.0                    # every stencil deactivated
        return w

    # -- states -------------------------------------------------------------
    def _value_at(self, k, coeff, f, idx, offset):
        """Candidate ``k`` of particles ``idx`` evaluated at physical
        ``offset`` (P, dim) from the particle."""
        c = self.c[k]
        s = c.op.s
        xi = offset / c.h
        P = torch.prod(xi[:, None, :] ** s.exps.to(xi.dtype)[None], dim=-1)
        v = (coeff[idx] * P).sum(-1)
        return v if c.constant else v + f[idx]

    def interface_states(self, f, radius_factor: float = 1.6):
        """Left / right states at the midpoints of all pairs closer than
        ``radius_factor * sqrt(V)``. Returns ``i, j, f_L, f_R, mid_offset``."""
        s0 = self.c[0].op.s
        h0 = self.c[0].h
        pair = s0.i_idx != s0.j_idx
        i, j = s0.i_idx[pair], s0.j_idx[pair]
        d = s0.xi[pair] * h0
        keep = torch.linalg.norm(d, dim=-1) < radius_factor * self.sqrtV
        i, j, d = i[keep], j[keep], d[keep]

        coeffs = self._coeffs(f)
        beta = self.smoothness(coeffs)
        omega = self.weights(beta)
        self.last_omega, self.last_beta = omega, beta

        def state(idx, off):
            val = torch.zeros(idx.shape[0], dtype=f.dtype, device=f.device)
            for k in range(self.n_cand):
                wk = omega[idx, k]
                nz = wk != 0
                if nz.any():
                    v = self._value_at(k, coeffs[k], f, idx[nz], off[nz])
                    val[nz] += wk[nz] * v
            return val

        return i, j, state(i, 0.5 * d), state(j, -0.5 * d), 0.5 * d


def _make(kind, positions, volumes, box, cands_spec, ell, sqrtV, params,
          kernel, cond_max):
    cands = []
    for spec in cands_spec:
        sys_ = build_system(positions, volumes, spec["h"], spec["degree"],
                            kernel=kernel, box=box, constant=spec["constant"],
                            select=spec["select"],
                            unit_weights=spec["unit_weights"],
                            cond_max=cond_max)
        # fewer particles than 1.2x basis functions: not robust (Avesani: 2 nc)
        sys_.deficient |= sys_.num_nbrs < spec["min_nbrs"]
        cands.append(_Candidate(RKPMOperator(sys_), spec["h"], spec["degree"],
                                spec["constant"]))
    return Reconstructor(kind, cands, ell, sqrtV, params)


def teno_reconstructor(positions, volumes, box=None, order="O4",
                       kernel="wendland2", cond_max=1e10,
                       dir_radius: float = 4.5,
                       dir_min: int | None = None) -> Reconstructor:
    """MLS-TENO (Gao et al. 2023). 2-D (nine stencils) or 1-D (three).

    ``dir_radius`` (units of sqrt(V)) and ``dir_min`` (minimum particles in a
    sector stencil) are exposed because the paper's pair (4.5 sqrt(V), >= 10)
    is inconsistent on a uniform lattice: a 45-degree sector of radius
    4.5 sqrt(V) holds ~pi 4.5^2 / 8 ~ 8 particles, so with ``dir_min = 10``
    most sectors are deactivated (found 2026-10-09: the scheme then falls
    back to the central fit next to a discontinuity and overshoots). The
    default ``dir_min`` is the smallest solvable degree-2 fit (nc + 1 = 7).
    """
    deg, rc, CT = TENO_ORDERS[order]
    dim = positions.shape[1]
    sqrtV = float(volumes.mean()) ** (1.0 / dim)
    n_dir = 8 if dim == 2 else 2
    # 1-D is not in the paper (2-D scheme): there a radius of rc*sqrt(V) holds
    # only ~2 rc points, so widen it to keep a robust central fit.
    spec = [dict(h=(rc if dim == 2 else rc + 1.5) * sqrtV, degree=deg,
                 constant=True, select=None, unit_weights=False,
                 min_nbrs={3: 18, 4: 28, 5: 40}[deg] if dim == 2 else deg + 2)]
    nc2 = math.comb(dim + 2, 2)
    for s in range(n_dir):
        spec.append(dict(h=dir_radius * sqrtV, degree=2, constant=True,
                         select=_sector(dim, s), unit_weights=False,
                         min_nbrs=(nc2 + 1) if dir_min is None else dir_min))
    return _make("teno", positions, volumes, box, spec, ell=sqrtV, sqrtV=sqrtV,
                 params=dict(eps=1e-12, C_T=CT), kernel=kernel,
                 cond_max=cond_max)


def weno_reconstructor(positions, volumes, box=None, degree=2, sigma_mls=4.0,
                       kernel="wendland2", cond_max=1e10) -> Reconstructor:
    """MLS-WENO (Avesani et al. 2014): nine unweighted no-constant fits."""
    dim = positions.shape[1]
    sqrtV = float(volumes.mean()) ** (1.0 / dim)
    hm = sigma_mls * sqrtV
    n_dir = 8 if dim == 2 else 2
    nc = math.comb(dim + degree, degree)
    spec = [dict(h=hm, degree=degree, constant=False, select=None,
                 unit_weights=True, min_nbrs=2 * nc)]
    for s in range(n_dir):
        spec.append(dict(h=2.0 * hm, degree=degree, constant=False,
                         select=_sector(dim, s), unit_weights=True,
                         min_nbrs=nc + 1))
    return _make("weno", positions, volumes, box, spec, ell=hm, sqrtV=sqrtV,
                 params=dict(eps=1e-14, r=4, lambda0=1e5), kernel=kernel,
                 cond_max=cond_max)
