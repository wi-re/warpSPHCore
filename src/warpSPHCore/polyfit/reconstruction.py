"""Non-oscillatory interface reconstruction for Riemann-SPH: MLS-TENO and
MLS-WENO on top of :class:`PolyFit`.

Per particle, nine candidate polynomial fits (one central, eight 45-degree
sector stencils in 2-D; one central + two one-sided in 1-D) are combined by a
nonlinear weighting into one reconstruction polynomial, whose value at the
pair interface point ``r_ij = (h_j r_i + h_i r_j) / (h_i + h_j)`` is the left /
right state a Riemann solver consumes (the solver coupling itself is the
caller's: warpSPH's Godunov SPH). Two weightings share the candidates:

* **TENO** -- Gao, Liang & Fu (2023, JCP 489:112270): kernel-weighted MLS fits,
  central degree 3 / 4 / 5 ("O4 / O5 / O6") at radius 2.5 / 3.2 / 4.0 sqrt(V),
  sector fits of degree 2 at radius 4.5 sqrt(V); smoothness ``beta_s`` is the
  exact integral of the squared derivatives of the stencil polynomial over the
  square of edge 2 (Eq. 26-27); ``gamma_s = 1/(beta_s + 1e-12)^6``,
  ``chi_0 = gamma_0 / sum_s gamma_s``; the central stencil is used alone when
  ``chi_0 >= C_T`` (1e-5 / 1e-6 / 1e-7), otherwise the sector stencils are
  blended with ``omega_s = gamma_s / sum_{s>=1} gamma_s``.
* **WENO** -- Avesani, Dumbser & Bertaux (2014, JCP 270:278): unweighted
  least-squares fits of ``Q_j - Q_i`` (no constant), central ``r <= h_mls``,
  sectors ``r <= 2 h_mls``, ``h_mls = sigma_mls sqrt(V)``; ``sigma_s =
  sum w_m^2`` in common coordinates, ``omega_s ~ lambda_s / (1e-14 +
  sigma_s)^4``, ``lambda_0 = 1e5``, ``lambda_s = 1``.

The weights are evaluated as a softmax over ``log gamma`` (stable in float32,
where ``1/(beta + 1e-12)^6`` overflows), which is algebraically the same.

Where the papers are not explicit (flagged for review; see
``higherOrderSPH/harness/reconstruct.py`` for the pure-torch reference this was
ported from, and ``higher_order.md`` Phase 6): (a) the TENO ``beta`` sums all
partial derivatives of order 1..p with unit weight over the square
``[-sqrt(V), sqrt(V)]^d`` common to every stencil; (b) a hard radius around
``sqrt(V)`` selects the stencil members; (c) the interface point is Eq. 25 with
the particle supports ``queryParticles.supports``; (d) a stencil with fewer
members than required, or ``cond > cond_max``, is deactivated (weight 0), and
if every stencil is deactivated the central fit is used regardless.

The sector stencil sizes of Gao et al. are not consistent on a uniform lattice
(a 45-degree sector of radius 4.5 sqrt(V) holds ~8 particles, they require 10;
with that minimum the sectors near a discontinuity were deactivated and the
scheme overshot): ``dirMin`` defaults to the smallest solvable degree-2 fit.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.autograd.forward_ad as fwAD

from ..dataTypes import DomainDescription, ParticleState
from ..enumTypes import KernelFunctions, SupportScheme
from .polyfit import PolyFit

__all__ = ["Reconstructor", "smoothnessGram"]

TENO_ORDERS = {  # name -> (central degree, central radius / sqrt V, C_T, min members)
    "O4": (3, 2.5, 1e-5, 18),
    "O5": (4, 3.2, 1e-6, 28),
    "O6": (5, 4.0, 1e-7, 40),
}


def smoothnessGram(exps: torch.Tensor) -> torch.Tensor:
    """``G[a, b] = sum_{1 <= |alpha| <= p} int_{[-1,1]^d} D^alpha m_a D^alpha m_b``
    for the monomials ``m = xi^e`` (Gao et al. Eq. 27, exact instead of by
    quadrature)."""
    E = [tuple(e) for e in exps.tolist()]
    dim = len(E[0])
    pmax = max(sum(e) for e in E)
    alphas = [a for a in itertools.product(range(pmax + 1), repeat=dim) if 1 <= sum(a) <= pmax]
    G = torch.zeros(len(E), len(E), dtype=torch.float64)

    def deriv(e, a):
        if any(ek < ak for ek, ak in zip(e, a)):
            return 0.0, None
        coef = math.prod(math.factorial(ek) // math.factorial(ek - ak) for ek, ak in zip(e, a))
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
                    continue
                G[i, j] += ci * cj * math.prod(2.0 / (k + 1) for k in u)
    return G


class Reconstructor:
    """Nine-candidate nonlinear reconstruction. Build with :meth:`teno` or
    :meth:`weno`."""

    def __init__(self, kind: str, fits: list, ell: torch.Tensor, params: dict,
                 interfaceSupports: torch.Tensor):
        self.kind = kind
        self.fits = fits                      # list[PolyFit], index 0 = central
        self.ell = ell                        # (N,) common length for the indicators
        self.params = params
        self.interfaceSupports = interfaceSupports
        self.dim = fits[0].dim
        self._gram: dict = {}
        self.last_central: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _candidateState(q: ParticleState, radius: torch.Tensor) -> ParticleState:
        return ParticleState(positions=q.positions, supports=radius, masses=q.masses,
                             densities=q.densities, kinds=q.kinds)

    @classmethod
    def _build(cls, kind, q, domain, kernel, specs, ell, params, adjacency, cond_max):
        maxR = torch.stack([s["radius"] for s in specs]).amax(0)
        if adjacency is None:
            from ..radiusSearch import radiusSearchCompactHashMap
            # the neighbour search is structural: build it from the primals (forward-mode
            # duals on the particle tensors are handled by PolyFit.build, not here)
            prim = lambda t: t if t is None else fwAD.unpack_dual(t).primal
            adjacency = radiusSearchCompactHashMap(
                ParticleState(positions=prim(q.positions), supports=prim(maxR),
                              masses=prim(q.masses), densities=prim(q.densities), kinds=q.kinds),
                domain, mode=SupportScheme.SuperSymmetric)
        fits = []
        for sp in specs:
            fits.append(PolyFit.build(
                cls._candidateState(q, sp["radius"]), domain, kernel, sp["degree"],
                constant=sp["constant"], adjacency=adjacency, cond_max=cond_max,
                sector=sp["sector"], numSectors=sp["numSectors"],
                unitWeights=sp["unitWeights"], minNeighbors=sp["minNbrs"]))
        return cls(kind, fits, ell, params, q.supports)

    @classmethod
    def teno(cls, queryParticles: ParticleState, domain: DomainDescription,
             kernel: KernelFunctions, order: str = "O4", adjacency=None,
             dirRadius: float = 4.5, dirMin: Optional[int] = None,
             cond_max: float = 1e10) -> "Reconstructor":
        """MLS-TENO (Gao et al. 2023). ``order`` is ``"O4"`` / ``"O5"`` / ``"O6"``."""
        deg, rc, CT, cmin = TENO_ORDERS[order]
        q = queryParticles
        dim = q.positions.shape[1]
        sqrtV = (q.masses / q.densities) ** (1.0 / dim)
        nSec = 8 if dim == 2 else 2
        # 1-D is not in the paper: a radius of rc*sqrt(V) holds only ~2 rc points
        r0 = (rc if dim == 2 else rc + 1.5) * sqrtV
        specs = [dict(radius=r0, degree=deg, constant=True, sector=-1, numSectors=nSec,
                      unitWeights=False, minNbrs=cmin if dim == 2 else deg + 2)]
        nc2 = math.comb(dim + 2, 2)
        for s in range(nSec):
            specs.append(dict(radius=dirRadius * sqrtV, degree=2, constant=True, sector=s,
                              numSectors=nSec, unitWeights=False,
                              minNbrs=(nc2 + 1) if dirMin is None else dirMin))
        return cls._build("teno", q, domain, kernel, specs, sqrtV,
                          dict(eps=1e-12, C_T=CT), adjacency, cond_max)

    @classmethod
    def weno(cls, queryParticles: ParticleState, domain: DomainDescription,
             kernel: KernelFunctions, degree: int = 2, sigmaMls: float = 4.0,
             adjacency=None, cond_max: float = 1e10) -> "Reconstructor":
        """MLS-WENO (Avesani et al. 2014)."""
        q = queryParticles
        dim = q.positions.shape[1]
        sqrtV = (q.masses / q.densities) ** (1.0 / dim)
        hm = sigmaMls * sqrtV
        nSec = 8 if dim == 2 else 2
        nc = math.comb(dim + degree, degree)
        specs = [dict(radius=hm, degree=degree, constant=False, sector=-1, numSectors=nSec,
                      unitWeights=True, minNbrs=2 * nc)]
        for s in range(nSec):
            specs.append(dict(radius=2.0 * hm, degree=degree, constant=False, sector=s,
                              numSectors=nSec, unitWeights=True, minNbrs=nc + 1))
        return cls._build("weno", q, domain, kernel, specs, hm,
                          dict(eps=1e-14, r=4, lambda0=1e5), adjacency, cond_max)

    # ------------------------------------------------------------------
    def smoothness(self, coeffs: list) -> torch.Tensor:
        """``beta`` (N, K); deactivated stencils get +inf."""
        cols = []
        for k, (pf, cf) in enumerate(zip(self.fits, coeffs)):
            deg = torch.tensor([sum(e) for e in pf.exps.tolist()], dtype=cf.dtype, device=cf.device)
            ct = cf * (self.ell / pf.h).unsqueeze(-1).to(cf.dtype) ** deg
            if self.kind == "weno":
                beta = (ct ** 2).sum(-1)
            else:
                if k not in self._gram:
                    self._gram[k] = smoothnessGram(pf.exps).to(cf)
                beta = torch.einsum("na,ab,nb->n", ct, self._gram[k], ct)
            cols.append(beta)
        beta = torch.stack(cols, dim=1)
        active = torch.stack([~pf.deficient for pf in self.fits], dim=1)
        return torch.where(active, beta, torch.full_like(beta, float("inf")))

    def weights(self, beta: torch.Tensor) -> torch.Tensor:
        """Nonlinear weights ``omega`` (N, K), rows summing to 1."""
        act = torch.isfinite(beta)
        big = torch.finfo(beta.dtype).max
        if self.kind == "weno":
            logw = torch.full_like(beta, float("-inf"))
            lam = torch.ones_like(beta)
            lam[:, 0] = self.params["lambda0"]
            logw = torch.where(act, torch.log(lam) - self.params["r"] * torch.log(self.params["eps"] + beta.clamp(max=big)),
                               logw)
            w = torch.softmax(logw, dim=1)
        else:
            lg = torch.where(act, -6.0 * torch.log(beta.clamp(max=big) + self.params["eps"]),
                             torch.full_like(beta, float("-inf")))
            chi0 = torch.softmax(lg, dim=1)[:, 0]
            central = (chi0 >= self.params["C_T"]) & act[:, 0]
            side = lg[:, 1:]
            sidew = torch.softmax(side, dim=1)
            sidew = torch.nan_to_num(sidew, nan=0.0)          # all sectors deactivated
            w = torch.zeros_like(beta)
            w[:, 1:] = sidew
            dead = (~central) & ~torch.isfinite(side).any(1)
            use0 = central | dead
            w[use0] = 0.0
            w[use0, 0] = 1.0
            self.last_central = central
        w = torch.nan_to_num(w, nan=0.0)
        s = w.sum(1, keepdim=True)
        dead = s.squeeze(1) == 0                              # every stencil deactivated
        w = w / s.clamp_min(torch.finfo(w.dtype).tiny)
        w[dead, 0] = 1.0
        return w

    def interfaceStates(self, f: torch.Tensor, i: torch.Tensor, j: torch.Tensor, d: torch.Tensor):
        """Left / right states of the scalar field ``f`` at the Eq. 25 interface
        points of the pairs ``(i, j)`` (``d = r_j - r_i``, minimum image).
        Returns ``(f_L, f_R)``; the stencil weights and indicators of the last
        call are kept in ``last_omega`` / ``last_beta``."""
        coeffs = [pf.coefficients(f) for pf in self.fits]
        beta = self.smoothness(coeffs)
        omega = self.weights(beta)
        self.last_beta, self.last_omega = beta, omega
        h = self.interfaceSupports.to(d.dtype)
        oi, oj = PolyFit.interfaceOffsets(d, h[i], h[j])

        def state(idx, off):
            val = torch.zeros(idx.shape[0], dtype=f.dtype, device=f.device)
            for k, pf in enumerate(self.fits):
                wk = omega[idx, k]
                v = pf.evaluate(f, idx, off, coeffs[k])
                val = val + torch.where(wk != 0, wk * v, torch.zeros_like(v))
            return val

        return state(i, oi), state(j, oj)
