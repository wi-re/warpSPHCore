"""Order-p local polynomial fit operator (MLS/RKPM and LABFM) on the core.

``PolyFit`` assembles the moment system with the warp kernels of
``wp_polyfit.py``, solves the small dense systems batched on the torch side
(Jacobi-equilibrated, symmetric eigen-decomposition for the pseudo-inverse
and the condition number), and reads values / gradients / Hessians /
Laplacians off the fit coefficients:

    f(x) ~ sum_{|a| <= p} c_a * xi^a ,   xi = (x - x_i) / h_i ,
    M c = b ,  M = sum_j w P P^T ,  b = sum_j w P (f_j - kappa f_i) ,
    d^a f = a! c_a / h_i^|a|

``constant=True`` is the MLS/RKPM fit (``c_0`` is the reproducing-kernel
value, kappa = 0); ``constant=False`` is the LABFM form (no constant basis
function, data taken as differences, kappa = 1, ``f_i`` reproduced exactly).
Both reproduce polynomials through degree p exactly wherever ``M`` has full
rank; rows with fewer neighbours than basis functions or condition number
above ``cond_max`` are flagged in ``deficient`` (their coefficients are the
minimum-norm solution -- callers decide whether to mask them).

The pure-torch reference (and the full derivation / tests) is
``higherOrderSPH/harness/rkpm.py``; this module is validated against it in
``tests/operations/test_polyfit.py``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..dataTypes import DomainDescription, ParticleState
from ..enumTypes import OperationDirection, SupportScheme, KernelFunctions
from ..dataTypes import OperationProperties
from .wp_polyfit import _computePolyMoments_stateBackend, _computePolyRHS_stateBackend

__all__ = ["PolyFit", "monomialExponents"]


def monomialExponents(dim: int, order: int) -> list[tuple[int, ...]]:
    """Exponent tuples of the monomials of total degree <= ``order`` in
    ``dim`` variables: by total degree, reverse-lexicographic within a
    degree; index 0 is the constant."""
    if dim < 1 or order < 0:
        raise ValueError(f"need dim >= 1 and order >= 0 (got {dim}, {order})")
    out = []
    for deg in range(order + 1):
        combos = [e for e in itertools.product(range(deg + 1), repeat=dim)
                  if sum(e) == deg]
        out.extend(sorted(combos, reverse=True))
    return out


@dataclass
class PolyFit:
    """Assembled order-p moment system of one particle set. Build with
    :meth:`build`."""
    dim: int
    order: int
    constant: bool
    exps: torch.Tensor             # (n, dim) int32 (device) -- what the kernels read
    h: torch.Tensor                # (N,) per-particle scale (the query support)
    M: torch.Tensor                # (N, n, n)
    Minv: torch.Tensor             # (N, n, n)
    cond: torch.Tensor             # (N,) of the Jacobi-equilibrated matrix
    num_nbrs: torch.Tensor         # (N,) neighbours counted by the kernel
    deficient: torch.Tensor        # (N,) bool
    # call context for the right-hand-side kernel
    _args: dict

    @property
    def n_basis(self) -> int:
        return self.exps.shape[0]

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, queryParticles: ParticleState, domain: DomainDescription,
              kernel: KernelFunctions, order: int, constant: bool = True,
              adjacency=None, referenceParticles: Optional[ParticleState] = None,
              queryVolumes: Optional[torch.Tensor] = None,
              referenceVolumes: Optional[torch.Tensor] = None,
              operationMode: OperationDirection = OperationDirection.AllToAll,
              equilibrate: bool = True, rtol: float = 1e-12,
              cond_max: float = 1e10, sector: int = -1, numSectors: int = 8,
              unitWeights: bool = False, minNeighbors: Optional[int] = None) -> "PolyFit":
        pos = queryParticles.positions
        N, dim = pos.shape
        exps_all = monomialExponents(dim, order)
        exps_list = exps_all if constant else exps_all[1:]
        exps = torch.tensor(exps_list, dtype=torch.int32, device=pos.device)
        n = exps.shape[0]

        if sector >= 0 and dim == 3:
            raise NotImplementedError("sector stencils are defined for 1-D and 2-D")
        props = OperationProperties(kernel=kernel, supportMode=SupportScheme.Gather,
                                    operationMode=operationMode)
        stencil = dict(sector=sector, numSectors=numSectors, unitWeights=unitWeights)
        Mp, nnb = _computePolyMoments_stateBackend(
            queryParticles, props, domain, exps,
            queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
            adjacency=adjacency, referenceParticles=referenceParticles, **stencil)

        rows, cols = torch.tril_indices(n, n, device=pos.device)
        M = torch.zeros(N, n, n, dtype=Mp.dtype, device=pos.device)
        M[:, rows, cols] = Mp
        M = M + torch.tril(M, -1).transpose(1, 2)

        if equilibrate:
            d = torch.sqrt(torch.diagonal(M, dim1=1, dim2=2).abs().clamp_min(1e-300))
        else:
            d = torch.ones(N, n, dtype=M.dtype, device=M.device)
        dd = d[:, :, None] * d[:, None, :]
        Me = M / dd
        # Me is symmetric PSD: eigenvalues == singular values. eigvalsh, not
        # svdvals -- torch's batched CUDA SVD falls off its fast path above
        # 32 x 32 (19 s vs 0.02 s for 2200 matrices of size 44).
        S = torch.linalg.eigvalsh(Me).abs()
        cond = S.max(1).values / S.min(1).values.clamp_min(1e-300)
        Minv = torch.linalg.pinv(Me, rtol=rtol, hermitian=True) / dd
        deficient = (nnb < (n if minNeighbors is None else minNeighbors)) \
            | (cond > cond_max) | ~torch.isfinite(cond)

        args = dict(queryParticles=queryParticles, props=props, domain=domain,
                    queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                    adjacency=adjacency, referenceParticles=referenceParticles,
                    stencil=stencil)
        return cls(dim=dim, order=order, constant=constant, exps=exps,
                   h=queryParticles.supports, M=M, Minv=Minv, cond=cond,
                   num_nbrs=nnb, deficient=deficient, _args=args)

    # ------------------------------------------------------------------
    def coefficients(self, f: torch.Tensor) -> torch.Tensor:
        """Fit coefficients ``(N, n)`` for a scalar field ``(N,)`` or
        ``(N, n, D)`` for ``(N, D)``."""
        scalar = f.dim() == 1
        F = (f[:, None] if scalar else f).contiguous()
        a = self._args
        b = _computePolyRHS_stateBackend(
            a["queryParticles"], a["props"], a["domain"], self.exps, F,
            subtractCenter=not self.constant,
            queryVolumes=a["queryVolumes"], referenceVolumes=a["referenceVolumes"],
            adjacency=a["adjacency"], referenceParticles=a["referenceParticles"],
            **a["stencil"])
        b = b.reshape(F.shape[0], self.n_basis, F.shape[1])
        c = torch.einsum("nab,nbd->nad", self.Minv, b)
        return c[:, :, 0] if scalar else c

    def _col(self, c: torch.Tensor, e: tuple[int, ...]) -> torch.Tensor:
        idx = self._index(e)
        return c[:, idx]

    def _index(self, e: tuple[int, ...]) -> int:
        if not hasattr(self, "_idx"):
            self._idx = {tuple(r): k for k, r in enumerate(self.exps.tolist())}
        return self._idx[e]

    def _scale(self, power: int, like: torch.Tensor) -> torch.Tensor:
        h = self.h.to(like.dtype) ** power
        return h.reshape(-1, *([1] * (like.dim() - 1)))

    def values(self, f: torch.Tensor) -> torch.Tensor:
        if not self.constant:
            return f.clone()
        return self._col(self.coefficients(f), (0,) * self.dim)

    def gradient(self, f: torch.Tensor) -> torch.Tensor:
        """``(N, dim)`` for scalar f; ``(N, D, dim)`` for ``(N, D)``."""
        c = self.coefficients(f)
        cols = []
        for k in range(self.dim):
            e = tuple(1 if a == k else 0 for a in range(self.dim))
            col = self._col(c, e)
            cols.append(col / self._scale(1, col))
        return torch.stack(cols, dim=-1)

    def hessian(self, f: torch.Tensor) -> Optional[torch.Tensor]:
        """``(N, dim, dim)`` for a scalar field, ``out[:, a, b] = d_a d_b f``;
        None for ``order < 2``."""
        if self.order < 2 or f.dim() != 1:
            return None
        c = self.coefficients(f)
        H = torch.zeros(f.shape[0], self.dim, self.dim, dtype=c.dtype, device=c.device)
        for a in range(self.dim):
            for b in range(self.dim):
                e = [0] * self.dim
                e[a] += 1
                e[b] += 1
                fact = math.prod(math.factorial(x) for x in e)
                H[:, a, b] = fact * self._col(c, tuple(e)) / self.h.to(c.dtype) ** 2
        return H

    def laplacian(self, f: torch.Tensor) -> Optional[torch.Tensor]:
        H = self.hessian(f)
        return None if H is None else torch.diagonal(H, dim1=1, dim2=2).sum(-1)

    def derivative(self, f: torch.Tensor, alpha: tuple[int, ...]) -> torch.Tensor:
        """The partial derivative ``d^alpha f`` (``alpha`` a multi-index of
        total degree 1..order; e.g. ``(2, 0)`` is ``d^2 f / dx^2``). This is
        the operator hyperviscosity needs: the highest-order derivatives the
        fit carries (``order`` itself), e.g. ``(order, 0)`` -- whose
        coefficient, ``h**order`` scaling and time-step limit belong to the
        caller (King et al. 2020 Sec. 4.3)."""
        alpha = tuple(int(a) for a in alpha)
        if len(alpha) != self.dim or not (1 <= sum(alpha) <= self.order):
            raise ValueError(f"alpha must have {self.dim} entries and total degree in [1, {self.order}]")
        c = self.coefficients(f)
        fact = math.prod(math.factorial(a) for a in alpha)
        col = self._col(c, alpha)
        return fact * col / self._scale(sum(alpha), col)

    # ------------------------------------------------------------------
    @staticmethod
    def interfaceOffsets(d: torch.Tensor, hi: torch.Tensor, hj: torch.Tensor):
        """Offsets of the pair interface point
        ``r_ij = (h_j r_i + h_i r_j) / (h_i + h_j)`` (Gao et al. 2023 Eq. 25;
        Avesani et al. 2014 use the arithmetic midpoint, which is the same
        point for equal supports) from each particle: ``(r_ij - r_i,
        r_ij - r_j)`` given the pair vector ``d = r_j - r_i`` (minimum image).
        With unequal supports the interface sits closer to the particle with
        the *smaller* support, where the two kernels' influence balances; the
        arithmetic midpoint is a position error of O(h_i - h_j) there."""
        w = (hi + hj).unsqueeze(-1)
        return d * (hi.unsqueeze(-1) / w), -d * (hj.unsqueeze(-1) / w)

    def evaluate(self, f: torch.Tensor, idx: torch.Tensor, offset: torch.Tensor,
                 coefficients: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Particle ``idx``'s fit of the scalar field ``f`` evaluated at
        ``x_idx + offset`` (``offset`` of shape ``(P, dim)``)."""
        c = self.coefficients(f) if coefficients is None else coefficients
        xi = offset / self.h[idx].unsqueeze(-1).to(offset.dtype)
        e = self.exps.to(offset.dtype)
        P = torch.prod(xi.unsqueeze(1) ** e.unsqueeze(0), dim=-1)          # (P, n)
        v = (c[idx] * P).sum(-1)
        return v if self.constant else v + f[idx]

    def interfaceStates(self, f: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                        d: torch.Tensor):
        """Left / right states of a scalar field at the interface points of
        the pairs ``(i, j)`` (``d = r_j - r_i``, minimum image): ``f_L`` is
        ``i``'s fit and ``f_R`` is ``j``'s fit, both evaluated at the Eq. 25
        point. In a smooth region both equal ``f(r_ij)`` to O(h^(order+1))."""
        c = self.coefficients(f)
        oi, oj = self.interfaceOffsets(d, self.h[i].to(d.dtype), self.h[j].to(d.dtype))
        return (self.evaluate(f, i, oi, c), self.evaluate(f, j, oj, c))
