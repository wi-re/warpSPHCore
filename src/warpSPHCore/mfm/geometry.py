"""Meshless volume partition and effective faces (Hopkins 2015, Sec. 2.1-2.4).

Every particle ``i`` owns the fraction ``psi_i(x) = W(x - x_i, h(x)) / omega(x)``
of space (a partition of unity, ``omega(x) = sum_j W(x - x_j, h)``); to second
order its effective volume is ``V_i = 1 / omega_i`` with ``omega_i = omega(x_i)``
(Eq. 27). Gradients are the weighted least-squares matrix gradients

    (grad f)_i = B_i sum_j (f_j - f_i) (x_j - x_i) psi_j(x_i) = Ehat_i^-1 sum_j W_ij (f_j - f_i) d_ij,
    Ehat_i = sum_j W_ij d_ij d_ij^T ,   d_ij = x_j - x_i ,   W_ij = W(|d_ij|, h_i)

(Eq. 12-14; the volume factor of ``psi`` cancels), exact for linear fields on
any particle configuration. The discretised conservation law is

    d(V_i U_i)/dt = - sum_j A_ij . F~_ij ,
    A_ij = V_i psi~_j(x_i) - V_j psi~_i(x_j) = V_i W_ij(h_i) Ehat_i^-1 d_ij + V_j W_ij(h_j) Ehat_j^-1 d_ij

(Eq. 17-19): ``A_ij = -A_ji`` (so every conserved quantity is conserved to
round-off whatever the fluxes are) and ``F~_ij`` is a Riemann-solver flux
through the *effective face*. ``A_ij`` points from ``i`` towards ``j``: the left
state of the face Riemann problem is ``i``'s, the right state ``j``'s.

This module is pure torch on a symmetric pair list (``interfacePairs``: each
unordered pair once, within ``max(h_i, h_j)``), so the whole chain is
differentiable (reverse mode and ``torch.autograd.forward_ad``) in positions
and supports. The cost is memory proportional to the number of pairs, not an
additional neighbour search.

**Closure.** ``sum_j A_ij`` is the discrete closed-surface integral of the
cell. It vanishes for the exact (Lanson-Vila) volume partition, but the
second-order quadrature of Eq. 10-11 makes ``A_ij`` close only on particle
configurations with ``sum_j d_ij psi_j(x_i) = 0`` (regular lattices). On
disordered sets the residual is not small -- measured on a jittered 2-D lattice
(jitter 0.3 dx, 28 neighbours) ``|sum_j A_ij|`` is 1.3 times a typical single
``|A_ij|`` -- and acts on a uniform pressure as a spurious force
``-P sum_j A_ij`` that does not decrease with resolution. ``closure="project"``
(default) removes it with the smallest antisymmetric correction,
``A_ij += kappa (lambda_j - lambda_i)`` with ``L lambda = -sum_j A_ij`` solved by
conjugate gradients on the pair graph: still antisymmetric (conservation is
untouched), closed to the CG tolerance, and the divergence of a linear flux
field, the consistency the matrix gradient guarantees, is unchanged to the
measured digit. ``closure="none"`` is the paper's face vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..dataTypes import DomainDescription, ParticleState
from ..enumTypes import KernelFunctions
from ..polyfit.polyfit import interfacePairs
from .kernels import kernelWeight

__all__ = ["MeshlessGeometry"]


def _scatter(n: int, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((n,) + val.shape[1:], dtype=val.dtype, device=val.device)
    return out.index_add(0, idx, val)


def _closeFaces(N: int, i: torch.Tensor, j: torch.Tensor, A: torch.Tensor,
                iters: int = 200, tol: Optional[float] = None, power: float = 1.0) -> torch.Tensor:
    """Antisymmetric correction ``C_ij = kappa_ij (lambda_j - lambda_i)``,
    ``kappa_ij = |A_ij|``, with
    ``sum_j (A_ij + C_ij) = 0`` for every particle (graph-Laplacian Poisson
    problem, conjugate gradients; the right-hand side sums to zero because
    ``A`` is antisymmetric, so the singular constant mode is harmless).
    Converged when the residual has dropped by ``tol`` relative to the initial
    one (default 1e-6 in float64, 1e-4 in float32) -- the closure error acts as
    ``P sum_j A_ij`` and must stay far below the physical signal, so it is
    reduced relative to itself, not to the face size; a set closed to round-off
    (``1e-12`` of the face norm in float64, 1e-5 in float32) returns immediately."""
    if tol is None:
        tol = 1e-6 if A.dtype == torch.float64 else 1e-4
    floor = 1e-12 if A.dtype == torch.float64 else 1e-5
    a = _scatter(N, i, A) - _scatter(N, j, A)                  # sum_j A_ij
    a = a - a.mean(0, keepdim=True)                            # compatibility: the sum over particles is zero
    bnorm = a.norm()
    if not bool(bnorm > floor * A.norm()):                     # already closed
        return A
    target = tol * bnorm

    # weights ~ the face size: a pair on the kernel edge (|A| ~ 0) must stay a small face
    # of unchanged orientation; an unweighted correction would dominate it and flip its sign
    kappa = torch.linalg.norm(A, dim=-1, keepdim=True) ** power

    def neg_lap(l):                                           # -(sum_j kappa (l_j - l_i))
        t = kappa * (l[j] - l[i])
        return -(_scatter(N, i, t) - _scatter(N, j, t))

    x = torch.zeros_like(a)
    r = a.clone()
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(iters):
        Ap = neg_lap(p)
        pAp = (p * Ap).sum()
        if not bool(pAp > 1e-30 * rs):                         # converged (or null-space direction)
            break
        al = rs / pAp
        x = x + al * p
        r = r - al * Ap
        rn = (r * r).sum()
        if bool(rn.sqrt() <= target):
            break
        p = r + rn / rs * p
        rs = rn
    return A + kappa * (x[j] - x[i])


@dataclass
class MeshlessGeometry:
    """Pair-resolved meshless geometry of one particle set. Build with :meth:`build`."""
    dim: int
    N: int
    positions: torch.Tensor        # (N, dim)
    supports: torch.Tensor         # (N,)
    i: torch.Tensor                # (P,) pair list, i < j
    j: torch.Tensor                # (P,)
    d: torch.Tensor                # (P, dim)  x_j - x_i (minimum image)
    Wi: torch.Tensor               # (P,)  W(|d|, h_i)
    Wj: torch.Tensor               # (P,)  W(|d|, h_j)
    omega: torch.Tensor            # (N,)  sum_j W_ij(h_i), self included
    volume: torch.Tensor           # (N,)  V_i = 1 / omega_i
    Einv: torch.Tensor             # (N, dim, dim)  Ehat_i^-1
    cond: torch.Tensor             # (N,)  Hopkins N_cond (Eq. C1)
    num_nbrs: torch.Tensor         # (N,)  neighbours inside h_i
    deficient: torch.Tensor        # (N,)  bool: too few neighbours or N_cond > cond_max
    A: torch.Tensor                # (P, dim)  effective face vector, points i -> j (closed)
    A0: torch.Tensor               # (P, dim)  the paper's face vector (before closure)
    domain: DomainDescription
    kernel: KernelFunctions

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, particles: ParticleState, domain: DomainDescription, kernel: KernelFunctions,
              adjacency=None, cond_max: float = 1.0e3, rtol: float = 1.0e-12,
              closure: str = "project", closure_power: float = 1.0) -> "MeshlessGeometry":
        """``particles.positions`` / ``particles.supports`` are used; masses and
        densities are not (the volumes come from the kernel sums). Without
        ``adjacency`` the pair list is a symmetric radius search at the
        supports. ``cond_max`` flags ill-conditioned rows (Hopkins App. C, the
        paper suggests 100-1000). ``closure``: ``"project"`` (default) or
        ``"none"`` (the paper's face vector), see the module docstring."""
        if closure not in ("project", "none"):
            raise ValueError("closure must be 'project' or 'none'")
        x, h = particles.positions, particles.supports
        N, dim = x.shape
        i, j, d = interfacePairs(particles, domain, adjacency, halfList=True)
        r = torch.linalg.norm(d, dim=-1)
        Wi = kernelWeight(r, h[i], dim, kernel)
        Wj = kernelWeight(r, h[j], dim, kernel)
        W0 = kernelWeight(torch.zeros_like(h), h, dim, kernel)

        omega = W0 + _scatter(N, i, Wi) + _scatter(N, j, Wj)
        dd = d.unsqueeze(-1) * d.unsqueeze(-2)                         # (P, dim, dim)
        Ehat = _scatter(N, i, Wi[:, None, None] * dd) + _scatter(N, j, Wj[:, None, None] * dd)
        nnb = 1 + _scatter(N, i, (Wi > 0).to(x.dtype)) + _scatter(N, j, (Wj > 0).to(x.dtype))

        # the dimensionless, scale-free matrix E = Ehat / (omega h^2)
        scale = (omega * h ** 2)[:, None, None]
        Es = Ehat / scale
        Es_inv = torch.linalg.pinv(Es, rtol=rtol, hermitian=True)
        Einv = Es_inv / scale
        cond = torch.linalg.matrix_norm(Es) * torch.linalg.matrix_norm(Es_inv) / dim
        deficient = (nnb < dim + 1) | (cond > cond_max) | ~torch.isfinite(cond)

        volume = 1.0 / omega
        Ai = (volume[i] * Wi)[:, None] * torch.einsum("pab,pb->pa", Einv[i], d)
        Aj = (volume[j] * Wj)[:, None] * torch.einsum("pab,pb->pa", Einv[j], d)
        A0 = Ai + Aj
        A = _closeFaces(N, i, j, A0, power=closure_power) if closure == "project" else A0
        return cls(dim=dim, N=N, positions=x, supports=h, i=i, j=j, d=d, Wi=Wi, Wj=Wj,
                   omega=omega, volume=volume, Einv=Einv, cond=cond, num_nbrs=nnb,
                   deficient=deficient, A=A, A0=A0, domain=domain, kernel=kernel)

    # ------------------------------------------------------------------
    @property
    def numPairs(self) -> int:
        return self.i.shape[0]

    def faceFraction(self) -> torch.Tensor:
        """``h_i / (h_i + h_j)``: the face (quadrature point, Eq. 20) sits at
        ``x_i + faceFraction * d_ij``."""
        hi, hj = self.supports[self.i], self.supports[self.j]
        return hi / (hi + hj)

    def gradient(self, f: torch.Tensor, mask_deficient: bool = True) -> torch.Tensor:
        """Matrix least-squares gradient: ``(N, dim)`` for ``f`` of shape
        ``(N,)``, ``(N, K, dim)`` for ``(N, K)``. Exact for linear fields.
        Rows flagged ``deficient`` return zero (first-order reconstruction;
        Hopkins falls back to the plain SPH gradient there)."""
        scalar = f.dim() == 1
        F = f[:, None] if scalar else f
        df = F[self.j] - F[self.i]                                        # (P, K)
        # both sides of a pair see the same (f_j - f_i) d, weighted by their own W
        s = (_scatter(self.N, self.i, self.Wi[:, None, None] * df[:, :, None] * self.d[:, None, :])
             + _scatter(self.N, self.j, self.Wj[:, None, None] * df[:, :, None] * self.d[:, None, :]))
        g = torch.einsum("nab,nkb->nka", self.Einv, s)                   # (N, K, dim)
        if mask_deficient:
            g = torch.where(self.deficient[:, None, None], torch.zeros_like(g), g)
        return g[:, 0] if scalar else g

    def closure(self) -> torch.Tensor:
        """``sum_j A_ij`` per particle (``(N, dim)``): zero for a closed cell
        (``closure="project"`` guarantees it to the CG tolerance; the paper's
        face vector leaves an O(1)-relative residual on disordered sets)."""
        return _scatter(self.N, self.i, self.A) - _scatter(self.N, self.j, self.A)

    def divergence(self, flux: torch.Tensor) -> torch.Tensor:
        """``dQ_i/dt = - sum_j |A_ij| Fn_ij`` for per-pair normal fluxes
        ``flux`` of shape ``(P, K)`` (flux of component ``k`` from ``i`` to
        ``j`` per unit ``|A|``): returns ``(N, K)``. Antisymmetric by
        construction, so every column sums to zero over the particles."""
        a = torch.linalg.norm(self.A, dim=-1, keepdim=True)
        t = a * flux
        return -_scatter(self.N, self.i, t) + _scatter(self.N, self.j, t)
