#!/usr/bin/env python3
"""Order-p MLS / RKPM reproducing-kernel operator (Phase 4 reference, pure torch).

A generic, order-parametrised moment-matrix builder and local polynomial
fit, so the same machinery serves Phase 4 (MLS/RKPM, p = 1..3) and Phase 5
(LABFM reuses the moment system with anisotropic basis functions).

For each particle ``i`` the reference fits the local polynomial

    f(x) ~ sum_{|a| <= p} c_a(i) * xi^a ,      xi = (x - x_i) / h

by kernel-weighted least squares over its neighbours,

    M_i c_i = b_i ,
    M_i = sum_j w_ij P(xi_ij) P(xi_ij)^T ,   b_i = sum_j w_ij P(xi_ij) f_j ,
    w_ij = V_j W(|x_ij|/h)  ,   P = the monomials of total degree <= p,

and reads the operators off the coefficients (the "diffuse derivative" of
Nayroles et al. / Taylor-MLS form; ``c_0`` is the reproducing-kernel value):

    f_i = c_0 ,   d^a f = a! c_a / h^|a|   (so grad = c_{e_k}/h, d_k d_l f = ...).

Every polynomial of degree <= p is reproduced exactly, value / gradient /
Hessian, as long as the moment matrix has full rank (>= dim-appropriate
neighbour count, not collinear). The kernel *normalisation* cancels in
``M^-1 b`` (it only rescales w), so the unnormalised Wendland shapes suffice.

Differences from the CRKSPH column (documented because the parent plan asks
for a p = 1 cross-check): the *value* at p = 1 is the CRK reproducing value
``A (1 + B.x) W`` summed over neighbours -- identical. The *gradient* here is
the fit coefficient (the diffuse derivative, the derivative of the fit at
x_i with the moment matrix frozen); CRK's gradient additionally
differentiates A and B (the full RKPM kernel-gradient). Both are exact to
degree 1; they differ at O(h) on non-polynomial fields.

**LABFM variant** (``constant=False``, King et al. 2020): the same weighted
least-squares problem with the constant removed and the data taken as
differences, ``M' c = sum_j w_ij P'(xi_ij) (f_j - f_i)``, ``P'`` = the
non-constant monomials. The derivative weights are
``w_ij e_L^T M'^-1 P'(xi_ij)`` -- the LABFM weights (anisotropic basis
function = kernel x polynomial), exact to degree p like the MLS fit and with
``f_i`` reproduced identically; the paper's order k is the polynomial degree
here. Consistency constraints are the same, the neighbour requirement is
``n_basis - 1`` instead of ``n_basis``.

Scaling: the basis is built from ``xi = x_ij / h`` so ``M`` is O(1) and the
condition number measures geometry, not units; ``equilibrate`` additionally
applies Jacobi (diagonal) scaling before the solve.

Rank handling: the solve uses a pseudo-inverse on the equilibrated matrix
(``rtol``), and flags particles with fewer neighbours than basis functions
or a condition number above ``cond_max`` in ``RKPMSystem.deficient`` (their
coefficient rows are still the minimum-norm solution; callers decide
whether to mask them). Pair data is stored sparsely (COO) so the system is
built once per case and reused for any number of fields.

Pure torch (CPU-testable, float64-native); pair search is blocked brute
force with optional periodic minimum image -- fine at harness sizes
(N <= ~1e5), and the algorithm is neighbour-search agnostic: only
``(i, j, w_ij, P_ij)`` enter the system.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

try:  # imported as `harness.rkpm` (tests) or as a top-level module (scripts)
    from .distribution import kernel_shape
except ImportError:  # pragma: no cover
    from distribution import kernel_shape

__all__ = ["monomial_exponents", "RKPMSystem", "build_system", "RKPMOperator"]


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------

def monomial_exponents(dim: int, order: int) -> list[tuple[int, ...]]:
    """Exponent tuples of the monomials of total degree <= ``order`` in
    ``dim`` variables, ordered by total degree, then reverse-lexicographic
    within a degree (degree 0 first, so index 0 is the constant)."""
    if dim < 1 or order < 0:
        raise ValueError(f"need dim >= 1 and order >= 0 (got {dim}, {order})")
    out = []
    for deg in range(order + 1):
        combos = [e for e in itertools.product(range(deg + 1), repeat=dim)
                  if sum(e) == deg]
        out.extend(sorted(combos, reverse=True))
    return out


def _basis(xi: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """Monomials ``prod_k xi_k^e_k`` at ``xi`` (P, dim) -> (P, n_basis)."""
    # xi[:, None, :] ** exps[None] with 0**0 = 1 (torch.pow gives 1 there)
    return torch.prod(xi[:, None, :] ** exps[None, :, :], dim=-1)


# ---------------------------------------------------------------------------
# pair search + system
# ---------------------------------------------------------------------------

def _pairs(positions: torch.Tensor, h: float, box: torch.Tensor | None,
           block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All ordered pairs (i, j) with |x_j - x_i| < h (self pair included),
    returned as ``i_idx, j_idx, dx`` with ``dx = x_j - x_i`` (minimum image
    where ``box`` is given, per dimension; ``box[d] <= 0`` marks that
    dimension open)."""
    N = positions.shape[0]
    I, J, D = [], [], []
    for lo in range(0, N, block):
        q = positions[lo:lo + block]                       # (B, dim)
        dx = positions[None, :, :] - q[:, None, :]         # (B, N, dim)
        if box is not None:
            per = box > 0
            Lsafe = torch.where(per, box, torch.ones_like(box))
            wrapped = dx - Lsafe * torch.round(dx / Lsafe)
            dx = torch.where(per, wrapped, dx)
        mask = (dx * dx).sum(-1) < h * h
        bi, jj = mask.nonzero(as_tuple=True)
        I.append(bi + lo)
        J.append(jj)
        D.append(dx[bi, jj])
    return torch.cat(I), torch.cat(J), torch.cat(D)


@dataclass
class RKPMSystem:
    """The assembled order-p moment system of one particle set.

    ``Minv`` (N, n, n) is the equilibrated pseudo-inverse already unscaled
    (i.e. ``c = Minv @ b``); ``M`` is the raw moment matrix; ``cond`` the
    2-norm condition number of the *equilibrated* matrix.
    """
    dim: int
    order: int
    h: float
    exps: torch.Tensor            # (n, dim) int64
    has_constant: bool            # False: LABFM variant (no c_0, differences)
    xi: torch.Tensor              # (P, dim) pair vectors (x_j - x_i) / h
    i_idx: torch.Tensor           # (P,)
    j_idx: torch.Tensor           # (P,)
    wP: torch.Tensor              # (P, n)  w_ij * P(xi_ij)
    M: torch.Tensor               # (N, n, n)
    Minv: torch.Tensor            # (N, n, n)
    cond: torch.Tensor            # (N,)
    num_nbrs: torch.Tensor        # (N,) neighbours incl. self
    deficient: torch.Tensor       # (N,) bool: rank-deficient / ill-conditioned

    @property
    def n_basis(self) -> int:
        return self.exps.shape[0]


def build_system(positions: torch.Tensor, volumes: torch.Tensor, h: float,
                 order: int, kernel: str = "wendland2",
                 box: torch.Tensor | None = None, equilibrate: bool = True,
                 rtol: float = 1e-12, cond_max: float = 1e10,
                 block: int = 512, constant: bool = True,
                 pair_chunk: int = 20000, select=None,
                 unit_weights: bool = False) -> RKPMSystem:
    """Assemble the order-``order`` moment system.

    ``positions`` (N, dim); ``volumes`` (N,) the quadrature weights V_j;
    ``h`` the support radius (uniform); ``kernel`` a name known to
    ``distribution.kernel_shape``; ``box`` (dim,) periodic box sides
    (``<= 0`` entry = open in that dimension) or None for an open domain.
    ``constant=False`` builds the LABFM variant (see module docstring).
    ``select(i_idx, j_idx, dx)`` -> bool mask restricts the pairs (the sector
    stencils of the Phase 6/7 reconstruction); ``unit_weights`` replaces the
    kernel weight by 1 inside the support (Avesani et al. 2014's plain
    least-squares stencil fit). The self pair is always kept.
    """
    N, dim = positions.shape
    dtype, device = positions.dtype, positions.device
    exps_list = monomial_exponents(dim, order)
    if not constant:
        exps_list = exps_list[1:]            # drop the constant (index 0)
    exps = torch.tensor(exps_list, dtype=torch.int64, device=device)
    n = exps.shape[0]

    box_t = None if box is None else torch.as_tensor(box, dtype=dtype,
                                                     device=device)
    i_idx, j_idx, dx = _pairs(positions, h, box_t, block)
    if select is not None:
        keep = select(i_idx, j_idx, dx) | (i_idx == j_idx)
        i_idx, j_idx, dx = i_idx[keep], j_idx[keep], dx[keep]
    xi = dx / h
    q = torch.linalg.norm(xi, dim=-1)
    if unit_weights:
        w = torch.ones_like(q)
    else:
        w = volumes[j_idx] * kernel_shape(q, kernel, dim)        # (P,)
    P = _basis(xi, exps.to(dtype))                               # (P, n)
    wP = w[:, None] * P

    M = torch.zeros(N, n, n, dtype=dtype, device=device)
    for lo in range(0, i_idx.shape[0], pair_chunk):      # bound the (P, n, n) temp
        sl = slice(lo, lo + pair_chunk)
        M.index_add_(0, i_idx[sl], wP[sl][:, :, None] * P[sl][:, None, :])
    num_nbrs = torch.bincount(i_idx, minlength=N)

    if equilibrate:
        d = torch.sqrt(torch.diagonal(M, dim1=1, dim2=2).abs().clamp_min(1e-300))
    else:
        d = torch.ones(N, n, dtype=dtype, device=device)
    Me = M / (d[:, :, None] * d[:, None, :])
    # Me is symmetric PSD, so its singular values are its eigenvalues. eigvalsh
    # (not svdvals): torch's batched CUDA SVD drops off its fast path above
    # 32 x 32 -- 19 s vs 0.02 s for 2200 matrices of size 44 (k = 8 in 2-D).
    S = torch.linalg.eigvalsh(Me).abs()
    cond = S.max(1).values / S.min(1).values.clamp_min(1e-300)
    Minv = torch.linalg.pinv(Me, rtol=rtol, hermitian=True) \
        / (d[:, :, None] * d[:, None, :])
    deficient = (num_nbrs < n) | (cond > cond_max) | ~torch.isfinite(cond)
    return RKPMSystem(dim=dim, order=order, h=h, exps=exps,
                      has_constant=constant, xi=xi, i_idx=i_idx,
                      j_idx=j_idx, wP=wP, M=M, Minv=Minv, cond=cond,
                      num_nbrs=num_nbrs, deficient=deficient)


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------

class RKPMOperator:
    """Operators from an `RKPMSystem`: ``values`` / ``gradient`` /
    ``hessian`` / ``laplacian`` of a scalar ``(N,)`` or vector ``(N, D)``
    field. ``hessian`` / ``laplacian`` need ``order >= 2`` and return None
    otherwise (the harness skips the row)."""

    def __init__(self, system: RKPMSystem):
        self.s = system
        self._index = {tuple(e): k for k, e in enumerate(system.exps.tolist())}

    def coefficients(self, f: torch.Tensor) -> torch.Tensor:
        """Fit coefficients ``c`` (N, n) [or (N, n, D) for vector f]."""
        s = self.s
        scalar = f.dim() == 1
        F = f[:, None] if scalar else f
        b = torch.zeros(s.M.shape[0], s.n_basis, F.shape[1],
                        dtype=F.dtype, device=F.device)
        data = F[s.j_idx] if s.has_constant else F[s.j_idx] - F[s.i_idx]
        for lo in range(0, s.i_idx.shape[0], 20000):
            sl = slice(lo, lo + 20000)
            b.index_add_(0, s.i_idx[sl], s.wP[sl][:, :, None] * data[sl][:, None, :])
        c = torch.einsum("nab,nbd->nad", s.Minv, b)
        return c[:, :, 0] if scalar else c

    def _col(self, c: torch.Tensor, exps: tuple[int, ...]) -> torch.Tensor:
        return c[:, self._index[exps]]

    def values(self, f: torch.Tensor) -> torch.Tensor:
        if not self.s.has_constant:
            return f.clone()      # LABFM: f_i is reproduced identically
        return self._col(self.coefficients(f), (0,) * self.s.dim)

    def gradient(self, f: torch.Tensor) -> torch.Tensor:
        """(N, dim) for scalar f; (N, D, dim) for vector f (matches the
        harness's ``run_probe`` layout)."""
        s = self.s
        c = self.coefficients(f)
        cols = []
        for k in range(s.dim):
            e = tuple(1 if a == k else 0 for a in range(s.dim))
            cols.append(self._col(c, e) / s.h)
        return torch.stack(cols, dim=-1)

    def hessian(self, f: torch.Tensor) -> torch.Tensor | None:
        """(N, dim, dim) for scalar f, out[:, a, b] = d_a d_b f."""
        s = self.s
        if s.order < 2 or f.dim() != 1:
            return None
        c = self.coefficients(f)
        H = torch.zeros(f.shape[0], s.dim, s.dim, dtype=f.dtype, device=f.device)
        for a in range(s.dim):
            for b in range(s.dim):
                e = [0] * s.dim
                e[a] += 1
                e[b] += 1
                fact = math.prod(math.factorial(x) for x in e)
                H[:, a, b] = fact * self._col(c, tuple(e)) / s.h ** 2
        return H

    def interface_states(self, f: torch.Tensor):
        """Left / right states of a scalar field at every pair midpoint --
        the MLS reconstruction a Riemann-SPH scheme feeds its solver
        (Phase 6/7). For each ordered pair (i, j), i != j, in the support:
        ``f_L`` = particle i's local fit evaluated at the midpoint
        ``(x_i + x_j)/2`` and ``f_R`` = particle j's fit at the same point.
        Returns ``i_idx, j_idx, f_L, f_R``. In a smooth region both equal
        ``f(x_mid)`` to O(h^(p+1)) and the jump ``f_L - f_R`` vanishes at
        that order (the "smooth-region convergence" Phase 6 asks for)."""
        s = self.s
        c = self.coefficients(f)                       # (N, n)
        pair = s.i_idx != s.j_idx
        i, j = s.i_idx[pair], s.j_idx[pair]
        xi = s.xi[pair]
        e = s.exps.to(xi.dtype)

        def basis(x):                                   # (P', n)
            return torch.prod(x[:, None, :] ** e[None], dim=-1)

        f_L = (c[i] * basis(0.5 * xi)).sum(-1)
        f_R = (c[j] * basis(-0.5 * xi)).sum(-1)
        if not s.has_constant:                          # LABFM: fit of f - f_i
            f_L = f_L + f[i]
            f_R = f_R + f[j]
        return i, j, f_L, f_R

    def laplacian(self, f: torch.Tensor) -> torch.Tensor | None:
        H = self.hessian(f)
        return None if H is None else torch.diagonal(H, dim1=1, dim2=2).sum(-1)
