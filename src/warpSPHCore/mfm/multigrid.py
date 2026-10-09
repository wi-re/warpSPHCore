"""Cell-aggregation multigrid for the MFM face-closure solve.

The closure correction solves a graph-Laplacian problem
``L lam = a``, ``L = sum_pairs kappa_ij (e_i - e_j)(e_i - e_j)^T``, on the particle graph.
Plain (Jacobi-preconditioned) conjugate gradients need ``O(sqrt N)`` iterations -- the
high-frequency part of the residual dies in ~20, the long-wavelength tail does not -- which
at ``N = 10^6`` costs 60x the flux pass of the step. This module builds the missing coarse
levels, matrix-free on the particles and as dense stencils above them:

* **Level 1**: aggregate the particles by a regular grid of cells no smaller than the largest
  support, so every pair couples a cell to itself or to one of its ``3^dim - 1`` neighbours.
  The Galerkin coarse operator ``P^T L P`` of this piecewise-constant aggregation keeps only
  the cross-cell pairs, ``W[I, delta] = sum kappa_ij`` over the pairs from cell ``I`` into
  ``I + delta``; the per-particle shares come out of one warp kernel
  (``mfmCoarseStencil``) and are summed per cell here.
* **Levels 2...**: blocks of ``2^dim`` cells, again Galerkin (a stencil of the same width), until
  at most ``maxCoarse`` cells remain; the coarsest problem is a dense Cholesky solve.
* A V-cycle (damped Jacobi on the particles with the warp matvec, damped Jacobi on the
  stencils) is the preconditioner of the outer CG; restriction is the transpose of the
  prolongation, so the preconditioner is symmetric.

The grid maps (cell ids, neighbour tables, aggregation index maps) depend only on the cell
geometry, not on the face weights, and are rebuilt only when the particles move to another
cell count; the weights are re-assembled each solve.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from typing import Callable, List, Optional

import torch

__all__ = ["CellMultigrid"]


def _slots(dim: int):
    return list(itertools.product((-1, 0, 1), repeat=dim))


class _Level:
    pass


@contextlib.contextmanager
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def _sumInto(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``out.index_add_(0, index, src)`` with a fixed summation order (CUDA ``index_add_`` uses float atomics,
    whose order -- and so the last bits of the closure potential -- changes from run to run)."""
    with _deterministic():
        return out.index_add_(0, index, src)


class CellMultigrid:
    """Multigrid hierarchy for one particle configuration. Build with :meth:`__init__`, set the face
    weights with :meth:`setWeights`, apply with :meth:`vcycle`."""

    def __init__(self, positions: torch.Tensor, hmax: float, domain, dim: int, maxCoarse: int = 512,
                 smooth: int = 1, omega: float = 0.8):
        self.dim, self.omega, self.smooth = dim, omega, smooth
        dev, dt = positions.device, positions.dtype
        lo = domain.min.to(dev, dt)
        hi = domain.max.to(dev, dt)
        per = domain.periodic.to(dev)
        # non-periodic dimensions: the particle bounding box (the domain box may be vastly larger)
        plo = torch.where(per, lo, positions.amin(0))
        phi = torch.where(per, hi, positions.amax(0) + 1e-9 * (hi - lo).abs().clamp_min(1.0))
        ext = (phi - plo)
        n = torch.clamp((ext / hmax).floor().to(torch.int64), min=1)
        # divisible by 2^k so that block coarsening stays aligned (and periodic)
        k = 0
        while all(int(x) >= 2 ** (k + 2) for x in n) and math.prod(int(x) // 2 ** k for x in n) > maxCoarse:
            k += 1
        n = torch.tensor([max(1, (int(x) // 2 ** k) * 2 ** k) for x in n], dtype=torch.int64, device=dev)
        self.ncells = [int(x) for x in n] + [1] * (3 - dim)
        self.periodic = [int(bool(p)) for p in per.tolist()] + [0] * (3 - dim)
        cs = ext / n
        c = torch.clamp(((positions - plo) / cs).floor().to(torch.int64), min=0)
        c = torch.minimum(c, n - 1)
        cid = torch.zeros(positions.shape[0], dtype=torch.int64, device=dev)
        for a in range(dim):
            cid = cid * n[a] + c[:, a]
        self.cellId = cid.to(torch.int32)
        self.cellId64 = cid
        self.cellSize = cs
        self.n0 = [int(x) for x in n]
        self._buildLevels(dev, maxCoarse)
        self.W = None

    # ------------------------------------------------------------------
    def _neighbourTable(self, shape, periodic, dev):
        dim = self.dim
        coords = torch.stack(torch.meshgrid(*[torch.arange(s, device=dev) for s in shape], indexing="ij"), -1).reshape(-1, dim)
        nb = []
        for delta in _slots(dim):
            q = coords + torch.tensor(delta, device=dev)
            valid = torch.ones(coords.shape[0], dtype=torch.bool, device=dev)
            for a in range(dim):
                if periodic[a]:
                    q[:, a] = q[:, a] % shape[a]
                else:
                    valid &= (q[:, a] >= 0) & (q[:, a] < shape[a])
                    q[:, a] = q[:, a].clamp(0, shape[a] - 1)
            idx = torch.zeros(coords.shape[0], dtype=torch.int64, device=dev)
            for a in range(dim):
                idx = idx * shape[a] + q[:, a]
            # an out-of-range neighbour (non-periodic edge) points at the cell itself; its weight is zero
            idx = torch.where(valid, idx, torch.arange(coords.shape[0], device=dev))
            nb.append(idx)
        return torch.stack(nb, 1), coords

    def _buildLevels(self, dev, maxCoarse):
        dim = self.dim
        shape = self.n0[:]
        levels: List[_Level] = []
        while True:
            L = _Level()
            L.shape = shape
            L.size = math.prod(shape)
            L.nb, coords = self._neighbourTable(shape, self.periodic[:dim], dev)
            levels.append(L)
            if L.size <= maxCoarse or any(s % 2 for s in shape) or any(s < 2 for s in shape):
                break
            cshape = [s // 2 for s in shape]
            # fine cell -> coarse cell, and where each (cell, delta) entry lands in the coarse stencil
            ccoords = coords // 2
            cidx = torch.zeros(L.size, dtype=torch.int64, device=dev)
            for a in range(dim):
                cidx = cidx * cshape[a] + ccoords[:, a]
            L.toCoarse = cidx
            slotsIdx, target, keep = [], [], []
            for sidx, delta in enumerate(_slots(dim)):
                nbc = L.nb[:, sidx]
                ncoords = torch.stack([(nbc // math.prod(shape[a + 1:])) % shape[a] for a in range(dim)], 1)
                jc = ncoords // 2
                dd = jc - ccoords
                for a in range(dim):
                    if self.periodic[a]:
                        dd[:, a] = torch.where(dd[:, a] > 1, dd[:, a] - cshape[a], dd[:, a])
                        dd[:, a] = torch.where(dd[:, a] < -1, dd[:, a] + cshape[a], dd[:, a])
                inRange = (dd.abs() <= 1).all(1)
                same = (dd == 0).all(1)
                slot = torch.zeros(L.size, dtype=torch.int64, device=dev)
                for a in range(dim):
                    slot = slot * 3 + (dd[:, a] + 1).clamp(0, 2)
                keep.append(inRange & ~same)
                target.append(cidx * (3 ** dim) + slot)
            L.keep = torch.stack(keep, 1)                       # (size, 3^dim)
            L.target = torch.stack(target, 1)
            shape = cshape
        self.levels = levels
        # dense coarsest operator support
        self.nLevels = len(levels)

    # ------------------------------------------------------------------
    def setWeights(self, S: torch.Tensor):
        """Assemble the coarse operators from the per-particle stencil sums ``S`` ``(N, 3^dim)``
        (:func:`mfmCoarseStencilWarp`)."""
        dim = self.dim
        W0 = torch.zeros(self.levels[0].size, 3 ** dim, dtype=S.dtype, device=S.device)
        _sumInto(W0, self.cellId64, S)
        Ws = [W0]
        for lev in self.levels[:-1]:
            W = Ws[-1]
            flat = torch.zeros(math.prod([s // 2 for s in lev.shape]) * 3 ** dim, dtype=W.dtype, device=W.device)
            _sumInto(flat, lev.target[lev.keep], W[lev.keep])
            Ws.append(flat.view(-1, 3 ** dim))
        self.W = Ws
        self.diag = [w.sum(1) for w in Ws]
        # dense coarsest operator (Cholesky; the constant mode is regularised)
        Lc = self.levels[-1]
        W, n = Ws[-1], Lc.size
        M = torch.zeros(n, n, dtype=W.dtype, device=W.device)
        rows = torch.arange(n, device=W.device)[:, None].expand(-1, 3 ** dim)
        with _deterministic():
            M.index_put_((rows.reshape(-1), Lc.nb.reshape(-1)), -W.reshape(-1), accumulate=True)
        M += torch.diag(self.diag[-1])
        reg = max(1e-10, 30.0 * torch.finfo(W.dtype).eps) * float(self.diag[-1].abs().max().clamp_min(1e-30))
        M += reg * torch.eye(n, dtype=W.dtype, device=W.device)
        self.chol, info = torch.linalg.cholesky_ex(M)
        self.cholOk = bool((info == 0).all())

    def _stencilApply(self, level: int, x: torch.Tensor) -> torch.Tensor:
        """``L_c x = diag x - sum_delta W_delta x[c + delta]`` on a dense level; ``x`` ``(cells, C)``."""
        lev, W = self.levels[level], self.W[level]
        xn = x[lev.nb]                                       # (cells, 3^dim, C)
        return self.diag[level][:, None] * x - (W[:, :, None] * xn).sum(1)

    def _cycle(self, level: int, b: torch.Tensor) -> torch.Tensor:
        lev = self.levels[level]
        if level == self.nLevels - 1:
            b0 = b - b.mean(0, keepdim=True)
            if self.cholOk:
                return torch.cholesky_solve(b0, self.chol)
            return b0 / self.diag[level].clamp_min(1e-30)[:, None]
        dinv = 1.0 / self.diag[level].clamp_min(1e-30)[:, None]
        x = self.omega * dinv * b
        for _ in range(self.smooth - 1):
            x = x + self.omega * dinv * (b - self._stencilApply(level, x))
        r = b - self._stencilApply(level, x)
        rc = torch.zeros(self.levels[level + 1].size, b.shape[1], dtype=b.dtype, device=b.device)
        _sumInto(rc, lev.toCoarse, r)
        x = x + self._cycle(level + 1, rc)[lev.toCoarse]
        for _ in range(self.smooth):
            x = x + self.omega * dinv * (b - self._stencilApply(level, x))
        return x

    def vcycle(self, r: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], diag: torch.Tensor) -> torch.Tensor:
        """One symmetric V-cycle for ``L x = r`` on the particles (zero initial guess); ``matvec`` is the fine
        operator (warp kernel), ``diag`` its diagonal."""
        dinv = (self.omega / diag.clamp_min(1e-30))[:, None]
        x = dinv * r
        for _ in range(self.smooth - 1):
            x = x + dinv * (r - matvec(x))
        res = r - matvec(x)
        rc = torch.zeros(self.levels[0].size, r.shape[1], dtype=r.dtype, device=r.device)
        _sumInto(rc, self.cellId64, res)
        x = x + self._cycle(0, rc)[self.cellId64]
        for _ in range(self.smooth):
            x = x + dinv * (r - matvec(x))
        return x
