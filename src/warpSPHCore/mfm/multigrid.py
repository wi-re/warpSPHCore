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
import warp as wp

from ..type_config import scalar_t

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


@wp.kernel
def _segmentSum_Kernel(perm: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64), ends: wp.array(dtype=wp.int64),  # type: ignore
                       src: wp.array2d(dtype=scalar_t), out: wp.array2d(dtype=scalar_t)):  # type: ignore
    o, c = wp.tid()
    acc = out[o, c]
    for k in range(wp.int32(starts[o]), wp.int32(ends[o])):
        acc += src[perm[k], c]
    out[o, c] = acc


class _SegmentSum:
    """``_sumInto(out, index, src)`` for a fixed ``index`` into ``nrows`` target rows: the stable sort of the index
    is done once, and each call sums every target row's sources sequentially in the original order -- the order
    (and so the bits) of torch's deterministic ``index_add_`` on CUDA, which re-sorts the index on every call (most
    of the V-cycle's cost). Verified bitwise against it for row widths >= 2 (2-, 3-component potentials, 3^dim
    stencil rows) up to hundreds of duplicates per row; torch's width-1 kernel splits long runs, so width 1 (and
    CPU) stays on ``_sumInto``. Every target row has a (possibly empty) range, so the buffers have a fixed size and
    :meth:`update` refills them in place (the closure workspace's CUDA graph reads them at fixed addresses)."""

    def __init__(self, index: torch.Tensor, nrows: int):
        self.nrows = int(nrows)
        self.cuda = index.is_cuda
        self.index = index.clone() if self.cuda else index
        if self.cuda:
            dev = index.device
            self.perm = torch.empty_like(index)
            self.starts = torch.empty(self.nrows, dtype=torch.int64, device=dev)
            self.ends = torch.empty(self.nrows, dtype=torch.int64, device=dev)
            self._rows = torch.arange(self.nrows, dtype=index.dtype, device=dev)
            self._wa = (wp.from_torch(self.perm), wp.from_torch(self.starts), wp.from_torch(self.ends))
            self._fill(index)

    def _fill(self, index):
        perm = torch.argsort(index, stable=True)
        srt = index[perm]
        self.perm.copy_(perm)
        self.starts.copy_(torch.searchsorted(srt, self._rows))
        self.ends.copy_(torch.searchsorted(srt, self._rows, right=True))

    def update(self, index: torch.Tensor):
        """Re-point at a new index of the same length and row count, in place."""
        self.index.copy_(index)
        if self.cuda:
            self._fill(index)

    def __call__(self, out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if not self.cuda or src.dim() != 2 or src.shape[1] < 2 or src.dtype != out.dtype:
            return _sumInto(out, self.index, src)
        src = src.contiguous()
        wp.launch(_segmentSum_Kernel, dim=(self.nrows, src.shape[1]),
                  inputs=[*self._wa, wp.from_torch(src)], outputs=[wp.from_torch(out)],
                  device=wp.device_from_torch(out.device), stream=wp.stream_from_torch(torch.cuda.current_stream(out.device)))
        return out


class CellMultigrid:
    """Multigrid hierarchy for one particle configuration. Build with :meth:`__init__`, set the face
    weights with :meth:`setWeights`, apply with :meth:`vcycle`."""

    #: level maps per (cell counts, periodicity, dim, maxCoarse, device); read-only once built
    _levelCache: dict = {}

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
        # one device read for the grid shape (and the periodicity): every later integer is host-side, so
        # building the hierarchy does not stall the stream several times per closure solve
        hm = hmax if isinstance(hmax, torch.Tensor) else torch.as_tensor(hmax, dtype=dt, device=dev)
        host = torch.cat([torch.clamp((ext / hm).floor(), min=1), per.to(dt)]).tolist()
        nh = [int(x) for x in host[:dim]]
        perh = [bool(x) for x in host[dim:]]
        # divisible by 2^k so that block coarsening stays aligned (and periodic)
        k = 0
        while all(x >= 2 ** (k + 2) for x in nh) and math.prod(x // 2 ** k for x in nh) > maxCoarse:
            k += 1
        nh = [max(1, (x // 2 ** k) * 2 ** k) for x in nh]
        self.ncells = nh + [1] * (3 - dim)
        self.periodic = [int(p) for p in perh] + [0] * (3 - dim)
        cs = torch.stack([ext[a] / float(nh[a]) for a in range(dim)])
        cid = torch.zeros(positions.shape[0], dtype=torch.int64, device=dev)
        for a in range(dim):
            ca = torch.clamp(((positions[:, a] - plo[a]) / cs[a]).floor().to(torch.int64), min=0, max=nh[a] - 1)
            cid = cid * nh[a] + ca
        self.cellId = cid.to(torch.int32)
        self.cellId64 = cid
        self._toCells = None                      # built below, once the level-0 cell count is known
        self.cellSize = cs
        self.n0 = list(nh)
        # the level maps depend only on the cell counts (not on the particles), so they are shared by every
        # hierarchy with the same grid -- in a time loop the grid rarely changes and the maps (~7 ms of small
        # torch ops at 4096 cells) are built once
        key = (tuple(self.n0), tuple(self.periodic), dim, int(maxCoarse), str(dev))
        cached = CellMultigrid._levelCache.get(key)
        if cached is None:
            self._buildLevels(dev, maxCoarse)
            if len(CellMultigrid._levelCache) >= 8:
                CellMultigrid._levelCache.pop(next(iter(CellMultigrid._levelCache)))
            CellMultigrid._levelCache[key] = self.levels
        else:
            self.levels = cached
            self.nLevels = len(cached)
        self._toCells = _SegmentSum(cid, self.levels[0].size)
        for l, lev in enumerate(self.levels[:-1]):     # the restrictions are fixed maps: built once, with the levels
            if getattr(lev, 'restrict', None) is None:
                lev.restrict = _SegmentSum(lev.toCoarse, self.levels[l + 1].size)
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
    def sameGrid(self, other: "CellMultigrid") -> bool:
        return (self.n0 == other.n0 and self.periodic == other.periodic and self.levels is other.levels
                and self.cellId64.shape == other.cellId64.shape)

    def assign(self, other: "CellMultigrid") -> None:
        """Take ``other``'s particle-to-cell map and operators (same grid, same particle count, weights set),
        copied into this hierarchy's tensors in place -- so a CUDA graph captured on this hierarchy replays the
        V-cycle of ``other``. Bitwise the same V-cycle."""
        self.cellId.copy_(other.cellId)
        self.cellId64.copy_(other.cellId64)
        self.cellSize = other.cellSize
        self._toCells.update(other.cellId64)
        for mine, theirs in zip(self.W, other.W):
            mine.copy_(theirs)
        for mine, theirs in zip(self.diag, other.diag):
            mine.copy_(theirs)
        self.chol.copy_(other.chol)
        self.cholOk = other.cholOk

    def setWeights(self, S: torch.Tensor):
        """Assemble the coarse operators from the per-particle stencil sums ``S`` ``(N, 3^dim)``
        (:func:`mfmCoarseStencilWarp`)."""
        dim = self.dim
        W0 = torch.zeros(self.levels[0].size, 3 ** dim, dtype=S.dtype, device=S.device)
        self._toCells(W0, S)
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
        lev.restrict(rc, r)
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
        self._toCells(rc, res)
        x = x + self._cycle(0, rc)[self.cellId64]
        for _ in range(self.smooth):
            x = x + dinv * (r - matvec(x))
        return x
