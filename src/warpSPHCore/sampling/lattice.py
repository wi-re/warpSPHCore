"""Host-side densest-packing lattice samplers for periodic domains.

Densest packings of equal point particles, returned as float64 numpy
positions in [0, box):

* 1D: uniform lattice (trivially densest), N = n exactly.
* 2D: hexagonal (triangular) lattice. The fundamental supercell is
  2a x sqrt(3)a with 4 points, so N = 4 sx sy. A periodic box is
  exactly commensurate iff Lx / Ly = 2 sx / (sqrt(3) sy) for integers
  (sx, sy) -- a *square* box is NOT exactly commensurate (sqrt(3) is
  irrational). When a box is requested the x axis is fitted exactly
  and the achieved y size (snapped to whole supercells) is reported
  in the result with `exact = False`.
* 3D: FCC (face-centered cubic). A p x p x p grid of conventional
  cubic cells of side b = L/p, each carrying the corner plus the three
  face centers, so N = 4 p^3 and the nearest-neighbour distance is
  b/sqrt(2). Exactly commensurate with a cubic box (a non-cubic box is
  fitted on its first axis and the achieved cubic size is reported).

`jitter` displaces each particle by uniform noise on
[-jitter/2, +jitter/2] in units of the nearest-neighbour distance;
a `seed` is required for jittered output (reproducibility is the
point of a seeded sampler).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ['LatticeSample', 'sampleDensestLattice']


@dataclass
class LatticeSample:
    """A sampled packing plus the achieved (possibly snapped) geometry."""
    positions: np.ndarray  # (N, dim), float64, in [0, box)
    count: int
    spacing: float         # nearest-neighbour distance
    box: np.ndarray        # (dim,) achieved periodic box size
    cells: np.ndarray      # (dim,) integer cell counts: 1D cells; 2D supercells (sx, sy); 3D conventional FCC cells (p, p, p)
    jitter: float
    seed: int | None
    exact: bool            # achieved box equals the requested box


def _applyJitter(positions: np.ndarray, jitter: float, spacing: float,
                 seed: int | None) -> np.ndarray:
    if jitter > 0.0:
        if seed is None:
            raise ValueError(
                "a seed is required for jittered output (reproducibility)")
        rng = np.random.default_rng(seed)
        positions = positions + \
            rng.uniform(-0.5, 0.5, size=positions.shape) * (jitter * spacing)
    return positions


def _checkCommon(n: int, L: float, dim: int, box) -> None:
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2 or 3 (got {dim})")
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    if L <= 0.0:
        raise ValueError(f"L must be > 0 (got {L})")
    if box is not None:
        box = np.asarray(box, dtype=float)
        if box.shape != (dim,):
            raise ValueError(f"box must have {dim} entries (got {box.shape})")
        if np.any(box <= 0.0):
            raise ValueError(f"box entries must be > 0 (got {box})")


def sampleDensestLattice(n: int, L: float, dim: int, box=None,
                         jitter: float = 0.0,
                         seed: int | None = None) -> LatticeSample:
    """Sample a densest packing of ~n particles in a periodic box.

    n      target particle count; the achieved count is snapped to the
           lattice geometry (exact in 1D; N = 4 sx sy in 2D; N = 4 p^3
           in 3D)
    L      box size: the full box in 1D/3D, the x size in 2D
    dim    1, 2 or 3
    box    optional explicit periodic box sizes; the first axis is
           fitted exactly, the rest are snapped to whole lattice cells
           (the achieved box is reported in the result)
    jitter uniform displacement in units of the nearest-neighbour
           distance (a seed is required when > 0)
    seed   RNG seed (required for jitter > 0)
    """
    _checkCommon(n, L, dim, box)
    box = None if box is None else np.asarray(box, dtype=float)

    if dim == 1:
        N = int(n)
        b = (box[0] if box is not None else L) / N
        positions = ((np.arange(N) + 0.5) * b).reshape(-1, 1)
        positions = _applyJitter(positions, jitter, b, seed)
        achieved = np.array([(box[0] if box is not None else L)])
        return LatticeSample(positions, N, b, achieved,
                             np.array([N]), jitter, seed, True)

    if dim == 2:
        Lx = box[0] if box is not None else L
        # (sx, sy): N = 4 sx sy ~= n at the hexagonal aspect
        # sy/sx = 2/sqrt(3) (the exact-commensurability ratio).
        best = None
        for sx in range(1, max(2, int(math.sqrt(n)) + 2)):
            sy = max(1, round(sx * 2.0 / math.sqrt(3.0)))
            err = abs(4 * sx * sy - n)
            if best is None or err < best[0]:
                best = (err, sx, sy)
        sx, sy = best[1], best[2]
        a = Lx / (2.0 * sx)
        Ly = math.sqrt(3.0) * a * sy
        # Supercell (i, j) origin is (2i a, sqrt(3) j a); the 4 basis
        # points fill one 2a x sqrt(3)a supercell.
        bx = np.array([0.0, 1.0, 0.5, 1.5])
        by = np.array([0.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 2.0])
        ii, jj = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
        x = (2.0 * ii[..., None] + bx[None, None, :]) * a
        y = (math.sqrt(3.0) * jj[..., None] + by[None, None, :]) * a
        positions = np.stack([x.ravel(), y.ravel()], axis=1)
        positions = _applyJitter(positions, jitter, a, seed)
        achieved = np.array([Lx, Ly])
        exact = box is None or \
            np.allclose(achieved, box, rtol=1e-12, atol=0.0)
        return LatticeSample(positions, 4 * sx * sy, a, achieved,
                             np.array([sx, sy]), jitter, seed, exact)

    # dim == 3: FCC — conventional cubic cell of side b with the corner
    # plus the three face centers (4 points per cell, N = 4 p^3,
    # nearest-neighbour distance b/sqrt(2)).
    L0 = box[0] if box is not None else L
    p = max(1, round((n / 4.0) ** (1.0 / 3.0)))
    b = L0 / p
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=float)
    ii, jj, kk = np.meshgrid(np.arange(p), np.arange(p), np.arange(p),
                             indexing='ij')
    coords = [((grid[..., None] + basis[:, d].reshape(1, 1, 1, 4)) * b).ravel()
              for d, grid in enumerate((ii, jj, kk))]
    positions = np.stack(coords, axis=1)
    positions = _applyJitter(positions, jitter, b / math.sqrt(2.0), seed)
    achieved = np.array([L0, L0, L0])
    exact = box is None or np.allclose(achieved, box, rtol=1e-12, atol=0.0)
    return LatticeSample(positions, 4 * p**3, b / math.sqrt(2.0), achieved,
                         np.array([p, p, p]), jitter, seed, exact)
