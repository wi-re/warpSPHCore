"""Tests for warpSPHCore.sampling.sampleDensestLattice (1D uniform,
2D hexagonal, 3D FCC densest packings for periodic domains)."""

import math

import numpy as np
import pytest

from warpSPHCore.sampling import LatticeSample, sampleDensestLattice


def _pairwiseMinDist(positions: np.ndarray) -> float:
    """Smallest distance between distinct points (small N only)."""
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _sortedKey(positions: np.ndarray, tol: float = 1e-12) -> tuple:
    return tuple(sorted(map(tuple, np.round(positions / tol) * tol)))


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

def test_1d_uniform():
    s = sampleDensestLattice(100, 2.0, 1)
    assert s.count == 100
    assert s.spacing == pytest.approx(0.02, rel=1e-14)
    assert s.exact
    assert np.allclose(s.box, [2.0])
    assert np.allclose(s.positions.ravel(), (np.arange(100) + 0.5) * 0.02)
    assert np.all(s.positions >= 0.0) and np.all(s.positions < 2.0)


def test_1d_box():
    s = sampleDensestLattice(10, 1.0, 1, box=[4.0])
    assert s.count == 10
    assert s.spacing == pytest.approx(0.4, rel=1e-14)
    assert np.allclose(s.box, [4.0])
    assert s.exact


# ---------------------------------------------------------------------------
# 3D (FCC)
# ---------------------------------------------------------------------------

def test_3d_counts_and_spacing():
    s = sampleDensestLattice(32, 1.0, 3)
    assert s.count == 32                      # p = 2 -> 4 p^3
    assert s.spacing == pytest.approx(0.5 / math.sqrt(2.0), rel=1e-14)
    assert s.exact
    assert np.allclose(s.box, [1.0, 1.0, 1.0])
    assert np.all(s.positions >= 0.0) and np.all(s.positions < 1.0)


def test_3d_snaps_to_4p_cubed():
    s = sampleDensestLattice(4096, 1.0, 3)
    assert s.count == 4 * 10**3               # p = round((4096/4)^(1/3)) = 10
    assert s.spacing == pytest.approx(0.1 / math.sqrt(2.0), rel=1e-14)


def test_3d_nearest_neighbour():
    s = sampleDensestLattice(32, 1.0, 3)
    d = _pairwiseMinDist(s.positions)
    assert d == pytest.approx(s.spacing, rel=1e-10)


def test_3d_translation_invariance():
    # A periodic lattice is invariant under a shift by one cell: the
    # shifted point set (mod L) must equal the original one.
    s = sampleDensestLattice(32, 1.0, 3)
    b = s.box[0] / s.cells[0]
    shifted = (s.positions + np.array([b, 0.0, 0.0])) % s.box[0]
    assert _sortedKey(shifted) == _sortedKey(s.positions)


def test_3d_volume_per_particle():
    s = sampleDensestLattice(32, 1.0, 3)
    # Each particle owns one conventional cell (b^3) divided by 4.
    b = s.box[0] / s.cells[0]
    assert np.prod(s.box) / s.count == pytest.approx(b**3 / 4.0, rel=1e-14)


def test_3d_noncubic_box_reported():
    s = sampleDensestLattice(32, 1.0, 3, box=[1.0, 2.0, 2.0])
    assert not s.exact
    assert np.allclose(s.box, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 2D (hexagonal)
# ---------------------------------------------------------------------------

def test_2d_counts_spacing_box():
    s = sampleDensestLattice(16, 1.0, 2)
    assert s.count == 16                      # sx = sy = 2 -> 4 sx sy
    assert s.spacing == pytest.approx(0.25, rel=1e-14)
    assert s.exact                            # no box requested beyond Lx
    assert s.box[0] == pytest.approx(1.0, rel=1e-14)
    # Hexagonal commensurability: Ly = sqrt(3) a sy
    assert s.box[1] == pytest.approx(math.sqrt(3.0) * 0.25 * 2.0, rel=1e-14)
    assert np.all(s.positions >= 0.0)
    assert np.all(s.positions < s.box[None, :] + 1e-15)


def test_2d_nearest_neighbour():
    s = sampleDensestLattice(16, 1.0, 2)
    d = _pairwiseMinDist(s.positions)
    assert d == pytest.approx(s.spacing, rel=1e-10)


def test_2d_six_neighbours_per_point():
    # Every point of a periodic hexagonal lattice has 6 neighbours at
    # distance a (some only through the periodic wrap).
    s = sampleDensestLattice(16, 1.0, 2)
    p = s.positions
    box = s.box
    for i in range(p.shape[0]):
        delta = p - p[i]
        # minimum image
        delta -= box * np.round(delta / box)
        d = np.sqrt((delta ** 2).sum(axis=1))
        n = int(np.sum(np.isclose(d, s.spacing, rtol=0.0, atol=1e-11)))
        assert n == 6, f"point {i} has {n} neighbours at distance a"


def test_2d_square_box_not_exact():
    s = sampleDensestLattice(1000, 1.0, 2, box=[1.0, 1.0])
    # x axis fitted exactly; y snapped to whole supercells
    assert s.box[0] == pytest.approx(1.0, rel=1e-14)
    assert not s.exact
    assert abs(s.box[1] - 1.0) > 1e-3
    # still a true hexagonal lattice: the achieved box has the
    # commensurate aspect 2 sx / (sqrt(3) sy)
    sx, sy = s.cells
    assert s.box[1] / s.box[0] == \
        pytest.approx(math.sqrt(3.0) * sy / (2.0 * sx), rel=1e-14)
    assert s.count % 4 == 0


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------

def test_jitter_reproducible():
    a = sampleDensestLattice(32, 1.0, 3, jitter=0.1, seed=42)
    b = sampleDensestLattice(32, 1.0, 3, jitter=0.1, seed=42)
    c = sampleDensestLattice(32, 1.0, 3, jitter=0.1, seed=43)
    assert np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


def test_jitter_bounded():
    s0 = sampleDensestLattice(32, 1.0, 3)
    s = sampleDensestLattice(32, 1.0, 3, jitter=0.1, seed=7)
    disp = np.abs(s.positions - s0.positions)
    assert disp.max() <= 0.5 * 0.1 * s0.spacing + 1e-15


def test_jitter_requires_seed():
    with pytest.raises(ValueError, match="seed"):
        sampleDensestLattice(32, 1.0, 3, jitter=0.1)


def test_no_jitter_is_exact_lattice():
    s = sampleDensestLattice(32, 1.0, 3, jitter=0.0)
    assert s.seed is None
    assert np.array_equal(s.positions, sampleDensestLattice(32, 1.0, 3).positions)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_invalid_inputs():
    with pytest.raises(ValueError):
        sampleDensestLattice(8, 1.0, 4)
    with pytest.raises(ValueError):
        sampleDensestLattice(0, 1.0, 3)
    with pytest.raises(ValueError):
        sampleDensestLattice(8, 0.0, 3)
    with pytest.raises(ValueError):
        sampleDensestLattice(8, 1.0, 3, box=[1.0, 1.0])


def test_returns_lattice_sample():
    s = sampleDensestLattice(8, 1.0, 1)
    assert isinstance(s, LatticeSample)
