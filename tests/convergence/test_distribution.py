"""CPU unit tests for the particle-distribution anisotropy measures
(higherOrderSPH/harness/distribution.py).

The measures are properties of the distribution alone, so the tests use
analytic particle sets with known symmetry: a perfect hexagonal lattice is
exactly isotropic (zero anisotropy), a 1D chain is maximally anisotropic
(second anisotropy 1), and a truncated cloud has a first moment pointing
toward the missing mass.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "higherOrderSPH"))

from harness.distribution import (  # noqa: E402
    anisotropy_summary,
    distribution_moments,
    kernel_shape,
)


# ---------------------------------------------------------------------------
# kernel shapes
# ---------------------------------------------------------------------------

def test_kernel_shape_support_and_positivity():
    for kernel in ("wendland2", "wendland4", "wendland6"):
        for dim in (1, 2, 3):
            q = torch.linspace(0.0, 1.0, 21)
            w = kernel_shape(q, kernel, dim)
            assert (w >= 0).all(), (kernel, dim)
            assert w[0].item() == pytest.approx(1.0, abs=1e-12)
            assert w[-1].item() == pytest.approx(0.0, abs=1e-12)
            # compact support: zero beyond q = 1
            assert (kernel_shape(torch.tensor([1.5]), kernel, dim) == 0.0).all()


def test_kernel_shape_matches_core_polynomials_2d():
    # The unnormalised 2D shapes in warpSPHCore's kernelFunctions.
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999], dtype=torch.float64)
    w2 = (1 - q) ** 4 * (1 + 4 * q)
    w4 = (1 - q) ** 6 * (1 + 6 * q + (35.0 / 3.0) * q ** 2)
    assert torch.allclose(kernel_shape(q, "wendland2", 2), w2, atol=1e-14)
    assert torch.allclose(kernel_shape(q, "wendland4", 2), w4, atol=1e-14)


# ---------------------------------------------------------------------------
# symmetry: hexagonal lattice is exactly isotropic
# ---------------------------------------------------------------------------

def _hex_lattice(nx, ny, dx=1.0):
    """Periodic triangular (hexagonal) lattice: nearest-neighbour distance
    dx, row spacing sqrt(3)/2 dx, odd rows shifted by dx/2. Box
    (nx*dx, ny*sqrt(3)/2*dx) is a lattice period for even ny."""
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = (i + (0.5 if j % 2 else 0.0)) * dx
            y = j * math.sqrt(3.0) / 2.0 * dx
            pts.append((x, y))
    pos = torch.tensor(pts, dtype=torch.float64)
    L = torch.tensor([nx * dx, ny * math.sqrt(3.0) / 2.0 * dx],
                     dtype=torch.float64)
    return pos, L


def test_hexagonal_lattice_is_isotropic():
    # h = 3 dx completes four full 6-fold-symmetric rings around every
    # particle (the r = 3 dx ring sits at q = 1 where W = 0), so both
    # anisotropy measures vanish to round-off for every particle.
    pos, L = _hex_lattice(8, 8)
    n = pos.shape[0]
    m = torch.ones(n, dtype=torch.float64)
    mom = distribution_moments(pos, m, h=3.0, L=L, dim=2,
                               kernel="wendland4", periodic=True)
    assert mom.first_anisotropy.max().item() < 1e-9
    assert mom.second_anisotropy.max().item() < 1e-9
    # every particle sees the same (translationally invariant) neighbourhood
    assert mom.kernel_sum.max().item() - mom.kernel_sum.min().item() < 1e-9


def test_anisotropy_invariant_under_translation():
    pos, L = _hex_lattice(6, 6)
    m = torch.ones(pos.shape[0], dtype=torch.float64)
    base = distribution_moments(pos, m, h=3.0, L=L, dim=2,
                                kernel="wendland4", periodic=True)
    # shift every particle by an incommensurate offset (still periodic)
    shifted = (pos + torch.tensor([0.37, 0.71], dtype=torch.float64)) % L
    moved = distribution_moments(shifted, m, h=3.0, L=L, dim=2,
                                 kernel="wendland4", periodic=True)
    assert torch.allclose(base.first_anisotropy, moved.first_anisotropy,
                          atol=1e-9)
    assert torch.allclose(base.second_anisotropy, moved.second_anisotropy,
                          atol=1e-9)


def test_second_anisotropy_rotation_invariant():
    # The measures depend only on the relative configuration: rigidly
    # rotating the whole cloud leaves every per-particle value unchanged.
    # (Guards against dropping off-diagonal components of M.)
    g = torch.Generator().manual_seed(7)
    pos = torch.randn(60, 2, generator=g, dtype=torch.float64) * 1.5
    pos -= pos.mean(dim=0)
    m = torch.ones(pos.shape[0], dtype=torch.float64)
    base = distribution_moments(pos, m, h=3.0, L=None, dim=2, periodic=False)
    th = 0.71
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]], dtype=torch.float64)
    rotated = distribution_moments(pos @ R.T, m, h=3.0, L=None, dim=2,
                                   periodic=False)
    assert torch.allclose(base.first_anisotropy, rotated.first_anisotropy,
                          atol=1e-12)
    assert torch.allclose(base.second_anisotropy, rotated.second_anisotropy,
                          atol=1e-12)


# ---------------------------------------------------------------------------
# anisotropy: 1D chain is maximally anisotropic, truncation breaks symmetry
# ---------------------------------------------------------------------------

def test_1d_chain_is_maximally_anisotropic():
    # A chain along x in 2D space: every neighbourhood lies on the x-axis,
    # so the second-moment tensor is diag(*, 0) and the shape anisotropy is
    # exactly 1, while the first moment vanishes (the chain is symmetric).
    n = 24
    L = float(n) * 1.0
    pos = torch.zeros(n, 2, dtype=torch.float64)
    pos[:, 0] = torch.arange(n, dtype=torch.float64)
    m = torch.ones(n, dtype=torch.float64)
    mom = distribution_moments(pos, m, h=3.0, L=L, dim=2,
                               kernel="wendland4", periodic=True)
    assert mom.second_anisotropy.min().item() > 0.99
    assert mom.first_anisotropy.max().item() < 1e-9


def test_truncated_cloud_first_moment_points_toward_mass():
    # Particles on the +x side of a query at the origin: the first moment
    # must point toward the existing mass (+x), and the anisotropy must be
    # large (an open half-cloud).
    pts = [(0.0, 0.0)]
    for i in range(1, 8):
        pts.append((float(i), 0.0))
    pos = torch.tensor(pts, dtype=torch.float64)
    m = torch.ones(pos.shape[0], dtype=torch.float64)
    mom = distribution_moments(pos, m, h=4.0, L=None, dim=2,
                               kernel="wendland4", periodic=False)
    f0 = mom.first_moment[0]
    assert f0[0].item() > 0.0 and abs(f0[1].item()) < 1e-12
    assert mom.second_anisotropy[0].item() > 0.9


def test_periodic_wrap_minimum_image():
    # A particle near the x = 0 wall with neighbours on both sides: the
    # minimum-image wrap must see the left neighbour across the boundary,
    # restoring the symmetric (zero first moment) configuration.
    L = torch.tensor([10.0, 10.0], dtype=torch.float64)
    pos = torch.tensor([
        [0.1, 5.0],        # query, near the x=0 wall
        [1.1, 5.0],        # right neighbour
        [9.1, 5.0],        # left neighbour, wraps to x = -0.9
        [0.1, 6.0],        # top
        [0.1, 4.0],        # bottom
    ], dtype=torch.float64)
    m = torch.ones(5, dtype=torch.float64)
    mom = distribution_moments(pos, m, h=3.0, L=L, dim=2,
                               kernel="wendland4", periodic=True)
    assert mom.first_anisotropy[0].item() < 1e-9
    # all four neighbours are inside the support
    assert mom.kernel_sum[0].item() > 4.0 * 0.5  # W(1/3) > 0.5 for the shapes here


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_keys_and_finite():
    pos, L = _hex_lattice(6, 6)
    m = torch.ones(pos.shape[0], dtype=torch.float64)
    mom = distribution_moments(pos, m, h=3.0, L=L, dim=2,
                               kernel="wendland4", periodic=True)
    s = anisotropy_summary(mom)
    for k in ("first_anisotropy_rms", "first_anisotropy_mean",
              "first_anisotropy_max", "second_anisotropy_rms",
              "second_anisotropy_mean", "second_anisotropy_max",
              "kernel_sum_min", "kernel_sum_max"):
        assert k in s
        assert math.isfinite(s[k]), k


# ---------------------------------------------------------------------------
# saved TGV distributions (higherOrderSPH/harness/data): structure, the
# disorder physics, and self-consistency of the stored summaries
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "higherOrderSPH" / "harness" / "data"
SNAP_T = (0.0, 0.5, 1.0, 1.5, 2.0)
SUMMARY_KEYS = ("first_anisotropy_rms", "first_anisotropy_max",
                "second_anisotropy_rms", "second_anisotropy_max",
                "kernel_sum_min", "kernel_sum_max")


def _load_data(name):
    if not (DATA_DIR / name).exists():
        pytest.skip(f"test data {name} not present")
    return np.load(DATA_DIR / name)


def _a2(d, t):
    return float(d[f"t{t:g}_second_anisotropy_rms"])


@pytest.mark.parametrize("name",
                         ["tgv2d_noshift_nx128.npz", "tgv2d_fullshift_nx128.npz"])
def test_saved_tgv_distributions_structure(name):
    d = _load_data(name)
    L = float(d["L"])
    N = int(d["nx"]) ** 2
    assert int(d["dim"]) == 2
    assert d["kernel"].item() == "wendland4"
    for t in SNAP_T:
        k = f"t{t:g}"
        p, m = d[f"{k}_positions"], d[f"{k}_masses"]
        assert p.shape == (N, 2) and p.dtype == np.float64
        assert m.shape == (N,) and m.dtype == np.float64
        assert np.isfinite(p).all() and np.isfinite(m).all() and (m > 0).all()
        # positions are raw simulation coordinates (the code does not
        # re-wrap them into the box); the net drift stays within one box
        # length of the box in these snapshots (it grows ~linearly in t)
        assert (p > -0.5 * L).all() and (p < 1.5 * L).all()
        for s in SUMMARY_KEYS:
            assert math.isfinite(float(d[f"{k}_{s}"])), s
        assert 0.0 <= float(d[f"{k}_second_anisotropy_max"]) <= 1.0
        assert 0.0 < float(d[f"{k}_kernel_sum_min"]) \
            < float(d[f"{k}_kernel_sum_max"])


def test_saved_tgv_disorder_physics():
    a = _load_data("tgv2d_noshift_nx128.npz")
    c = _load_data("tgv2d_fullshift_nx128.npz")
    # (1) the uncorrected flow accumulates disorder over the run
    assert _a2(a, 2.0) > 1.5 * _a2(a, 0.0)
    # (2) full shifting re-regularises: its (deliberately disordered) start
    # collapses by >>10x within 500 steps and keeps drifting down
    assert _a2(c, 0.5) < 0.1 * _a2(c, 0.0)
    assert _a2(c, 0.5) > _a2(c, 1.0) > _a2(c, 1.5) > _a2(c, 2.0)
    # (3) at t=2 the uncorrected distribution is markedly more anisotropic
    # than the shifted one (PST keeps the neighbourhood near-isotropic)
    assert _a2(a, 2.0) > 1.5 * _a2(c, 2.0)
    # (4) the two files start from DIFFERENT t=0 states: the case's
    # shuffleParticles relaxation honours the scheme's shift strength (and
    # scales each iteration by 10), so the full-shift IC is far more
    # disordered -- a trap for anyone comparing the legs at t=0.
    assert _a2(c, 0.0) > 10 * _a2(a, 0.0)


def test_saved_tgv_distributions_self_consistent():
    # The stored anisotropy summaries must be reproducible from the stored
    # positions/masses by the current module (guards the test-data contract
    # against silent module changes). One full snapshot recompute (~10 s).
    d = _load_data("tgv2d_noshift_nx128.npz")
    pos = torch.tensor(d["t0_positions"], dtype=torch.float64)
    m = torch.tensor(d["t0_masses"], dtype=torch.float64)
    L = torch.tensor([float(d["L"])] * 2, dtype=torch.float64)
    mom = distribution_moments(pos, m, h=float(d["t0_h"]), L=L, dim=2,
                               kernel="wendland4", periodic=True)
    s = anisotropy_summary(mom)
    for key in SUMMARY_KEYS:
        assert s[key] == pytest.approx(float(d[f"t0_{key}"]), rel=1e-12)
