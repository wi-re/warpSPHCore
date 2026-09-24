"""CI gate for the Pass-2 PDE benchmark (higherOrderSPH/harness/pde/).

Two layers, mirroring tests/convergence/test_harness.py:

1. Direct-import unit tests for the pure modules (conservation, field_error,
   pde_cases analytic fields) -- CPU-only, no warp, fast.
2. A session-scoped subprocess run of `run_pde.py --smoke --cases linearWave`
   in float64 on a fast 1D analytic case. The process must exit 0 and write
   the report + CSV.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PDE_DIR = REPO_ROOT / "higherOrderSPH" / "harness" / "pde"
DRIVER = PDE_DIR / "run_pde.py"
REPORT = PDE_DIR / "REPORT_pde.md"
ROWS_CSV = PDE_DIR / "results" / "pde_rows.csv"

sys.path.insert(0, str(PDE_DIR))

from conservation import ConservedQuantities, conserved, drift  # noqa: E402
from field_error import grid_l2_error                            # noqa: E402
from pde_cases import (CASES, linearWave_analytic,                # noqa: E402
                       tgv_analytic_velocity)


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------

def test_conserved_2d_known_state():
    # Two unit-mass particles; check mass, momentum, KE, angular momentum.
    m = torch.tensor([1.0, 2.0])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    c = conserved(m, v, x)
    assert c.mass == pytest.approx(3.0)
    # momentum = sum m v = [1, 2]
    assert c.momentum_norm == pytest.approx(np.hypot(1.0, 2.0))
    # KE = 0.5*(1*1 + 2*1) = 1.5
    assert c.kinetic_energy == pytest.approx(1.5)
    # Lz = sum m (x vy - y vx) = 2*(1*1 - 0*0) = 2
    assert c.angular_momentum_norm == pytest.approx(2.0)


def test_conserved_1d_has_no_angular_momentum():
    m = torch.tensor([1.0, 1.0])
    v = torch.tensor([[1.0], [-1.0]])
    x = torch.tensor([[0.0], [1.0]])
    c = conserved(m, v, x)
    assert c.momentum_norm == pytest.approx(0.0)   # symmetric velocities
    assert c.angular_momentum_norm == pytest.approx(0.0)


def test_drift_relative_and_absolute():
    def cq(mass, ke, mom_norm, lm_norm, ie=0.0):
        return ConservedQuantities(
            mass=mass, momentum=torch.tensor([mom_norm, 0.0]),
            momentum_norm=mom_norm, kinetic_energy=ke,
            angular_momentum=torch.tensor([lm_norm]),
            angular_momentum_norm=lm_norm, internal_energy=ie)
    # incompressible: IE = 0, so total energy == KE
    init = cq(3.0, 1.5, 0.0, 0.0)
    final = cq(3.0, 1.4, 0.2, 0.1)
    d = drift(init, final)
    assert d["mass_drift"] == pytest.approx(0.0)
    assert d["ke_drift"] == pytest.approx((1.4 - 1.5) / 1.5)
    assert d["total_energy_drift"] == pytest.approx((1.4 - 1.5) / 1.5)
    # ~0-initial quantities are reported as absolute final norms
    assert d["momentum_norm_final"] == pytest.approx(0.2)
    assert d["angmom_norm_final"] == pytest.approx(0.1)
    assert d["momentum_norm_init"] == pytest.approx(0.0)


def test_drift_total_energy_compressible():
    # compressible: KE and IE exchange, but total energy (KE + IE) is the
    # conserved quantity -- its drift is ~0 even though KE drifts a lot.
    def cq(ke, ie):
        return ConservedQuantities(
            mass=1.0, momentum=torch.tensor([0.0, 0.0]), momentum_norm=0.0,
            kinetic_energy=ke, angular_momentum=torch.tensor([0.0]),
            angular_momentum_norm=0.0, internal_energy=ie)
    # KE rises 1->2, IE falls 2->1: total stays 3
    d = drift(cq(1.0, 2.0), cq(2.0, 1.0))
    assert d["ke_drift"] == pytest.approx(1.0)            # KE doubled
    assert d["total_energy_drift"] == pytest.approx(0.0)  # total conserved


# ---------------------------------------------------------------------------
# field_error (grid projection)
# ---------------------------------------------------------------------------

def _uniform_1d(n):
    p = (torch.linspace(0, 1, n) - 0.5)[:, None]      # (n,1) in [-0.5,0.5]
    m = torch.full((n,), 1.0 / n)                      # total mass 1
    return m, p


def test_grid_l2_error_identical_is_zero():
    m, p = _uniform_1d(64)
    d = torch.ones(64)
    assert grid_l2_error(d, m, p, d, m, p, L=1.0, dim=1, periodic=True,
                         n_grid=128) == 0.0


def test_grid_l2_error_scalar_density_field():
    # A scalar density field (as for the Sod/Sedov reference metric) projects
    # via the same mass-weighted CIC average as the vector fields; identical
    # sets -> zero error.
    torch.manual_seed(0)
    p = (torch.rand(128) - 0.5)[:, None]
    m = torch.full((128,), 1.0 / 128)
    rho = 1.0 + 0.5 * torch.sin(2 * np.pi * p[:, 0])
    assert grid_l2_error(rho, m, p, rho, m, p, L=1.0, dim=1,
                         periodic=True, n_grid=256) < 1e-12


def test_grid_l2_error_smooth_field_small_error():
    # A smooth field sampled at two resolutions; the CIC projections onto a
    # common fine grid agree to within the coarser sampling's discretisation
    # error (small, and it shrinks as the coarse resolution increases).
    def sample(n):
        p = (torch.linspace(0, 1, n + 1)[:-1] - 0.5)[:, None]
        m = torch.full((n,), 1.0 / n)
        return m, p, 1.0 + 0.1 * torch.sin(2 * np.pi * p[:, 0])
    mf, pf, ff = sample(512)
    for n_coarse, tol in ((64, 5e-3), (256, 5e-4)):
        mc, pc, fc = sample(n_coarse)
        err = grid_l2_error(fc, mc, pc, ff, mf, pf, L=1.0, dim=1,
                            periodic=True, n_grid=512)
        assert err < tol, f"n_coarse={n_coarse}: err={err}"


def test_grid_l2_error_detects_difference():
    m, p = _uniform_1d(64)
    d_lo = torch.ones(64) * 1.0
    d_hi = torch.ones(64) * 2.0          # twice the density everywhere
    err = grid_l2_error(d_lo, m, p, d_hi, m, p, L=1.0, dim=1, periodic=True,
                        n_grid=128)
    # scalar (non-density) mass-weighted: cell value = field itself -> |1-2|=1
    assert err == pytest.approx(1.0, abs=1e-9)


def test_grid_l2_error_vector_field():
    n = 64
    m, p = _uniform_1d(n)
    v1 = torch.stack([torch.zeros(n), torch.ones(n)], dim=1)   # (n,2)
    v2 = torch.stack([torch.ones(n), torch.zeros(n)], dim=1)
    err = grid_l2_error(v1, m, p, v2, m, p, L=1.0, dim=1, periodic=True,
                        n_grid=128)
    # per component the difference is 1; RMS over 2 components = 1
    assert err == pytest.approx(1.0, abs=1e-9)


def test_grid_l2_error_periodic_wrap():
    # A particle just below -L/2 must wrap to the +L/2 edge (periodic).
    n = 32
    m, p = _uniform_1d(n)
    p_wrapped = p.clone()
    p_wrapped[0, 0] = p_wrapped[0, 0] - 1.0        # shift one particle a full L
    d = torch.ones(n)
    # identical field, same mass distribution (the wrap is a no-op on the
    # periodic grid) -> zero error
    assert grid_l2_error(d, m, p, d, m, p_wrapped, L=1.0, dim=1,
                         periodic=True, n_grid=64) < 1e-9


# ---------------------------------------------------------------------------
# pde_cases analytic fields
# ---------------------------------------------------------------------------

def test_tgv_analytic_t0_matches_ic():
    # At t=0 the decay is 1; check against the case IC form directly.
    params = dict(k=2, nu=0.01, uMag=1.0)
    x = torch.tensor([[0.0, 0.0], [0.3, -0.7]], dtype=torch.float64)
    v = tgv_analytic_velocity(x, 0.0, params)
    kTgv = 1.0
    phase = np.pi / 2.0
    u = torch.cos(kTgv * x[:, 0] + phase) * torch.sin(kTgv * x[:, 1] + phase)
    w = -torch.sin(kTgv * x[:, 0] + phase) * torch.cos(kTgv * x[:, 1] + phase)
    assert torch.allclose(v[:, 0], u, atol=1e-12)
    assert torch.allclose(v[:, 1], w, atol=1e-12)


def test_tgv_analytic_decays():
    params = dict(k=2, nu=0.01, uMag=1.0)
    x = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    v0 = tgv_analytic_velocity(x, 0.0, params)
    v1 = tgv_analytic_velocity(x, 1.0, params)
    # decay factor exp(-2 nu kTgv^2 t) with kTgv=1, t=1
    expect = np.exp(-2 * 0.01 * 1.0 * 1.0)
    assert v1.norm().item() == pytest.approx(v0.norm().item() * expect,
                                             rel=1e-9)


def test_linearWave_analytic_t0():
    params = dict(A=1e-2, lamda=1.0, c_s=1.0, rho0=1.0)
    x = torch.linspace(-0.5, 0.5, 17, dtype=torch.float64)[:, None]
    v = linearWave_analytic(x, 0.0, params)
    expect = 1e-2 * torch.sin(2 * np.pi * x[:, 0])
    assert torch.allclose(v[:, 0], expect, atol=1e-12)


def test_linearWave_analytic_travels_right():
    params = dict(A=1e-2, lamda=1.0, c_s=1.0, rho0=1.0)
    x = torch.tensor([[0.0]], dtype=torch.float64)
    # right-travelling: v(x, t) = v_initial(x - c_s t). So the value at
    # x=0, t=0.25 equals the t=0 value at x = -c_s t = -0.25.
    v_t = linearWave_analytic(x, 0.25, params)
    v_ref = linearWave_analytic(torch.tensor([[-0.25]], dtype=torch.float64),
                                0.0, params)
    assert v_t[0, 0].item() == pytest.approx(v_ref[0, 0].item(), abs=1e-12)


def test_cases_registry_complete():
    expected = {"tgv", "tgv-wc", "linearWave", "gresho",
                "kelvinHelmholtz", "sod", "sedov"}
    assert set(CASES) == expected
    for name, e in CASES.items():
        assert len(e.nx_ladder) == 4, name
        assert e.metric in ("analytic", "reference"), name
        if e.metric == "analytic":
            assert e.analytic is not None, name


# ---------------------------------------------------------------------------
# smoke (subprocess, float64, fast 1D analytic case)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_run():
    if not torch.cuda.is_available():
        pytest.skip("PDE smoke requires a CUDA device")
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--smoke", "--cases", "linearWave"],
        env=env, cwd=str(PDE_DIR),
        capture_output=True, text=True, timeout=900,
    )
    return proc


def test_pde_smoke_exits_zero(smoke_run):
    proc = smoke_run
    assert proc.returncode == 0, (
        f"smoke exited {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


def test_pde_smoke_writes_outputs(smoke_run):
    if smoke_run.returncode != 0:
        pytest.skip("smoke did not succeed")
    assert REPORT.exists() and REPORT.stat().st_size > 0
    assert ROWS_CSV.exists()
    with ROWS_CSV.open() as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    # header + exactly one row (smoke = single lowest resolution)
    assert len(lines) == 2, f"expected 2 CSV lines, got {len(lines)}"
    assert "linearWave" in lines[1]
