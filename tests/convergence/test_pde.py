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
# smoke writes its own artifacts so it never clobbers the full-suite results
SMOKE_REPORT = PDE_DIR / "REPORT_pde_smoke.md"
SMOKE_CSV = PDE_DIR / "results" / "pde_rows_smoke.csv"

sys.path.insert(0, str(PDE_DIR))

from conservation import (KE_REBOUND_TOL, ConservedQuantities,  # noqa: E402
                          conserved, drift, ke_rebound)
from field_error import (aligned_1d_l2_error, grid_error_p,      # noqa: E402
                         grid_l2_error, particle_error_norms)
from pde_cases import (CASES, linearWave_analytic,                # noqa: E402
                       tgv_analytic_velocity)
from report_pde import (load_rows, merge_rows,                    # noqa: E402
                        observed_orders, reference_corrected_orders)


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
    # init/final norms are reported as-is (absolute, not relative)
    assert d["momentum_norm_final"] == pytest.approx(0.2)
    assert d["angmom_norm_final"] == pytest.approx(0.1)
    assert d["momentum_norm_init"] == pytest.approx(0.0)
    # from a zero initial state the vector drift equals the final norm
    assert d["momentum_drift_abs"] == pytest.approx(0.2)
    assert d["angmom_drift_abs"] == pytest.approx(0.1)


def test_drift_vector_drift_with_nonzero_initial_momentum():
    # KH-like: the IC already carries |p| ~ 0.23 that is conserved. The final
    # norm alone looks like a big "drift"; the vector drift is the real one.
    def cq(p, lz):
        p = torch.tensor(p, dtype=torch.float64)
        return ConservedQuantities(
            mass=1.0, momentum=p, momentum_norm=float(p.norm()),
            kinetic_energy=1.0,
            angular_momentum=torch.tensor([lz], dtype=torch.float64),
            angular_momentum_norm=abs(lz))
    d = drift(cq([0.23, 0.0], 0.06), cq([0.23, 1e-3], 0.0598))
    assert d["momentum_norm_final"] == pytest.approx(np.hypot(0.23, 1e-3))
    assert d["momentum_drift_abs"] == pytest.approx(1e-3)
    assert d["angmom_drift_abs"] == pytest.approx(2e-4)


def test_ke_rebound_catches_dip_then_spin_up():
    # monotone decay: no rebound
    assert ke_rebound([1.0, 0.9, 0.8, 0.75]) == 0.0
    # the CRKSPH Gresho signature: -3.8% dip, then +7.3% net -> the final
    # drift (+0.073) understates the anti-dissipation (0.038 + 0.073)
    ke = [1.0, 0.962, 0.97, 1.0, 1.04, 1.073]
    assert ke_rebound(ke) == pytest.approx(0.111)
    assert ke_rebound(ke) > KE_REBOUND_TOL
    # sub-tolerance wiggles (acoustics) pass
    assert ke_rebound([1.0, 0.99, 0.995, 0.98, 0.983]) < KE_REBOUND_TOL
    assert np.isnan(ke_rebound([])) and np.isnan(ke_rebound([0.0, 1.0]))


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
# field_error: shock metrics (L1, aligned L2)
# ---------------------------------------------------------------------------

def test_grid_error_p_l1_l2_zero_and_constant():
    m, p = _uniform_1d(64)
    d = torch.ones(64)
    for pnorm in (1, 2):
        assert grid_error_p(d, m, p, d, m, p, L=1.0, dim=1, periodic=True,
                            n_grid=128, p=pnorm) == 0.0
    d_hi = torch.ones(64) * 2.0
    for pnorm in (1, 2):
        assert grid_error_p(d, m, p, d_hi, m, p, L=1.0, dim=1, periodic=True,
                            n_grid=128, p=pnorm) == pytest.approx(1.0, abs=1e-9)


def _full_coverage_1d(n, offset=0.3):
    """One particle per grid cell, offset so every cell gets mass from two
    particles (a full-coverage CIC projection = a smooth field on all
    cells). Positions in [-0.5, 0.5), float64."""
    pos = ((torch.arange(n, dtype=torch.float64) + offset) / n - 0.5)[:, None]
    m = torch.full((n,), 1.0 / n, dtype=torch.float64)
    return m, pos


def test_grid_error_p_l1_linear_in_shift_l2_sqrt():
    # The reason L1 is the standard shock metric: a step of height Delta
    # shifted by delta costs O(delta) in L1 but O(sqrt(delta)) in L2.
    n = 1024
    m, pos = _full_coverage_1d(n)

    def step(x0):
        return torch.where(pos[:, 0] >= x0,
                           torch.ones_like(pos[:, 0]),
                           torch.zeros_like(pos[:, 0]))
    e1 = {}
    e2 = {}
    for delta in (0.02, 0.005):
        e1[delta] = grid_error_p(step(0.1), m, pos, step(0.1 + delta), m, pos,
                                 L=1.0, dim=1, periodic=True, n_grid=n, p=1)
        e2[delta] = grid_error_p(step(0.1), m, pos, step(0.1 + delta), m, pos,
                                 L=1.0, dim=1, periodic=True, n_grid=n, p=2)
    assert e1[0.02] / e1[0.005] == pytest.approx(4.0, rel=0.15)
    assert e2[0.02] / e2[0.005] == pytest.approx(2.0, rel=0.15)


def test_aligned_1d_l2_recovers_shift_and_reduces_error():
    # Coarse field = reference field translated by a fractional number of
    # grid cells: the aligned error must collapse and the search must
    # recover the shift.
    n = 1024
    m, pos = _full_coverage_1d(n)
    dxg = 1.0 / n
    delta_true = 2.7 * dxg

    def f(x):
        return 1.0 + 0.5 * torch.sin(2.0 * np.pi * x[:, 0]) \
            + 0.3 * torch.cos(4.0 * np.pi * x[:, 0])
    fc = f(pos + delta_true)      # coarse features left of the reference
    fr = f(pos)
    e_un = grid_l2_error(fc, m, pos, fr, m, pos, L=1.0, dim=1, periodic=True,
                         n_grid=n)
    e_al, shift = aligned_1d_l2_error(fc, m, pos, fr, m, pos, L=1.0,
                                      periodic=True, n_grid=n,
                                      max_shift=0.05)
    assert shift == pytest.approx(delta_true, abs=0.2 * dxg)
    assert e_al < 0.05 * e_un


def test_aligned_1d_l2_gappy_projection():
    # Sparse particles on integer grid fractions: each fills exactly ONE
    # grid cell, so half the projection is empty (the real Sedov situation:
    # a particle spreads over 2 of the ~4-8 cells between neighbours). The
    # densification must keep the shift search well-posed.
    n_grid = 1024
    dxg = 1.0 / n_grid
    delta_true = 5.3 * dxg
    n = 512
    pos = ((torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5)[:, None]
    m = torch.full((n,), 1.0 / n, dtype=torch.float64)

    def f(x):
        return 1.0 + 0.5 * torch.sin(2.0 * np.pi * x[:, 0]) \
            + 0.3 * torch.cos(4.0 * np.pi * x[:, 0])
    fc = f(pos + delta_true)
    fr = f(pos)
    e_al, shift = aligned_1d_l2_error(fc, m, pos, fr, m, pos, L=1.0,
                                      periodic=True, n_grid=n_grid,
                                      max_shift=0.05)
    assert shift == pytest.approx(delta_true, abs=0.2 * dxg)
    assert e_al < 1e-4


def test_aligned_1d_rejects_2d():
    n = 32
    m = torch.full((n,), 1.0 / n)
    p = (torch.rand(n, 2) - 0.5)
    d = torch.ones(n)
    with pytest.raises(ValueError):
        aligned_1d_l2_error(d, m, p, d, m, p, L=1.0, periodic=True,
                            n_grid=64, max_shift=0.1)


def test_observed_orders_alt_yattr_and_blank_columns():
    rows = []
    for dx in (0.1, 0.05, 0.025, 0.0125):
        rows.append({"case": "a", "N": int(1.0 / dx), "dx": dx,
                     "error_l2": 0.1 * dx ** 3, "error_l1": 0.1 * dx ** 2})
        rows.append({"case": "b", "N": int(2.0 / dx), "dx": dx,
                     "error_l2": 0.05 * dx, "error_l1": ""})   # blank (old row)
    o2 = observed_orders(rows)
    assert o2["a"]["slope"] == pytest.approx(3.0, abs=1e-6)
    assert o2["b"]["slope"] == pytest.approx(1.0, abs=1e-6)
    o1 = observed_orders(rows, yattr="error_l1")
    assert set(o1) == {"a"}
    assert o1["a"]["slope"] == pytest.approx(2.0, abs=1e-6)


def test_load_rows_roundtrip_blank_alt_column(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text(
        "case,nx,N,dx,error_l2,error_l1\n"
        "a,32,32,0.03,1.5e-3,\n"       # error_l1 blank (pre-metric row)
        "a,64,64,0.015,3.75e-4,7.5e-4\n")
    rows = load_rows(csv)
    assert rows[0]["error_l1"] == ""
    assert rows[1]["error_l1"] == pytest.approx(7.5e-4)
    # one blank -> the case is excluded from the L1 fit, not a crash
    assert "a" not in observed_orders(rows, yattr="error_l1")


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
                "kelvinHelmholtz", "sod", "sedov",
                # matched-scheme legs (2026-09-26)
                "linearWave-std", "gresho-std", "kelvinHelmholtz-std",
                "sedov-std", "sod-crk"}
    assert set(CASES) == expected
    # the legs differ from their base case only in name / scheme / notes
    for leg, base, scheme in [("gresho-std", "gresho", "CompSPH"),
                              ("sedov-std", "sedov", "CompSPH"),
                              ("sod-crk", "sod", "CRKSPH")]:
        a, b = CASES[leg], CASES[base]
        assert a.scheme == scheme
        assert (a.module, a.nx_ladder, a.metric, a.field, a.exact,
                a.t_star) == (b.module, b.nx_ladder, b.metric, b.field,
                              b.exact, b.t_star)
    # KE-monotonicity check applies to the unforced steady / decaying flows
    assert {n for n, e in CASES.items() if e.ke_nonincreasing} == {
        "tgv", "tgv-wc", "gresho", "gresho-std"}
    # CRKSPH needs symmetric support for exact energy conservation
    assert CASES["sod-crk"].spec == {"supportMode": "KernelMeanSymmetric"}
    for name, e in CASES.items():
        assert len(e.nx_ladder) == 4, name
        assert e.metric in ("analytic", "reference"), name
        if e.metric == "analytic":
            assert e.analytic is not None, name


# ---------------------------------------------------------------------------
# exact-solution metric (Sod / Sedov) + finite-reference-corrected orders
# ---------------------------------------------------------------------------

def test_particle_error_norms_volume_weighted():
    # 3 dense particles (V=1/3 each) with error 1, 1 sparse particle (V=1)
    # with error 0: by count the L1 would be 0.75, by volume it is 0.5.
    f = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    ex = torch.zeros(4, dtype=torch.float64)
    V = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0], dtype=torch.float64)
    n = particle_error_norms(f, ex, V)
    assert n["l1"] == pytest.approx(0.5)
    assert n["l2"] == pytest.approx(0.5 ** 0.5)


def test_merge_rows_replaces_only_run_cases(tmp_path):
    import csv
    path = tmp_path / "rows.csv"
    with path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "nx", "error_l2"])
        w.writeheader()
        for case, nx, e in [("tgv", 32, 0.1), ("sod", 200, 0.5),
                            ("sedov", 200, 0.7)]:
            w.writerow({"case": case, "nx": nx, "error_l2": e})
    new = [{"case": "sod", "nx": 200, "error_l2": 0.4},
           {"case": "sod", "nx": 400, "error_l2": 0.2}]
    out = merge_rows(path, new, ["sod"], order=["tgv", "sod", "sedov"])
    assert [(r["case"], r["nx"]) for r in out] == [
        ("tgv", 32), ("sod", 200), ("sod", 400), ("sedov", 200)]
    assert out[1]["error_l2"] == pytest.approx(0.4)
    # no existing file: just the new rows
    assert merge_rows(tmp_path / "none.csv", new, ["sod"],
                      order=["sod"]) == new


def test_exact_rows_keep_reference_rung_in_fit():
    # the finest (reference) rung is excluded from reference-metric fits but
    # is a real point for the exact-solution columns
    rows = [{"case": "sod", "N": n, "dx": 1.0 / n, "error_l2": e,
             "error_l1_exact": 2.0 / n, "is_reference": ref}
            for n, e, ref in [(200, 0.3, False), (400, 0.2, False),
                              (800, 0.1, False), (1600, 0.0, True)]]
    exact = observed_orders(rows, yattr="error_l1_exact")["sod"]
    assert exact["slope"] == pytest.approx(1.0)
    corr = reference_corrected_orders(rows, yattr="error_l2")
    assert "sod" in corr and not np.isnan(corr["sod"]["slope"])


def _sod_params():
    return dict(left_pressure=1.0, left_rho=1.0, left_velocity=0.0,
                right_pressure=0.1795, right_rho=0.25, right_velocity=0.0,
                gamma=5 / 3)


def test_sod_exact_density_geometry():
    pytest.importorskip("warpSPH")
    from pde_cases import sod_exact_density
    x = torch.tensor([[-0.95], [-0.1], [0.0], [0.1], [0.95]],
                     dtype=torch.float64)
    rho = sod_exact_density(x, 0.15, _sod_params())
    # undisturbed left state at the centre, right state near the wrap
    assert rho[2] == pytest.approx(1.0)
    assert rho[0] == pytest.approx(0.25) and rho[4] == pytest.approx(0.25)
    # mirror symmetric
    assert rho[1] == pytest.approx(rho[3])
    # at t -> 0 the profile is the initial step at |x| = 0.5
    rho0 = sod_exact_density(torch.tensor([[0.49], [0.51]],
                                          dtype=torch.float64),
                             1e-6, _sod_params())
    assert rho0[0] == pytest.approx(1.0) and rho0[1] == pytest.approx(0.25)


def test_gresho_exact_velocity_profile():
    from pde_cases import gresho_exact_velocity
    x = torch.tensor([[0.1, 0.0], [0.0, 0.3], [0.45, 0.0], [0.0, 0.0]],
                     dtype=torch.float64)
    v = gresho_exact_velocity(x, 1.7, {})
    # counter-clockwise: at (0.1, 0) v = (0, 0.5); at (0, 0.3) v = (-0.5, 0)
    assert torch.allclose(v[0], torch.tensor([0.0, 0.5], dtype=torch.float64))
    assert torch.allclose(v[1], torch.tensor([-0.5, 0.0], dtype=torch.float64))
    assert torch.equal(v[2], torch.zeros(2, dtype=torch.float64))
    assert torch.isfinite(v[3]).all() and v[3].abs().max() == 0   # centre


def test_sedov_exact_density_shock_state():
    pytest.importorskip("warpSPH")
    from pde_cases import sedov_exact_density
    from warpSPH.caseUtils.compressible.sedov.sedovSolution import (
        SedovSolution)
    params = dict(gamma=5 / 3, rho0=1.0, E0=1.0)
    t = 0.5
    _, r2, _, rho2, _ = SedovSolution(nDim=1, gamma=5 / 3).shockState(t)
    x = torch.tensor([[-(r2 + 0.05)], [r2 * 0.999], [r2 + 0.05]],
                     dtype=torch.float64)
    rho = sedov_exact_density(x, t, params)
    assert rho[0] == pytest.approx(1.0) and rho[2] == pytest.approx(1.0)
    # just inside the front: the strong-shock jump (g+1)/(g-1) = 4
    assert rho[1] == pytest.approx(rho2, rel=2e-2)
    assert rho2 == pytest.approx(4.0)


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
    assert SMOKE_REPORT.exists() and SMOKE_REPORT.stat().st_size > 0
    assert SMOKE_CSV.exists()
    with SMOKE_CSV.open() as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    # header + exactly one row (smoke = single lowest resolution)
    assert len(lines) == 2, f"expected 2 CSV lines, got {len(lines)}"
    assert "linearWave" in lines[1]


def test_pde_smoke_measures_initial_state(smoke_run):
    # The driver must measure the t=0 conserved quantities on the real state
    # (it used to zero-fill momentum/angmom and copy the final mass) and
    # record the scheme the case actually ran.
    if smoke_run.returncode != 0:
        pytest.skip("smoke did not succeed")
    (row,) = load_rows(SMOKE_CSV)
    assert row["scheme"] == "CRKSPH"          # linearWave's frontend default
    assert "momentum_drift_abs" in row and "angmom_drift_abs" in row
    # 1D periodic acoustic wave: CRKSPH conserves momentum to round-off
    assert abs(row["momentum_drift_abs"]) < 1e-10
    assert row["mass_drift"] == pytest.approx(0.0, abs=1e-14)


def test_pde_smoke_does_not_clobber_full_suite_results(smoke_run):
    # A smoke run must not overwrite the full-suite results file (it used
    # to, silently destroying the committed local results on every CI run).
    if smoke_run.returncode != 0:
        pytest.skip("smoke did not succeed")
    if not ROWS_CSV.exists():
        pytest.skip("no full-suite results file present")
    assert ROWS_CSV.read_bytes() != SMOKE_CSV.read_bytes(), (
        "smoke run clobbered the full-suite pde_rows.csv")


# ---------------------------------------------------------------------------
# CRK limiter-constant derivation (derive_crk_limiter.py)
# ---------------------------------------------------------------------------

def test_crk_limiter_derivation_units():
    import derive_crk_limiter as D
    # n_h = 4 (H = 4 dx): the paper's CRKSPH constants (1/n_h, 0.2) in r/h
    # with h = dx become (0.25, 0.05) in the frontend's r/H
    d = D.derive(0.25, ks=2.291288)
    assert d["spacing"] == pytest.approx((0.25, 0.05))
    # D&A-sigma reading of h: fold 0.2 / ks
    assert d["sigma"] == pytest.approx((0.25, 0.2 / 2.291288))
    # the frontend's current hard-coded pair is 1/n_h for n_h = 3
    assert D.CURRENT[0] == pytest.approx(1 / 3, rel=1e-6)


def test_crk_limiter_sweep_cases_have_derivation_entries():
    import derive_crk_limiter as D
    import run_crk_limiter_sweep as S
    assert set(S.SWEEP) <= set(D.CASES)
