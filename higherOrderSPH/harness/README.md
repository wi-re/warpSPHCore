# Convergence harness

Shared (non-paper) infrastructure for the higher-order effort in
`../../higher_order.md`: the Phase 0 evaluation harness plus the
Phase 1 (standard SPH), Phase 2 (CRKSPH), and Phase 3 (Bonet–Lok /
`renormVal`) baselines. Read `PLAN.md` for scope, components, and
success criteria.

Status: **first pass implemented 2026-09-22** (components + CI gate);
**Phase 3 `renormVal` mode added 2026-09-23** (additive versioning
event — the three prior modes are byte-identical). `REPORT.md` +
`FINDINGS.md` are the frozen "before column", generated/curated by the
driver.

## Layout

| File | Role |
|---|---|
| `particle_sets.py` | `build_case`: densest-packing lattices (1D/2D-hex/3D-FCC, jitter, seeded) via the shipped `sampleDensestLattice`, open/periodic domains, interior/boundary masks, densities, adjacency |
| `test_fields.py` | monomials (degree 0–4, scalar + one linear vector field) and smooth fields (commensurate sinusoid modes, wrapped Gaussian), each with analytic value/gradient/Laplacian |
| `operators.py` | the common probe interface over `warpOperation`: modes `standard` / `crk` (CRK apparent volume, per the validated frontend usage) / `renorm` (Bonet–Lok corrected gradient) / `renormVal` (full Phase 3: renorm gradient + Randles–Libersky value renormalization `f̂/S`, derived from two `Interpolate` calls); probes `interpolate` / `gradient` (Difference) / `laplacian` (Brookshaw, scalar) |
| `metrics.py` | masked L1/L2/L∞, log-log order extraction with saturation **and exact-zero** detection (pure numpy/torch) |
| `conditioning.py` | renorm condition numbers from the shipped eigenvalues, identity-fallback row detection (pure torch) |
| `report.py` | standardized CSV / markdown / log-log plot output |
| `run_baseline.py` | sweep driver: suites `patch`, `resolve-open`, `resolve-periodic`, `smoothing`, `conditioning`; writes `results/`, `figures/`, `REPORT.md`; `--smoke` is the CI matrix |
| `FINDINGS.md` | curated interpretation of the baseline report (the frozen "before column") |

```
python run_baseline.py --smoke     # CI gate (float64, minimal matrix)
python run_baseline.py             # full baseline -> REPORT.md
python run_baseline.py --suites patch,resolve-open --kernels Wendland2 CubicSpline
```

## Pass 2: PDE benchmark (`pde/`)

Multi-resolution convergence + conservation for the real `warpSPH` frontend
cases (imported, never re-implemented). Each case runs a 4-point resolution
ladder at its full simulated time; the metric is the L2 field error — vs the
analytic solution for the smooth closed-form cases (TGV, linearWave), or vs
the finest-ladder run (projected onto a common grid) for the rest (Gresho,
Kelvin–Helmholtz, Sod, Sedov) — plus the conservation drift (mass / KE /
momentum / angular momentum).

| File | Role |
|---|---|
| `pde/pde_cases.py` | case registry: `PDECase` (module, dim, metric, 4-point `nx_ladder`, `t_star`, analytic field, measured state field) + the TGV / linearWave analytic fields |
| `pde/conservation.py` | driver-side conserved quantities + drift (pure torch; mass exact, KE relative, momentum/angmom as absolute final norms for the ~0-symmetric ICs) |
| `pde/field_error.py` | the reference metric: cloud-in-cell (CIC) projection of both particle sets onto a common grid, RMS L2 over the shared cells (pure torch; density = mass/cell volume) |
| `pde/run_pde.py` | the driver: float64, per-(case, nx) rows, observed order per case, `REPORT_pde.md` + `results/pde_rows.csv`; `--smoke` is the CI gate |

```
python pde/run_pde.py --cases sod sedov   # fast 1D cases (minutes) -> dev loop
python pde/run_pde.py                     # all 6 cases, full t* (~3-4 h overnight)
python pde/run_pde.py --cases tgv --smoke # CI gate
```

The 2D incompressible cases (TGV/Gresho/KH) cost ~nx³ under adaptive dt and
dominate the runtime; the 1D cases finish in minutes and are the fast
dev/test path.

## Phases 4-7: high-order operators and reconstruction (`rkpm.py`, `reconstruct.py`)

Added 2026-10-09 (parent plan `../../higher_order.md`, Phases 4-7). These are
**pure-torch reference implementations** (CPU-testable, float64-native); they
do not touch `src/warpSPHCore` and do not change any frozen baseline output.

| file | role |
|---|---|
| `rkpm.py` | order-p moment system + local polynomial fit: MLS/RKPM (Phase 4) and the LABFM variant `constant=False` (Phase 5); value / gradient / Hessian / Laplacian, `interface_states` |
| `rkpm_modes.py` | registers `rkpm1..3` and `labfm<k>` through `operators.register_mode` |
| `run_rkpm.py` -> `REPORT_rkpm.md` | Phase 4: the unchanged static suites with the new modes + p=1 vs CRK cross-check + conditioning vs order (`--smoke` is the CI gate) |
| `run_labfm.py` -> `REPORT_labfm.md` | Phase 5: observed order vs scheme order k and stencil size (King et al. 2020) |
| `run_frozen_hi.py` -> `REPORT_frozen_hiorder.md` | frozen-particle PDE leg for the new modes (own report; `REPORT_frozen.md` untouched) |
| `reconstruct.py` | Phases 6-7: nine-stencil TENO (Gao 2023) / WENO (Avesani 2014) interface-state reconstruction |
| `run_reconstruct.py` -> `REPORT_reconstruct.md`, `run_interface.py` -> `REPORT_interface.md` | reconstruction-only evidence (smooth order, step overshoot, jump capture); **no Riemann solver** |

CI: `tests/convergence/test_rkpm.py`, `tests/convergence/test_reconstruct.py`.

## Conventions

- float64 (`warpSPHCore_PRECISION=float64`, set by the driver before
  import); Wendland2 default kernel; jitter 0.3; default target
  neighbors 40 (h/Δx ≈ 3.6 in 2D).
- Monomial patch tests run on the **open** domain (a linear field is not
  periodic, so the Difference gradient is seam-contaminated on periodic
  domains); the periodic resolution sweep uses periodic smooth fields.
- `results/` and `figures/` are git-ignored (regenerable).

Like the replications, code here imports the installed `warpSPHCore`
package and never re-implements a kernel or operator from `src/`.
CI gate: `tests/convergence/test_harness.py` (repo root) -- pure-module
unit tests plus a session-scoped `--smoke` subprocess run.
