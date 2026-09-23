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
