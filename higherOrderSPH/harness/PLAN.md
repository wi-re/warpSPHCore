# Convergence harness — first-pass plan (Phase 0 static + Phases 1–2 baselines)

> Parent plan: `../../higher_order.md`. This folder is **shared
> infrastructure**, not a paper replication — it exists for the
> higher-order effort as a whole and is reused unmodified by every
> later phase (per the parent plan's cross-cutting notes: a harness
> change mid-plan is a versioning event that re-runs all prior phases).

Goal: build the evaluation harness once, then run it against the two
existing reference operators — **standard SPH** (Phase 1) and
**CRKSPH** (Phase 2) — to produce the "before" column every later
phase (Bonet–Lok, MLS, LABFM, TENO, WENO) is compared against.

## Pass 2 (PDE benchmark) — implemented 2026-09-23

The pass-2 scope below is now built in the `pde/` subfolder (see its
section in `README.md`): a multi-resolution driver over the real
`warpSPH` frontend cases, measuring the L2 field error (vs analytic for
TGV/linearWave, vs the finest-ladder run projected onto a common grid for
Gresho/KH/Sod/Sedov) plus the conservation drift. The 1D cases are the
fast dev/test path (minutes); the full 6-case run is an overnight job.

## Scope decision (2026-09-22)

- **In:** static consistency only — patch tests, convergence rates,
  condition numbers, interior vs boundary, both refinement modes.
- **Out (pass 2):** the PDE benchmark suite (TGV, Gresho, standing
  acoustic, Sod, Sedov, KH) and conservation diagnostics. Frontend
  assets already exist (`warpSPH/cases/tgv.py`,
  `warpSPH/caseUtils/compressible/{greshoVortex,linearWave,sod,sedov,
  kelvinHelmholtz}`, runner `RunResult.trajectory` carries per-step
  diagnostics) — the pass-2 work is the multi-resolution driver +
  drift measurement, not the cases themselves. *(Now implemented — see
  "Pass 2" above and the `pde/` subfolder.)*
- **Out (later phases):** all new operators (parent-plan Phases 3–7).

## Key finding — the operator stack for this pass already exists

The core library already ships everything the static pass probes;
nothing in `src/` needs to change:

| Need | Where |
|---|---|
| Lattice particle sets (1D / 2D hex / 3D FCC, jitter, seeded) | `warpSPHCore.sampling.sampleDensestLattice` |
| Grid set + domain + h from target neighbor count | `warpSPHCore.util.generateNeighborTestData`, `volumeToSupport` |
| Operator dispatch (Interpolate/Gradient/Laplacian/Divergence/Curl/Density/Covariance) | `warpOperation` + `OperationProperties` |
| Neighbor search | `radiusSearchCompactHashMap` |
| CRKSPH correction factors | `warpSPHCore.crk.computeCRKFactors` |
| Correction matrix + eigenvalues + pseudo-inverse | `warpSPHCore.renorm.computeRenormalizationMatrices` → `(C, eigVals, L)` |
| Open (non-periodic) domains | `DomainDescription.periodicity` (tested in `tests/operations/`) |
| float64 runs | `warpSPHCore_PRECISION=float64` env var (same mechanism the kernel audit uses) |

**Bonus column:** the parent plan's Phase 3 (Bonet–Lok corrected
gradient, "local d×d matrix solve per particle") is already the
covariance/renorm machinery above — `warpOperation(Gradient,
renormalizationState=…)` consumes `computeRenormalizationMatrices`'s
output. Phase 3's implementation work therefore likely collapses to
validation + gap analysis; this pass runs it as a **third reference
column at ~zero cost** and pre-answers most of Phase 3's patch tests.

## Components (all new code lives in this folder)

Conventions: same as the dehnen2012 replication — import the installed
`warpSPHCore` package, never re-implement a kernel or operator that
exists in `src/`.

1. **`particle_sets.py`**
   - Periodic ordered + jittered sets: wrap `sampleDensestLattice`
     (jitter fraction + seed configurable).
   - Open-boundary sets: same lattice on a non-periodic domain;
     exposes the interior mask and the boundary band (particles within
     one support of an open edge — the truncated-support region).
   - Refinement parameterization, decoupled:
     (a) N fixed, vary h/Δx via `volumeToSupport` (smoothing error);
     (b) h/Δx fixed, vary N (discretization error).
2. **`test_fields.py`** — monomial fields x^a y^b (z^c) up to degree 4
   (1D/2D/3D, scalar; vector fields for gradient-order tests) and
   smooth C^∞ fields (sinusoids, Gaussians), each with analytic value
   / gradient / Laplacian.
3. **`operators.py`** — thin adapter over `warpOperation` implementing
   the common interface the parent plan's Phase 0 deliverable requires:
   - modes: `standard` (no correction state), `crk`
     (`computeCRKFactors`), `renorm` (`computeRenormalizationMatrices`);
   - probes: Interpolate, Gradient, Laplacian (the plan's named set;
     Divergence/Curl only if trivially wired — no Hessian probe: the
     core has no Hessian operator, deferred).
4. **`metrics.py`** — L1/L2/L∞ of (probe − analytic) with the
   interior/boundary masks; log-log least-squares slope extraction for
   observed order; explicit handling of the float64 noise floor (a run
   whose error flattens is reported as "saturated", not fitted).
5. **`conditioning.py`** — per-particle condition number
   max|λ|/min|λ| from the renorm `eigVals`, tracked vs jitter fraction,
   neighbor count, and open-boundary position; flags the
   `num_nbrs < dim + 2` identity-fallback region.
6. **`report.py` + `run_baseline.py`** — the sweep driver and the
   standardized CSV + plot + `REPORT.md` generator (order vs scheme vs
   test). Every later phase's driver reuses `report.py` unmodified so
   outputs are directly comparable.

**CI gate:** `tests/convergence/test_harness.py` in the repo root
`tests/`, subprocess pattern copied from
`tests/kernels/test_kernel_audit.py` (run the harness in float64 once
per session, small N). Checks at minimum:
- CRK reproduces degree-1 scalar/vector fields to machine precision on
  an ordered AND a jittered periodic set (CRK's core claim);
- standard SPH does NOT (guards against a harness that accidentally
  corrects everything);
- renorm reproduces linear gradients (interior);
- the order extractor returns the exact order on a synthetic
  E = C h^p series.

## Conventions

- **Precision:** float64 (`warpSPHCore_PRECISION=float64`) for all
  convergence runs — float32's ~1e-7 floor caps observable order.
- **h convention:** code h = support radius; paper 2σ convention
  converted via the shipped `eval_kernelScale` (same pin as the
  dehnen2012 `scripts/common.py`).
- **Kernel(s):** Wendland2 as the default probe kernel (the repo's
  notebook/test default); one additional kernel (Wendland6 or Cubic
  Spline) for a kernel-dependence spot check.
- **Device:** GPU (shared box — the local LLM holds ~69 % of SMs; keep
  per-point N ≲ 10⁴ so the whole sweep is minutes, and cap BLAS
  threads for any host-side numerics).
- **Outputs:** `results/` (CSVs) and `figures/` (PNGs), git-ignored,
  like the dehnen2012 folder.

## Success criteria (first pass)

1. Patch tests: exact (machine-precision) degree-1 reproduction by CRK
   and renorm in the interior on ordered + jittered sets; standard SPH
   behavior at degrees 0–1 documented (expected: kernel-dependent
   zeroth-order value, sub-first-order gradient on disordered sets).
2. Observed convergence orders measured for Interpolate/Gradient/
   Laplacian × {standard, crk, renorm} × both refinement modes ×
   {interior, boundary band}, degree 0–4 fields, in a single
   comparable table (`REPORT.md`).
3. Renorm condition number characterized vs jitter and neighbor count,
   including the open-boundary degradation and the identity-fallback
   region — this is the documented ill-conditioning boundary the parent
   plan's Phase 3 asks for.
4. `REPORT.md` is the frozen "before" column: Phases 3–7 diff against
   it and may not re-baseline silently.

## Requirements

- conda env `warp` (Python 3.13, NumPy 2.x, warp, torch);
  `OMP/OPENBLAS/MKL_NUM_THREADS=4` for host-side numerics.
- No new dependencies, no `src/` changes, no new kernels.
- `warpSPHCore` installed in the env (it is).

## Non-goals (explicit)

- PDE benchmarks + conservation diagnostics → pass 2 (frontend
  driver), assets listed in the scope decision above.
- Hessian probe, MLS/RKPM, LABFM, TENO, WENO → parent-plan Phases 4–7.
- Any change to kernel constants or operator code in `src/` —
  discrepancies found here are logged in `REPORT.md`'s findings
  section, fixed separately and deliberately, per the replication
  convention.
