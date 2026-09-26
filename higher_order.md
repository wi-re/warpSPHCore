# SPH Higher-Order Convergence — Implementation Plan

> **Status (2026-09-26).** Phases 0–3 are at *first pass*: the Phase 0
> harness is built and CI-gated (`higherOrderSPH/harness/`), and the three
> reference operator columns — standard SPH (Phase 1), CRKSPH (Phase 2),
> Bonet–Lok/`renormVal` (Phase 3) — exist for the *static* benchmarks, plus
> the Pass-2 PDE benchmark suite (all six requested cases + a delta+-SPH
> leg). **Phases 4–7 (MLS/RKPM, LABFM, TENO, WENO) have not started.** A
> parallel *paper-replication validation track* (see "Tracks added outside
> this plan") preceded the new-operator work and is mostly closed.
>
> **Where the numbers live:** `harness/REPORT.md` + `FINDINGS.md` (frozen
> static "before column"), `harness/pde/REPORT_pde.md` (PDE orders +
> conservation), `harness/pde/TGV_NOTES.md` (TGV floor + delta+ scheme
> study), `harness/pde/SEDov_NOTES.md` (shock metrics + Sedov diagnosis).

---

## Phase 0 — Convergence Test Harness

**Goal:** Build the evaluation infrastructure once, before touching any operator implementation, so every subsequent phase is judged against the same fixed tests.

**Status: DONE** (first pass 2026-09-22; PDE pass 2026-09-23; CI gate
`tests/convergence/test_harness.py` + `tests/convergence/test_pde.py`).
Two deliberate scope cuts vs the original list (monomial degree cap 4,
no Hessian probe) are noted on the tasks.

**Tasks:**
- [x] Particle set generators — `particle_sets.py`: densest-packing
  lattices (1D / 2D-hex / 3D-FCC, jitter, seeded) via the shipped
  `sampleDensestLattice`, open/periodic domains, interior/boundary masks.
  (Densest-packing rather than plain regular lattices — the shipped
  sampler is the one the real cases use.)
- [x] Test function library — `test_fields.py`: monomials $x^a y^b (z^c)$
  degree **0–4** (cap is 4, not 4–5) + one linear vector field, and smooth
  $C^\infty$ fields (commensurate sinusoids, wrapped Gaussian), each with
  analytic value/gradient/Laplacian.
- [x] Operator probes — `operators.py`: value interpolation, gradient
  (Difference), **scalar Laplacian** (Brookshaw). *Hessian probe deferred —
  the core has no Hessian operator.*
- [x] Error metrics — `metrics.py`: masked $L_1/L_2/L_\infty$ vs $h$;
  log-log order extraction with explicit **saturation** and **exact-zero**
  detection (float64-floor aware).
- [x] Decoupled refinement modes — `run_baseline.py` suites: `smoothing`
  (vary $h/\Delta x$ at fixed $N$) and `resolve-open`/`resolve-periodic`
  (vary $N$ at fixed $h/\Delta x$).
- [x] Boundary-region variant — open (non-periodic) domain suites with a
  boundary-band mask (truncated kernel support).
- [x] Condition-number tracking — `conditioning.py`: renorm condition
  number from the shipped eigenvalues vs jitter / neighbor count /
  open-boundary position, with `num_nbrs < dim+2` identity-fallback row
  detection.
- [x] PDE-level benchmark suite — implemented as *Pass 2* in `pde/`
  (2026-09-23), over the real `warpSPH` frontend cases: TGV (analytic),
  Gresho, standing acoustic wave (linearWave, analytic), Sod, Sedov,
  Kelvin–Helmholtz — **plus a 7th leg, tgv-wc (delta+-SPH), registered
  2026-09-24.** 4-point resolution ladders at full simulated time.
- [x] Conservation diagnostics — `conservation.py` (driver-side, pure
  torch): mass / KE / **total energy (KE+IE)** / momentum / angular
  momentum drift.
- [x] Reporting — `report.py` + `report_pde.py`: standardized CSV /
  markdown / log-log; `REPORT_pde.md` regenerable from `pde_rows.csv`
  without importing warp.

**Deliverable:** ✅ Harness runnable against any operator/solver satisfying a common interface, producing order-of-convergence tables + condition-number diagnostics + PDE benchmark plots.

**References:** methodology drawn from `frontiere_2017_crksph`, `king_lind_2020_labfm`, `lind_rogers_stansby_2020_review` (survey of evaluation practice across the field).

---

## Phase 1 — Standard SPH Baseline

**Goal:** Establish the reference point everything else is measured against.

**Status: DONE** — static column (`REPORT.md` mode `standard`) + full PDE
suite (`REPORT_pde.md`) + two root-cause deep-dives on the weak cases.

**Tasks:**
- [x] Run Phase 0 harness against uncorrected SPH kernel
  interpolation/gradient (existing warpSPH frontend) — `REPORT.md`,
  `standard` column (Wendland2 default; CubicSpline spot check).
- [x] Confirm expected behavior — sub-first-order (often not even
  zeroth-order) consistency on disordered particles, degraded further at
  boundaries: confirmed and documented in `FINDINGS.md`.
- [x] Run full PDE benchmark suite — 2026-09-23 full run, no divergences.
  Observed L2 orders: TGV **0.69**, Gresho **1.54**, linearWave **~1.05**
  (near the float64 floor), KH **saturated** (non-self-similar
  instability), Sod **1.31** (L1 **1.46**), Sedov **0.47** raw (L1
  **0.92**; raw-L2 non-monotonicity root-caused as a common-cells
  ensemble artifact — `SEDov_NOTES.md`).
- [x] Record baseline numbers — `REPORT_pde.md` + `results/pde_rows.csv`
  are the comparison table header for every later phase.

**Deep-dives (beyond the original task list):**
- TGV 0.69 root-caused (`TGV_NOTES.md` §1–3): a **non-converging ~3e-3
  error floor** = residual compressibility of the DFSPH velocity field
  (spurious (2,0)/(0,2) Fourier modes that cannot exist in a div-free
  flow), not the integrator. User-scoped as the standing *vd+ps* problem
  (out of benchmark scope).
- Sedov/Sod shock metrics added 2026-09-25 (`field_error.py`: L1 area
  norm + sub-cell shift-aligned L2): shock cases are first-order
  shock-capturing; judge them on L1 / full-grid L2, not common-cells
  raw L2 (`SEDov_NOTES.md`).

**Deliverable:** ✅ Baseline report — the "before" column for all following phases.

---

## Phase 2 — CRKSPH Baseline (existing frontend)

**Goal:** Since CRKSPH is already implemented in the compressible frontend, use it as the second reference point and validate it against the harness rather than implementing it fresh.

**Status: DONE at static level; PDE-level CRKSPH comparison NOT done**
(the "improved scheme" PDE leg was filled by delta+-SPH, tgv-wc — see
Phase 1 deep-dives and `TGV_NOTES.md` §4).

**Tasks:**
- [x] Run Phase 0 harness against the existing CRKSPH operators
  (interpolation + gradient) — `REPORT.md`, `crk` column (CRK apparent
  volume per the validated frontend usage).
- [x] Verify exact linear-field reproduction (patch test to degree 1) —
  met to machine precision, ordered **and** jittered, and CI-gated
  (`test_harness.py`).
- [ ] Verify exact conservation of mass/momentum/energy to machine
  precision; measure angular momentum drift — not run as a CRKSPH
  operator-level check (PDE-suite conservation is per *scheme*, and no
  CRKSPH PDE scheme exists in the frontend).
- [ ] Run PDE benchmark suite, compare dissipation/order against Phase 1 —
  not done (no CRKSPH PDE case).
- [x] Document gaps between harness results and paper-reported behavior —
  `FINDINGS.md` (static); the delta+ leg's floor analysis is the
  de-facto "what does a corrected scheme still fail at" study.

**Deliverable:** ✅ (partial) CRKSPH validated as second reference column for the static harness; harness itself validated against a known-good implementation.

**References:** `frontiere_2017_crksph`.

---

## Phase 3 — Bonet–Lok Gradient Correction

**Goal:** Add the cheapest first-order consistency correction as the entry point into the "corrected kernel" family.

**Status: DONE** — with the key scoping fact discovered up front: the
correction machinery **already ships in the core**
(`warpOperation(Gradient, renormalizationState=…) +
computeRenormalizationMatrices` → `(C, eigVals, L)`), so the phase
collapsed to validation + gap analysis at ~zero implementation cost, run
as a **third reference column**. The full Phase-3 operator (renorm
gradient + Randles–Libersky value renormalization `f̂/S`) landed as the
`renormVal` mode (2026-09-23 — an additive harness versioning event; the
three prior modes byte-identical).

**Tasks:**
- [x] Implement corrected kernel gradient enforcing
  $\sum_j (r_j - r_i)\otimes\nabla W_{ij} V_j = I$ — pre-existing in
  `warpSPHCore.renorm`; no `src/` change needed.
- [x] Add correction-matrix condition-number logging — `conditioning.py`
  (per-particle condition number vs jitter, neighbor count, open boundary;
  identity-fallback region flagged).
- [x] Run patch tests + convergence tests — `REPORT.md` `renorm` /
  `renormVal` columns: exact linear-gradient reproduction (interior);
  value zeroth-order unless also corrected (`renormVal` fixes the value,
  as the task anticipated).
- [x] Boundary-region tests — open-domain suites: documented
  ill-conditioning boundary + identity-fallback failure region.
- [ ] Run PDE benchmark suite, compare against Phase 1/2 — not done (no
  Bonet–Lok PDE scheme in the frontend).

**Deliverable:** ✅ (static) Corrected-gradient SPH operator, benchmarked, with documented ill-conditioning boundaries. PDE-level comparison pending a frontend scheme.

**References:** `bonet_lok_1999_variational_momentum`, `randles_libersky_1996_...`, optionally `chen_beraun_1999_cspm` and `liu_liu_2006_restoring_consistency` if extending correction to the kernel value itself.

---

## Phase 4 — MLS / RKPM Reproducing Kernel

**Goal:** Generalize Phase 3's single-order correction into an arbitrary-polynomial-order reproducing-kernel operator generator — this becomes the shared moment-matrix machinery reused in Phase 5 (LABFM).

**Status: NOT STARTED.** (Explicit non-goal of the first pass, `harness/PLAN.md`.)

**Tasks:**
- [ ] Implement generic moment-matrix builder for a chosen monomial basis (parametrized consistency order $p$), following MLS/RKPM formalism.
- [ ] Implement kernel correction function $C(x; x-x_j)$ so the kernel itself (not just gradient) reproduces polynomials to order $p$.
- [ ] Note non-symmetry of resulting kernel ($W_{ij} \neq W_{ji}$) — decide whether to carry forward a naive (non-conservative) MLS operator as a standalone mode, or route straight into a CRKSPH-style conservative reformulation reusing this moment matrix.
- [ ] Run Phase 0 harness at multiple orders $p = 1, 2, 3$: patch tests to degree $p$, convergence-rate tests, condition-number tracking vs. $p$ and neighbor count (expect conditioning to worsen with $p$ on disordered particles).
- [ ] Run PDE benchmark suite; if non-conservative, explicitly report conservation-diagnostic failures (expected) as a documented tradeoff vs. Phase 2.

**Deliverable:** Order-parametrized reproducing-kernel operator generator, validated across $p=1..3$, with conservation-vs-accuracy tradeoff documented.

**References:** `liu_jun_zhang_1995_rkpm`, `dilts_1999_mlsph`, `dilts_2000_mlsph2`.

---

## Phase 5 — LABFM

**Goal:** Push the moment-matching idea from Phase 4 to arbitrary order using anisotropic basis functions and compact stencils, decoupled from kernel-summation framing.

**Status: NOT STARTED.**

**Tasks:**
- [ ] Implement anisotropic basis function (ABF) construction and the local linear system solved per stencil (reuse Phase 4's moment-matrix infrastructure where architecturally possible).
- [ ] Implement one-sided/boundary stencil handling for incomplete support.
- [ ] Run Phase 0 harness at increasing order (target: reproduce paper's reported 4th order at ~25 neighbors in 2D, up to 8th–10th order at larger stencils) — this is the most discriminating test of harness correctness, since the paper gives concrete stencil-size/order pairs to match.
- [ ] Stability check: add hyperviscosity option and confirm it stabilizes hyperbolic test PDEs per `king_lind_2020_labfm`.
- [ ] Run PDE benchmark suite; compare compute cost vs. order against Phase 4's correction-matrix approaches at matched order.

**Deliverable:** Arbitrary-order LABFM operator, cross-validated against the paper's specific stencil-size/order table.

**References:** `king_lind_2020_labfm`, `king_lind_2021_labfm_isothermal`.

---

## Phase 6 — TENO Reconstruction (Riemann-SPH layer)

**Goal:** Add high-order shock-capturing reconstruction as a separate solver-level module (not part of the core operator library), per the earlier architectural note.

**Status: NOT STARTED.** Note: the dehnen2012 replication's Phase 6
delivered the *prerequisite* shock-capturing validation (C&D 2010 switch
+ R&H 2012 conductivity as validated frontend options) — see
`dehnen2012.../REPORT.md` §6.

**Tasks:**
- [ ] Implement MLS-based local polynomial reconstruction at particle-pair interfaces, interface position $\overline{r}_{ij} = (h_j r_i + h_i r_j)/(h_i + h_j)$.
- [ ] Implement TENO smoothness indicators / stencil selection and blending weights.
- [ ] Couple to existing Riemann solver in the compressible frontend for left/right state resolution.
- [ ] Run Phase 0 smooth-region convergence tests (target: 4th-order-class reconstruction accuracy) and shock-tube/Sedov/KH tests for non-oscillatory behavior and reduced dissipation vs. WENO of matched order.

**Deliverable:** TENO-SPH reconstruction module, validated for both smooth-region order and shock robustness.

**References:** `fu_2016_teno` (original TENO), MLS-TENO-SPH paper in your literature folder (arXiv 2306.00514).

---

## Phase 7 — WENO Reconstruction

**Goal:** Implement WENO as the baseline shock-capturing comparison point against Phase 6's TENO.

**Status: NOT STARTED.**

**Tasks:**
- [ ] Implement MLS-WENO reconstruction (same interface-reconstruction structure as Phase 6, swap smoothness-indicator/weighting scheme).
- [ ] Run identical Phase 0 test set used for TENO — smooth-region order, shock-tube/Sedov/KH dissipation comparison.
- [ ] Direct head-to-head report: WENO vs. TENO dissipation at matched formal order (this is the comparison the TENO papers themselves make).

**Deliverable:** WENO-SPH reconstruction module; final comparison table across all phases (Standard SPH / CRKSPH / Bonet-Lok / MLS / LABFM / WENO-SPH / TENO-SPH) on the same harness.

**References:** `avesani_dumbser_bertaux_2014_mlswenosph`, follow-up "Investigations on a high order SPH scheme using WENO reconstruction" paper in your literature folder.

---

## Tracks added outside this plan's structure

Work that landed before/during the first pass and isn't covered by the
phase list above:

- **Paper-replication validation track** —
  `higherOrderSPH/dehnen2012_convergence-without-pairing-instability/`
  (Dehnen & Aly 2012, MNRAS 425, 1068). Purpose: prove the shipped
  kernels *are* the paper's kernels and behave as claimed, before building
  new operators. **Phases 0–6 + closeout complete (2026-09-20)** —
  10/10 kernels audited to machine precision, all static evaluations
  reproduced, Gaussian + HOCT4 on-boarded via the repo-level
  onboarding/audit pipeline, C&D 2010 / R&H 2012 dissipation validated as
  frontend options (`REPORT.md`). **Phase 7 (dynamic tests, paper
  Figs 7–13) open** — separate effort.
- **Delta+-SPH PDE leg + scheme study** — `tgv-wc` (delta-SPH + PST)
  registered as the 7th PDE case (order ~1.69–1.88, r²≈1.0, vs standard
  TGV 0.69 — the first "after" column). `TGV_NOTES.md` §4–6: higher-order
  RK does NOT lift the ~8.5e-3 floor; the floor **is the particle
  shifting** (PST causal test — velocity decreases monotonically with
  shift strength; full-strength `sun2017DeltaSPH` is 30% lower on
  velocity, 4.5× better on volume). User decision 2026-09-24: frontend
  default shift stays 1/8 (free-surface balance); the harness leg runs
  full strength via a `PDECase.scheme` override, and the `shuffleEq7`
  knob removes the disordered-IC penalty (clean ladder
  5.48/2.47/1.63/6.60e-2, fit ≈2.0).
- **Distribution anisotropy instrumentation** — `harness/distribution.py`
  (first-moment residual + second-moment shape anisotropy, pure torch,
  CI-tested) + saved TGV particle distributions as committed test data
  (`data/tgv2d_{noshift,fullshift}_nx128.npz`): uncorrected flow
  accumulates disorder, full shifting re-regularises it.
- **Disorder probe** — `harness/run_disorder_probe.py` (CI-gated): static
  operator probes on five distributions. Headline: CRK interpolation is
  distribution-*independent*; Bonet–Lok renorm is the most robust
  gradient; higher-order Wendland kernels are *worse* under strong
  disorder for standard operators (`TGV_NOTES.md` §7.5).
- **Validated optimal-sampling start** — the `SamplingScheme.optimal`
  glass sampler (0.1·dx jitter + kernel-dependent delta-shift relaxation)
  is wired into the frontend tgv-wc build path (warpSPH commit 765ab55,
  2026-09-26): glass IC a2-rms 7.9e-3 vs shuffle 8.4e-3, kernel-sum
  spread 3× tighter, stable run — `samplingScheme='optimal'` is now a
  turn-key knob.

---

## Cross-cutting notes

- Every phase reuses Phase 0's harness unmodified — if a harness change is needed mid-plan, treat it as a harness-versioning event and re-run all prior phases before proceeding.
  **Versioning events that have occurred:** (1) `renormVal` mode added
  2026-09-23 — additive, the three prior modes byte-identical, no
  re-baseline; (2) `pde/` subfolder added 2026-09-23 (separate driver,
  static `REPORT.md` untouched); (3) shock metrics (L1 / aligned L2)
  added to `field_error.py` 2026-09-25 (new columns, old columns
  unchanged).
- Phases 3–5 share moment-matrix infrastructure; consider implementing that shared layer once in Phase 3/4 rather than duplicating in Phase 5.
  (Phase 3 discovered the moment-matrix machinery already ships in
  `warpSPHCore.renorm` — Phases 4/5 should start from
  `computeRenormalizationMatrices`, not a fresh builder.)
- Phases 6–7 are architecturally separate (solver-level, not operator-library-level) — fine to parallelize with Phase 4/5 if useful.
