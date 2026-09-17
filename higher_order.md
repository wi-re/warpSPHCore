Here's the plan, structured as phases you can paste directly into a doc. I've used `author_year_slug`-style placeholders for citations — swap in your actual filenames from the literature subfolder where they differ.

---

# SPH Higher-Order Convergence — Implementation Plan

## Phase 0 — Convergence Test Harness

**Goal:** Build the evaluation infrastructure once, before touching any operator implementation, so every subsequent phase is judged against the same fixed tests.

**Tasks:**
- [ ] Particle set generators: regular lattice, randomly perturbed lattice (configurable jitter fraction), and a bounded-domain set with open boundaries (truncated kernel support).
- [ ] Test function library: monomials $x^a y^b (z^c)$ up to degree 4–5 for patch tests; smooth $C^\infty$ fields (sinusoids, Gaussians) for convergence-rate tests.
- [ ] Operator probes: value interpolation, gradient, Laplacian/Hessian — evaluated independently, since consistency order can differ between them (this is the CRKSPH motivating point — see `frontiere_2017_crksph`).
- [ ] Error metrics: $L_1$, $L_2$, $L_\infty$ vs. $h$ (or $1/\sqrt{N}$ / $1/\sqrt[3]{N}$), log-log slope extraction for observed order.
- [ ] Decoupled refinement modes: vary $h/\Delta x$ at fixed $N$ (smoothing error) and vary $N$ at fixed $h/\Delta x$ (discretization error) separately.
- [ ] Boundary-region variant of all above (near open boundary / incomplete support).
- [ ] Condition-number tracking for any local linear system (correction matrix, moment matrix, LABFM ABF system) as a function of particle disorder and neighbor count.
- [ ] PDE-level benchmark suite (reuse existing compressible frontend where possible): 2D Taylor-Green vortex (smooth, exact solution — primary order-of-convergence benchmark), Gresho vortex, standing acoustic wave, Sod shock tube, Sedov blast, Kelvin-Helmholtz (dissipation/order-swamping diagnostic).
- [ ] Conservation diagnostics: mass/momentum/energy exact-conservation check, angular momentum drift vs. time and vs. resolution.
- [ ] Reporting: standardized plot/table generator (order vs. scheme vs. test) so all later phases produce directly comparable output.

**Deliverable:** Harness runnable against any operator/solver satisfying a common interface, producing order-of-convergence tables + condition-number diagnostics + PDE benchmark plots.

**References:** methodology drawn from `frontiere_2017_crksph`, `king_lind_2020_labfm`, `lind_rogers_stansby_2020_review` (survey of evaluation practice across the field).

---

## Phase 1 — Standard SPH Baseline

**Goal:** Establish the reference point everything else is measured against.

**Tasks:**
- [ ] Run Phase 0 harness against uncorrected SPH kernel interpolation/gradient (existing warpSPH frontend).
- [ ] Confirm expected behavior: sub-first-order (often not even zeroth-order) consistency on disordered particles, degraded further at boundaries.
- [ ] Run full PDE benchmark suite with standard SPH to get baseline order-of-convergence and dissipation characteristics (especially Taylor-Green, Kelvin-Helmholtz).
- [ ] Record baseline numbers as the comparison table header for every later phase.

**Deliverable:** Baseline report — this is the "before" column for all following phases.

---

## Phase 2 — CRKSPH Baseline (existing frontend)

**Goal:** Since CRKSPH is already implemented in the compressible frontend, use it as the second reference point and validate it against the harness rather than implementing it fresh.

**Tasks:**
- [ ] Run Phase 0 harness against the existing CRKSPH operators (interpolation + gradient).
- [ ] Verify exact linear-field reproduction (patch test to degree 1) — this is CRKSPH's core claim.
- [ ] Verify exact conservation of mass/momentum/energy to machine precision; measure angular momentum drift.
- [ ] Run PDE benchmark suite, compare dissipation/order against Phase 1 baseline (expect reduced artificial-viscosity diffusion per the limiter in `frontiere_2017_crksph`).
- [ ] Document any gaps between harness results and paper-reported behavior — flags either harness bugs or frontend implementation drift worth fixing before building further on top.

**Deliverable:** CRKSPH validated as second reference column; harness itself validated against a known-good implementation.

**References:** `frontiere_2017_crksph`.

---

## Phase 3 — Bonet–Lok Gradient Correction

**Goal:** Add the cheapest first-order consistency correction as the entry point into the "corrected kernel" family.

**Tasks:**
- [ ] Implement corrected kernel gradient enforcing $\sum_j (r_j - r_i)\otimes\nabla W_{ij} V_j = I$ (local $d\times d$ matrix solve per particle).
- [ ] Add correction-matrix condition-number logging into the harness's per-particle diagnostics.
- [ ] Run Phase 0 patch tests (expect exact linear-gradient reproduction) and convergence tests (expect first-order gradient, zeroth-order value unless value is also corrected — decide whether to also apply Randles–Libersky-style kernel renormalization, `randles_libersky_1996_...`, for the value itself).
- [ ] Boundary-region tests — expect this to be where the correction matrix becomes ill-conditioned; log failure modes.
- [ ] Run PDE benchmark suite, compare against Phase 1/2.

**Deliverable:** Corrected-gradient SPH operator, benchmarked, with documented ill-conditioning boundaries.

**References:** `bonet_lok_1999_variational_momentum`, `randles_libersky_1996_...`, optionally `chen_beraun_1999_cspm` and `liu_liu_2006_restoring_consistency` if extending correction to the kernel value itself.

---

## Phase 4 — MLS / RKPM Reproducing Kernel

**Goal:** Generalize Phase 3's single-order correction into an arbitrary-polynomial-order reproducing-kernel operator generator — this becomes the shared moment-matrix machinery reused in Phase 5 (LABFM).

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

**Tasks:**
- [ ] Implement MLS-WENO reconstruction (same interface-reconstruction structure as Phase 6, swap smoothness-indicator/weighting scheme).
- [ ] Run identical Phase 0 test set used for TENO — smooth-region order, shock-tube/Sedov/KH dissipation comparison.
- [ ] Direct head-to-head report: WENO vs. TENO dissipation at matched formal order (this is the comparison the TENO papers themselves make).

**Deliverable:** WENO-SPH reconstruction module; final comparison table across all phases (Standard SPH / CRKSPH / Bonet-Lok / MLS / LABFM / WENO-SPH / TENO-SPH) on the same harness.

**References:** `avesani_dumbser_bertaux_2014_mlswenosph`, follow-up "Investigations on a high order SPH scheme using WENO reconstruction" paper in your literature folder.

---

## Cross-cutting notes
- Every phase reuses Phase 0's harness unmodified — if a harness change is needed mid-plan, treat it as a harness-versioning event and re-run all prior phases before proceeding.
- Phases 3–5 share moment-matrix infrastructure; consider implementing that shared layer once in Phase 3/4 rather than duplicating in Phase 5.
- Phases 6–7 are architecturally separate (solver-level, not operator-library-level) — fine to parallelize with Phase 4/5 if useful.