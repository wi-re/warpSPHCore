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
> **Verification pass (2026-09-26)** — see the section of that name below.
> The *static* results hold up. The *PDE* results needed three corrections:
> (1) the PDE suite is **not** a standard-SPH column — 4 of 7 cases run
> CRKSPH; (2) momentum / angular-momentum / mass drift were not actually
> measured (fixed in the driver, versioning event #4); (3) the
> reference-metric orders (Gresho/KH/Sod/Sedov) are unreliable (finite
> reference; now replaced by exact-solution metrics for Sod/Sedov/Gresho —
> Sod and Sedov are clean first-order shock capturing, L1 ≈ 1). A **pre-Phase-4 harness versioning event** is now
> required (checklist below) before any new-operator phase starts.
>
> **Follow-up (same day):** most of that checklist is now done — incl. a
> **core CRK kernel bug** found and fixed (`correctGradientCRK` contracted
> `gradB` on the wrong axis since 2026-08-11; CRK linear gradients are
> machine-exact again, the static `crk` gradient column was re-baselined).
> Every CRKSPH PDE row produced before the fix is being re-run.
>
> **Where the numbers live:** `harness/REPORT.md` + `FINDINGS.md` (frozen
> static "before column"), `harness/pde/REPORT_pde.md` (PDE orders +
> conservation; local, git-ignored), `harness/pde/TGV_NOTES.md` (TGV floor +
> delta+ scheme study), `harness/pde/SEDov_NOTES.md` (shock metrics + Sedov
> diagnosis).

---

## Phase 0 — Convergence Test Harness

**Goal:** Build the evaluation infrastructure once, before touching any operator implementation, so every subsequent phase is judged against the same fixed tests.

**Status: DONE for Phases 1–3; NOT sufficient for Phases 4–7** (first pass
2026-09-22; PDE pass 2026-09-23; CI gate `tests/convergence/test_harness.py`
+ `tests/convergence/test_pde.py`). Two deliberate scope cuts vs the original
list (monomial degree cap 4, no Hessian probe) were fine for the p ≤ 1
reference columns but block Phases 4/5 — see the pre-Phase-4 checklist.

**Tasks:**
- [x] Particle set generators — `particle_sets.py`: densest-packing
  lattices (1D / 2D-hex / 3D-FCC, jitter, seeded) via the shipped
  `sampleDensestLattice`, open/periodic domains, interior/boundary masks.
  (Densest-packing rather than plain regular lattices — the shipped
  sampler is the one the real cases use.)
- [x] Test function library — `test_fields.py`: monomials $x^a y^b (z^c)$
  degree **0–4** (cap is 4, not 4–5) + one linear vector field, and smooth
  $C^\infty$ fields (commensurate sinusoids, wrapped Gaussian), each with
  analytic value/gradient/Laplacian. *Cap must rise for Phase 5 (8th–10th
  order patch tests).*
- [x] Operator probes — `operators.py`: value interpolation, gradient
  (Difference), **scalar Laplacian** (Brookshaw). *Hessian probe deferred —
  the core has no Hessian operator. Needed before Phase 4 (p ≥ 2 operators
  produce second derivatives, and the Brookshaw Laplacian is the largest
  baseline defect — `FINDINGS.md` §3).*
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
  detection. *Reads the d×d renorm eigenvalues only — must generalise to an
  arbitrary moment matrix for Phase 4.*
- [x] PDE-level benchmark suite — implemented as *Pass 2* in `pde/`
  (2026-09-23), over the real `warpSPH` frontend cases: TGV (analytic),
  Gresho, standing acoustic wave (linearWave, analytic), Sod, Sedov,
  Kelvin–Helmholtz — **plus a 7th leg, tgv-wc (delta+-SPH), registered
  2026-09-24.** 4-point resolution ladders at full simulated time. Each
  case runs its **frontend default scheme** unless `PDECase.scheme`
  overrides it (see Phase 1).
- [x] Conservation diagnostics — `conservation.py` (driver-side, pure
  torch): mass / KE / **total energy (KE+IE)** / momentum / angular
  momentum. **Fixed 2026-09-26:** the driver used to zero-fill the initial
  momentum / angular momentum and copy the final mass into the initial
  one, so `mass_drift` was 0 by construction and the reported final
  |p| / |L| were carried-through *initial* values (KH, Gresho), not drift.
  It now measures the true t=0 state (one-shot `extraData` hook, no second
  build) and reports vector drifts `momentum_drift_abs = |p_f − p_0|`,
  `angmom_drift_abs = |L_f − L_0|`. Rows run before the fix show those
  columns blank.
- [x] Reporting — `report.py` + `report_pde.py`: standardized CSV /
  markdown / log-log; `REPORT_pde.md` regenerable from `pde_rows.csv`
  without importing warp. Now shows the `scheme` column (backfilled
  2026-09-26 for the 16 rows that predate it).
- [x] **Finite-reference-bias-aware order fit** (2026-09-26) —
  `metrics.reference_corrected_order` fits $e(h) = C\,(h^p - h_{ref}^p)$
  (C profiled out, p scanned + golden-section refined; exact recovery on
  synthetic series, CI-tested); `REPORT_pde.md` gains a "plain vs corrected"
  table for every reference-metric case/column. A fit on the lower search
  bound is shown as `≤ 0.05` (read r²: low = model misfit, high = slower
  than any power law). **Validated against the exact metrics below it is
  NOT reliable:** Sod L1 corrected 1.07 vs exact 0.98 (works), Sedov L1
  corrected 0.23 vs exact **1.03** (badly over-corrects — the shock-position
  error changes sign between rungs, breaking the model's same-sign
  assumption). Kept as a diagnostic with that warning in the report; the
  exact-solution metric is the answer wherever it exists.
- [x] **Exact-solution metric for Sod, Sedov and Gresho** (2026-09-26) —
  Gresho is a *steady* Euler solution (exact field = initial v_φ(r) at all
  t — the standard Gresho metric; the registry had it as "no clean
  analytic"); for Sod and Sedov the
  frontend ships both exact solutions (`sodSolution.solve`,
  `SedovSolution.solution`); `pde_cases.PDECase.exact` scores every rung
  (incl. the finest) against them at the run's *own* `t_final`,
  volume-weighted (`field_error.particle_error_norms`), as new columns
  `error_l1_exact` / `error_l2_exact`. No reference bias, no t_final
  mismatch, 4-point fits.
- [x] **Kinetic-energy monotonicity check** (2026-09-26) —
  `conservation.ke_rebound` = largest KE rise above its running minimum /
  KE(0), computed from each run's KE trajectory; cases declared
  `ke_nonincreasing` (TGV, tgv-wc, Gresho — unforced steady / decaying
  flows) are flagged above 1 % in `REPORT_pde.md`. The final-state
  `ke_drift` could not see the CRKSPH Gresho spin-up where an early
  dissipative dip and the later gain partly cancel. Full ladders
  (2026-09-26): TGV, tgv-wc and CompSPH-Gresho rebound exactly 0 at every
  rung; CRKSPH-Gresho is flagged at nx = 48 / 64 / 96 (4.7 / 11.2 /
  15.1 %) — including nx=48, whose final drift (−3.9 %) looks healthy.
- [x] **Partial runs merge into the results CSV** (2026-09-26) —
  `run_pde.py --cases X` used to overwrite `pde_rows.csv` with case X
  only; it now replaces just those cases' rows (`report_pde.merge_rows`).
- [x] **Frozen-particle linear PDE leg** (2026-09-26) —
  `harness/run_frozen.py` → `REPORT_frozen.md`: linear advection (gradient
  probe) and diffusion (Laplacian probe) of a commensurate sinusoid on a
  fixed periodic jittered (0.3) lattice, RK4, exact solutions; any mode incl.
  registered external ones; divergence (|u| > 10³·|u₀|) recorded as the
  stability signal; CI smoke. First results (5-rung ladder N 576→9216):

  | | standard | crk | renorm / renormVal |
  |---|---|---|---|
  | advection | 1.05 (pairwise → 0.64) | **1.93** | **1.98** |
  | diffusion | 0.88 (pairwise → 0.47) | **2.00** | 1.72 |

  No mode diverges. CRK/renorm *diffusion* converges at ~2 although the
  static Brookshaw probe does not converge — **explained** by
  `harness/run_modal_probe.py` (below).
- [x] **Modal-error probe** (2026-09-26) — `harness/run_modal_probe.py`
  splits a static operator error on one Fourier mode into the modal part
  (projection onto the mode: effective diffusivity / phase-speed error) and
  the residual. Brookshaw: the residual is ~all of the pointwise error and
  grows ~1/dx (jitter noise, damped by diffusion at rate ~ν/dx²); the modal
  part converges at **2.00 (crk) / 2.10 (renorm)** / 0.88 → 0.45
  (standard). Gradient: modal 1.98 (crk/renorm), residual ~1.0. The modal
  orders predict the frozen-leg PDE orders for both operators — a cheap
  static proxy for Phase 4/5. `FINDINGS.md` §3 updated: "Brookshaw does
  not converge" is a pointwise statement about noise; judge new Laplacians
  on modal *and* residual error.
- [x] **Hessian probe + pluggable operator modes** (2026-09-26) —
  `operators.py`: `hessian` probe = grad-of-grad through each mode's own
  gradient (the baseline a p ≥ 2 operator is compared against), analytic
  Hessians by autodiff of each field's closed-form gradient
  (`test_fields.analytic_hessian`, exact for every field); and
  `register_mode(name, fn)` so a Phase 4/5 operator supplies its **own**
  interpolate/gradient/Laplacian/Hessian without editing the dispatch.
  Wired into the patch + resolve suites (additive rows; static re-run
  pending with the other re-baseline).
- [x] **General condition numbers** (2026-09-26) —
  `conditioning.matrix_condition_numbers(M, equilibrate=...)`: SVD κ₂ of any
  (N, n, n) batch (Phase 4/5 moment matrices, non-symmetric OK), optional
  Jacobi equilibration so monomial-basis h-scaling is not mistaken for
  geometric ill-conditioning.
- [x] **Monomial degree cap** — there is no hard cap in code
  (`monomial_fields(dim, degree_max)`); `--degree-max` just defaults to 4.
  Phase 5 passes `--degree-max 10`.

**Deliverable:** ✅ Harness runnable against any operator/solver satisfying a common interface, producing order-of-convergence tables + condition-number diagnostics + PDE benchmark plots. *(Extensions required for Phases 4–7: see the pre-Phase-4 checklist.)*

**References:** methodology drawn from `frontiere_2017_crksph`, `king_lind_2020_labfm`, `lind_rogers_stansby_2020_review` (survey of evaluation practice across the field).

---

## Phase 1 — Standard SPH Baseline

**Goal:** Establish the reference point everything else is measured against.

**Status: DONE (static + matched-scheme PDE, post CRK fix, 2026-09-26).**
The static column (`REPORT.md` mode `standard`) is a true standard-SPH
baseline. The Pass-2 PDE suite originally ran each case's frontend default
scheme (4 of 7 CRKSPH); it now has matched legs, so every case except TGV
has a standard-SPH (CompSPH) *and* a CRKSPH row on the same case / ladder /
metric / IC. Numbers below are from the post-fix re-run (CRK gradient fix;
conservation measured from t=0; sampler fix for TGV). Exact-solution
metrics where they exist; KH has only the finest-run reference.

| case | metric | standard (CompSPH) | CRKSPH | reading |
|---|---|---|---|---|
| linearWave | L2 vs analytic | 0.85 (err 5.3e-8 → 9.4e-9) | 1.05 (1.6e-8 → 1.8e-9) | CRKSPH 3–5× more accurate |
| sod | L1 / L2 vs exact Riemann | **0.98 / 0.53** | **0.96 / 0.56** (symmetric support; errors ~1.3× CompSPH's) | both first-order shock capturing |
| sedov | L1 / L2 vs exact self-similar | 0.43 / sat. (L1 9.6e-2 → 4.0e-2) | **1.03 / 0.59** (1.1e-1 → 1.3e-2) | CRKSPH converges, CompSPH barely |
| gresho | L1 vs exact (steady v_φ) | 0.44 (2.2e-1 → 1.3e-1; below the saturation flag's 2× range); KE −90 … −62 % | **non-monotone** 9.1e-2 → 2.9e-2 → 2.9e-2 → **3.4e-2**; KE **+7.3 / +13 %** at nx 64 / 96 | CompSPH over-dissipates; CRKSPH spins up (anomaly below) |
| kelvinHelmholtz | L2 vs finest run | 1.69 | sat. (7.6e-2 → 4.4e-2) | reference metric only; KE loss at nx=96: CompSPH −36 %, CRKSPH −3.6 % |
| tgv | L2 vs analytic | — | — | DFSPH: 0.68 (r² 0.96), floor-limited (TGV_NOTES) |
| tgv-wc | L2 vs analytic | — | — | δ⁺-SPH: 1.89 (r² 0.99), pre-floor |

Conservation (post-fix, measured from t=0): total energy 1e-16 – 4e-14 in
every CompSPH / CRKSPH row (except `sod-crk` under Gather, fixed —
Phase 2); linear momentum |Δp| ≤ 2e-15 in every compressible row, but
**not** in the incompressible legs: DFSPH TGV 2.1e-3 → 4.8e-4 (iterative
solve tolerance), δ⁺-SPH tgv-wc 7e-4 → **1.2e-2, growing with resolution**
(the particle shifting runs *without* the ALE-style `correctdrhodt` /
`correctdvdt` terms — both default False, no scheme enables them; an A/B
at nx=64 with both on **doubles** |Δp| (1.3e-2 → 2.7e-2) at unchanged L2
error, so in their current form they do not make shifting more
conservative — a conservative (pairwise-antisymmetric) δr·∇v form would
be needed; |ΔL| is not an invariant on the periodic TGV box and positions
are never re-wrapped, so ignore it there); angular momentum (open
question only for Gresho): CRKSPH 4.5e-3 → 3.9e-5, CompSPH 2.4e-2 →
4.0e-3 over the ladder — both converge, CRKSPH faster. Consistency check
on the CRK fix: the 1D rows (linearWave, Sedov) are **bit-identical**
before and after it (`gradB` is 1×1 in 1D, the transpose a no-op).

**Gresho anomaly — CRKSPH spins the vortex up (investigated 2026-09-26;
source located, fix open).** At nx=64 kinetic energy dips −1.6 % by
t≈0.5, then rises monotonically to +7.3 % at t=3 (+13 % at nx=96), total
energy exact; the exact-solution error *rises* at the finest rung while the
finest-run reference metric reported a clean "2.06" (every rung shares the
growing error). Present before the CRK gradB fix as well. Experiments
(nx=64 unless noted; scratch hooks, no frontend change kept):

| experiment | KE(t=3) | reading |
|---|---|---|
| baseline (van Leer limiter, C_l = C_q = 1) | +7.3 % | spin-up in the **mean** v_φ profile (+6.8 %), not noise (0.35 %) or radial (0.19 %) |
| CFL / 2 | +7.8 % | **not** the time integrator |
| AV × 0.5 / × 2 | +5.3 % / +9.0 % | scales with AV … |
| AV off (nx=48) | +199 %, vortex destroyed | … but without AV the scheme is violently unstable (CompSPH w/o AV: +37 %) |
| limiter φ forced 0 / 1 | −96 % / **+52 %** | the velocity reconstruction in Q strongly shapes it |
| Q gated on raw compression | −36 % (φ=1: −25 %) | spin-up gone, vortex now over-damped |
| velocity-gradient `.mT` removed | bit-identical | only x·Gx enters μ_ij — transpose-invariant, not a bug |
| AV gradient projected radial | +5.4 % | AV's tangential work is not the source |
| **power budget** (P_i = m_i v_i·Σ a_ij, t ∈ [0, 1.5]) | — | **viscosity is net dissipative (−0.37 KE₀/time); the *pressure* term injects +0.39 (φ=1: +0.52) KE₀/time**, concentrated in 0.22 < r < 0.38 |
| **pressure forces projected central** | −72 %, no spin-up | pressure power flips to −0.045 — **source confirmed** — but the vortex collapses (the non-central CRK components are needed for an accurate pressure gradient), so not a fix |

**Mechanism:** CRKSPH's pair pressure forces
(P_i+P_j)(∂W^R_ij − ∂W^R_ji) are *non-central* (W^R_ij ≠ W^R_ji — the
paper notes this). In an exact equilibrium vortex the pressure force is
radial and does no work on the azimuthal flow; here the tangential
components do net positive work in the annulus, i.e. they redistribute
angular momentum radially (|ΔL| stays small and converges, but the
profile shows inner-core and r≈0.26 bands too fast, r≈0.2/0.36 too
slow). The viscosity removes most of the injected energy; the residue is
the spin-up, and it grows with resolution. **Open question** — inherent to
CRKSPH (the paper shows no spin-up at 64², t=5, but also used Cullen-type
limiting), or an implementation deviation. **Follow-up (same day)**, every
change measured on Gresho *and* on the CRK-Sod ripple probe
(`pde/run_sod_ripple.py`, the paper's Sod: (400, 100) per tube, t = 0.15 —
per guidance, any CRK limiter / AV change must be checked for Sod ripple
and Sod must end up matching the paper's Fig. 4):

- ruled out: viscosity switches (CRKSPH evaluations use none, by design);
  the pair-force prefactor (code ≡ Eq. 38); single-support vs kernel-mean
  pair gradients (Gresho supports are exactly uniform at t=0 and vary
  0.8 % (std) by t=3 — second-order at most).
- **limiter constants are in the wrong units.** The paper's CRKSPH kernel
  has extent η_max = 4 in units of h with n_h = 1 (≈ 4 radial neighbours) —
  the same resolution as our n_h = 4, H = 4 Δx — so its
  (η_crit, η_fold) = (1/n_h, 0.2) = (1, 0.2) in r/h is **(0.25, 0.05)** in
  our r/H. The code uses (1/3, 0.2): every nearest-neighbour pair is
  limited (factor ≈ 0.83) and the fall-off is 4× too wide. In units of the
  nominal spacing the paper's values are kernel-independent (η_crit = 1 Δx,
  η_fold = 0.2 Δx), so they should be derived from n_h, not hard-coded.

| (η_crit, η_fold) | Gresho KE(3) / rebound | Sod plateau u std / max | Sod TV excess | Sod entropy err |
|---|---|---|---|---|
| (1/3, 0.2) current | +7.3 % / 0.112 | 8.0e-3 / 3.1e-2 | 0.076 | 1.9e-3 |
| (1/4, 0.2) | +27 % / 0.292 | 1.05e-2 / 4.4e-2 | 0.099 | 1.7e-3 |
| **(1/4, 0.05) paper** | **+0.7 % / 0.056** | 9.4e-3 / 3.9e-2 | 0.092 | **1.6e-3** |

  The paper-consistent constants cut the Gresho gain ~10× (still flagged:
  rebound 5.6 %) at the cost of ~17–22 % more Sod ringing (profile still
  visually the paper's Fig. 4 CRKSPH column; L1 unchanged, entropy
  better). **Not applied** — a limiter trade-off for a decision. Remaining
  spin-up after it: the non-central pressure work (source above).
  **Check that now catches it:** `ke_rebound` (Phase 0 / PDE driver).

**Tasks:**
- [x] Run Phase 0 harness against uncorrected SPH kernel
  interpolation/gradient (existing warpSPH frontend) — `REPORT.md`,
  `standard` column (Wendland2 default; CubicSpline spot check).
- [x] Confirm expected behavior — no exact reproduction even of constants
  (~5e-2 lattice-sum bias), O(h) smooth-field gradients (~0.95), degraded
  further at boundaries (4–15×): confirmed and documented in `FINDINGS.md`.
- [x] Run full PDE benchmark suite — 2026-09-23 full run, no divergences
  (numbers above; per-scheme, not standard SPH).
- [x] **Standard-SPH PDE legs on the four CRKSPH cases** — `<case>-std`
  (CompSPH), full ladders run 2026-09-26 (table above).
- [x] Re-run the full suite with the fixed conservation driver
  (2026-09-26; all drift columns filled).
- [x] Record baseline numbers — `REPORT_pde.md` + `results/pde_rows.csv`
  (now with the `scheme` column).

**Deep-dives (beyond the original task list):**
- TGV 0.69 root-caused (`TGV_NOTES.md` §1–3): a **non-converging error
  floor** = residual compressibility of the DFSPH velocity field
  (spurious (2,0)/(0,2) Fourier modes that cannot exist in a div-free
  flow), not the integrator. User-scoped as the standing *vd+ps* problem
  (out of benchmark scope).
- Sedov/Sod shock metrics added 2026-09-25 (`field_error.py`: L1 area
  norm + sub-cell shift-aligned L2): judge shock cases on L1 / full-grid
  L2, not common-cells raw L2 (`SEDov_NOTES.md`). *Caveat added
  2026-09-26: those orders are still measured against the nx=1600 run
  and carry the finite-reference bias (Verification pass, finding 3).*

**Anomalies found 2026-09-26:**
- **Gresho KE gain — confirmed, open:** a secular CRKSPH spin-up, not a
  transient or a metric artifact (see the anomaly paragraph above).
- **linearWave "float64 floor" is not supported:** A = 1e-6, so the
  relative velocity error is ~2e-3 … 2e-2, far above round-off. All four
  rungs run exactly 999 steps (fixed dt), so a temporal floor is the more
  likely cause of the declining pairwise order (untested).
- **TGV particle count — fixed** (frontend sampler round-off, warpSPH
  `bb88b46`; see the checklist).

**Deliverable:** ✅ static baseline report — the "before" column. ✅
matched-scheme PDE baseline (standard SPH vs CRKSPH on every compressible
case).

---

## Phase 2 — CRKSPH Baseline (existing frontend)

**Goal:** Since CRKSPH is already implemented in the compressible frontend, use it as the second reference point and validate it against the harness rather than implementing it fresh.

**Status: static DONE (with one open gradient question); PDE numbers
EXIST** — the corrected reading of the Pass-2 suite is that CRKSPH is the
default scheme of Gresho, KH, linearWave and Sedov, so their Phase-1-table
rows *are* the CRKSPH PDE column. What is missing is the standard-SPH
counterpart (Phase 1 task) and a CRKSPH run of Sod / TGV-like cases.

**Tasks:**
- [x] Run Phase 0 harness against the existing CRKSPH operators
  (interpolation + gradient) — `REPORT.md`, `crk` column (CRK apparent
  volume per the validated frontend usage).
- [x] Verify exact linear-field reproduction (patch test to degree 1) —
  **values only**: constant + linear interpolation machine-exact, ordered
  **and** jittered, interior **and** boundary band, CI-gated
  (`test_harness.py`).
- [x] **Resolve the CRK linear-gradient residual — it was a core bug,
  FIXED 2026-09-26.** `crk/kernel.py` `correctGradientCRK` (and its JVP)
  contracted `gradB` — stored [derivative, component] by `crk_terms.py` —
  on the wrong axis: `matmul(wp.transpose(gradBi), x_ij)` instead of
  `matmul(gradBi, x_ij)`. Regression from 5ca1882 (2026-08-11), which
  replaced a *correct* explicit loop with `matmul` to fix an adjoint bug and
  transposed it. `gradB` is symmetric on a regular lattice, so only
  jittered/boundary particles saw it. After the fix CRK linear gradients
  are 7e-16 (interior) / 1.4e-15 – 6.3e-15 (wall band; was 0.12 – 0.25), and
  CRK becomes the best mode for degree 2–4 gradients in the band.
  Verified: gradcheck gate 42/42, CRK gradcheck script, 293 forward-mode /
  CRK tests, the tier-2 CRK JVP spike (reference updated to match). New
  guard: `run_baseline.py --smoke` asserts CRK linear-gradient exactness
  (< 1e-10) on a jittered open set, interior + boundary. Static re-run:
  only the 134 `crk`×`gradient` rows changed (every other row
  byte-identical); `REPORT.md` / `FINDINGS.md` updated (versioning event
  #5). Lesson recorded in `docs/lessons_learned.md`. **Every CRKSPH
  frontend result produced 2026-08-11 → 09-26 used the buggy gradient.**
- [~] Verify exact conservation of mass/momentum/energy to machine
  precision; measure angular momentum drift — **total energy is already
  demonstrated** in the existing CRKSPH rows (drift 1e-16 … 3e-14 on
  Gresho / KH / linearWave / Sedov); momentum / angular momentum need the
  re-run with the fixed driver (the old columns were initial values).
  Lowest-rung check 2026-09-26 (Verification pass, finding 2): linear
  momentum conserved to round-off (KH |Δp| 6e-17, Gresho 9e-17);
  Gresho angular momentum drifts +7.8% at nx=32, converging with
  resolution (~0.3% at nx=96 by the old columns). Full-ladder re-run
  still needed for the table.
- [x] Run PDE benchmark suite, compare dissipation/order against Phase 1 —
  done 2026-09-26 with matched legs (Phase 1 table): CRKSPH is 3–5× more
  accurate on linearWave, converges on Sedov where CompSPH barely does
  (L1 1.03 vs 0.43), dissipates ~10× less on KH, matches CompSPH on Sod —
  but **spins up the Gresho vortex** (open anomaly, Phase 1).
- [x] Document gaps between harness results and paper-reported behavior —
  `FINDINGS.md` (static); the delta+ leg's floor analysis is the
  de-facto "what does a corrected scheme still fail at" study.

**Deliverable:** ✅ static reference column (values); ⬜ gradient
exactness question; ⬜ matched-scheme PDE comparison.

**References:** `frontiere_2017_crksph`.

---

## Phase 3 — Bonet–Lok Gradient Correction

**Goal:** Add the cheapest first-order consistency correction as the entry point into the "corrected kernel" family.

**Status: DONE (static)** — with the key scoping fact discovered up front:
the correction machinery **already ships in the core**
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
  `renormVal` columns: exact linear-gradient reproduction (interior *and*
  boundary band); value only zeroth-order-exact under `renormVal` (linear
  values ~2e-3, not exact — only the 0th moment is corrected).
- [x] Boundary-region tests — open-domain suites. *Correction to the
  earlier wording:* the tested regime is **well-conditioned**, not an
  ill-conditioning boundary — κ ≈ 1.0–1.34 interior, max 2.3–4.7 in the
  wall band; identity fallback on ≤ 0.7% of band rows (jitter ≥ 0.3, 12
  neighbours). Free surfaces, corners and near-empty supports were not
  tested; that is where Phase 4's p ≥ 2 conditioning problems will show.
- [ ] Run PDE benchmark suite, compare against Phase 1/2 — not done (no
  Bonet–Lok PDE scheme in the frontend; the frozen-particle PDE leg from
  Phase 0 would sidestep this).

**Deliverable:** ✅ (static) Corrected-gradient SPH operator, benchmarked, conditioning characterised in the tested regime. PDE-level comparison pending a frontend scheme or the frozen-particle leg.

**References:** `bonet_lok_1999_variational_momentum`, `randles_libersky_1996_...`, optionally `chen_beraun_1999_cspm` and `liu_liu_2006_restoring_consistency` if extending correction to the kernel value itself.

---

## Verification pass (2026-09-26)

A consistency check of Phases 0–3 against the CSVs, notes and code. The
static results reproduce (`FINDINGS.md` tables, disorder-probe verdict,
TGV 0.69, tgv-wc 1.90, Sedov L1/aligned numbers). Findings:

1. **PDE suite is per-scheme, not standard SPH.** Gresho, KH, linearWave
   and Sedov default to `CRKSPH`; Sod to `CompSPH`; TGV to DFSPH. The
   `scheme` CSV column was blank for the pre-column rows, which hid this.
   → Phase 1/2 tasks rewritten; `scheme` backfilled and shown in the report.
2. **Conservation drift was not measured.** Initial momentum / angular
   momentum were hard-coded to 0 and initial mass copied from final.
   → Fixed in `run_pde.py` (versioning event #4). Smoke re-runs with the
   fixed driver (lowest rung) confirm the old "final" values were the
   initial ones:
   - linearWave (CRKSPH, nx=100): |p_0| = 5.0099e-13 (old "final":
     5.010e-13); true |Δp| < 1e-10.
   - KH (CRKSPH, nx=32): |p_0| = |p_f| = 0.2266 (old "final": 0.2266);
     true |Δp| = 5.9e-17 — momentum conserved to round-off.
   - Gresho (CRKSPH, nx=32): |p_0| = 2e-19, |Δp| = 8.6e-17; angular
     momentum L_0 = 0.0586 → L_f = 0.0632 (**+7.8%**). The old "final"
     values at nx=32/48/64/96 (0.0632 / 0.0598 / 0.0591 / 0.0588) approach
     L_0, i.e. a *resolution-convergent* angular-momentum error (~7.8% /
     2% / 0.8% / 0.3%) — the expected CRKSPH behaviour (exact linear
     momentum + energy, not angular momentum).
   - KE / total-energy drifts reproduce the old values exactly (KH −0.2017,
     Gresho −0.3398, linearWave −0.02867), so switching the KE source from
     the trajectory to the measured state is consistent.
3. **Finite-reference bias.** The `reference` metric compares against the
   finest run, only 1.5–2× finer than the last fitted rung. For a
   systematic error $e(h) = C(h^p - h_{ref}^p)$ the fitted slope reads:

   | ladder | true p = 0.5 | p = 1 | p = 2 |
   |---|---|---|---|
   | Sod/Sedov (200/400/800 vs 1600) | 1.07 | **1.40** | 2.20 |
   | Gresho/KH (32/48/64 vs 96) | 1.68 | **1.98** | 2.66 |

   So the plain reference-metric orders over-read systematic errors.
   **Correction (same day, after scoring against exact solutions):** the
   model itself is not reliable either — it recovers Sod (L1 plain 1.46 →
   corrected 1.07; exact 0.98) but over-corrects Sedov (plain 0.92 →
   corrected 0.23; exact **1.03**), because Sedov's shock-position error
   changes sign between rungs. The earlier conclusions drawn from it
   ("Gresho < 0.5", "Sedov below first order") are withdrawn. Resolution:
   exact-solution metrics for Sod, Sedov and Gresho (Phase 0); only KH
   still depends on a finite reference.
4. **CRK linear exactness is values-only** (Phase 2 task split).
5. **`computeRenormalizationMatrices` is not a generic moment-matrix
   builder.** It is the d×d covariance $\sum_j (r_j-r_i)\otimes\nabla W_{ij}V_j$
   with closed-form 1×1/2×2/3×3 pseudo-inverses (`src/warpSPHCore/pinv/`).
   An MLS basis of order p=2/3 needs 6×6/10×10 (2D) or 10×10/20×20 (3D).
   What *is* reusable: the neighbour-loop / operation-dispatch pattern and
   CRK's (A, B, ∇A, ∇B) structure as the p=1 special case.
   → Cross-cutting note corrected.
6. **The PDE suite cannot currently discriminate orders above ~2.** Both
   TGV legs sit on floors unrelated to the spatial operator (DFSPH
   compressibility; δ⁺ PST accumulation ~6e-3); TGV/tgv-wc/linearWave run
   fixed step counts; shock cases are first-order by construction;
   reference cases are biased (3). → frozen-particle PDE leg added to
   Phase 0.
7. **Phase 6 premise:** there is no Riemann solver between particles in
   the frontend (the only Riemann code is the exact Sod solution used for
   validation). → Phase 6 task rewritten.
8. Minor numeric corrections: tgv-wc clean ladder is
   5.48e-2 / 2.47e-2 / 1.63e-2 / **6.60e-3**, fit **1.90** (not "≈2.0");
   KH "saturated" is a borderline threshold hit; open anomalies listed
   under Phase 1.

### Pre-Phase-4 harness versioning event (required)

Do these, then re-run Phases 1–3 (static + PDE) before Phase 4 starts:

- [x] Measure t=0 conserved quantities; report vector drifts (2026-09-26).
- [x] Record + display `scheme` per row; backfill old rows (2026-09-26).
- [x] Bias-aware order fit for the `reference` metric; Sod **and Sedov**
  → exact-solution metric (2026-09-26).
- [x] Standard-SPH legs `gresho-std` / `kelvinHelmholtz-std` /
  `linearWave-std` / `sedov-std` (CompSPH) + `sod-crk` (CRKSPH) registered
  as derived registry entries (same case/ladder/metric/IC, scheme swapped);
  all five smoke-run without divergence (2026-09-26).
- [x] Frozen-particle linear PDE leg (`run_frozen.py`).
- [x] Hessian probe + pluggable operator modes (`register_mode`).
- [x] `conditioning.matrix_condition_numbers` for any n×n.
- [x] Degree cap: none in code; `--degree-max`.
- [x] CRK linear-gradient residual — core bug, fixed (Phase 2).
- [x] **Re-run** (2026-09-26): full PDE ladders for the four CRKSPH cases
  (post-fix), the five matched-scheme legs, TGV / tgv-wc (drift columns,
  sampler fix); static sweep for the additive Hessian rows (320 rows, no
  existing row changed; grad-of-grad Hessian does not converge in any mode
  — Hessian of x² is off by ~2 against a true value of 2).
- [x] `sod-crk` total-energy drift −7.3e-6 — **root-caused**: CRKSPH's
  compatible-energy update needs symmetric pair interactions, and Sod's
  default `supportMode='Gather'` is asymmetric (CRKSPH+Gather −7.3e-6;
  CRKSPH+KernelMeanSymmetric +2.5e-16; CompSPH exact under both). The leg
  now overrides `supportMode='KernelMeanSymmetric'` (the mode every
  CRKSPH-default case uses) via a new `PDECase.spec` field. Frontend
  (warpSPH `6574d01`): `crkSPH_step` now warns (`CRKSupportWarning`, once
  per mode) for any non-kernel-mean support. Frontiere et al. 2017 *define*
  the CRKSPH pair kernel as the kernel mean (Eq. 8), so Gather is outside
  the method; a Gather-conserving variant would be derivation work (their
  Eq. 60 keeps both a_ij and a_ji in general) and is not pursued. Pinning
  just the f_ij balance term to the kernel-mean support does *not* remove
  the drift (the Owen adaptive-support solve is the other consumer). Side
  finding: `geometry/sdfFunctionality/implicitFunctions.py` installs a
  blanket `warnings.filterwarnings("ignore")` at import, silencing **every**
  warning in any process that loads it — **removed** (warpSPH `5d2ce17`): it
  hid exactly one warning, a deprecated `.T` on a 1-D tensor in
  `sdHorseshoe` (fixed, same value); representative Sod / Gresho / tgv-wc /
  dambreak runs emit no warnings without it; `sdPolygon` export fixed on
  the way (`sdStar` / `sdRing` remain broken: `device=` passed to numpy).
- [x] Brookshaw static-vs-diffusion discrepancy — explained (modal probe).
- [x] TGV 49²/97² particles at nx=48/96 — **root-caused, frontend bug**:
  `warpSPH/src/warpSPH/sample/regular.py:112` (`buildPointCloud`'s default
  nx-driven branch) does `ceil(l / dx)` with no tolerance; float64 round-off
  gives l/dx = 48.00000000000001 → 49 cells. The explicit-dx branch of the
  same function already uses `ceil(l/dx - 1e-3)` for exactly this reason;
  **fixed in warpSPH `bb88b46`** with the same tolerance (+ a regression
  test over L × nx × dtype × CPU/CUDA — the round-off only shows on CUDA;
  float32 L=3 at nx=100/200 was hit too). TGV / tgv-wc re-run with the
  corrected counts (48², 96²).

---

## Phase 4 — MLS / RKPM Reproducing Kernel

**Goal:** Generalize Phase 3's single-order correction into an arbitrary-polynomial-order reproducing-kernel operator generator — this becomes the shared moment-matrix machinery reused in Phase 5 (LABFM).

**Status: NOT STARTED.** (Explicit non-goal of the first pass, `harness/PLAN.md`.) Blocked on the pre-Phase-4 checklist above.

**Tasks:**
- [ ] Implement generic moment-matrix builder for a chosen monomial basis (parametrized consistency order $p$), following MLS/RKPM formalism. **New code** — `computeRenormalizationMatrices` is d×d-only (Verification pass, finding 5). Needs a batched small dense solve (n = 3…20) with explicit regularisation / rank handling, float64-capable, and its own reverse-mode story (see the pinv2x2 adjoint history in `warpier_core.md`).
- [ ] Implement kernel correction function $C(x; x-x_j)$ so the kernel itself (not just gradient) reproduces polynomials to order $p$.
- [ ] Derivative operators from the corrected kernel: gradient **and Laplacian / Hessian** (p ≥ 2), measured through `register_mode` against the `hessian` (grad-of-grad) and Brookshaw baselines. The Brookshaw Laplacian does not converge in the *static* probes (`FINDINGS.md` §3), but does at ~2 in the frozen diffusion leg for CRK (1.72 renorm) — settle which error component matters (Phase 0 frozen-leg note) before making "a converging Laplacian" the headline target; the static pointwise norm and the PDE-level error disagree.
- [ ] Note non-symmetry of resulting kernel ($W_{ij} \neq W_{ji}$) — decide whether to carry forward a naive (non-conservative) MLS operator as a standalone mode, or route straight into a CRKSPH-style conservative reformulation reusing this moment matrix. *(Phase 2 settled: the CRK linear-gradient residual was a bug, not a cost of the CRK form — the corrected-kernel derivative is exact to degree 1. Note that CRKSPH's* momentum equation *uses an antisymmetrised pair form for conservation, which is a separate question from the operator's exactness and should be evaluated the same way here.)*
- [ ] Run Phase 0 harness at multiple orders $p = 1, 2, 3$: patch tests to degree $p$, convergence-rate tests, condition-number tracking vs. $p$ and neighbor count (expect conditioning to worsen with $p$ on disordered particles). Include free-surface / corner / thin-support sets, which Phase 3 did not probe. Cross-check p=1 against the `crk` column (should match CRK values exactly).
- [ ] Run the frozen-particle PDE leg (primary order evidence). Full PDE suite only if a frontend scheme uses the operator; if non-conservative, explicitly report conservation-diagnostic failures (expected) as a documented tradeoff vs. Phase 2.

**Deliverable:** Order-parametrized reproducing-kernel operator generator, validated across $p=1..3$, with conservation-vs-accuracy tradeoff documented.

**References:** `liu_jun_zhang_1995_rkpm`, `dilts_1999_mlsph`, `dilts_2000_mlsph2`.

---

## Phase 5 — LABFM

**Goal:** Push the moment-matching idea from Phase 4 to arbitrary order using anisotropic basis functions and compact stencils, decoupled from kernel-summation framing.

**Status: NOT STARTED.**

**Tasks:**
- [ ] Implement anisotropic basis function (ABF) construction and the local linear system solved per stencil (reuse Phase 4's moment-matrix infrastructure — the *new* n×n builder, not the d×d renorm path).
- [ ] Implement one-sided/boundary stencil handling for incomplete support.
- [ ] Harness prerequisites specific to this phase: monomial fields to degree ~10; ladder design that stays above the float64 floor at 8th–10th order (few, coarse rungs; smooth fields with O(1) high derivatives); Laplacian probe on the LABFM operator itself.
- [ ] Run Phase 0 harness at increasing order (target: reproduce paper's reported 4th order at ~25 neighbors in 2D, up to 8th–10th order at larger stencils) — this is the most discriminating test of harness correctness, since the paper gives concrete stencil-size/order pairs to match.
- [ ] Stability check: add hyperviscosity option and confirm it stabilizes hyperbolic test PDEs per `king_lind_2020_labfm`.
- [ ] Run the frozen-particle PDE leg at matched order vs Phase 4; compare compute cost vs. order against Phase 4's correction-matrix approaches. (The current PDE suite cannot show orders > ~2 — Verification pass, finding 6.)

**Deliverable:** Arbitrary-order LABFM operator, cross-validated against the paper's specific stencil-size/order table.

**References:** `king_lind_2020_labfm`, `king_lind_2021_labfm_isothermal`.

---

## Phase 6 — TENO Reconstruction (Riemann-SPH layer)

**Goal:** Add high-order shock-capturing reconstruction as a separate solver-level module (not part of the core operator library), per the earlier architectural note.

**Status: NOT STARTED.** Note: the dehnen2012 replication's Phase 6
delivered the *prerequisite* shock-capturing validation (C&D 2010 switch
+ R&H 2012 conductivity as validated frontend options) — see
`dehnen2012.../REPORT.md` §6. **Scope correction (2026-09-26):** the
frontend has no pairwise Riemann solver (CRKSPH uses a "Riemann-like"
pseudo-viscosity; the only Riemann solver is the exact Sod solution used
for validation), so a Riemann-SPH base scheme must be built first.

**Tasks:**
- [ ] **Implement a first-order Riemann-SPH (Godunov-SPH) base scheme** in the compressible frontend (pairwise interface states, approximate Riemann solver — e.g. HLLC or an exact solver generalised from `sodSolution.py`). This is the zero-reconstruction baseline TENO/WENO are compared against.
- [ ] Implement MLS-based local polynomial reconstruction at particle-pair interfaces, interface position $\overline{r}_{ij} = (h_j r_i + h_i r_j)/(h_i + h_j)$ (reuse Phase 4's n×n moment builder).
- [ ] Implement TENO smoothness indicators / stencil selection and blending weights.
- [ ] Couple reconstruction to the Riemann-SPH scheme for left/right state resolution.
- [ ] Run Phase 0 smooth-region convergence tests (target: 4th-order-class reconstruction accuracy) and shock-tube/Sedov/KH tests for non-oscillatory behavior and reduced dissipation vs. WENO of matched order. *Use the exact-solution metrics (Sod, Sedov); compare within the Riemann-SPH family (the current Sod baseline is CompSPH and Sedov is CRKSPH — not like-for-like).*

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

**Deliverable:** WENO-SPH reconstruction module; final comparison table across all phases (Standard SPH / CRKSPH / Bonet-Lok / MLS / LABFM / WENO-SPH / TENO-SPH) on the same harness — **with the scheme stated per row** (the Pass-2 suite showed that "per-case default scheme" is not a column).

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
  registered as the 7th PDE case — the first "after" column vs standard
  TGV 0.69. History: ~1.69 (r² 1.000) at the 1/8 default shift; current
  leg (full-strength `sun2017DeltaSPH` + `shuffleEq7=False`) fits **1.90
  (r² 0.995)** on the clean ladder 5.48e-2 / 2.47e-2 / 1.63e-2 /
  6.60e-3, flattening to ~0.4 order at 96→128 where the ~6.3e-3 per-step
  accumulation floor takes over (so 1.90 is pre-floor). `TGV_NOTES.md`
  §4–6: higher-order RK does NOT lift the (1/8-shift) ~8.5e-3 floor; the
  floor **is the particle shifting** (PST causal test — velocity error
  decreases monotonically with shift strength; full-strength
  `sun2017DeltaSPH` is 30% lower on velocity, 4.5× better on volume).
  User decision 2026-09-24: frontend default shift stays 1/8 (free-surface
  balance); the harness leg runs full strength via a `PDECase.scheme`
  override, and the `shuffleEq7` knob removes the disordered-IC penalty.
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
  unchanged); (4) **conservation driver fix + `scheme` column in the
  report** 2026-09-26 — t=0 state now measured, new
  `momentum_drift_abs` / `angmom_drift_abs` columns; error columns
  unchanged; the full PDE suite must be re-run to fill the drift columns;
  (5) **core CRK gradient fix** 2026-09-26 — `crk`×`gradient` static rows
  re-baselined (all other rows byte-identical), all CRKSPH PDE rows re-run;
  (6) additive 2026-09-26: `hessian` probe rows, exact-solution and
  bias-corrected PDE columns, matched-scheme PDE legs, frozen-particle leg
  (`REPORT_frozen.md`) — no existing number changes.
  **Pending:** the pre-Phase-4 event (checklist under Verification pass).
- Phases 3–5 share moment-matrix infrastructure; implement that shared
  layer once in Phase 4 rather than duplicating in Phase 5.
  *(Corrected 2026-09-26: the shipped `computeRenormalizationMatrices` is a
  d×d covariance with closed-form ≤3×3 pseudo-inverses and cannot hold an
  order-p moment matrix. Phase 4 builds a new n×n builder; reuse from the
  core is the neighbour-loop / dispatch pattern and CRK's p=1 structure.)*
- Phases 6–7 are architecturally separate (solver-level, not operator-library-level) — fine to parallelize with Phase 4/5 if useful, but both now start with building a Riemann-SPH base scheme.
- Every PDE-level comparison must state the scheme per row and compare
  like-for-like (same scheme family, only the operator varied).
