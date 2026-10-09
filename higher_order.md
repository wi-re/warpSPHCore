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
> **Frontend sync (2026-10-09).** Three things in warpSPH moved after the
> 09-26 text below was written:
> (1) **CRKSPH limiter default changed** — `(eta_crit, eta_fold) = (1/n_h,
> 0.2/n_h)`, derived per step (warpSPH `aa40291`, user-decided 2026-10-01; the
> "wrong units" finding below is *fixed*); CRK run-to-run nondeterminism was
> also fixed (`cfb03a6`, `segment_sum` instead of atomic scatter). Every
> CRKSPH PDE row dated 09-26 or earlier used the old (1/3, 0.2) limiter —
> re-run 2026-10-09 (done; see the Phase 1 table; the 09-26 CSV/report are
> kept as `results/pde_rows_pre-limiter-default_2026-09-26.csv` and
> `REPORT_pde_pre-limiter-default_2026-09-26.md`). Net effect: linearWave
> order 1.05 → 1.91 and KE loss ~10³× smaller; Sod / Sedov / KH essentially
> unchanged; Gresho spin-up reduced (~35 %) but not removed. (2) **The Gresho spin-up is closed
> as intrinsic, not open** (`c9a5133`, warpSPH `CRKSPH_LIMITER_PLAN.md`): a
> slowly converging pressure pump the viscosity cancels; cusps, pair weights
> and support consistency were ruled out; no fix. `ke_rebound` stays as the
> detector and a frontend regression guard. (3) **The frontend now has
> pairwise Riemann solvers** (`modules/riemann`: Acoustic / PVRS / TRRS / TSRS
> / Adaptive / HLLC; `modules/godunov`: simplified GSPH + Inutsuka 2002, per
> warpSPH `GODUNOV_SPH_PLAN.md`) — the Phase 6 premise below is partly
> obsolete. Phases 4–7 are still not started.
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
  15.1 % at the old limiter; **3.2 / 8.7 / 11.7 %** at the n_h-derived default,
  re-run 2026-10-09) — including nx=48, whose final drift (−3.9 % / −5.8 %)
  looks healthy.
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
| linearWave | L2 vs analytic | 0.85 (err 5.3e-8 → 9.4e-9) | **1.91** (1.1e-8 → 2.1e-10; was 1.05 at the old limiter) | CRKSPH 5× (nx=100) … **44×** (nx=800) more accurate; KE loss −2.6e-3 → −4.5e-6 (was −2.9e-2 → −3.4e-3) |
| sod | L1 / L2 vs exact Riemann | **0.98 / 0.53** | **0.97 / 0.58** (symmetric support; errors ~1.3× CompSPH's; unchanged by the limiter default) | both first-order shock capturing |
| sedov | L1 / L2 vs exact self-similar | 0.43 / sat. (L1 9.6e-2 → 4.0e-2) | **1.10 / 0.71** (1.1e-1 → 1.1e-2; was 1.03 / 0.59) | CRKSPH converges, CompSPH barely |
| gresho | L1 vs exact (steady v_φ) | 0.44 (2.2e-1 → 1.3e-1; below the saturation flag's 2× range); KE −90 … −62 % | **non-monotone** 8.5e-2 → 3.0e-2 → 2.8e-2 → **3.6e-2** (fit 0.75, r² 0.46); KE **+4.5 / +9.6 %** at nx 64 / 96 (old limiter +7.3 / +13 %); `ke_rebound` 3.2 / 8.7 / 11.7 % at nx 48 / 64 / 96, still flagged | CompSPH over-dissipates; CRKSPH spins up, reduced but **not** cured (intrinsic — anomaly below) |
| kelvinHelmholtz | L2 vs finest run | 1.69 | sat. (8.0e-2 → 5.7e-2; was 7.6e-2 → 4.4e-2) | reference metric only; KE loss at nx=96: CompSPH −36 %, CRKSPH −3.5 % |
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
question only for Gresho): CRKSPH 4.0e-3 → 1.8e-5 (n_h-derived limiter;
4.5e-3 → 3.9e-5 at the old one), CompSPH 2.4e-2 → 4.0e-3 over the ladder —
both converge, CRKSPH faster. Consistency check
on the CRK fix: the 1D rows (linearWave, Sedov) are **bit-identical**
before and after it (`gradB` is 1×1 in 1D, the transpose a no-op).

**Gresho anomaly — CRKSPH spins the vortex up (investigated 2026-09-26;
source located, fix open).** *The frontend follow-up — limiter
parameter-stability map, formulation variants, spin-up source, acceptance
criteria — lives in warpSPH `CRKSPH_LIMITER_PLAN.md` (branch `acsph-plan`);
this section keeps the harness-side record.* At nx=64 kinetic energy dips −1.6 % by
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
limiting), or an implementation deviation. **RESOLVED 2026-10-01 (frontend,
`c9a5133`): closed as intrinsic** — a slowly converging pressure pump that the
viscosity cancels; cusps, pair weights and support consistency ruled out; the
limiter constants were changed to the n_h-derived default (below) but
modulate, not cure, it. The text from here to "Check that now catches it"
is the 09-26 investigation record, kept as history; the numbers in its
tables are at the old (1/3, 0.2) limiter. **Follow-up (same day)**, every
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
- **Derivation + full compressible sweep (2026-09-26).**
  `pde/derive_crk_limiter.py` measures Δx (from particle volume) and H on
  each case's sampled particles and derives η_crit = Δx/H, η_fold =
  0.2 Δx/H (the paper's CRKSPH constants in the frontend's r/H units;
  kernel-independent in units of spacing — the D&A-σ reading of h is
  reported alongside). All 14 compressible cases measure Δx/H = 0.245 –
  0.2505 (nearest-neighbour η equal to it), so derived ≈ (0.25, 0.05) in
  1D, (0.246, 0.049) in 2D. `pde/run_crk_limiter_sweep.py` runs every case
  under CRKSPH with current (1/3, 0.2) vs derived:

  | case | reference metric | current → derived |
  |---|---|---|
  | Kidder | L1 ρ vs exact | 2.96e-5 → **5.33e-6 (×0.18)** |
  | linearWave | L2 v vs analytic / KE loss | 6.7e-9 → **3.0e-9** / −1.36 % → **−0.03 %** |
  | Yee vortex | L1 v vs exact (core) / KE loss | 2.90e-3 → 2.73e-3 / −1.9 % → −1.6 % |
  | Sod | L1 ρ / entropy / plateau u std / TV excess | 2.95e-3 → 2.90e-3 / **−14 %** / **+18 %** / **+22 %** |
  | Sedov | L1 ρ vs exact | 6.09e-2 → 6.63e-2 (**+9 %**) |
  | Noh | L1 ρ vs exact | 0.106 → 0.110 (+3 %) |
  | Gresho | KE(3) / ke_rebound / L1 v | +7.3 % → +4.8 % / 0.112 → 0.091 / +10 % |
  | KH | KE loss / rebound | −5.8 % → −5.6 % / 2.1e-3 → 7.2e-4 |
  | hydrostatic | spurious max |v| | 5.7e-7 → 6.3e-7 |
  | Woodward–Colella, triple point, RT, shearing Noh, 2D Sod | no reference | stable, energy at round-off, final-state Lagrangian Δρ 0.02 – 3.7 % |

  Reading: the derived constants remove the extra nearest-neighbour
  damping — large gains on the smooth / weakly nonlinear cases (Kidder,
  linearWave, Yee), a cost on the shock cases (Sod ringing +18–28 %,
  Sedov +9 %, Noh +3 %). **Gresho is threshold-sensitive:** η_crit = 0.25
  gave +0.7 %, the per-case derived 0.2458 (the measured 2D nearest-
  neighbour η after the support solve) +4.8 % — a 2 % shift in the
  threshold changes the spin-up ~7×, so the limiter constants modulate
  the spin-up but are not a robust fix for it. Still a decision, now with
  the full trade-off on the table.
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
- [x] Verify exact conservation of mass/momentum/energy to machine
  precision; measure angular momentum drift — **total energy is already
  demonstrated** in the existing CRKSPH rows (drift 1e-16 … 3e-14 on
  Gresho / KH / linearWave / Sedov); momentum / angular momentum need the
  re-run with the fixed driver (the old columns were initial values).
  Lowest-rung check 2026-09-26 (Verification pass, finding 2): linear
  momentum conserved to round-off (KH |Δp| 6e-17, Gresho 9e-17);
  Gresho angular momentum drifts +7.8% at nx=32, converging with
  resolution (~0.3% at nx=96 by the old columns). **Full ladder done
  2026-10-09** (n_h-derived limiter): |ΔL| = 4.0e-3 / 7.4e-4 / 1.8e-4 /
  1.8e-5 at nx 32 / 48 / 64 / 96 (resolution-convergent); |Δp| ≤ 1e-16 and
  total energy ≤ 1e-13 in every CRKSPH row.
- [x] Run PDE benchmark suite, compare dissipation/order against Phase 1 —
  done 2026-09-26 with matched legs, CRKSPH rows re-run 2026-10-09 at the
  n_h-derived limiter (Phase 1 table): CRKSPH is 5–44× more accurate on
  linearWave (order 1.91), converges on Sedov where CompSPH barely does
  (L1 1.10 vs 0.43), dissipates ~10× less on KH, matches CompSPH on Sod —
  but **spins up the Gresho vortex** (+4.5 / +9.6 % KE at nx 64 / 96, down
  from +7.3 / +13 %; intrinsic per the frontend close-out).
- [x] Document gaps between harness results and paper-reported behavior —
  `FINDINGS.md` (static); the delta+ leg's floor analysis is the
  de-facto "what does a corrected scheme still fail at" study.

**Deliverable:** ✅ static reference column (values); ✅ gradient
exactness question (core bug, fixed); ✅ matched-scheme PDE comparison.
⬜ Full-ladder angular-momentum table (the `[~]` conservation task).

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
  existing row changed; *the grad-of-grad Hessian "does not converge in
  any mode — off by ~2 against a true value of 2" reading was wrong:*
  `analytic_hessian` returned zeros under the drivers'
  `torch.set_grad_enabled(False)`, so the "error" was the full Hessian
  magnitude. Fixed 2026-10-09, see versioning event #7: the corrected
  grad-of-grad Hessian converges at ~1.5 (crk / renorm, smooth fields) and
  ~1.3 (standard).)
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

**Status: FIRST PASS DONE (2026-10-09) — pure-torch reference operator + static evidence, and the operator is now in the core (`warpSPHCore.polyfit.PolyFit`, warp kernels, reverse mode; see "Core port" below). Frozen-leg run, conservative reformulation and forward-mode (JVP) open.**
Delivered in `higherOrderSPH/harness/`: `rkpm.py` (order-p moment system + local polynomial fit; pure torch, float64, CPU-testable), `rkpm_modes.py` (registers `rkpm1/2/3` through `register_mode`), `run_rkpm.py` → `REPORT_rkpm.md` (own driver; the frozen baseline outputs are untouched), `tests/convergence/test_rkpm.py` (20 CPU unit tests + a harness smoke). Static results, 2-D jittered 0.3, Wendland2, 40 target neighbours:

| | patch (interior and boundary band) | interpolate order | gradient order | Laplacian / Hessian order |
|---|---|---|---|---|
| crk | value+grad exact to degree 1 | 1.84–1.95 | 1.4–1.7 | Brookshaw: saturated / negative |
| rkpm1 | exact to degree 1 (**values identical to crk, 8e-15**) | 1.84–1.97 | 1.65–1.77 | n/a |
| rkpm2 | exact to degree 2 (value, grad, Laplacian, Hessian ~1e-11) | **2.7–3.1** | **1.86–1.97** | **1.6–1.7** |
| rkpm3 | exact to degree 3 | **3.8–3.96** | **2.9–3.3** | **1.8–1.96** |

Orders read from the periodic and open resolution ladders at fixed h/Δx (interior). Interpolation converges at p+1 and the gradient at p (the Laplacian / Hessian at p−1 as expected: ≥ 1.6 for p = 2, ≈ 1.9 for p = 3), so the generator hits the textbook orders; the Brookshaw Laplacian, which does not converge pointwise, is replaced by one that does. Conditioning (equilibrated moment matrix, `results/cond_rkpm.csv`): interior κ ≈ 1.0–2 (p=1), 5–7 (p=2), 15–30 (p=3) at ≥ 20 neighbours; **12 neighbours is not enough for p = 3** (interior κ median 2.6e3, boundary rows 38–64 % rank-deficient); in the wall band the 95th-percentile κ is ~55–80 (p = 2) and ~1.3e3–3e3 (p = 3) at ≥ 35 neighbours, and 3 % of the p = 3 band rows are rank-deficient at 20 neighbours (0 % at ≥ 35). The `smoothing` suite (N fixed, h varied) is uninformative for p ≥ 2 (order vs h at a fixed discretisation reads negative: more neighbours dominate).

**Core port (2026-10-09, `src/warpSPHCore/polyfit/`, commit `acae773` + follow-ups).** `PolyFit.build(...)` assembles the order-p moment system with a warp kernel (one thread per matrix entry; arbitrary monomial basis from an exponent table; adjacency and compact-hash traversal; per-particle supports, so variable h works), solves the small systems batched in torch (Jacobi-equilibrated `eigvalsh`/`pinv`, condition numbers, rank flags), and provides `values / gradient / hessian / laplacian / derivative(alpha) / evaluate / interfaceStates`; `constant=False` is LABFM. Agrees with the pure-torch reference to 1e-12 (gradient) / 1e-11 (Laplacian) through k = 8 in 2-D; at k = 8, 2,200 particles: 122 ms build, 4 ms apply. **Reverse mode through the Warp tape works for the field, positions, supports, masses and densities** (`scripts/gradcheck/gradcheck_polyfit_native.py`, MLS and LABFM, in the CI gradcheck list). Not done: forward-mode JVP (the apply is exactly linear in the field, so the field tangent is the same kernel on the tangent array — needs a `JVPSpec` hook; the geometry tangent needs the kernels' JVP), `warpOperation` dispatch integration, a multi-lane variant, pair-list/neighbour-list extraction for the interface states (the caller supplies `(i, j, r_j - r_i)`). Building it found and fixed three bugs (in `docs/lessons_learned.md`): a dynamic loop updating a variable multiplicatively (position/support adjoints silently wrong), read-modify-write of a differentiated output array (corrupted on every backward), and the AD bridge zeroing the caller's `grad_output` in place and inferring the seed dtype from rank.

**Tasks:**
- [x] *(done in the core 2026-10-09, `warpSPHCore.polyfit`)* Implement generic moment-matrix builder for a chosen monomial basis (parametrized consistency order $p$), following MLS/RKPM formalism. **New code** — `computeRenormalizationMatrices` is d×d-only (Verification pass, finding 5). Needs a batched small dense solve (n = 3…20) with explicit regularisation / rank handling, float64-capable, and its own reverse-mode story (see the pinv2x2 adjoint history in `warpier_core.md`).
- [x] Implement kernel correction function $C(x; x-x_j)$ so the kernel itself (not just gradient) reproduces polynomials to order $p$. *(Value: $c_0 = e_0^T M^{-1} b$, reproduces degree p exactly; at p = 1 identical to CRK. The full RKPM kernel-gradient — differentiating $M^{-1}$ as well — is **not** implemented; the shipped derivatives are the local-fit (diffuse) ones, exact to degree p. Obtainable by autodiff w.r.t. the query point if wanted.)*
- [x] Derivative operators from the corrected kernel: gradient **and Laplacian / Hessian** (p ≥ 2), measured through `register_mode` against the `hessian` (grad-of-grad) and Brookshaw baselines *(done as diffuse derivatives of the local fit, see the table; the frozen-diffusion cross-check below is still open)*. The Brookshaw Laplacian does not converge in the *static* probes (`FINDINGS.md` §3), but does at ~2 in the frozen diffusion leg for CRK (1.72 renorm) — settle which error component matters (Phase 0 frozen-leg note) before making "a converging Laplacian" the headline target; the static pointwise norm and the PDE-level error disagree.
- [ ] Note non-symmetry of resulting kernel ($W_{ij} \neq W_{ji}$) — decide whether to carry forward a naive (non-conservative) MLS operator as a standalone mode, or route straight into a CRKSPH-style conservative reformulation reusing this moment matrix. *(Phase 2 settled: the CRK linear-gradient residual was a bug, not a cost of the CRK form — the corrected-kernel derivative is exact to degree 1. Note that CRKSPH's* momentum equation *uses an antisymmetrised pair form for conservation, which is a separate question from the operator's exactness and should be evaluated the same way here.)*
- [~] *(static done; free-surface / corner / thin-support sets beyond the wall band not yet)* Run Phase 0 harness at multiple orders $p = 1, 2, 3$: patch tests to degree $p$, convergence-rate tests, condition-number tracking vs. $p$ and neighbor count (expect conditioning to worsen with $p$ on disordered particles). Include free-surface / corner / thin-support sets, which Phase 3 did not probe. Cross-check p=1 against the `crk` column (should match CRK values exactly).
- [ ] Run the frozen-particle PDE leg (primary order evidence). Full PDE suite only if a frontend scheme uses the operator; if non-conservative, explicitly report conservation-diagnostic failures (expected) as a documented tradeoff vs. Phase 2.

**Deliverable:** Order-parametrized reproducing-kernel operator generator, validated across $p=1..3$, with conservation-vs-accuracy tradeoff documented.

**References:** `liu_jun_zhang_1995_rkpm`, `dilts_1999_mlsph`, `dilts_2000_mlsph2`.

---

## Phase 5 — LABFM

**Goal:** Push the moment-matching idea from Phase 4 to arbitrary order using anisotropic basis functions and compact stencils, decoupled from kernel-summation framing.

**Status: STATIC + FROZEN-LEG EVIDENCE DONE (2026-10-09, pure-torch reference); hyperviscosity, boundary-stencil work and strong-disorder tests open.**
LABFM is the same weighted-LS moment system as Phase 4 with the constant dropped and the data taken as differences (`build_system(constant=False)`, harness modes `labfm<k>`, k = the paper's order = the polynomial degree); `run_labfm.py` → `REPORT_labfm.md`. Weights are exactly the ABF weights (kernel × polynomial, coefficients solving the moment equations), so no separate ABF construction is needed. 2-D periodic, jitter 0.3, observed order of the interior L∞ gradient error on a 5-rung ladder at fixed N (neighbours):

| k | N=25 | N=40 | N=60 | N=80 | N=150 | paper (King et al. 2020) | smallest N reaching k−0.5 here |
|---|---|---|---|---|---|---|---|
| 2 | 1.95 | 1.95 | 1.94 | 1.93 | 1.88 | — | 25 |
| 4 | **3.94** | 3.91 | 3.92 | 3.93 | 3.86 | 4th order at N ≈ 25 | **25** |
| 6 | sat. (singular) | 5.90 | 5.91 | 5.92 | 5.86 | k ≤ 6 at h/δr = 2 (N ≈ 50) | 40 |
| 8 | sat. (singular) | sat. (singular) | **7.88** | 7.88 | 7.88 | 8th order at N ≈ 60 (shown at N ≈ 78) | **60** |

The paper's headline pairs reproduce: 4th order at ~25 neighbours, 8th at ~60. Below the critical stencil the moment matrix is singular (cond ~1e17, flagged `deficient`) and the gradient does not converge, matching the paper's critical sizes (N_crit ≈ {8, 21, 37, 57} for k = {2, 4, 6, 8}, vs N_poly = {6, 15, 28, 45} for plain polynomial reconstruction — here k = 6 is already fine at N = 40 and k = 4 at N = 25, consistent with both). Moment-matrix κ (median, equilibrated): 2 / 42 / 8e2 / 1.7e4 for k = 2 / 4 / 6 / 8 — float64 is enough through k = 8 (finest-rung gradient error 6e-10 at k = 8, N = 60, well above the floor). The Laplacian converges at about k−1…k (k = 4: 2.8 → 3.9 for N = 25 → 150; k = 8: 6.8 → 7.3). The gradient error constant grows with N at fixed h/Δx (a larger h), as expected.

**Tasks:**
- [x] Implement anisotropic basis function (ABF) construction and the local linear system solved per stencil (reuse Phase 4's moment-matrix infrastructure — the *new* n×n builder, not the d×d renorm path). *(Done as the `constant=False` variant; chunked pair accumulation keeps k = 8's 44-function system in memory.)*
- [ ] Implement one-sided/boundary stencil handling for incomplete support. *(Not measured separately. The wall-band patch tests in `REPORT_rkpm.md` show the underlying fit is exact there wherever the moment matrix is full rank; the LABFM paper's claim — one-sided stencils of the same order up to 4th — is not yet checked, and the open-domain `labfm` patch/order suites have not been run.)*
- [~] Harness prerequisites specific to this phase: monomial fields to degree ~10 *(no cap in code; `--degree-max 10` not run)*; ladder design that stays above the float64 floor at 8th–10th order *(done: five rungs, coarse; 10th order not attempted)*; Laplacian probe on the LABFM operator itself *(done: `labfm<k>` provides its own Laplacian/Hessian)*.
- [x] Run Phase 0 harness at increasing order (target: reproduce paper's reported 4th order at ~25 neighbors in 2D, up to 8th–10th order at larger stencils) — *(done to k = 8, table above; 10th not run)*.
- [~] Stability check: add hyperviscosity option and confirm it stabilizes hyperbolic test PDEs per `king_lind_2020_labfm`. *(Split decided 2026-10-09: the core provides the operator — `PolyFit.derivative(f, alpha)` returns any `d^alpha f` of total degree ≤ order, i.e. the highest-order derivative hyperviscosity needs; the coefficient α h^m, its scaling and the time-step limit belong to the frontend. Not exercised yet, and not needed so far: **No run diverged** in the frozen leg (26 of 26, jitter 0.3, periodic), so nothing here needs stabilising yet; the paper's instability is for noisy distributions at large k (ε/δr ≳ 0.5) and for incomplete support, neither of which this leg exercises. Next test: jitter ≥ 0.5 and the saved TGV distributions in `data/`, then Dirichlet walls.)*
- [x] Run the frozen-particle PDE leg at matched order vs Phase 4 *(done, below)*; compare compute cost vs. order against Phase 4's correction-matrix approaches *(done for the pure-torch reference only: setup 7 / 40 / 47 ms and apply 0.14 / 0.15 / 0.18 ms for k = 4 / 6 / 8 at N = 2200, ~1e5–2e5 pairs — cost is not a constraint here, and says nothing about a warp implementation)*.

**Frozen-particle leg** (`run_frozen_hi.py` → `REPORT_frozen_hiorder.md`; RK4, cfl 0.05 advection / 0.02 diffusion, 288 → 2304 particles, jitter 0.3, periodic): LABFM converges in time at its design order, **advection 1.95 / 3.93 / 5.90 / 7.87 and diffusion 2.02 / 3.95 / 5.91 / 7.88 for k = 2 / 4 / 6 / 8** (k = 8 advection L2: 5.6e-5 → 1.8e-8). MLS (Phase 4) is far less efficient: advection rkpm1 **1.94** and rkpm2 **1.94 with an identical error** (3.5e-1 → 4.9e-2: on near-symmetric stencils the quadratic term does not enter a first derivative), rkpm3 3.91; diffusion rkpm2 2.05 and rkpm3 2.06 (the cubic does not enter the Laplacian). I.e. odd-degree MLS behaves like the even degree below it, and the even-k LABFM schemes are the efficient ones: k = 4 (14 coefficients) gives ~3.9 for both operators where rkpm3 (10 coefficients) gives 3.9 / 2.1. The gradient order k does not rely on that symmetry: it holds at jitter 0.5 (3.87–3.88 for k = 4, 5.85–5.92 for k = 6; Laplacian 2.9–3.3 and 4.7–4.8, i.e. ~k−1); strong disorder (the saved TGV distributions) is untested.

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
**Superseded 2026-10-09:** the frontend built one outside this plan —
`modules/riemann` (Acoustic, PVRS, TRRS, TSRS, Adaptive, HLLC) plus
`modules/godunov` (simplified Cha–Whitworth GSPH, `InutsukaGSPH`), with
state reconstruction + limiters (warpSPH `GODUNOV_SPH_PLAN.md`, layers 1–3
built and validated 2026-10-08, committed in `abdaf11`; remaining: blob /
Wengen KH, solver ablation). The first Phase 6 task is therefore largely
done; what is left is the MLS/TENO reconstruction on top of it. Reference
numbers for the like-for-like comparison: Gresho 0.076–0.091 and Sod
velocity 0.0030 with no AV (warpSPH `docs/av/godunov_l3_2026-10-08/`).

**Status (2026-10-09): reconstruction layer built, tested standalone, and ported to the core (`warpSPHCore.polyfit.Reconstructor`: `.teno(...)`, `.weno(...)`, `.interfaceStates(f, i, j, d)`, Eq. 25 interface point, per-particle volumes/radii, softmax-of-log weights so float32 does not overflow); NOT coupled to a Riemann solver — that coupling is a frontend (Godunov SPH) integration, base scheme decided 2026-10-09: the existing Godunov SPH, MFM not implemented.** The core reproduces the pure-torch table to every printed digit in float64 (TENO O4 / O6 and WENO M = 3: orders 4.08 / 5.94 / 4.01, finest errors 7.69e-6 / 4.73e-8 / 6.19e-5, step overshoot ≤ 2.5e-8); `tests/operations/test_reconstruction.py` (CPU + CUDA, float32). The candidate stencils needed two options on the fit kernels: an angular sector filter and an unweighted (plain least squares) mode. Gradients flow through the nonlinear weights (finite, non-zero); a gradcheck is not meaningful across TENO's hard cutoff.
**Eq. 25 versus the arithmetic midpoint (decided 2026-10-09):** the core uses Eq. 25, `r_ij = (h_j r_i + h_i r_j)/(h_i + h_j)` — the same point for equal supports (every harness lattice), but with unequal supports it sits closer to the particle with the smaller support where the two kernels' influence balances, whereas the arithmetic midpoint has an O(h_i − h_j) position error that does not vanish with resolution at refinement jumps or in adaptive-h regions; tested with a smoothly varying support (`test_interface_states_exact_with_variable_support`).
 `reconstruct.py` (pure torch, 2-D and 1-D; nine candidate stencils per particle on the Phase 4 moment systems), `run_reconstruct.py` → `REPORT_reconstruct.md`, `run_interface.py` → `REPORT_interface.md`, `tests/convergence/test_reconstruct.py`. The Riemann coupling waits on the base-scheme decision (Phase 8 / MFM note) and a frontend integration choice.

Static results (2-D hex lattice, jitter 0.3, periodic; particle counts 288 / 1152 / 4608; TENO per Gao et al. 2023, central radius 2.5 / 3.2 / 4.0 √V for O4 / O5 / O6):

| scheme | smooth-region order | step overshoot | jump capture ¹ | false jump ² |
|---|---|---|---|---|
| plain MLS p = 3 / 4 / 5 (unlimited) | 4.08 / 4.94 / 5.94 | **0.40 / 0.26 / 0.21** | 0.17 / 0.15 / 0.04 | 0.38 / 0.19 / 0.10 |
| TENO O4 / O5 / O6 | **4.08 / 4.94 / 5.94** | 1.5e-10 / 2.6e-8 / 2.4e-8 | 1.000 | ≤ 7e-8 |
| WENO M = 3 / 4 / 5 | 4.01 / 4.85 / 5.93 | 2e-16 | 1.000 | 4e-16 |

¹ mean |f_L − f_R| over pairs straddling a unit step (1 = fully resolved). ² max |f_L − f_R| over same-side pairs (any value is a spurious Riemann problem). In smooth data TENO reproduces plain MLS **exactly** (it keeps the central stencil everywhere), whereas matched-order WENO is 5–12× less accurate (O4/M=3: 7.7e-6 vs 6.2e-5 at the finest rung; O5/M=4: 6.5e-7 vs 7.7e-6; O6/M=5: 4.7e-8 vs 2.2e-7) — the WENO weighting (λ₀ = 1e5, unit-weight fits) mixes the sectors in even where the data are smooth. Interface states themselves converge at order p+1 (`run_interface.py`: MLS p = 1/2/3: 1.8 / 2.9 / 3.9; LABFM k = 2/4: 2.9 / 4.8). **Caveat:** a perfectly sharp step is the easy case for both; they differ near the jump: a smooth background (amplitude 0.5) superposed on the step, evaluated ≥ 2 spacings from it, keeps max errors of 6.6e-3 – 8.6e-3 for TENO O4/O5/O6 (identical to plain MLS at O4, 3× / 4× below it at O5 / O6) and 2.6e-2 / 7.5e-3 / 1.4e-4 for WENO M = 3 / 4 / 5 (the no-jump errors are ~1e-5 – 1e-7): TENO's cutoff keeps the central stencil where it only marginally touches the jump, so its error there does not improve with order, while the WENO sectors do. The real head-to-head dissipation comparison (the Phase 6/7 deliverable) needs the solver.

**Findings from building it** (flagged in the module docstring):
- **The paper's sector-stencil numbers are inconsistent on a uniform lattice.** Gao et al. choose directional stencils of radius 4.5 √V with "at least 10 particles", but a 45° sector of that radius holds π·4.5²/8 ≈ 8. With a minimum of 10, five of the eight sectors were deactivated near a discontinuity, the scheme fell back to the central fit and **overshot by 0.24** (plain MLS: 0.41). Using the smallest solvable degree-2 fit (nc+1 = 7) as the minimum removes the overshoot (1e-8). `dir_radius` / `dir_min` are exposed.
- 1-D is not in either paper; a central radius of 2.5 √V holds ~5 points, below any sensible minimum, so TENO O4 silently stayed at 3rd order (the sectors) until the central radius was widened for 1-D.
- Where the papers are not explicit (flagged `(a)–(d)` in `reconstruct.py`): the TENO β sums all partial derivatives of order 1..p with unit weight on a common square; stencils use a hard radius around √V; the interface point is the arithmetic midpoint (equal supports); a deactivated stencil gets weight 0.

**Tasks:**
- [x] *(done in the frontend, 2026-10-08, warpSPH `abdaf11` — see the scope note above)* **Implement a first-order Riemann-SPH (Godunov-SPH) base scheme** in the compressible frontend (pairwise interface states, approximate Riemann solver — e.g. HLLC or an exact solver generalised from `sodSolution.py`). This is the zero-reconstruction baseline TENO/WENO are compared against.
- [x] Implement MLS-based local polynomial reconstruction at particle-pair interfaces (`RKPMOperator.interface_states`), interface position $\overline{r}_{ij} = (h_j r_i + h_i r_j)/(h_i + h_j)$ *(arithmetic midpoint for the uniform-support harness lattices; the h-weighted form is not yet implemented)* (reuse Phase 4's n×n moment builder).
- [x] Implement TENO smoothness indicators / stencil selection and blending weights. *(Exact Gram-matrix β, γ = 1/(β+ε)⁶, cutoff C_T; see findings above.)*
- [ ] Couple reconstruction to the Riemann-SPH scheme for left/right state resolution.
- [~] Run Phase 0 smooth-region convergence tests (target: 4th-order-class reconstruction accuracy) *(done: 4.08 / 4.94 / 5.94)* and shock-tube/Sedov/KH tests for non-oscillatory behavior and reduced dissipation vs. WENO of matched order *(non-oscillatory step test done; the shock-tube / Sedov / KH runs need the solver coupling)*. *Use the exact-solution metrics (Sod, Sedov); compare within the Riemann-SPH family (the current Sod baseline is CompSPH and Sedov is CRKSPH — not like-for-like).*

**Deliverable:** TENO-SPH reconstruction module, validated for both smooth-region order and shock robustness.

**References:** `fu_2016_teno` (original TENO), MLS-TENO-SPH paper in your literature folder (arXiv 2306.00514).

---

## Phase 7 — WENO Reconstruction

**Goal:** Implement WENO as the baseline shock-capturing comparison point against Phase 6's TENO.

**Status (2026-10-09): reconstruction built (shares `reconstruct.py` with Phase 6); solver-level comparison not done.**

**Tasks:**
- [x] Implement MLS-WENO reconstruction (same interface-reconstruction structure as Phase 6, swap smoothness-indicator/weighting scheme). *(Avesani et al. 2014: nine unweighted no-constant fits, central r ≤ h_mls, sectors r ≤ 2 h_mls, σ = Σ w², ω ∝ λ/(ε+σ)⁴ with λ₀ = 1e5; `weno_reconstructor`.)*
- [~] Run identical Phase 0 test set used for TENO — smooth-region order *(done, Phase 6 table)*, shock-tube/Sedov/KH dissipation comparison *(needs the solver)*.
- [~] Direct head-to-head report: WENO vs. TENO dissipation at matched formal order. *(Static reconstruction-only proxy in the Phase 6 table: same non-oscillatory behaviour; TENO 5–12× more accurate in smooth data; near a jump the picture is mixed (WENO M = 5 is far better 2 spacings from the jump). The solver-level comparison is open.)*

**Deliverable:** WENO-SPH reconstruction module; final comparison table across all phases (Standard SPH / CRKSPH / Bonet-Lok / MLS / LABFM / WENO-SPH / TENO-SPH) on the same harness — **with the scheme stated per row** (the Pass-2 suite showed that "per-case default scheme" is not a column).

**References:** `avesani_dumbser_bertaux_2014_mlswenosph`, follow-up "Investigations on a high order SPH scheme using WENO reconstruction" paper in your literature folder.

---

## Phase 8 — MFM column (outlook, not scheduled)

**Goal:** Add Hopkins (2015) meshless finite mass (MFM) as one more scheme
row in the final comparison table. Added 2026-10-09; the frontend side is
tracked in warpSPH `PESPH_PLAN.md` §7 (MFM is its stated longer-term goal).

**Status: NOT STARTED, no commitment.** It needs core work (MFM is nearly a
second core: per-face quantities with a Riemann solve inside the pair loop).

**Why it is not an operator phase.** MFM is a *solver*, not an operator. Its
formal order is that of CRKSPH (a first-order-consistent matrix gradient,
limited linear reconstruction, Riemann flux: ~2 on smooth flow, ~1 at
shocks), so it is not "higher order" in the sense of Phases 4–5. It fits the
harness as follows:
- **Static operator probes:** only the matrix gradient applies, and that is
  the existing `renorm` (Bonet–Lok) column. The face-flux sum has no
  standalone operator to probe.
- **Frozen-particle leg:** does not exercise the Riemann machinery.
- **Compressible PDE cases:** apply directly (Sod, Sedov, Gresho, KH,
  linearWave with the exact-solution metrics). Gresho is the one to watch: if
  MFM does not spin up the vortex, that isolates the CRK pair pressure force
  as the cause (Phase 1 anomaly).

**Relationship to other phases:**
- **Phase 6–7:** TENO / WENO produce left/right interface states for a
  Riemann solver; MFM is another consumer of that interface, with Hopkins'
  limited linear reconstruction as its baseline. PESPH_PLAN §7.2: MFM is
  CRKSPH's conservative differencing with the averaged flux replaced by a
  Riemann flux, i.e. a Riemann-SPH base scheme on the CRK substrate. **If the
  MFM core work is going to happen, build Phase 6–7 reconstruction against its
  face-state interface instead of the SPH pair interface** — decide this
  before starting Phase 6.
- **Phase 4:** independent. MFM needs only the existing d×d p=1 machinery
  (`computeRenormalizationMatrices` / CRK), not Phase 4's new n×n builder, so
  Phase 4 does not wait on it.

**Tasks (when scheduled):**
- [ ] Decide whether the core grows an MFM face-flux path (user).
- [ ] Register MFM as a scheme in the compressible frontend; add it to the PDE
  registry as `<case>-mfm` legs (same case / ladder / metric / IC).
- [ ] Report it as one more row of the Phase 7 final table, scheme stated.

**References:** `hopkins2015_new-class-meshfree-hydrodynamic-methods`,
`hopkins2013_general-class-lagrangian-sph`.

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
  (7) **2026-10-09, `analytic_hessian` under `no_grad`**: it silently
  returned zeros when the caller had grad disabled, as both drivers do, so
  every `hessian` row of `baseline_rows.csv` / `REPORT.md` since 2026-09-26
  was scored against a zero target. Fixed (`torch.enable_grad()` inside), a
  `no_grad` regression test added, the static sweep re-run: **only the 320
  `hessian` rows changed**; every other row agrees to ≤ 1.2e-15 absolute
  (GPU atomic-ordering noise at the round-off floor, rows that are machine
  exact). The old `REPORT.md` is in git history; the old CSV is kept locally as
  `results/baseline_rows_pre-hessian-fix_2026-10-09.csv`. The regenerated
  `REPORT.md` also shows round-off-floor churn in rows that are machine
  exact (≤ 1.2e-15), not a change in any result. Also (additive):
  `run_baseline.measure` now skips a probe an external mode returns `None`
  for (the `register_mode` contract; unused until Phase 4).
  The pre-Phase-4 event is done.
- Phases 3–5 share moment-matrix infrastructure; implement that shared
  layer once in Phase 4 rather than duplicating in Phase 5.
  *(Corrected 2026-09-26: the shipped `computeRenormalizationMatrices` is a
  d×d covariance with closed-form ≤3×3 pseudo-inverses and cannot hold an
  order-p moment matrix. Phase 4 builds a new n×n builder; reuse from the
  core is the neighbour-loop / dispatch pattern and CRK's p=1 structure.)*
- Phases 6–7 are architecturally separate (solver-level, not operator-library-level) — fine to parallelize with Phase 4/5 if useful, but both now start with building a Riemann-SPH base scheme.
- Every PDE-level comparison must state the scheme per row and compare
  like-for-like (same scheme family, only the operator varied).
