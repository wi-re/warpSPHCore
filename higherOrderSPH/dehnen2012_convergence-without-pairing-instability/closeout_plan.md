# Closeout plan — D&A 2012 replication (post-Phase 6)

> **Resume protocol:** Phases 0–6 are complete. Read `PLAN.md` (Phase 5
> STATUS, Phase 6 status, Phase 8), then this file. Check `git status` /
> `git log -n 5` in BOTH repos (`warpSPHCore` and sibling `warpSPH`) to see
> what has landed; continue at the first unchecked item. Append dated
> progress notes to the **Progress log** at the bottom as items complete.
>
> **Environment:** conda env `warp` (`/home/lu26029/miniconda3/envs/warp/
> bin/python`); cap BLAS threads (`OMP_NUM_THREADS=2`) for all numerics —
> shared box; everything here runs on CPU. Do NOT push. Scratch files go in
> the repo's `.tmp/` (gitignored). Commit per completed item; check `git log`
> in each repo for message style.

## Decisions (user, 2026-09-19)

- The five non-D&A kernels (poly6/spiky/adhesion/cohesion/viscosity) are
  CG-specialty force/Laplacian kernels, **not** general-purpose density
  kernels → **excluded** from the audit pipeline and from any D&A-convention
  re-scaling. No code change; document the exclusion in `REPORT.md`.
- Fix `sampleOptimal` (warpSPH) — explicitly wanted.
- Fix the `support.py` float32-BinOp constant (warpSPHCore).
- The eq.-18 ε `renorm` feature is **scoped only** (design note), not
  built. *(Superseded 2026-09-19: the user then asked to build it — item 7.)*
- The high-res 10-kernel stability sweep comes **before** `REPORT.md`.
- **The high-res sweep SKIPS the Gaussian** (item 5, 2026-09-19): it is by
  far the largest (N_H to ~82 000 over h/d_nn 0.9–3.0; the oracle build
  OOMs on this box above N_H≈7 000 — measured peak RSS 23.8 GB at N_H=7000)
  and is not practical in an actual simulation. Run the other nine
  `STABILITY_ORDER` kernels at full h/d_nn 0.9–3.0. Document the exclusion
  in `REPORT.md`.
- **Run the sweep STAGED, a few kernels at a time** (item 5, 2026-09-19):
  monitor box memory between stages (the box is shared and a memory spike
  has been breaking tmux sessions). Per-kernel `results/*.npz` caching
  makes an interrupted stage resumable.

## Items

### 1. PLAN.md bookkeeping — [x] DONE 2026-09-19
- Retired the "LIVE WORK LOG (Phase 5)" header banner → now points to the
  closed logs (here + warpSPH) and this file.
- Findings log: C&D sign-note row `open — Phase 6` → resolved 2026-09-19.
- Findings log: Phase-5 force-law/P-matrix row `open — Phase 5` → resolved
  2026-09-19 (entries d–k).
- Phase-5 section: STATUS `entries a–h` → `a–l`; the checkbox P-matrix
  formula was the pre-entry-(i)/(k) form (doubled K, wrong sign,
  absolute-coordinate phase) → corrected to the validated form
  `P = 2mB̄ Σ(1−cos k·d0_j)∇∇W(d0_j) + mB̄(γ−2)/ρ̄ Re[Σ ∇W(d0_j)⊗B_j]`.

### 2. Fix the float32-BinOp constants in `src/warpSPHCore/util/support.py` — [x] DONE 2026-09-19
- `volumeToSupport_warp`, dim == 3 branch: `scalar_t(np.pi * 3.0 /4.0)` and
  `scalar_t(1.0/3.0)` are evaluated as **float32** constants by the warp
  tracer even in float64 builds (same bug class as the B7 knot; findings log
  2026-09-18). Corrupts the (4π/3) volume factor at ~1e-8 relative.
  **Found while fixing:** the function did not compile in *any* build — the
  tracer has no `int32 × float64` mul overload (`targetNeighbors * volume`
  raised "Input types must be the same"), so the float32-constant bug was
  latent behind a compile error (which is why it had zero callers).
- Fix (committed): module-level Python-float constants `_VOL_3D_FACTOR` /
  `_CUBE_ROOT_EXP` referenced by name (the `kernelFunctions/B7.py` knot
  pattern) + `n = scalar_t(targetNeighbors)` cast before the muls.
- Verified: `.tmp/support_const_probe.py` (float64 build) — all three dims
  now **exact** (max |rel| = 0.000e+00) vs the double reference; float32
  build gives ~7.5e-8 (= f32 round-off, the pre-fix expectation).
  `.tmp/support_const_probe2.py` isolates the pre-fix corruption: the two
  BinOp constants came out as float32-rounded values (2.5e-9 / 2.98e-8 rel)
  in a float64 build, while the `scalar_t(np.pi)` Name form was exact.
  Full suite: 451 passed, 1 skipped (one unrelated pre-existing failure in
  `test_field_abstraction.py::test_tangent_slot_inert_when_unset`, present
  on the unmodified file too).

### 3. Fix `sampleOptimal` (`warpSPH/src/warpSPH/sample/optimal.py`) — [x] DONE 2026-09-19
Three bugs (findings log 2026-09-18):
1. passes a `ParticleSet` where `warpOperation` / `computeDeltaShiftWarp`
   need a `ParticleState` → `AttributeError: ... 'kinds'`;
2. applies the raw delta-shift term (O(1–10), pointing TOWARD neighbours)
   without the `−CFL·Ma·2·h²` scaling `modules/shifting/delta.py` uses →
   relaxation diverges (density std 24 %, clumping);
3. overwrites its lattice start with uniform random points.
Fix: build/accept a `ParticleState`, use the delta.py-style scaling
(`scale = −CFL·Ma·2·h²`, CFL 0.3, Ma 0.1 = the no-velocity fallback), keep
the jittered-lattice start.
**Fixed (committed in warpSPH):** all three bugs; also (a) a fourth latent
bug — `ParticleSet` was used unimported at the end of the function (would
`NameError` after the other fixes), now `from ..geometry import ParticleSet`;
(b) the `kernel` arg was ignored (Wendland2 hardcoded) — now used for both
the density op and the shift + `sphKernelScale`; (c) a `seed` kwarg (default
`None`) makes the Gaussian jitter reproducible (the recipe uses seed 42);
positions are wrapped into `[min, max)` on periodic axes each iter.
**Verified** (`.tmp/verify_sampleOptimal.py` + `.tmp/ref_glass_cpu.py`,
warpSPH `.tmp/`, CPU/float64): the fixed function reproduces the recipe
algorithm **exactly** — per-particle positions match an unmodified
CPU run of the original `glass_recipe_probe.py` logic to max |Δpos| =
4.4e-15 (float64 round-off). Recipe acceptance on CPU: density std/mean =
0.283 % (recipe ~0.30 %), nn/dx = 0.8863 (recipe ~0.86), no clumping.
The cached CUDA npz differs from BOTH the CPU recipe run and the fixed
function by the same 3.4e-2 max (a device-level ulp/jitter-generator fork
of the relaxation into a neighbouring glass basin — both are valid glasses
meeting the recipe criteria; not an implementation deviation). Float32
build also runs (50-iter smoke). Regression: `test_densestSampling.py` +
`test_latticeDensity.py` 151 passed.

### 4. Scope the `renorm` ε feature (warpSPHCore) — [x] DONE 2026-09-19 (scope ONLY)
**Deliverable: `renorm_eps_design_note.md` (this folder)** — answers all six
questions. Key scoping outcomes: (a) naming — do NOT call it `renorm`
(`warpSPHCore/renorm.py` is gradient renormalization); use
`densityCorrection`/`selfTermCorrection`; (b) hook — post-process OUTSIDE
the density operator (pure-torch elementwise remap, immediately after the
raw density, before the EOS), keeping the continuity update standard
(justified by the paper's constant-in-h Lagrangian argument);
**post-scoping addendum (2026-09-19, user option B):** the originally
"two scheme sites" are now ONE — monaghan/compSPH density calls ported
onto `computeDensities` (optional `supportMode` param added, default
Gather; monaghan passes `config.supportMode` = SuperSymmetric → zero
behavior change, bit-exact vs the old direct calls on non-uniform-support
states, probe warpSPH `.tmp/probe_density_port_equiv.py`), so the hook is
a single call site inside `computeDensities`; (c) N_H from the raw estimate
`N_H = V_ν h^ν ρ̂/m` (no iteration — second order); (d) W0 =
C_d·f(0)/h^ν at the code support = paper H via `eval_k(0)`/`eval_C_d`
(kernelScale³ gotcha); (e) constants — ship only the 3 Wendland 3D pairs
(the correction only helps over-estimating kernels; B-splines
under-estimate), W2 the default candidate, config override + KeyError;
(f) `calibrateNormalization` — different axis (constant 1/L lattice
quadrature vs per-particle N_H^−α self-term fraction), partial overlap at
small n_h → both default off, refit ε before combining; (g) tests spec:
fig03 corrected-curve unit check (0.10–2.40 % over 40≤N_H≤400), cross-repo
consistency, Monaghan integration smoke. NOT built (per user decision).

*Original scope (for reference):*
Paper eq. 18/19 (`data/da2012_reference.yaml` → `density_correction`):
`rho_corr = rho_hat − ε·m·W(0, H)` with `ε = ε₁₀₀·(N_H/100)^(−α)`; 3D
constants W2 (0.0294, 0.977), W4 (0.01342, 1.579), W6 (0.0116, 2.236); our
Phase-4 refits agree within ×1.04. Deliverable = a short design note
(REPORT.md appendix or its own file) answering:
- **Hook point:** where in the density path the correction applies
  (warpSPHCore density function vs warpSPH scheme); it is per-particle.
- **N_H at density time:** N_H must be recovered per particle from support +
  density, `N_H = V_3·H³·ρ̂/m` — check what is available at the hook point.
- **W(0,H) convention:** must be the physical central value
  `W(0) = C_d·f(0)/H³` at the support radius (parameterisation-invariant);
  evaluating at h = H/kernelScale is wrong by kernelScale³ (findings log
  2026-09-18, fig03 gotcha).
- **Constants:** only the three Wendland kernels have paper constants —
  per-kernel table vs refit; W2 is the default-kernel candidate.
- **`calibrateNormalization` interaction:** that is a *different* correction
  (lattice renormalisation) — do not conflate; decide coexistence.
- **Tests (for when it is built):** unit check reproducing the fig03 result
  (corrected FCC within 0.10–2.40 % of 1 over 40 ≤ N_H ≤ 400, all three
  Wendland kernels).
Not implemented in the closeout; `REPORT.md` carries the scope +
recommendation.

### 5. 10-kernel stability sweep, high-res fig04/05/06 — [x] DONE 2026-09-20
Current `fig04_fig05_stability_contours.py` grid is **14 h/d_nn × 26
|k|d_nn** per direction — too coarse for the paper's contours ("fig 4 needs
much higher nx/ny resolution"). Steps:
- Add a CLI resolution flag (e.g. `--grid NH×KDN`) to the fig04/05 script
  (keep the current grid as default). Target ≈ 60 N_H (log) × 200 |k|d_nn
  (log); tune after the timing probe.
- **Timing probe first:** time one kernel (cubic_b4) at the target grid,
  extrapolate to 10 kernels × 2 directions. The Gaussian (N_H up to 5120)
  is the most expensive point — if its cost is unreasonable, document a
  reduced N_H range for it.
- **Cache:** save each kernel's (lon, tr; 111, 110) fields to
  `results/stability_<kernel>.npz` (gitignored) and load-if-present, so an
  interrupted sweep resumes and figures regenerate without recompute.
- Run all ten `STABILITY_ORDER` kernels (`--kernel` enables split runs).
- Verify the Phase-5 acceptance boundaries from the fields: cubic ≲ 55,
  quartic ≈ 67, quintic ≈ 190 (+ small-N_H island near 100), Wendland C²
  island near 40, HOCT4 island near 150, clean otherwise; the cubic long-λ
  dip (|k|d_nn ≈ 0.3–0.6, all N_H 40–100) must be clearly resolved — it is
  the documented Phase-5 discrepancy.
- Regenerate fig04/05 (per-kernel, as now) and fig06 (all kernels; raise its
  kdn grid from 30 points to match) at the new resolution.

**As-built (2026-09-20):** the nine non-Gaussian kernels (Gaussian skipped
per the decision above) at 60 N_H (log, per-kernel h/d_nn 0.9–3.0) ×
200 |k|d_nn (log), both k-directions, per-kernel `results/*.npz` cache
(~370 KB each, two-mode: lon + smallest transverse per direction); the
masked B_j (phase-5 log entry (m)) made the large-N_H rows tractable;
staged in two sets of 4/5, ~7 h wall-clock, 0 errors.
`check_stability_boundaries.py` verifies the boundaries from the cached
fields (two-mode separation: longitudinal = the "accessible N_H" metric,
transverse = the generic no-shear-stiffness pathology, reported
separately). Longitudinal onsets: cubic 62 (≲55 ✓), quartic 66 (≈67 ✓✓),
quintic 225 (≈190 ~), W2 clean (3-pt edge artifact near 40 ✓), b7 193,
b8 549, W4/W6 fully clean (0.00 %), HOCT4 island N_H 114–185 (centre
≈145 — the "island near 150", ✓, at |k|d_nn 5.6–6.0). The b7/b8 onsets
sit at h/d_nn ≥ 1.35/1.80: the Ŵ(H|k|)<0 pairing region pushed to large
N_H by the higher order (both clean in the practical h/d_nn ≲ 1.2
regime) — consistent with the paper's mechanism; the plan's "clean
otherwise" was stricter than the paper's own contours. The cubic long-λ
dip box (|k|d_nn 0.3–0.6, N_H 40–100) is EMPTY (0 longitudinal unstable
points) → the Phase-5 discrepancy was the phase-reference bug (entry i),
not physics. fig04/05 (per-kernel) + fig06 (all nine, 200 kdn points)
regenerated at the new resolution.

### 6. `REPORT.md` — [ ] (after 5)
Per PLAN Phase 8:
- Per-figure comparison, paper vs replication: Figs 1–6 + figA, Tables 1–2
  (note the Phase-4 glass-proxy caveat and the no-image-input limitation:
  PNGs are for the user's visual check).
- The discrepancy table from the findings log, including the Phase-5 cubic
  long-λ instability (the one open `[ ]` in Phase 5) with its hypotheses
  (`phase5_stability_log.md` entry f).
- Implications for `warpSPHCore` kernel defaults: Wendland2 + its N_H; the
  B7→B8 rename + classical-B7 outcome; the `renorm` ε scope (item 4); the
  Phase-3 kernel additions; the Phase-6 frontend work (C&D sign resolution
  B11, R&H SPHS module, 1D/2D/3D Sod validation numbers from the warpSPH
  phase-6 log).
- Closeout `src/` decisions: the support.py BinOp fix (item 2), the
  `sampleOptimal` fix (item 3), and the exclusion of the five specialty
  kernels (decisions above).

### 7. Build the ε correction (warpSPHCore + warpSPH) — [x] DONE 2026-09-19
(User decision 2026-09-19, supersedes the item-4 scope-only decision:
"Go build the whole thing". Built 2026-09-19, before item 5.)
- **Refit** (`scripts/eps_constants_multidim.py`): 8 kernels (Wendland +
  B-splines) × 3 dims on the densest lattice (1D uniform / 2D hexagonal /
  3D FCC — the same source as fig03), fig03's N_H grid, fit windows
  40-400/40-800/20-800, dense band sweeps measured with the LIBRARY's
  runtime convention (ε at N_H,est = V_d·h^d·ρ̂/m, since the exact N_H
  is unknown at runtime; at the window edges the two conventions differ
  by O(α·bias), residual by α·bias²). Self-test: 3D reproduces fig03
  check #4 at x1.005 (identical computation) and the paper at x1.5.
  All 24 fits: `results/eps_constants_multidim.json` (gitignored).
- **Shipped constants** (`warpSPHCore/util/densityCorrection.py`, 6
  entries, window 40-400): the three Wendland in 2D AND 3D. NOT shipped:
  1D (raw estimate already ≤1e-6 of exact over 40-400; implied ε not a
  power law — crosses zero) and the B-splines 2D/3D (lattice bias
  oscillates with N_H; the fitted power law's "corrected" band is WORSE
  than the raw one for 8 of the 10 entries; quartic-3D fits a negative
  α). 3D agrees with the paper within x1.04; corrected band over the
  closed window 0.27-1.0 % (3D) / 3e-5-2.2e-4 (2D), mid-window ~1e-4 /
  1e-5.
- **Util** (warpSPHCore `bf14e56`): `densityCorrectionConstants` /
  `selfTermW0` (one-time host-call `C_d·f(0)` cache) /
  `applyDensityCorrection` (pure torch; W0 at the code support via the
  shipped host-callable eval fns) + `util/__init__.py` exports.
- **warpSPH frontend** (`3de37e7`): `DensityCorrection` dataclass
  (enabled/eps100/alpha) as a SimulationConfig SIBLING of
  `calibrateNormalization` (user: "making it a sibling next to
  calibrateNormalization is probably the easiest option"); buildConfig
  bool shorthand + the generic nested-dataclass encode/decode branches;
  CaseSpec bool (`--densityCorrection`); runner pass-through; the
  `computeDensities` hook (pure-torch post-process — the warp kernel is
  unchanged; off = the operator's tensor untouched, bit-exact).
- **Verified:** warpSPHCore 46/46 (unit, CPU f32); warpSPH 45/45
  (frontend, CPU f32, real operator + Verlet list on the refit's own
  lattice-box periodic domain); regression
  test_caseSpec/test_runner/test_latticeDensity: 170 passed / 1 failed —
  the failure is PRE-EXISTING and unrelated (test_runner.py::
  test_everyCaseDeclaresItsParamsAsScalarsOrLists: dambreak.
  surfacePressureProbes is a tuple, introduced in 252d857). Design note
  updated (status; §1 stale-quote fix 0.02834/0.01335/0.01220 →
  0.02949/0.01361/0.01131; §3 as-built config; §4 runtime-convention
  note + 2D extension; §6 shipped table + measured non-ship reasons;
  §8/§9 actuals).

## Progress log
- 2026-09-19: closeout plan created; item 1 (PLAN.md bookkeeping) done.
- 2026-09-19: item 2 (support.py float32-BinOp constants) DONE + committed.
  Module-level constants + int-cast fix; float64 probe now exact (0.0 rel),
  float32 ~7.5e-8 (f32 round-off); full suite 451 passed / 1 skipped / 1
  unrelated pre-existing failure. Also surfaced that the function previously
  did not compile in any build (no int32×float64 mul overload) — the
  float32-constant bug was latent behind that.
- 2026-09-19: item 3 (warpSPH `sampleOptimal`) DONE + committed in warpSPH.
  Fixed the three findings-log bugs (ParticleState, −CFL·Ma·2h² scaling,
  keep the jittered lattice) plus two more latent ones (unimported
  ParticleSet, ignored `kernel` arg); added a `seed` kwarg. Verified:
  per-particle match to 4.4e-15 vs an unmodified CPU run of the original
  glass-recipe probe; recipe acceptance met on CPU (std 0.283 %, nn/dx
  0.8863); the 3.4e-2 gap to the cached CUDA npz is device-level fork
  (same for the unmodified CPU recipe) — not an implementation deviation.
- 2026-09-19: item 4 (renorm ε scope) DONE — `renorm_eps_design_note.md`
  written (all six questions: naming collision with the existing gradient
  `renorm`, outside-operator hook at the two scheme density sites, N_H from
  the raw estimate, W0 convention + kernelScale³ gotcha, 3-Wendland-only
  constants table, calibrateNormalization coexistence rule, test spec).
  Not built, per the user decision.
- 2026-09-19: item-4 follow-up (user option B) — monaghan/compSPH density
  calls ported onto `computeDensities` (optional `supportMode` param,
  default Gather; monaghan passes `config.supportMode`, default
  SuperSymmetric → zero behavior change; warpSPH 27b9934). Verified
  bit-exact against the old direct `warpOperation(Density)` calls on a
  non-uniform-support state in both modes (mode spread 1.3e-1 rel —
  non-vacuous), imports clean in
  float64 + float32. The ε hook is now a single call site inside
  `computeDensities` (design note §3 updated). Scheme-level GPU tests
  (runner auto-selects CUDA) deferred — GPU held by the local LLM.
- 2026-09-19: item 7 (build the ε correction) DONE + committed in both
  repos (warpSPHCore bf14e56, warpSPH 3de37e7). Multi-dim refit (8
  kernels × 3 dims, densest lattice, fig03 grid + dense runtime-convention
  bands; 3D self-test vs fig03 check #4 x1.005 / paper x1.5); 6 Wendland
  2D/3D entries shipped, 1D + B-splines excluded with measured reasons
  (see item 7). warpSPHCore util (pure torch, warp kernel untouched) +
  46/46 unit tests; warpSPH sibling config (DensityCorrection dataclass
  next to calibrateNormalization, bool shorthand, --densityCorrection
  CLI flag) + computeDensities hook + 45/45 frontend tests — all green
  on CPU f32. Regression test_caseSpec/test_runner/test_latticeDensity:
  170 passed / 1 failed, the failure PRE-EXISTING and unrelated
  (dambreak.surfacePressureProbes tuple params, 252d857). Design note
  updated (status BUILT; §1 stale quote fixed; §3/§4/§6/§8/§9 as-built).
- 2026-09-20: item 5 (high-res sweep) STARTED. (a) Masked B_j in
  `stability.py::exact_p_matrix` (per-point O(n1·n2) → O(nc)~O(N_H²); the
  dropped terms are exactly zero — compact support at H): re-validated
  vs the ground-truth real-FD Jacobian (P sym ~1e-14, max|Pfd-Pex|
  0.000–0.19 = FD truncation). (b) fig04/05 gained `--grid ROWSXKDN`
  (default 14x26), LOG-spaced N_H rows, per-kernel `results/*.npz` caching
  (load-if-present, resume on interrupt), full-set default; fig06 kdn
  40→200. (c) Timing probe: cubic ~9 min, W2 ~13 min, b8 ~66 min at 60x200;
  the Gaussian's dense oracle BUILD OOMs above N_H≈7000 (measured peak RSS
  3.1→23.8 GB over N_H 2211→7000) — per user decision the sweep SKIPS the
  Gaussian (largest by far, not practical in a real sim) and runs STAGED
  (few kernels at a time, box is shared). (d) Stage 1 (cubic, quartic,
  quintic, W2) DONE at 60x200. (e) `check_stability_boundaries.py` written
  to verify the acceptance boundaries from the cached fields. KEY FINDING:
  the first check run conflated two modes — the TRANSVERSE (shear) mode is
  broadly ω²<0 (62–82 % of the field, EVERY kernel) as a generic SPH
  pathology (no shear stiffness), which is NOT the pairing instability; the
  longitudinal (pairing) mode is the small 5–8 % region and is the correct
  "accessible N_H" metric. After separating the two, the longitudinal
  onsets match the Phase-5 expectations: cubic ~62 (≲55 ✓), quartic 66
  (≈67 ✓✓), quintic 225 (≈190, ~), W2 essentially clean (3-pt edge artifact
  near N_H 40, ✓). The cubic long-λ dip box is EMPTY (0 longitudinal
  unstable points) → confirms entry (i) it was the phase-reference bug, not
  physics. (f) Stage 2 (b7, b8, W4, W6, HOCT4) launched. Two-mode cache
  format (lon + smallest transverse) confirmed sufficient by the user for
  threshold reprocessing — no all-three-eigenvalue storage.
- 2026-09-20: item 5 DONE + committed. Stage 2 cached b7 (09:32), b8
  (10:42), W4 (11:12), W6 (12:22), HOCT4 (12:48) — nine of nine kernels
  at 60x200, 0 errors, ~7 h wall-clock over the two stages. Stage-2
  boundary check: HOCT4 "island near 150" CONFIRMED (N_H 114–185, centre
  ≈145, |k|d_nn 5.6–6.0, 0.23 % of the field); W4/W6 fully clean (0.00 %
  longitudinal); b7 onset 193 (h/d_nn 1.35) and b8 onset 549 (h/d_nn
  1.80) are high-h/high-|k| Ŵ<0 pairing islands only — both clean below
  h/d_nn 1.2 (the plan's "clean otherwise" was stricter than the
  paper's own contours; see item-5 as-built note). All Phase-5
  acceptance boundaries now verified; the cubic long-λ dip box is EMPTY.
  fig04/05 (nine kernels) + fig06 (nine kernels, 200 kdn points)
  regenerated. Phase-5 log entry (n).
