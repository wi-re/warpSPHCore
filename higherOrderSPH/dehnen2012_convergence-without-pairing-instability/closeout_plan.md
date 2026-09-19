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

### 5. 10-kernel stability sweep, high-res fig04/05/06 — [ ]
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
