# Phase 6 prompt — shock-capturing frontend build-out in `warpSPH`

> Copy the block below into a **fresh Qwen Code session started in
> `/home/lu26029/dev/warpSPH`**. It is self-contained. This file is a planning
> artifact and lives in the dehnen2012 replication folder next to `PLAN.md`;
> reference it from the warpSPH session by its absolute path if you like.

---

```text
Task: Phase 6 of the Dehnen & Aly (2012) replication — build out the
compressible shock-capturing frontend in warpSPH: (1) validate + clean the
existing Cullen & Dehnen (2010) viscosity switch on the BASIC Monaghan
compressible scheme, and (2) add the Read & Hayfield (2012) SPHS
artificial-conductivity module as a first-class, selectable option. Then
validate both against the Sod shock tube (1D -> 2D -> 3D) with an
exact-solution overlay.

WORKING DIRECTORY: /home/lu26029/dev/warpSPH  (a git repo; do NOT push; do NOT
modify the sibling /home/lu26029/dev/warpSPHCore except to READ reference data).
Conda env: /home/lu26029/miniconda3/envs/warp/bin/python (warp 1.17.0).
warpSPH is editable-installed, so `import warpSPH` picks up local edits.

## KEEP A LOG (do this first, and keep doing it)
Sessions can crash mid-task on a context limit. BEFORE starting any work, create
`/home/lu26029/dev/warpSPH/phase6_shock_capturing_log.md` and keep it current at
EVERY step: what was tried, the result (numbers / which test passed), and the
next step. Newest entries at the BOTTOM. This is the single most important
resumability safeguard — if you lose the rest of the context, this log is what
lets the next session pick up cleanly. Re-read it at the top of any resumed
session.

## Background (read before editing)
The Cullen & Dehnen (2010) switch is a BASIC-COMPRESSIBLE (Monaghan-style)
construct — it is NOT a CRKSPH formulation. In warpSPH the scheme layout is:
- `src/warpSPH/schemes/monaghan.py` — the basic compressible scheme. It
  currently does NOT call the viscosity switch (no `shockCapturing` import); it
  uses the standard `computeViscosity` from `modules/dissipation`, which DOES
  accept per-particle alphas via `referenceAlphas`.
- `src/warpSPH/schemes/compSPH.py` — DOES call the switch:
    `currentState.alphas, switchState = computeViscositySwitchTerms(dt,
    currentState, config, schemeConfig, SupportScheme.SuperSymmetric, adjacency)`
    then passes `queryAlphas = currentState.alphas` into the accel/dudt, then
    `currentState.alpha0s, switchState = updateViscositySwitch(switchState, dt,
    dvdt, currentState, config, schemeConfig, SupportScheme.SuperSymmetric,
    adjacency)`. THIS IS THE WIRING PATTERN TO MIRROR in monaghan.py.
- `src/warpSPH/schemes/crkSPH.py` — also calls `computeViscositySwitchTerms`,
  but CRKSPH is the common DEFAULT case scheme and is NOT the switch's physical
  target. Investigate whether its call site is correct or vestigial, but treat
  the Monaghan scheme as the authoritative validation target.

So the FIRST job is to wire the C&D switch into `schemes/monaghan.py` (mirror the
compSPH.py pattern above) and validate it THERE, before any R&H 2012 or 3D-Sod
work.

## What already exists (build on it, don't rebuild)
- `src/warpSPH/modules/shockCapturing/`:
  * `CullenDehnen2010.py` (384 lines) — implements R (paper eq.17), Xi (eq.18),
    v_sig, `computeCullenTerms` (eqs.13,16), `computeCullenUpdate`. KNOWN ISSUES:
    (a) `computeSecondOrderV` carries the comment "The signs here should have
    been wrong, double check!" — the active path returns `divdotdvdt - tr(V^2)`;
    there is a DEAD alternative `return tr(A + V^2)` after the first return;
    (b) leftover commented-out SPHOperation lines + the dead branch.
  * `CullenHopkins.py` (implemented), `wrapper.py` (dispatch on
    `schemeConfig.viscositySwitchParams.scheme`), `switchState.py`
    (`ViscositySwitchState`), `common.py`, `wp_computeM.py`, `wp_vsig.py`.
- `src/warpSPH/enumTypes.py`: `ViscositySwitch` enum = {Balsara1995=0,
  Colagrossi2004=1, CullenDehnen2010=2, CullenHopkins=3, MorrisMonaghan1997=4,
  Rosswog2000=5, NoneSwitch=6}. NO Read & Hayfield 2012 entry yet.
- `src/warpSPH/configurations/moduleConfigurations/viscositySwitchParameters.py`:
  `ViscositySwitchConfig` (alpha_min/max, beta_c/d/xi, limitXi [declared twice —
  the 2nd, True, is live], correctVelocityGradient, divergenceScheme).
- `alpha0s` lives on `CompressibleState`
  (`src/warpSPH/systems/compressibleMonaghan.py`).
- Validated exact-solution Riemann solver:
  `src/warpSPH/caseUtils/compressible/sod/sodSolution.py` —
  `calculate_regions(pl, ul, rhol, pr, ur, rhor, gamma)` and top-level
  `solve(left_state, right_state, geometry, t, gamma, npts)`. USE it, do not
  re-derive. **It accepts ARBITRARY left/right Riemann states**, so the D&A IC
  is just a parameter choice — no special-casing needed.
- Existing 1D Sod case `src/warpSPH/cases/sod.py` DEFAULTS TO THE CLASSIC Sod IC
  (1,1,0)->(0.25,0.1795,0) — that is NOT the D&A problem (see reference data).
- Control case: `src/warpSPH/cases/greshoVortex.py` (2D CRKSPH, steady exact
  solution). Tests: flat `tests/test_*.py` (pytest).

## Reference data (READ-ONLY, sibling repo)
/home/lu26029/dev/warpSPHCore/higherOrderSPH/dehnen2012_convergence-without-
pairing-instability/data/da2012_reference.yaml
  * `rnh2012_sphs:` — the FULL Read & Hayfield (2012) SPHS transcription:
    switch eq.21, relaxation eqs.22-25, viscosity eqs.29-31, Balsara limiter
    eq.32, entropy-dissipation eqs.33-35, mass-dissipation eqs.36,39-40; params
    ns=0.05, alpha_max=1, alpha_min=0.2, balsara_const=1e-4; the 2nd-order
    Maron-Howes-style gradient estimator (10x10 moment matrix in 3D).
  * `sod:` — D&A Sod IC = the R&H 2012 problem: gamma=5/3,
    L=(rho=1.0,p=1.0,u=0), R=(rho=0.125,p=0.1,u=0), t=0.2. Exact Riemann
    (validated solver): p*=0.29394519, rho*=0.47968906, u*=0.84119485,
    rho_post=0.22980575, S_shock=1.84447337; discontinuities at t=0.2:
    head=-0.258199, foot=-0.033880, contact=0.168239, shock=0.368895 (paper
    quotes contact ~0.17, shock ~0.378).
  * `rnh2012_sphs.sod_setup_phase6`: "domain [-0.5,0.5], 3D, 32x32x400 +
    16x16x200 lattice, N_1D=600, t=0.2, initial alpha=1 on -0.05<x<0.05".

## Papers (local PDFs in /home/lu26029/dev/warpSPH/literature/)
- Cullen & Dehnen 2010: cullen2010_inviscid-sph.pdf
- Read & Hayfield 2012 (SPHS): read2012_sphs-higher-order-dissipation-switch.pdf

## Deliverables (in this order)
1. **Wire + validate the C&D switch on the basic Monaghan scheme FIRST.**
   - Mirror the compSPH.py call pattern into `schemes/monaghan.py`
     (computeViscositySwitchTerms -> queryAlphas into the accel ->
     updateViscositySwitch). Keep NoneSwitch as the off path.
   - Validate equation-by-equation against cullen2010: RESOLVE THE SIGN
     QUESTION in `computeSecondOrderV` (derive the correct 2nd-order divergence
     ∇·v̇ — paper eq. B11 `∇·v̇ = ∇·(dv/dt) - tr(V^2)` vs the alternative
     `tr(A+V^2)` — and document which is right and why). Remove (or clearly mark
     `_deprecated_`) the dead alternate formulations + commented-out
     SPHOperation lines. Verify R (eq.17), Xi (eq.18), target-alpha (eq.13), and
     the l=0.05 decay integration (eq.16).
   - **Debug in 1D and 2D FIRST** (cheap, fast iteration) before any 3D work:
     a small 1D Sod or 2D shock/rarefaction run to confirm the alpha field looks
     sane (switches off in smooth/shear flow, up at the shock), no instability,
     and the contact overshoot responds correctly. Only move to 3D once 1D/2D
     behave.
2. **Add `ReadHayfield2012.py`** implementing SPHS artificial conductivity from
   the `rnh2012_sphs` transcription (switch eq.21, relaxation eqs.22-25,
   viscosity eqs.29-31, Balsara limiter eq.32, entropy-dissipation eqs.33-35).
   Follow the CullenDehnen2010.py structure (compute*Terms / compute*Update + a
   ViscositySwitchState). Its job per D&A §4.3: suppress the thermal-energy
   overshoot at the contact discontinuity.
3. **Wire both in as first-class selectable options**: add `ReadHayfield2012` to
   the `ViscositySwitch` enum; extend the `wrapper.py` dispatch; add any new
   params to `ViscositySwitchConfig` (+ to-dict/from-dict). Keep the Monaghan
   path as baseline (NoneSwitch).
4. **Control test (cheap, 2D)**: run `cases/greshoVortex.py` with and without the
   C&D switch — the switch must remove the shear-driven artificial dissipation
   (match the no-viscosity solution).
5. **Sod validation (1D -> 2D -> 3D)**: run the D&A Sod IC (reference data) with
   (a) NoneSwitch, (b) CullenDehnen2010, (c) ReadHayfield2012 (conductivity on).
   Overlay the exact Riemann solution (sodSolution.solve) at t=0.2. Acceptance:
   the contact discontinuity's thermal-energy/pressure overshoot is SUPPRESSED by
   R&H 2012 vs NoneSwitch, and all three reproduce the exact discontinuity
   positions (head/foot/contact/shock) within SPH smearing.

## Validation harness
Add `tests/test_shockCapturing.py` (pytest): equation-level unit checks (R, Xi,
switch values on a manufactured state) + a 1D Sod overlay check asserting the
contact-overshoot metric. Save overlay PNGs for the user's visual check (the
model CANNOT view images — the user is the visual-check channel).

## Process
- Work incrementally; run the relevant pytest after each module.
- Do NOT push. Commit per deliverable (check `git log` for message style first).
- Keep `phase6_shock_capturing_log.md` current at every step (see top).
- Report: the sign-question resolution (with the derivation), the R&H equation
  mapping, the control-test result, the 1D/2D debug findings, and the Sod overlay
  numbers (discontinuity positions + contact overshoot before/after).
```


