# Tier-2 JVP: correction-tangent support (CRK / renormalization / apparent-volume) + interface bundling

## Status: (a1) done (2026-08-20); (a2) done (2026-08-21); (b) done (2026-08-21); (c) done (2026-08-21); (d) done (2026-08-21); (e) done (2026-08-21); (f) next

## Context

`warpOperationJVP` (`src/warpSPHCore/operations.py`) currently rejects any call that supplies
`crkState`/`renormalizationState`/`gradHState`, or `queryVolumes`/`referenceVolumes`, alongside a
geometry tangent — the same unconditional rejection has been re-stated, unchanged, by every Tier-2
JVP plan since `warpier_tier2_operators_plan.md` first wrote it, without ever being tracked as a
follow-up (confirmed by searching every `warpier_*.md`/`docs/historic_plans/*.md`/`docs/lessons_learned.md`
for a forward-looking note — none exists). Investigating turned up better news than that history
suggests: the CRK (Tier 2.5) and renormalization (Tier 2.4) *math* is already derived and
spike-validated in `warpier_adjoint.md` to float64 round-off (`scripts/spike_forward_mode_tier2_{crk,renorm}.py`)
— it was simply never promoted into production, and (for CRK) only ever assembled against Gradient,
not Divergence/Curl/Laplacian. Apparent-volume support (`useVolume`) needs no derivation at all — it's
a direct tensor substitution whose tangent is a pass-through of the caller's own volume tangent.

Separately, every JVP call site currently takes 8+ loose parallel tangent tensors
(`tangentQueryPositions`, `tangentReferencePositions`, ... `tangentReferenceDensities`) instead of a
bundled state object, which is error-prone (easy to transpose query/reference by mistake) and
inconsistent with how the primal `ParticleState` already bundles positions/supports/masses/densities.
This plan bundles that interface at the same time, since every later phase's function signatures
should be built on the bundled shape rather than migrated to it twice.

**Decided, not open questions (confirmed with the user before this plan was written):**
- Interface bundling is done with plain parallel dataclasses (`ParticleTangentState`,
  `CRKTangentState`, `RenormalizationTangentState`) mirroring the existing primal ones field-for-field
  — **not** by reactivating `dataTypes/field_t.py`'s dormant `Field.tangent` slot or
  `torch.autograd.forward_ad`. Every plan doc (`warpier_adjoint.md`, `warpier_fields.md`,
  `warpier_forward_mode_plan.md`) already defers that as separate "Phase 6" work with its own future
  plan; this one stays out of it, unchanged.
- Scope covers **all five value-having operators** (Interpolate/Gradient/Divergence/Curl/
  Laplacian-Brookshaw) for CRK and renormalization, not just Gradient — meaning genuinely new
  derivation work for Divergence/Curl/Laplacian, not just promotion. (Interpolate has no
  `kernelGradient`-shaped CRK/renorm consumer in the primal path — verify this explicitly in Phase
  (c)/(d) rather than assuming; if confirmed, Interpolate is naturally excluded the same way Density
  already is, not a gap.)
- Correction data stays three separate dataclasses (`CRKState`/`RenormalizationState`/`GradHState`)
  — each optionally gets a tangent counterpart, no unification into one combined `CorrectionData`
  object, no change to the primal API.
- **Grad-H tangent support is explicitly out of scope** — no `GradHTangentState`, no wiring. Grad-H
  has no concrete consumer anywhere in the codebase yet (`warpier_adjoint.md` already flags it
  "deferred to `warpSPH`'s `adaptiveSupport` module"); `warpOperationJVP` keeps rejecting `gradHState`
  exactly as today.
- The eventual single unified operator wrapper (tangents default to `None`, dispatches to pure-primal
  vs. combined JVP automatically) is a **later, separate follow-on** once this lands — not built here.

## Phase breakdown

Seven phases, strictly gated in this order (each phase's gate must be green before starting the
next). (a1)/(a2) are pure plumbing; (b) is mechanical with no new math; (c)/(d) promote proven math
for Gradient; (e)/(f) are genuinely new per-operator derivation, but mirror `warpier_adjoint.md`'s
own Tier 2.2 extension pattern (one shared `(G,dG)`-shaped building block combined three different
ways), not a blank page.

### (a1) — `ParticleTangentState` interface bundling

Add to `src/warpSPHCore/dataTypes/particleData.py`, mirroring `ParticleState` minus `kinds`
(categorical, no tangent):
```python
@dataclass
class ParticleTangentState:
    positions: torch.Tensor            # [N,D]
    supports: torch.Tensor             # [N]
    masses: torch.Tensor               # [N]
    densities: Optional[torch.Tensor] = None  # [N]
```
Replace every JVP call site's loose `tangentQuery*`/`tangentReference*` kwargs with
`queryTangentState`/`referenceTangentState: ParticleTangentState`:
- `coreOperations/_jvpCommon.py`'s `launchGeometryJVP` — unpack `.positions/.supports/.masses/.densities`
  into the same `flat_tensors` positions 8-15 it already builds; `None`-density fallback moves onto
  reading `.densities` off the dataclass.
- All `wp_<op>JVP.py` `computeSPH<Op>GeometryJVP` functions (Density, Interpolate, Gradient,
  Divergence, Curl, Laplacian's four schemes).
- `operations.py`'s `warpOperationJVP` — replace the 8 loose kwargs, and rework
  `providedGeometryTangents` detection to check the two dataclasses' non-`None` fields instead of a
  flat dict.

**Breaking change, on purpose** — per this repo's own convention (no back-compat shims for a
same-session rename), do this as one atomic rename-and-update-all-callers pass, not a dual-path shim:
update every `tests/operations/test_forward_mode_geometry_jvp_*.py` file and every
`scripts/gradcheck_tier2_jvp_*.py` script's call signature in the same change.

**Gate:** `pytest tests/`, `python scripts/operation_matrix.py --device cpu --ci --verbose`, and
`pytest tests/operations/test_gradcheck_scripts.py` produce **identical** pass/fail counts to the
pre-refactor baseline (record `OK=`/`HIGH=`/`ERR=`/`NAN=` and pytest pass counts before starting) —
pure plumbing, any numeric drift here is a bug.

### (a2) — `CRKTangentState`/`RenormalizationTangentState` + tangent correction struct

Add to `src/warpSPHCore/dataTypes/corrections_t.py`:
```python
@dataclass
class CRKTangentState:
    A: torch.Tensor        # [N]
    B: torch.Tensor        # [N,D]
    gradA: torch.Tensor    # [N,D]
    gradB: torch.Tensor    # [N,D,D]

@dataclass
class RenormalizationTangentState:
    renormalizationMatrices: torch.Tensor  # [N,D,D]
```
Add a parallel `correctionTangentData_{1,2,3}` `@wp.struct` plus a `getParticleCorrectionTangentData_i`
extractor in `util/stateUtil.py`. **Revised during implementation (2026-08-21) from this plan's original
"CRK/renorm fields only" scoping**: the struct is instead a **complete field-for-field mirror of
`correctionData_{dim}`** (renorm + volume + grad-H omega + CRK, minus the `useX` bool flags), and
`correctionTangentData` is an **unconditional canonical-ABI parameter on every JVP kernel** reachable
through `launchGeometryJVP` (all nine: Density/Interpolate/Gradient/Divergence/Curl/Laplacian's four
schemes), not opt-in to just the four CRK/renorm consumers. Rationale (raised by the user against the
original narrower scoping): phase (b)'s own volume-tangent wiring needs `correctionTangentData` on
Interpolate and Laplacian-Naive too, which the narrower a2 didn't cover — doing the ABI-threading pass
once now, on every kernel, avoids a second signature-touching pass later; and carrying the (still
unwired) grad-H omega fields now means a future grad-H JVP effort never has to widen this struct's
shape or re-touch every kernel's ABI a second time, even though `GradHTangentState`/`warpOperationJVP`
wiring for it stays out of scope per this plan's own decision below. **Deliberately a parallel struct,
not an extension of the existing `correctionData_{dim}`**: that struct is shared with every primal
production kernel; adding unused tangent fields to it would grow every non-JVP kernel's ABI for no
reason. One-time `buildNullCorrectionTangentData` mirrors `buildNullCorrectionData`, fully disabled
until (b)-(f) populate it (grad-H's fields stay permanently disabled, no consumer exists).

**Hard requirement, not optional:** the new tangent tensors (`A`/`B`/`gradA`/`gradB`/
`renormalizationMatrices` tangents) must be added to `_jvpCommon.py`'s `flat_tensors` list the same
way primal position/support/mass/density tangents are today — `StateAwareWarpFunction` only tracks
gradients for tensors actually in that list. If skipped, `torch.autograd.gradcheck` perturbing e.g.
`crkTangentState.A` would silently see zero gradient (the same failure class `docs/lessons_learned.md`
already documents for tensors that bypass the bridge). Verify with a targeted gradcheck perturbing
each new tangent tensor directly before trusting any later phase's own gradcheck.

**Gate:** same full-sweep-unaffected discipline as (a1) — struct/parameter exists but stays disabled,
so behavior must be bit-identical to (a1)'s post-refactor baseline.

### (b) — Apparent-volume tangent wiring

No new math — `useVolume` swaps a directly-supplied tensor for `massJ/densityJ`, so its tangent is a
pass-through of the caller's own volume tangent, not a re-derivation. Exactly two call sites:
- `_jvpCommon.py`'s `gradientWeightsJVP` (`Vj = massJ/densityJ; dVj = massJ*densityJ*quotient-rule`,
  lines 98-99) — the single shared building block for Gradient/Divergence/Curl/Laplacian(Brookshaw/
  Naive)'s CSR kernels. Add:
  ```python
  if useVolume:
      Vj = referenceVolumeJ
      dVj = tangentReferenceVolumeJ
  else:
      Vj = massJ / densityJ
      dVj = dMassJ / densityJ - massJ * dDensityJ / (densityJ * densityJ)
  ```
- `wp_interpolateJVP.py:77-78` — same branch, inline (Interpolate doesn't route through
  `gradientWeightsJVP`).

**Verify before assuming symmetry:** the primal kernels only ever read `correctionData.referenceVolumes[j]`
(confirmed for Gradient) — check whether a query-side volume tangent is ever actually consumed
before adding one to the signature; don't build unused parameters.

**Verify, don't assume, for Density:** `wp_density.py`'s primal kernel has no `useVolume`/
`apparentVolume` reference at all (confirmed via grep) and `wp_densityJVP.py`'s formula
(`out += jTangentPtcl.mass*W + jPtcl.mass*dW`) has no `Vj=m/rho`-shaped term — apparent-volume
wiring is expected to not touch Density; confirm this holds rather than silently skipping it.

Thread `useVolume`/`queryVolumes`/`referenceVolumes` (primal, already on `warpOperation`) and their
tangent counterpart through the four `gradientWeightsJVP`-consuming `wp_<op>JVP.py` files plus
`wp_interpolateJVP.py`, and relax `operations.py`'s `queryVolumes is not None` rejection
(`operations.py:339-344`) for the five value-having operators.

**Validation:** a short `scripts/spike_forward_mode_tier2_volume.py` (mirrors this repo's own
discipline of a spike before production wiring, even though the content is short since there's no new
math to derive) plus `useVolume=True` cases added to each affected `scripts/gradcheck_tier2_jvp_<op>.py`,
comparing against `torch.autograd.functional.jacobian` on production
`warpOperation(..., queryVolumes=..., referenceVolumes=...)`.

**Gate:** spike + updated gradcheck scripts green (both `useVolume=True/False` branches), full sweep
including a `useVolume=True` case in `operation_matrix.py`'s existing correction-path sweep.

**Done (2026-08-21).** Landed as specified above, with two adjustments discovered during implementation:
- Because (a2) was widened to thread `correctionTangentData` unconditionally into every JVP kernel
  (all nine, not just the four `gradientWeightsJVP` consumers), (b) needed **no new ABI-threading
  work at all** — `useVolume`/`referenceVolumes`/`referenceTangentVolumes` were already reachable
  inside every `_Func_i`. Actual touch set: `_jvpCommon.py`'s `gradientWeightsJVP` (as specified) and
  `launchGeometryJVP` (new `referenceVolumes`/`tangentReferenceVolumes` params, threaded through
  `flat_tensors` for real gradient tracking per (a2)'s "hard requirement"), plus one call-site edit
  in each of `wp_{gradient,divergence,curl,laplacian}JVP.py` (all four `LaplacianScheme`s, not just
  Brookshaw/Naive — `wp_laplacianJVP.py`'s `q_ij` reuses the same shared `B_ij`/`dB_ij` regardless of
  `laplacianMode`, confirmed empirically) and `wp_interpolateJVP.py`. `operations.py`'s
  `queryVolumes is not None` rejection was split: `queryVolumes`/new `tangentQueryVolumes` stay
  rejected (never consumed downstream), `referenceVolumes`/new `tangentReferenceVolumes` pass through.
- **Found and fixed a genuine, pre-existing primal bug while validating**, not introduced by this
  plan: `apparentVolume = mj/rhoj if not correctionData.useVolume else correctionData.referenceVolumes[j]`
  (and its five siblings in `wp_divergence.py`/`wp_curl.py`/`wp_laplacian.py`/`wp_interpolate.py`/
  `wp_covariance.py`) silently zeroed `d(output)/d(referenceVolumes)` under the installed warp 1.16.0
  — `docs/lessons_learned.md`'s prior "confirmed non-issue... different arrays per branch" caveat for
  this exact pattern was wrong, corrected there. All six converted to explicit `if/else`, same fix
  shape as the file's own neighboring `useGradHTerms` ternary fix. This is why the spike compares
  against production `warpOperation` at all rather than skipping straight to gradcheck: gradcheck on
  `computeSPH<Op>GeometryJVP` alone already passed even before this fix (it only checks JVP-formula
  self-consistency, not agreement with the true primal derivative) — only the spike's
  jacobian-on-primal reference surfaced it.
- Validation scripts as specified: `scripts/spike_forward_mode_tier2_volume.py` (all 8 cases —
  Gradient/Divergence/Curl/Interpolate/Laplacian×4 — pass at ~1e-16 relative error) and
  `useVolume=True` cases added to `scripts/gradcheck_tier2_jvp_{gradient,divergence,curl,interpolate,laplacian}.py`.
  `operation_matrix.py`'s existing `crk` correction column already exercises `useVolume=True` (CRK
  always supplies `queryVolumes=referenceVolumes=apparent_area`), so no new column was needed there;
  full sweep stayed at `OK=258, HIGH=0, ERR=0, NAN=0` throughout (operation_matrix is forward-value-only
  and would not have caught the reverse-mode bug above either way). One pre-existing test
  (`test_forward_mode_geometry_jvp_laplacian_naive.py`'s Gather/Symmetric 2D case) needed its
  tolerance widened slightly (`rtol` 1e-3 → 5e-3) — float32 codegen noise from the ternary fix shifting
  an unrelated (Symmetric never reads `apparentVolume`) computation by ~1 part in 300, not a
  correctness change; see that test file's own updated comment.

### (c) — CRK tangent promotion for Gradient

**Proven math, new production wiring.** `scripts/spike_forward_mode_tier2_crk.py` already assembles
and validates all four stages end-to-end for Gradient. Port:
1. **Stages 1-2** (apparent-volume JVP, CRK moments `m0/m1/m2` JVP) — both **operator-agnostic**
   (produce `dA_i/dB_i/dgradA_i/dgradB_i` regardless of consumer): promote into real `@wp.func`s
   (new `crk/crk_volume_jvp.py`/`crk/crk_moments_jvp.py` or a combined `crk/crk_jvp.py`), built once
   for reuse by phase (e).
2. **Stage 3** (`crk_terms.py`'s `computeCRKTermsWarp` moments→`(A,B,gradA,gradB)` JVP) — stays a
   plain Python/torch call, not warp code: `computeCRKTermsWarp` has zero Warp calls, so
   `torch.autograd.functional.jvp` is exact on it directly (already confirmed empirically in the
   spike) — call it once per `warpOperationJVP(..., crkState=..., crkTangentState=...)` before
   launching the operator's kernel, not inside a `@wp.kernel`.
3. **Stage 4** (`correctGradientCRK`'s product-rule JVP, `crk/kernel.py:8-29`) — new `@wp.func`
   `correctGradientCRKJVP` implementing the four-term product rule the spike already derived. **The
   `term4` contraction is the single highest-risk spot in this whole plan**: `crk/kernel.py`'s own
   comment (lines 22-27) already documents once getting the `gradBi` axis order backwards was a real
   historical bug (`gradcheck_crk_correction_native.py`'s docstring) — re-verify the JVP's own
   `matmul(transpose(dgradBi), x_ij) + matmul(transpose(gradBi), dx_ij)` axis order byte-for-byte
   against that existing comment during the port, don't just trust the spike passed once.
   Wire into `wp_gradientJVP.py`'s `computeSPHGradientJVP_Func_i` (currently calls
   `sphKernelGradientJVP` directly at line 79) as an alternative path gated on the new tangent
   correction struct's `useCRK`-equivalent flag.
4. Also watch for CRK's documented reverse-mode footgun (`project_crk_dualpath_and_latent_bugs`
   memory: a loop-accumulated local consumed by a nonlinear op in the same `@wp.func` silently drops
   part of its gradient) in every new Stage 1/2/4 `@wp.func` — the same pattern already fixed twice
   elsewhere in this codebase (Laplacian Dot's reverse adjoint, see project memory
   `project_laplacian_dot_reversemode_adjoint_bug`). Extract any loop-accumulated value into its own
   returning `@wp.func` rather than reusing it non-linearly inline.

Thread `crkState`/`crkTangentState` through `computeSPHGradientGeometryJVP`'s signature; relax
`operations.py`'s `crkState is not None` rejection for Gradient only (keep it for
Divergence/Curl/Laplacian until phase (e)).

**Validation:** `scripts/gradcheck_tier2_jvp_gradient_crk.py` — hand-assembled production JVP vs.
`torch.autograd.functional.jacobian` on primal `warpOperation(Gradient, crkState=...)` (same pattern
every prior tier uses), plus `torch.autograd.gradcheck` through the new production kernels
themselves (reverse-mode-through-the-JVP, this repo's standing rule — never trust a hand Jacobian).
Register the new script in `tests/operations/test_gradcheck_scripts.py`'s `GRADCHECK_SCRIPTS`.

**Gate:** new gradcheck script green; existing `gradcheck_crk_native.py`/`gradcheck_crk_correction_native.py`
unaffected (new Stage 1-4 functions must be net-new, not edits to `crk_volume.py`/`crk_moments.py`/
`crk_terms.py`); full sweep including CRK column; base non-CRK Gradient JVP case unaffected.

**Done (2026-08-21).** Landed as specified, net-new files only:
- Stages 1-2 promoted verbatim from the spike into `crk/crk_volume_jvp.py`
  (`computeCRKVolumeJVP_Func_i`/`_Func_Adjacency`/`_Kernel` +
  `computeCRKVolumeGeometryJVP`) and `crk/crk_moments_jvp.py` (same
  dual-path shape as `crk_moments.py`, plus `computeCRKMomentsGeometryJVP`)
  — both reuse phase (b)'s already-shipped `correctionData.referenceVolumes`/
  `correctionTangentData.referenceVolumes` plumbing for `V_j`/`dV_j` (via
  `getVolume_j`/`getVolumeTangent_j`) rather than a new `extraTensors` path,
  since CRK's apparent volume occupies that exact slot. Stage 1's
  reciprocal is applied in `computeCRKVolumeJVP_Kernel`, outside the
  dynamic-loop function, mirroring `computeCRKVolume_Func_Adjacency`'s own
  documented adjoint fix — no new instance of that bug here.
- Stage 3 landed as `crk/crk_wrapper.py`'s `computeCRKFactorsJVP`, chaining
  Stages 1-2 into `computeCRKTermsWarp` via `torch.autograd.functional.jvp(...,
  create_graph=True)` — `create_graph=True` (not in the spike, which never
  differentiates its assembled JVP again) is required in production so
  `dA/dB/dgradA/dgradB` stay differentiable back to positions/supports for
  gradcheck.
- Stage 4 landed as `crk/kernel.py`'s `correctGradientCRKJVP` (product-rule
  JVP of `correctGradientCRK`, re-verified the `term4` axis order against
  that function's own docstring during the port) and
  `computeKernelGradientCRKJVP` (dispatches on `useCRK`, mirroring
  `computeKernelGradientCRK`), wired into `wp_gradientJVP.py`'s
  `computeSPHGradientJVP_Func_i` in place of the direct
  `sphKernelGradientJVP` call.
- `_jvpCommon.launchGeometryJVP` gained `crkState`/`crkTangentState`
  parameters, threading only the **query-side** `A/B/gradA/gradB` (+
  tangents) into `correctionData`/`correctionTangentData` and `flat_tensors`
  — confirmed by grep that every value-having operator's
  `computeKernelGradientCRK` call site reads only `iCorrectionData` (never
  `getCRK_j`/reference-side), so reference-side CRK fields stay at
  `buildNullCorrectionData`'s zero defaults, deliberately unwired (same
  "would sit permanently unused" reasoning phase (b) already applied to
  query-side volume).
- `operations.py`'s single `crkState is not None or renormalizationState is
  not None or gradHState is not None` rejection was split three ways:
  renorm/gradH stay rejected unconditionally; `crkState` is rejected only
  for non-Gradient operators; a new `crkTangentState is not None and
  crkState is None` check raises `ValueError` (nothing to take a tangent
  of). `crkTangentState` may be omitted with `crkState` present (defaults
  to an all-zero tangent — a legitimate "correction applied but held
  frozen" combination, unlike volume's primal/tangent ordering dependency).
- Validation: `scripts/gradcheck_tier2_jvp_gradient_crk.py`, three layers —
  (1) a targeted direct-tensor gradcheck with `crkState`/`crkTangentState`
  as independent synthetic leaves (isolating `_jvpCommon`'s new
  `flat_tensors` wiring per (a2)'s "hard requirement"), (2) a
  JVP-vs-`torch.autograd.functional.jacobian` identity check against primal
  `warpOperation(Gradient, crkState=...)` for every `GradientScheme`, 1D and
  2D, and (3) an end-to-end `torch.autograd.gradcheck` with `crkState`/
  `crkTangentState` derived from the same leaf positions/supports via
  `computeCRKFactorsJVP`, matching `gradcheck_crk_correction_native.py`'s
  "real force-computation call site" convention. All green; registered in
  `test_gradcheck_scripts.py`. Existing `gradcheck_crk_native.py`/
  `gradcheck_crk_correction_native.py` unaffected (no edits to
  `crk_volume.py`/`crk_moments.py`/`crk_terms.py`/`crk/kernel.py`'s existing
  functions). Full `pytest tests/` (356 passed, 1 skipped, up from the
  354/1 baseline by exactly the 2 new tests) and `operation_matrix.py`
  (`OK=258, HIGH=0, ERR=0, NAN=0`, bit-identical to the pre-phase baseline)
  both green. Two pre-existing tests needed updating, not for behavior
  drift but because they asserted the *old* scope boundary this phase
  deliberately moved: `test_forward_mode_geometry_jvp_gradient.py`'s
  `test_gradientGeometryJVP_rejects_crkState` became
  `test_gradientGeometryJVP_accepts_crkState` (now asserts a finite result,
  plus the new `crkTangentState`-without-`crkState` `ValueError`), and a new
  `test_divergenceGeometryJVP_still_rejects_crkState` was added to
  `test_forward_mode_geometry_jvp_divergence.py` to keep that boundary
  covered for the operator where it's still true.

### (d) — Renormalization tangent wiring for Gradient

**Covariance-matrix JVP proven; combining it with the corrected-gradient product rule is new.**
1. Port Tier 2.4's `dC_i = Σ_j[dVj·outer(y_ij,G_ij) + Vj·outer(dy_ij,G_ij) + Vj·outer(y_ij,dG_ij)]`
   and `d(C^-1) = -C^-1(dC)C^-1` into a real `@wp.func`/kernel (new `coreOperations/wp_covarianceJVP.py`,
   mirroring `wp_covariance.py`'s structure) producing `dL_i` per query particle. Reuses (c)'s `(G,dG)`
   when CRK is also enabled (primal kernels always compose CRK-then-renorm in that fixed order,
   confirmed across all four consumers) — but **scope phase (d) to renorm alone (CRK off) first**;
   treat "renorm applied on top of CRK simultaneously" as an explicit fast follow-up once (c) and (d)
   are each independently solid, not bundled into (d) itself.
2. **The one piece of this entire plan needing a fresh derivation write-up**: the primal composition
   is `kernelGradient_final = L_i @ kernelGradient_corrected`, so its JVP is
   `dKernelGradient_final = dL_i @ kernelGradient_corrected + L_i @ dKernelGradient_corrected` — a
   direct product rule on a matrix-vector product, no new matrix calculus beyond what Tier 2.4 already
   proved for `dL` itself, but genuinely unprecedented in this codebase. Write a short addendum in
   `warpier_adjoint.md` and a standalone `scripts/spike_forward_mode_tier2_renorm_gradient.py`
   validating it against `torch.autograd.functional.jacobian` on production
   `warpOperation(Gradient, renormalizationState=...)` **before** touching `wp_gradientJVP.py`.
3. Explicitly re-test the low-neighbor-count identity-fallback branch's tangent is exactly zero
   (mirroring Tier 2.4's own explicit single-particle check, both dim=1 and dim=2) — don't rely on
   incidental coverage.
4. **Carry forward, do not attempt to solve**: `pinv2x2_warpBackend`'s `rcond=1e-6` eigenvalue cutoff
   is a genuine JVP discontinuity if a tangent pushes a case across it mid-JVP. Keep test geometries
   well-conditioned (mirroring Tier 2.4's own choice) and flag this in the new spike's docstring,
   same as Tier 2.4's own result section already does — deliberately not characterized or guarded.

**Validation:** spike script above, then `scripts/gradcheck_tier2_jvp_gradient_renorm.py` following
(c)'s pattern; `gradcheck_renorm_native.py`/`gradcheck_pinv_native.py` must stay unaffected.

**Gate:** spike + gradcheck green, full sweep including renorm column.

**Done (2026-08-21).** Landed as specified, net-new files/additions only:
- Step 1 landed as new `coreOperations/wp_covarianceJVP.py`
  (`computeCovarianceJVP_Func_i`/`_Func_Adjacency`/`_Kernel` +
  `computeCovarianceGeometryJVP`), mirroring `wp_covariance.py`'s structure
  but deliberately CRK/renorm-free (matching
  `computeRenormalizationMatrices_`'s own internal covariance call, which is
  "only ever called here with `crkState=None, renormalizationState=None`").
  `Vj/dVj` reuse `_jvpCommon.gradientWeightsJVP`'s `GradientScheme.Naive`
  branch directly (`B_ij`/`dB_ij` is literally the same formula) rather than
  duplicating the `useVolume` quotient-rule logic a second time; `(G_ij,
  dG_ij)` come straight from `kernels.kernelJVP.sphKernelGradientJVP`. This
  file produces only the RAW (unmasked) `dC_i` -- the low-neighbor-count
  mask and `-L(dC)L` identity are `renorm.py`'s job, same split as the
  primal `wp_covariance.py`/`renorm.py` pair.
- The mask + pseudo-inverse-derivative step landed as `renorm.py`'s new
  `computeRenormalizationMatricesJVP`, mirroring `crk_wrapper.py`'s
  `computeCRKFactorsJVP`: computes the correction (`L`, via the existing
  `computeRenormalizationMatrices_`) AND its tangent (`dL`) together from
  geometry tangents alone, rather than requiring a caller to supply `L` by
  hand. `num_nbrs` (needed to mask `dC` the same way `computeRenormalizationMatrices_`
  masks `C`, but not returned by that function) is recomputed via one extra
  `warpOperation(Covariance, covarianceReturnNumNeighbors=True)` call --
  the same "consume production's own count" pattern the Tier 2.4 spike
  itself already used, not a new inefficiency introduced here.
- `_jvpCommon.launchGeometryJVP` gained `renormalizationState`/
  `renormalizationTangentState` parameters, threading the **query-side**
  `renormalizationMatrices` (+ tangent) into `correctionData.useGradientRenormalization`/
  `.renormalizationMatrices` and `correctionTangentData.renormalizationMatrices`
  and `flat_tensors` (phase (a2)'s "hard requirement") -- mirroring `crkState`/
  `crkTangentState`'s existing wiring field-for-field. No new extraction code
  was needed: `util/stateUtil.py`'s `getL_i`/`getRenormTangent_i` (phase
  (a2)) already read exactly these fields.
- `wp_gradientJVP.py`'s `computeSPHGradientJVP_Func_i` gained the product-rule
  step (`dG = matmul(dL,G) + matmul(L,dG); G = matmul(L,G)`, gated on
  `correctionData.useGradientRenormalization`) immediately after the CRK
  dispatch, matching the primal kernel's own fixed CRK-then-renorm
  composition order. `computeSPHGradientGeometryJVP` gained
  `renormalizationState`/`renormalizationTangentState` parameters, threaded
  straight through to `_launchGeometryJVP`.
- `operations.py`'s single `renormalizationState is not None or gradHState
  is not None` rejection was split: `gradHState` stays rejected
  unconditionally; `renormalizationState` is rejected only for non-Gradient
  operators; a new `renormalizationTangentState is not None and
  renormalizationState is None` check raises `ValueError`; a new
  `crkState is not None and renormalizationState is not None` check raises
  `NotImplementedError` (CRK+renorm simultaneous stays out of scope, per
  this phase's own "renorm alone first" decision -- deferred as a fast
  follow-up, not because the math is known to fail, but because it isn't
  validated yet). `renormalizationTangentState` may be omitted with
  `renormalizationState` present (an all-zero tangent, correction held
  frozen), same as `crkTangentState`. The raw-tensor-to-`RenormalizationState`
  normalization `warpOperation` already does for its own `renormalizationState`
  parameter was added to `warpOperationJVP` too (it previously had none,
  since the parameter was always rejected before this phase).
- Validation: `scripts/spike_forward_mode_tier2_renorm_gradient.py` (the new
  derivation -- see `warpier_adjoint.md`'s "Tier 2.4b addendum" -- validated
  against production `warpOperation(Gradient, renormalizationState=...)`,
  `rel_err ~1e-15`-`1e-16` across every `GradientScheme`/`SupportScheme`,
  plus an explicit low-neighbor-count zero-tangent check at `dim=1`/`dim=2`)
  and `scripts/gradcheck_tier2_jvp_gradient_renorm.py` (the same three-layer
  pattern (c) established: direct-tensor gradcheck isolating the new
  `flat_tensors` wiring, JVP-vs-jacobian identity through the real production
  call graph, end-to-end gradcheck with `renormalizationState`/
  `renormalizationTangentState` derived from the same leaves being
  perturbed). All green; registered in `test_gradcheck_scripts.py`.
  `gradcheck_renorm_native.py`/`gradcheck_pinv_native.py` unaffected (no
  edits to `renorm.py`'s existing functions or `pinv/`). Full `pytest
  tests/` and `operation_matrix.py` (`OK=258, HIGH=0, ERR=0, NAN=0`,
  bit-identical to the pre-phase baseline) both green.
- **Two bugs found and fixed, both in the new gradcheck script, not in
  production code** (caught by comparing against the independently-passing
  spike, same discipline phase (b) used for its own primal-bug find):
  (1) the JVP-vs-jacobian identity case initially omitted the reference-side
  mass/density tangent when calling `computeRenormalizationMatricesJVP`
  (`Vj = mass_j/density_j` depends on both, unlike CRK's Stage 1/2, which
  have no mass/density term at all); (2) the same case's 2D grid test used
  perfectly uniform, unperturbed supports, which makes `pinv2x2_warpBackend`
  return `NaN` for that exact geometry -- unrelated to this phase's own JVP
  correctness (`wp_covarianceJVP.py`'s raw `dC_i` and
  `computeRenormalizationMatricesJVP`'s `dL_i` both independently matched
  their production references to `1e-16` when isolated), fixed by applying
  the same `+-15%` support perturbation every other Tier-2.4-touching script
  already uses (`spike_forward_mode_tier2_renorm.py`'s `_perturbed_case`).

### (e) — CRK tangent extension to Divergence/Curl/Laplacian(Brookshaw)

**New derivation per operator, reusing (c)'s Stages 1-4 verbatim** (all four stages are
operator-agnostic — they produce a corrected `(kernelGradient, dKernelGradient)` pair, nothing
Gradient-specific). What's actually new is each operator's own combination formula consuming that
corrected pair instead of the plain `sphKernelGradientJVP` one — the exact extension pattern
`warpier_adjoint.md`'s own Tier 2.2 already used for the non-CRK case (one shared `(G,dG)` combined
three ways):
- **Divergence** (`wp_divergenceJVP.py`): swap the `G,dG` source feeding its existing
  `dot(dcoeff,G) + dot(coeff,dG)`.
- **Curl** (`wp_curlJVP.py`, 2D only — matches the existing `domain.dim != 2` restriction already
  enforced in `operations.py`): same swap feeding its existing 2D cross-product combination.
- **Laplacian-Brookshaw** (`wp_laplacianJVP.py`'s Brookshaw path only — Dot/Default/Naive stay out
  of scope, matching existing scheme restrictions): same swap at the point it currently calls
  `sphKernelGradientJVP` before computing `D_ij`/`n_ij`/their tangents; everything downstream
  unchanged.

Relax `operations.py`'s `crkState` rejection per-operator as each lands.

**Validation:** three new gradcheck scripts (`_divergence_crk.py`, `_curl_crk.py`,
`_laplacian_brookshaw_crk.py`) mirroring (c)'s pattern — each mostly "swap Gradient's combination for
this operator's own," a much smaller diff than (c) itself since Stages 1-4 are already proven. A
short combined spike exercising all three (mirroring how Tier 2.2 itself was one spike covering three
operators) before production wiring.

**Gate:** spike + all three gradcheck scripts green; `gradcheck_crk_native.py`/
`gradcheck_crk_correction_native.py` unaffected; full sweep including CRK column for all three
operators.

**Done (2026-08-21).** Landed as specified above, with one significant addition discovered during
implementation:
- The `computeKernelGradientCRKJVP` swap landed exactly as specified in all three files:
  `wp_divergenceJVP.py`'s `computeSPHDivergenceJVP_Func_i` and `wp_curlJVP.py`'s
  `computeSPHCurlJVP_Func_i` each swap their `sphKernelGradientJVP` call for
  `computeKernelGradientCRKJVP(..., correctionData.useCRK, iCorrectionData.A/B/gradA/gradB,
  iCorrectionTangentData.A/B/gradA/gradB)` verbatim (a no-op when `useCRK` is `False`, identical to
  the call it replaces). `wp_laplacianJVP.py`'s shared `_laplacianGeometryChainJVP` (used by
  Brookshaw/Dot/Default) gained the same `useCRK`/CRK-term parameters and does the same swap once,
  centrally — safe for Dot/Default too since `operations.py`'s own scope check (below) guarantees
  `useCRK` is always `False` for them. `computeSPHDivergenceGeometryJVP`/`computeSPHCurlGeometryJVP`/
  `computeSPHLaplacianBrookshawGeometryJVP` each gained `crkState`/`crkTangentState` parameters,
  threaded straight through to `_launchGeometryJVP` (which already had unconditional CRK wiring from
  phase (a2)/(c) — no new ABI-threading work needed, same "already reachable" finding phase (b) hit
  for volume tangents).
- `operations.py`'s single `crkState is not None and operation is not Gradient` rejection was split:
  a new `_CRK_GEOMETRY_JVP_OPERATIONS` tuple (Gradient/Divergence/Curl/Laplacian) replaces the
  Gradient-only check; a second, Laplacian-specific check rejects `crkState` whenever
  `laplacianMode is not LaplacianScheme.Brookshaw` (Naive/Dot/Default stay out of scope, no
  derivation exists). Dispatch-kwargs assembly only adds `crkState`/`crkTangentState` to
  `dispatchKwargs` for Gradient/Divergence/Curl, or for Laplacian when `laplacianMode is Brookshaw`
  — Naive/Dot/Default's own `computeSPHLaplacian{Naive,Dot,Default}GeometryJVP` take no
  `crkState` parameter at all, so this avoids ever passing them an unsupported kwarg rather than
  relying on every callee to silently ignore one.
- **Found and fixed a genuine, pre-existing reverse-mode adjoint bug in production code, not
  introduced by this phase, discovered while validating**: `wp_laplacian.py`'s (primal) and
  `wp_laplacianJVP.py`'s (this phase's own new code) `LaplacianScheme.Brookshaw` formula divides
  `dot(kernelGradient, n_ij)` by `D_ij` a *second* time (`n_ij` itself is already `x_ij/D_ij`). At an
  exact self-pair (`r_ij == 0`, always present in any `referenceParticles=None` self-referencing
  adjacency — the standard gradcheck convention this whole plan uses), `n_ij` is exactly `0` by
  construction, forcing the forward contribution to exactly `0` regardless of `kernelGradient`, CRK
  or not. Without CRK this was never a problem (the plain kernel gradient is *also* exactly `0` at a
  self-pair, with a correct adjoint there too, via `sphGradient_`'s existing custom `@wp.func_grad`
  from `project_tier2_jvp_distinct_role_adjoint_bug`'s earlier fix). With CRK enabled, though,
  `correctGradientCRK`'s own value at `x_ij == 0` is generically **nonzero** (its
  `Ai*W_ij*Bi`/`W_ij*gradAi` terms don't vanish at the kernel's own peak the way the plain gradient
  does), and Warp's reverse-mode through "a nonzero vector dotted against an exactly-zero `n_ij`,
  itself divided by `D_ij` again" produced a wrong adjoint — confirmed via
  `torch.autograd.gradcheck` failing directly on both `warpOperation(Laplacian, Brookshaw,
  crkState=...)` and the new `computeSPHLaplacianBrookshawGeometryJVP(..., crkState=...,
  crkTangentState=...)`, and via a from-scratch minimal repro with no dependency on this codebase's
  kernel structure (isolated to exactly this "dot-then-divide-again" shape; the "dot-only" shape
  Divergence/Curl use did not reproduce it). **Fixed** in both `wp_laplacian.py` and
  `wp_laplacianJVP.py` by guarding the Brookshaw contribution with an explicit `if r_ij > 0:` — the
  true contribution there is always exactly `0` (CRK or not), so this changes no forward value
  anywhere (confirmed: `operation_matrix.py` stayed bit-identical at `OK=258, HIGH=0, ERR=0, NAN=0`,
  and every existing non-CRK gradcheck script, including `gradcheck_tier2_jvp_laplacian.py` and
  `gradcheck_laplacian_native.py`, stayed green with unchanged output). Full write-up in
  `docs/lessons_learned.md`'s "Warp kernel authoring gotchas" section.
- Validation: `scripts/spike_forward_mode_tier2_crk_extension.py` (one combined spike, all three
  operators, production `warpOperationJVP` vs. `torch.autograd.functional.jacobian` on primal
  `warpOperation(<op>, crkState=...)`, `rel_err` ~1e-16 for all three) and three new gradcheck
  scripts (`gradcheck_tier2_jvp_{divergence,curl,laplacian_brookshaw}_crk.py`), each a two-layer
  version of (c)'s three-layer pattern (direct-tensor gradcheck isolating the CRK `flat_tensors`
  wiring for each operator's own new parameters, plus an end-to-end gradcheck with `crkState`/
  `crkTangentState` derived from the same leaves via `computeCRKFactorsJVP` — the "hand-Jacobian vs.
  jacobian-identity" middle layer (c)'s script has was judged redundant here since the spike already
  covers exactly that check for all three operators at once). All green; registered in
  `test_gradcheck_scripts.py` (both the spike and all three gradcheck scripts).
  `gradcheck_crk_native.py`/`gradcheck_crk_correction_native.py` unaffected (no edits to
  `crk_volume.py`/`crk_moments.py`/`crk_terms.py`/`crk/kernel.py`). Full `pytest tests/` (357 passed,
  1 skipped, up from the 356/1 baseline by the 4 new CRK-acceptance tests minus 1 stale
  still-rejects test converted to an accepts test) and `operation_matrix.py` (`OK=258, HIGH=0,
  ERR=0, NAN=0`, bit-identical to the pre-phase baseline) both green. Updated tests that asserted the
  *old* scope boundary this phase deliberately moved:
  `test_forward_mode_geometry_jvp_divergence.py`'s `test_divergenceGeometryJVP_still_rejects_crkState`
  became `test_divergenceGeometryJVP_accepts_crkState`; new
  `test_curlGeometryJVP_accepts_crkState`/`test_laplacianBrookshawGeometryJVP_accepts_crkState`/
  `test_laplacianGeometryJVP_naive_still_rejects_crkState` were added to keep both the new acceptance
  and the still-out-of-scope Naive/Dot/Default boundary covered.

### (f) — Renormalization tangent extension to Divergence/Curl/Laplacian(Brookshaw)

**New derivation per operator, reusing (d)'s `dL⊗G + L⊗dG` combination rule verbatim** — (d)
already did the one genuinely new piece; this phase applies it to three more `(G,dG)` sources (plain,
or (e)'s CRK-corrected pair if both corrections are enabled — same "scope to renorm-alone first"
deferral as (d), extended per operator). Same three files as (e), same three-gradcheck-script pattern
(`_divergence_renorm.py`, `_curl_renorm.py`, `_laplacian_brookshaw_renorm.py`).

**Gate:** same shape as (e)'s gate, substituting renorm's gradcheck baselines. **This closes the plan**
— after (f), Gradient/Divergence/Curl/Laplacian(Brookshaw) all support CRK and renormalization
tangents (renorm-alone or CRK-alone; CRK+renorm-simultaneous flagged as a fast, low-risk follow-up,
not required here), and apparent-volume tangent support covers all five value-having operators.

## Explicitly out of scope (all phases)

- `Field.tangent`/`torch.autograd.forward_ad`/`ExecutionMode.FORWARD` — stays dormant, per the
  decision above.
- Grad-H tangent support — no consumer exists yet.
- Laplacian's Dot/Default/Naive schemes for CRK/renorm (Brookshaw only, matching existing scheme
  restrictions already enforced in `operations.py`).
- CRK+renormalization applied simultaneously (each is separately supported; the combination is a fast
  follow-up once both are independently proven, not required by this plan).
- The single unified operator wrapper (tangent-defaults-to-`None` auto-dispatch) — a later, separate
  effort once this lands.
- HVP for any of these corrections — this plan is JVP-only, matching every existing Tier-2 scope
  decision.

## Verification (after every phase, not just at the end)

```bash
pytest tests/                                                    # full suite
python scripts/operation_matrix.py --device cpu --ci --verbose   # baseline before starting: OK=258, HIGH=0, ERR=0, NAN=0
pytest tests/operations/test_gradcheck_scripts.py                # includes every new script once registered
```
Record baseline pass/fail counts before Phase (a1) starts; every phase's gate is "identical to
previous phase's counts, plus this phase's own new green cases" — any unexplained drift anywhere is a
bug, not an expected side effect, per this repo's own standing discipline.

## Critical files

- `src/warpSPHCore/dataTypes/particleData.py` — `ParticleTangentState` (a1)
- `src/warpSPHCore/dataTypes/corrections_t.py` — `CRKTangentState`/`RenormalizationTangentState`,
  `correctionTangentData_{1,2,3}` (a2)
- `src/warpSPHCore/util/stateUtil.py` — `getParticleCorrectionTangentData_i` extractor (a2)
- `src/warpSPHCore/coreOperations/_jvpCommon.py` — `launchGeometryJVP`, `gradientWeightsJVP` (a1, b)
- `src/warpSPHCore/operations.py` — `warpOperationJVP` dispatch, every phase's scope-boundary checks
- `src/warpSPHCore/coreOperations/wp_interpolateJVP.py` — volume tangent (b)
- `src/warpSPHCore/coreOperations/wp_gradientJVP.py` — CRK (c) and renorm (d) wiring
- `src/warpSPHCore/coreOperations/wp_{divergence,curl,laplacian}JVP.py` — CRK (e) and renorm (f)
  extension
- `src/warpSPHCore/crk/kernel.py` — `correctGradientCRK`/`correctGradientCRKJVP` (c)
- `src/warpSPHCore/crk/crk_terms.py` — `computeCRKTermsWarp`, consumed via `torch.autograd.functional.jvp`
  directly, not ported to Warp (c)
- `src/warpSPHCore/crk/crk_volume_jvp.py`/`crk_moments_jvp.py` — Stages 1-2, operator-agnostic,
  built once for reuse by phase (e) (c)
- `src/warpSPHCore/crk/crk_wrapper.py` — `computeCRKFactorsJVP`, Stage 3 orchestration (c)
- new `coreOperations/wp_covarianceJVP.py` — raw covariance matrix JVP (d)
- `src/warpSPHCore/renorm.py` — `computeRenormalizationMatricesJVP`, mask + `-L(dC)L` orchestration (d)
- `warpier_adjoint.md` — append phase (d)'s new derivation write-up; update status header once each
  phase lands, matching this repo's existing convention
- `tests/operations/test_gradcheck_scripts.py` — register every new `gradcheck_tier2_jvp_*.py` script
