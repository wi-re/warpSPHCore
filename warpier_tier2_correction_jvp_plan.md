# Tier-2 JVP: correction-tangent support (CRK / renormalization / apparent-volume) + interface bundling

## Status: (a1) done (2026-08-20); (a2) done (2026-08-21); (b) next

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
- new `coreOperations/wp_covarianceJVP.py` — renormalization matrix JVP (d)
- `warpier_adjoint.md` — append phase (d)'s new derivation write-up; update status header once each
  phase lands, matching this repo's existing convention
- `tests/operations/test_gradcheck_scripts.py` — register every new `gradcheck_tier2_jvp_*.py` script
