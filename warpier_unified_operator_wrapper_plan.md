# Unified operator wrapper: `torch.autograd.forward_ad` dual tensors drive JVP auto-dispatch via `StateAwareWarpFunction.jvp()`

## Status: planned, not started (2026-08-21)

## Context

`warpier_tier2_correction_jvp_plan.md` deliberately deferred "the eventual single unified operator
wrapper (tangents default to `None`, dispatches to pure-primal vs. combined JVP automatically)" as
later, separate work, once explicitly declining to reactivate `Field.tangent`/`torch.autograd.forward_ad`
for that plan's own scope. That later point has now arrived. The ask: let a caller wrap a leaf tensor
as a standard `torch.autograd.forward_ad` dual tensor (`fwAD.make_dual(x, v)`), pass it straight into
`warpOperation` in place of an ordinary tensor, and get back a dual tensor whose tangent is exactly
`warpOperationJVP`'s result — no separate JVP call, no `tangentQueryPositions=`-style parallel kwargs.

Two prior, independent attempts at composing this codebase's kernels with `torch.autograd.forward_ad`
already failed, both root-caused to the same cause: `spike_forward_mode_tier1.py` found
`fwAD.make_dual`/`dual_level` silently returns `tangent=None` through `warpOperation`; `wp_densityHVP.py`
found the same one level deeper. Both post-mortems land on one sentence: **`StateAwareWarpFunction`
(the `torch.autograd.Function` every operator launch goes through) never registered a `jvp()`
staticmethod**, so torch's forward-mode engine has nothing to consult when a dual tensor reaches it and
drops the tangent rather than erroring. Per the user's explicit choice (a manual `fwAD.unpack_dual`/
`make_dual` dispatch at the `warpOperation` call boundary was raised as a lower-risk alternative and
declined), this plan closes that gap for real — implementing `jvp()` on the functor itself.

**Why this is expected to actually work now, not a third repeat of the same failure:** the missing
piece was identified precisely (no `jvp()` registered), not "composition is fundamentally impossible."
Separately, `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md` (done, 2026-08-20) already
closed an adjacent, harder gap: every Tier-2 JVP kernel (`launchGeometryJVP`, `_jvpCommon.py`) now
routes through `StateAwareWarpFunction` itself, so a JVP result is already `.backward()`-capable
w.r.t. its own primal *and* tangent inputs — verified end-to-end by
`scripts/gradcheck_tier2_jvp_chained_backprop.py` (a `warpOperation` → `warpOperationJVP` chain,
`.backward()` reaching the original leaves). This plan's new `jvp()` hook will internally call exactly
those same, already-validated functions — it does not need to re-derive or re-validate the JVP math,
only wire torch's dual-tensor engine to trigger it.

## Design

**`OperatorSpec` (`autograd/operator_spec.py`) gains an optional `jvp` field** — a new `JVPSpec`
dataclass naming, per declared kernel, how to turn "which of this call's tensor arguments carry a live
tangent" into a tangent output, by delegating to the operator's *existing* JVP entry point
(`computeSPH<Op>GeometryJVP` for Tier-2 geometry tangents; a relaunch of the same primal kernel on the
tangent array in place of the value array for Tier-1 value tangents, exactly `warpOperationJVP`'s
existing linear-and-homogeneous shortcut). `JVPSpec` is opt-in per kernel — a spec without one behaves
exactly as today, and a dual-tensor argument reaching it raises a clear error rather than silently
returning a tangent-free dual output (mirrors `_VALUE_JVP_OPERATIONS`'s existing "raise, don't drop"
contract in `operations.py`).

**`StateAwareWarpFunction.forward` gains one new leading non-tensor argument, `jvp_fn`** (bumping
`_N_NON_TENSOR` 5→6), a closure built by `_launch` (`autograd/wrapper.py`) from `OperatorSpec.jvp` the
same way `build_fn` is built from the spec today. `forward()` itself needs no other change — torch
already hands it stripped primals regardless of whether the caller passed dual tensors. It additionally
stashes what `jvp()` will need via `ctx.save_for_forward(...)` (not `save_for_backward`, which forward
mode has its own bookkeeping for — confirmed in Phase 0, not assumed).

**`StateAwareWarpFunction.jvp(ctx, *tangents)`** (new): tangents align positionally with `forward`'s
`*flat_tensors` (torch guarantees this — confirmed against the installed torch 2.13 source: "as many
inputs as forward got, None for non-tensor/no-tangent ones"). If every tangent is `None`, return
`None` for every output (nothing to propagate). If any is live and `ctx.jvp_fn is None`, raise
clearly. Otherwise delegate to `ctx.jvp_fn(ctx, tangents)`, which reconstructs the semantic tangent
objects (`ParticleTangentState`/`CRKTangentState`/`RenormalizationTangentState`/tangent volumes) from
the flat positional list — mirroring `extractStateInfo`'s own index layout, since that's what built
`flat_tensors` in the first place — and calls the operator's real JVP function. That function's own
internals launch via `launchGeometryJVP`/`StateAwareWarpFunction.apply()` themselves, a **nested**
`.apply()` call made with ordinary (non-dual) tensors — an unremarkable, already-common pattern, not
new engine territory.

**Only two call sites construct `StateAwareWarpFunction.apply()` directly** (confirmed by grep — the
old pair-indexed COO JVP launchers were consolidated into the CSR path):
`autograd/wrapper.py:68` (`_launch`, shared by `warpWrapper2` and `launchOperator`) and
`coreOperations/_jvpCommon.py:547` (`launchGeometryJVP`). Both need the new argument threaded through;
`warpWrapper2` (no `OperatorSpec`, and no remaining callers in `coreOperations`) always passes
`jvp_fn=None`, unaffected. `warpOperation`/`OperatorSpec`/`launchOperator`'s own public signatures do
not change at all — callers keep calling them exactly as today; the only thing that's new is that the
tensors flowing in may themselves be `torch.autograd.forward_ad` dual tensors.

## Phases

### Phase 0 — go/no-go spike (gate; mirrors this repo's own "Step 0" discipline)

Throwaway script, one operator, the mathematically simplest case first (Tier-1 value tangent on
Interpolate — "same kernel relaunched on the tangent array," no new kernel needed, same starting
point the original Tier-1 promotion used). Register `jvp()` for real (not a mock) and confirm, before
writing any production code:

1. `fwAD.make_dual(x, v)` → `warpOperation(Interpolate, ...)` → `fwAD.unpack_dual(result).tangent`
   is no longer `None` and matches `warpOperationJVP`'s existing reference to float64 round-off
   (reversing `spike_forward_mode_tier1.py`'s finding).
2. `ctx.save_for_forward` vs. a plain `ctx.` attribute for the primal tensors `jvp_fn` needs — settle
   which is actually required (PyTorch's docs flag `save_for_forward` as the correct mechanism, but
   this repo verifies rather than assumes).
3. The **no-dual-input path is bit-identical and unaffected** by `jvp()` merely existing — primal
   forward/backward behavior for every existing caller must not change.
4. `.backward()` on `fwAD.unpack_dual(result).tangent` (outside the `dual_level` block, ordinary
   reverse-mode) reaches the original leaf `x`/`v` — the "differentiating through the forward tangent
   pass" property, exercised here through the *new* dual-tensor path specifically, not just the
   existing explicit one `gradcheck_tier2_jvp_chained_backprop.py` already covers.

If this doesn't reproduce cleanly, that's new information worth writing up (matching this repo's own
practice of correcting a plan against what the code actually does) before continuing.

### Phase 1 — wire Tier-1 (value) JVP for all five `_VALUE_JVP_OPERATIONS`

Promote Phase 0's spike into `OperatorSpec.jvp`/`_launch`/`StateAwareWarpFunction` for real, Interpolate
first, then Gradient/Divergence/Curl/Laplacian. No new kernel math — this tier is exactly the existing
"relaunch the same kernel with the tangent array in place of the value array" trick, now triggered by
`jvp()` instead of a caller's explicit `tangentQueryValues=` kwarg.

### Phase 2 — wire Tier-2 (geometry) JVP for `_GEOMETRY_JVP_OPERATIONS`, plus Covariance

Same treatment, `jvp_fn` closures calling the existing `computeSPH<Op>GeometryJVP` functions
(Density/Interpolate/Gradient/Divergence/Curl/Laplacian). Gate CRK/renormalization/volume tangent
combinations exactly against the existing `_CRK_GEOMETRY_JVP_OPERATIONS`/`_RENORM_GEOMETRY_JVP_OPERATIONS`/
`_LAPLACIAN_CORRECTION_SCHEMES` tables in `operations.py` — reused, not duplicated — so a
not-yet-supported combination raises the same clear error `warpOperationJVP` already raises, rather
than a dual tensor silently losing its tangent partway through.

**Covariance correction, found while grounding this plan against the actual code (not the stale
comment above `_GEOMETRY_JVP_OPERATIONS` in `operations.py`, which claims "no geometry-tangent formula
was ever derived for it"):** `coreOperations/wp_covarianceJVP.py`'s `computeCovarianceGeometryJVP`
already exists, already routes through `_jvpCommon.launchGeometryJVP` (so it's already bridge-backed
and reverse-mode differentiable, same as every other Tier-2 JVP function), and is already consumed
internally by `renorm.py`'s `computeRenormalizationMatricesJVP` — it was simply never registered in
`_GEOMETRY_JVP_DISPATCH`/`_GEOMETRY_JVP_OPERATIONS`, so `warpOperationJVP(Covariance, ...)` isn't a
supported *public* call today, even though the math and the kernel both already exist. Covariance has
no value-tangent path at all (no `queryValues`/`referenceValues` input, same shape as Density) — it
never belongs in `_VALUE_JVP_OPERATIONS` — but its geometry-tangent path is real and this phase should
promote it into the dispatch table alongside the other six, since the registration is mechanical and
the underlying function is already proven. This is new *public dispatch* coverage (needs its own
gradcheck-style test, since `warpOperationJVP(Covariance, ...)` was never exercised standalone before,
only as `renorm.py`'s internal building block) even though it is not new *math*.

### Phase 3 — tests

New `tests/operations/test_forward_mode_dual_wrapper.py`: for every operator/tangent combination
`warpOperationJVP` already covers (plus Covariance, now that Phase 2 exposes it), build the equivalent
case with `fwAD.make_dual`/`dual_level` through plain `warpOperation` and assert the unpacked tangent
matches the existing `warpOperationJVP` call bit-for-bit (it's calling the same underlying function —
this guards the new wiring, not the math). Plus: an unsupported combination raises instead of silently
dropping a tangent; a chained `warpOperation` → `warpOperation` case entirely inside one `dual_level`
(unlike `gradcheck_tier2_jvp_chained_backprop.py`, no explicit `warpOperationJVP` call anywhere in the
test) with `.backward()` reaching the original leaves. Register any new scripts in
`test_gradcheck_scripts.py` per convention. Full suite + `operation_matrix.py --ci` must stay at
baseline throughout every phase, not just at the end.

### Phase 4 — exploratory, not gated: nested `dual_level` for automatic HVP

`wp_densityHVP.py`'s own docstring says nested `dual_level`/`torch.func.jvp` composition failed for
the *same* reason (`jvp()` never registered) that this plan fixes. Once Phase 2 lands, retry that exact
experiment for Density (which has an independently hand-derived HVP reference to check against) and see
if it now reproduces automatically. If yes, this becomes the natural "hook into HVP" the JVP functions'
own tangent-state wrappers already invite — a big win, but no existing reference exists for the other
five operators, so treat a positive result here as a promising lead to scope as its own follow-up plan,
not something to build out now.

## Explicitly out of scope

- `torch.func.jvp`/functorch composition — separately, already-confirmed broken (`RuntimeError`,
  dual tensors with no storage for `wp.from_torch`) for reasons unrelated to `jvp()` registration;
  unaffected by this plan either way.
- Extending `warpWrapper2`'s legacy untyped entry point with JVP support — no production callers
  remain; it keeps passing `jvp_fn=None`.
- Grad-H tangent support — `warpOperationJVP` already rejects it outright; this plan's `jvp_fn`
  closures do the same, unchanged.
- Covariance's *value*-tangent path — it has no `queryValues`/`referenceValues` input, so there is
  nothing to wire at Tier 1 (its geometry-tangent path is in scope for Phase 2, see above — it already
  has a JVP formula, just not yet public dispatch).

## Verification

```bash
pytest tests/                                                    # full suite, baseline before Phase 0
python scripts/operation_matrix.py --device cpu --ci --verbose   # OK=258, HIGH=0, ERR=0, NAN=0 baseline
pytest tests/operations/test_gradcheck_scripts.py                # every registered gradcheck/spike script
pytest tests/operations/test_forward_mode_dual_wrapper.py        # new, Phase 3
```

## Critical files

- `src/warpSPHCore/autograd/stateAwareWarpFunction.py` — `jvp()` staticmethod, `forward()`'s new arg
- `src/warpSPHCore/autograd/wrapper.py` — `_launch` builds `jvp_fn` from `OperatorSpec.jvp`
- `src/warpSPHCore/autograd/operator_spec.py` — new `JVPSpec` dataclass, `OperatorSpec.jvp` field
- `src/warpSPHCore/coreOperations/_jvpCommon.py` — `launchGeometryJVP`'s second `StateAwareWarpFunction.apply()` call site, and the existing per-operator `computeSPH<Op>GeometryJVP` functions `jvp_fn` closures delegate to
- `src/warpSPHCore/coreOperations/wp_covarianceJVP.py` — `computeCovarianceGeometryJVP`, promoted into public dispatch in Phase 2
- `src/warpSPHCore/operations.py` — `_VALUE_JVP_OPERATIONS`/`_GEOMETRY_JVP_OPERATIONS`/`_GEOMETRY_JVP_DISPATCH`/`_CRK_GEOMETRY_JVP_OPERATIONS`/`_RENORM_GEOMETRY_JVP_OPERATIONS`/`_LAPLACIAN_CORRECTION_SCHEMES` — reused for scope gating, not duplicated; `_GEOMETRY_JVP_OPERATIONS`/`_GEOMETRY_JVP_DISPATCH` gain Covariance in Phase 2
- `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md` — the reverse-mode-through-JVP work this plan's `jvp_fn` closures lean on unmodified
