# JVP dual/inert argument pruning — scoping plan (2026-08-24)

## Context

Triggered by a user report: `warpSPH/benchmarks/wave`'s `sdirk2_jfnk_jvp_1e-6` scheme measured
~4x the cost of a pure forward-mode RHS evaluation (`msPerRhs`), against an expected ~2x (one
primal kernel launch plus one tangent-producing pass — the unavoidable minimum for an exact
forward-mode directional derivative). Investigated across one session (2026-08-24) and found
**three independent, layered cost centers**, not one. Two are root-caused, fixed, and validated
below. The third — the one this doc scopes — turned out to be architecturally the deepest, and is
**not implemented**, per explicit direction to write it up as a plan rather than build it this
session.

Everything below is about `warpSPHCore`'s autograd bridge (`src/warpSPHCore/autograd/`), reached
from `warpSPHIntegrators`' `JFNKSolver`'s `jvp_matvec` (`warpSPHIntegrators/src/warpSPHIntegrators/jfnk.py`),
exercised end-to-end by `warpSPH`'s wave-equation benchmark
(`warpSPH/benchmarks/wave/bench_performance.py`, scheme `sdirk2_jfnk_jvp_1e-6`).

## Reproducing this from scratch

Exact command (run from the `warpSPH` repo root, `warp` conda env), the one used for every number
quoted below:

```sh
python benchmarks/wave/bench_performance.py \
    --nxs 32 64 128 256 --steps 32 --warmup 50 \
    --schemes rk4 sdirk2_jfnk_jvp_1e-6 sdirk2_jfnk_fd_1e-6 \
    --out <some/output/dir> --no-plot
```

`msPerRhs` from `<out>/results.json`, at each stage of this session's work (all measured on the
same machine/GPU in one sitting; run-to-run noise is a few percent — treat differences under ~5%
as noise, not signal):

| nx  | rk4 (baseline) | jvp — original | jvp — after Fix 1 | jvp — after Fix 1+2 | fd — original | fd — after Fix 1+2 |
|-----|---------------:|----------------:|-------------------:|----------------------:|---------------:|---------------------:|
| 32  | 0.451          | 1.4671           | 1.4354              | 1.3950                 | 0.7580         | 0.7594                |
| 64  | 0.4512         | 1.6611           | 1.6743              | 1.6176                 | 0.7892         | 0.8299                |
| 128 | 0.4475         | 1.7153           | 1.6765              | 1.6293                 | 0.8265         | 0.8399                |
| 256 | 0.464          | 1.7734           | 1.6999              | 1.6643                 | 0.8359         | 0.8369                |

jvp/fd ratio, the number this whole investigation is chasing down toward ~1.0x: **~1.94-2.12x
originally → ~1.84-1.99x after both fixes** (computed from the table above; e.g. nx=256:
1.7734/0.8359=2.122 → 1.6643/0.8369=1.989).

Three checked-in scripts reproduce every isolated-call finding referenced throughout this doc,
independent of the full benchmark harness (each runnable standalone, `warp` conda env, needs the
sibling `warpSPH` repo checked out per this workspace's `dev/{warpSPHCore,warpSPH,warpSPHIntegrators}`
layout):

- `scripts/spike_jvp_wave_case_overhead_profile.py` — confirms 0/212 `warpOperationJVP` calls ever
  have a live geometry tangent in this benchmark (Fix 1's caveat), and profiles an isolated dual
  Laplacian(u,u) call against two plain calls (the "1.9x, 12 `zeros_like`/`empty_like` per call"
  finding below).
- `scripts/spike_jvp_dual_wrapping_alternatives.py` — the two ruled-out "wrap more as dual"
  directions (cached-zero tangents, all-live tangents) and the all-zero-tangent-dual crash repro.
- `scripts/spike_jvp_unpack_dual_pruning_mechanism.py` — the `unpack_dual` correctness/cost
  validation Phase 1 depends on.

## What's already fixed this session

### Fix 1 — combined geometry+value JVP kernel fusion (3 launches → 2)

**Root cause**: `warpOperationJVP` (`src/warpSPHCore/operations.py`) computed the "combined" JVP
(a geometry tangent — position/support/mass/density — and a value tangent both live on the same
operator call) as `geometryResult + valueResult`, where `valueResult` was a **full second
relaunch of the primal kernel** (`warpOperation(...)` with tangent arrays substituted for
`fi`/`fj`). Combined with the primal launch from `StateAwareWarpFunction.forward()` and the
dedicated geometry-JVP kernel, that's 3 full O(N×neighbors) passes where 2 should suffice.

**Fix**: each `computeSPH<Op>GeometryJVP` kernel's neighbor loop already computes `A`, `B`, `G`
(needed for the geometry term `dcoeff*G + coeff*dG`) — the value-tangent contribution
`(dfi*A + dfj*B)*G` folds into the same loop for free, using those same values. Implemented for
all 5 value-having operators (Interpolate, Gradient, Divergence, Curl, Laplacian's
Brookshaw/Naive/Dot/Default schemes) — each `wp_<op>JVP.py` now accepts optional
`tangentQueryValues`/`tangentReferenceValues` (default zero, reproducing the old geometry-only
partial exactly when omitted). `operations.py`'s new `_FUSED_VALUE_JVP_OPERATIONS` set gates
`warpOperationJVP` into using the fused single-pass result instead of the old relaunch-and-sum
when both tangents are live.

**Validated**: exact match against the old unfused computation (with and without CRK+renorm) to
float32 roundoff; full `tests/operations/` suite (403 passed, 1 pre-existing *unrelated* failure
in `test_field_abstraction.py`, confirmed via `git stash` to predate this work). One test's
`rtol=0,atol=0` bit-exactness assertion was loosened to `rtol=1e-5,atol=1e-6` — legitimately
different (still exact) floating-point summation order, not a correctness gap. Isolated Gradient
microbenchmark: **1.35x speedup** (0.94ms → 0.70ms/call, nx=128, combined tangent).

**Caveat, confirmed by direct instrumentation**: the wave-equation benchmark never exercises this
path. `WaveSystemStatev3` (`warpSPH/src/warpSPH/systems/waveSystem.py`) tags `u`/`v` as
`'position'`/`'velocity'` purely to reuse integrator plumbing — actual particle `positions` are a
`constant()` field, never part of the JFNK-integrated state. A patched counter around
`warpOperationJVP` over 3 timed `sdirk2_jfnk_jvp_1e-6` steps showed **0/212** calls had a live
geometry tangent — every call took the already-minimal value-only path. This fix matters for a
genuinely Lagrangian (moving-particle) JFNK case; it correctly shows ~0% effect on this one.
Reproduce with `scripts/spike_jvp_wave_case_overhead_profile.py`'s "Part 1".

### Fix 2 — batched `hasLiveTangent` (N syncs → 1 sync per call)

**Root cause**: `StateAwareWarpFunction.jvp()` computed
`live_mask = [hasLiveTangent(t) for t in flat_tangents]`, and `hasLiveTangent` does
`t.abs().max() > 0` — a `bool()` coercion that forces a GPU→CPU sync. Torch's forward-mode-AD
dispatch synthesizes a **non-None zero tensor** (not `None`) as the tangent for every real-Tensor
argument in a call that isn't itself dual, once *any* argument to that call is dual — so a
Laplacian(u, u) call with ~8-10 structurally-constant arguments (positions/supports/masses/
densities × query/reference role) paid ~8-10 separate host round-trips per call, mostly just to
confirm "no, this one isn't live either."

**Fix**: `_liveTangentMask` (`stateAwareWarpFunction.py`) computes every tensor's `.abs().max()`
reduction on-device (no sync — `Tensor.max()` returns a GPU-resident 0-d tensor), stacks the
results, and pays exactly **one** GPU→CPU round trip (`.tolist()`) for the whole call instead of
one per tensor. Wired in at both call sites (`stateAwareWarpFunction.py`'s primary path,
`operator_spec.py`'s defensive fallback).

**Validated**: full test suite green (same 403/1 split as Fix 1). Profiler-confirmed:
`cudaStreamSynchronize` count dropped from ~14/call to exactly 1/call on an isolated dual
Laplacian(u,u) call. Benchmark effect: modest but real — jvp/fd `msPerRhs` ratio improved from
~2.0-2.1x to ~1.84-1.99x across nx=32..256 (a direct instrumentation pass had suggested ~16% of
wall time, before the fix, was `hasLiveTangent`'s sync cost — that figure was itself inflated by
the instrumentation's own extra `torch.cuda.synchronize()` per call, so the true benchmark-level
win is smaller than that number implied, but is real and reproducible).

## What's still open: torch's forward-mode-AD dispatch overhead for non-dual arguments

### The mechanism, confirmed by profiling

Even after both fixes above, an isolated single dual-tensor Laplacian(u, u) JVP call still costs
**1.91x** what two plain (non-dual) Laplacian(u, u) calls cost — nearly the entire remaining
benchmark-level jvp/fd gap. Profiling that isolated call against an equivalent plain-call baseline
(same kernel, same particle count, same profiler) found the difference: the dual call shows **12
`aten::zeros_like` and 12 `aten::empty_like` calls** that are **completely absent** from the plain
baseline's profile. This matches the count of non-dual real-Tensor arguments a Laplacian call
carries (positions/supports/masses/densities × query/reference role, ~8-10 tensors) once the one
genuinely-dual argument (`u`) is excluded.

This is **torch's own `torch.autograd.Function.apply()`/`.jvp()` dispatch machinery**, not
warpSPHCore's bridge code: once any argument to an `apply()` call is a dual tensor, torch
allocates and zero-fills a **fresh tensor** as the synthesized tangent for every other real-Tensor
argument, every single call. This is the same synthesis behavior `hasLiveTangent`'s own docstring
already documented (Fix 2's context) — Fix 2 addressed the cost of *checking* those synthesized
zero tensors; this is the cost of torch *creating* them in the first place, which Fix 2 doesn't
touch. Reproduce with `scripts/spike_jvp_wave_case_overhead_profile.py`'s "Part 2" — it prints the
same 1.9x ratio and the exact `aten::zeros_like`/`aten::empty_like`/`cudaStreamSynchronize`/
`StateAwareWarpFunction` event counts for both the dual call and the plain-call baseline, side by
side.

### Directions considered and ruled out (both tested empirically, not just reasoned about)

All three findings in this subsection reproduce via `scripts/spike_jvp_dual_wrapping_alternatives.py`.

**"Wrap constant fields as dual too, with a cached zero tangent"** — tested directly
(positions/supports/masses/densities each explicitly `fwAD.make_dual(t, cached_zero_tensor)`,
zero tensors pre-allocated once and reused every call, not synthesized per-call by torch).
Result: **1.08x — marginally *worse*, not better**, than today's baseline. Some synthesis
persisted regardless (`zeros_like` count dropped from ~12/call to ~4/call, not to 0), and the
`make_dual`/`unpack_dual` bookkeeping added for the newly-wrapped fields offset what savings there
were.

**"Wrap the whole state as dual, relying on fields becoming genuinely live in a real coupled
simulation"** (motivated by: in a real Lagrangian fluid sim, density→pressure→force→velocity→
position coupling means nearly every field has some live tangent after the first step, so the
"is this exactly zero" ambiguity that costs syncs today mostly evaporates) — tested directly, with
all fields carrying genuinely random (non-zero) tangents. Result: **1.995x — essentially 2x worse**,
not better. Reason: this forces *every* operator call through the full combined geometry+value JVP
path (Fix 1's territory) — a real, physically-necessary second kernel pass (CRK/renorm-aware
chain-rule terms for the position/support tangent) — even for a matvec direction that doesn't
actually need that field differentiated for *this specific* Krylov vector. Dual-wrapping doesn't
make that compute cheaper; it just makes it happen unconditionally. **This is the load-bearing
finding against the "wrap everything" direction generally**: the fix has to reduce how much gets
differentiated per call, not change the mechanism by which non-differentiated fields reach the
call.

**"Statically classify fields as constant vs. differentiable"** (e.g. from a field's
`constant()`/`integrated()` tag in a `BaseState` subclass like `WaveSystemStatev3`) — proposed,
then **rejected on correction from the user**: `constant()` in this codebase describes constancy
*w.r.t. the time-stepping scheme*, not whether a tensor can ever carry a forward-mode tangent. A
field like mass is `constant()` at the integrator level but can still legitimately be dual if a
caller is differentiating the whole pipeline w.r.t. an upstream sizing/material parameter — exactly
the kind of design-sensitivity/shape-optimization use case this codebase's whole AD bridge exists
to support (see `warpier_core.md`/`warpier_adjoint.md`). A static per-field schema would silently
break gradient flow for that use case. **The classification has to be a runtime, per-call fact
about the actual tensor object handed in, not a property of what the field conceptually
represents.**

### The corrected mechanism — validated cheap and correct

`torch.autograd.forward_ad.unpack_dual(t)` returns `(primal, tangent)`, with `tangent is None`
iff `t` is not currently a dual tensor. Spiked directly, checked in as
`scripts/spike_jvp_unpack_dual_pruning_mechanism.py`:

- **Correctness**: correctly detects dual-ness from a callee 3 stack frames removed from wherever
  `make_dual` was called, with no knowledge of which `dual_level` created it. Correctly reports
  `tangent is None` for a tensor held onto *after* its creating `dual_level` has exited (i.e. it
  reflects genuine, currently-active dual status, not a stale flag).
- **Cost**: **0.14us** on a plain tensor, **0.89us** on a dual tensor — pure metadata inspection,
  no GPU sync, no kernel launch. For scale: `hasLiveTangent`'s own `.abs().max()>0` sync costs
  **14.3us** — `unpack_dual` is 15-100x cheaper.

So the mechanism is sound: at the point `_launch` (`wrapper.py`) assembles `flat_tensors` before
calling `StateAwareWarpFunction.apply()`, each tensor's actual dual-ness (`unpack_dual`) and
`requires_grad` can be checked, per call, for near-zero cost. A tensor that is **neither** dual
nor `requires_grad`, right now, for this specific call, carries no gradient information that needs
to cross the `torch.autograd.Function` boundary at all — excluding it from the tracked-argument
list removes it from what torch's dispatch has to synthesize a filler tangent for, with no
correctness cost, and no schema anywhere: the same field is naturally included when it genuinely
is dual (any reason, any caller) and naturally excluded when it genuinely isn't.

### The actual obstacle: `operator_spec.py`'s fixed positional convention

This is *not* a drop-in filter. `operator_spec.py`'s geometry-JVP dispatch reads `flat_tangents`
by **fixed absolute index** — `_QPOS, _RPOS, _QSUP, _RSUP, _QMAS, _RMAS, _QDEN, _RDEN = range(8)`,
`_RENORM_MAT = 10`, `_QVOL, _RVOL = 13, 14`, `_QCRK_A, _QCRK_B, _QCRK_GRADA, _QCRK_GRADB = 15, 16,
17, 18`, `_STATE_N = 36` — assuming `extractStateInfo`'s full 36-slot layout is always present,
in order, in every call. If inert tensors are dynamically dropped from the tracked-argument list
per call, those absolute indices stop meaning what they mean today: if `positions` happens to be
inert and gets excluded this call while `supports` stays tracked, position 2 no longer reliably
means "query support" — the mapping shifts per call, silently, based on which fields happened to
be dual *this time*.

Fixing this requires reworking that indexing from "fixed absolute position into a known-length
array" to something keyed by **stable semantic name**, resolved fresh per call against whichever
subset of `flat_tensors` is actually tracked this time (a dict or small dataclass built in
`_launch`/`extractStateInfo`, e.g. `{"queryPositions": <wp.array or None>, "queryMasses": ...}`,
rather than a positional convention). This touches:

- `arg_extract.py`'s `extractStateInfo` (the source of the 36-slot convention),
- `operator_spec.py`'s `_QPOS`/`_RPOS`/... constants and `_build_geometry_jvp_fn`'s `field(i)`
  lookups,
- all five `wp_<op>JVP.py` files' `_launchGeometryJVP` call sites (`_jvpCommon.py`), which build
  `flat_tensors` in the same fixed order today.

This is the genuinely architecturally deep part of this change — bigger and riskier than either
fix landed this session, which is why it's being scoped rather than built now.

## Phased plan

**Phase 1 — design** (not started): work out the tracked/inert split at the `_launch`/
`StateAwareWarpFunction.apply()` boundary, and the semantic-keyed replacement for
`operator_spec.py`'s fixed-position convention. This is the load-bearing decision; everything
else is comparatively mechanical once it's settled. Concretely:

- Decide the data shape for "whichever fields are tracked this call" — likely a small dataclass or
  dict, built once per `_launch` call, threaded through `build_fn` and into `_build_geometry_jvp_fn`
  in place of today's absolute-index `field(i)` lookups.
- Decide where the `unpack_dual`/`requires_grad` classification happens (`_launch` itself, or a
  small helper it calls) and confirm it composes cleanly with the existing `is_tensor` filtering
  (Field-registry placeholders) already in `StateAwareWarpFunction.forward()`.
- ~~Write a proper, checked-in spike script reproducing the `unpack_dual` correctness/cost
  findings above~~ **Done**: `scripts/spike_jvp_unpack_dual_pruning_mechanism.py`, matching this
  codebase's standard practice (`scripts/spike_forward_mode_tier1.py` and siblings) of a
  validate-before-build spike.
- Confirm reverse-mode (`ctx.save_for_backward`, `gradcheck`) correctness is unaffected: a tensor
  must stay in the tracked list if it's `requires_grad=True` even when not dual, so the pruning
  logic needs both checks, not just `unpack_dual`.

**Phase 2 — prototype on one operator**: implement on Laplacian specifically (the operator the
wave-equation benchmark actually exercises), validate against `gradcheck` and the jacobian-
reference suite (`tests/operations/test_forward_mode_geometry_jvp_laplacian_*.py`), and re-run
`scripts/spike_jvp_wave_case_overhead_profile.py`'s "Part 2" (one dual call vs two plain calls) to
confirm `aten::zeros_like`/`empty_like` drop to ~0 for inert arguments and the dual/plain ratio
approaches the true ~1.0x floor instead of today's ~1.9x.

**Phase 3 — rollout**: extend to the remaining four operators (Interpolate, Gradient, Divergence,
Curl), full test suite, re-benchmark the real wave-equation case end-to-end
(`bench_performance.py --schemes rk4 sdirk2_jfnk_jvp_1e-6 sdirk2_jfnk_fd_1e-6`) for the actual
`msPerRhs` effect, and a second benchmark on a genuinely Lagrangian (moving-particle) case if one
is available, since that's where Fix 1 *and* this fix should compound.

## Open questions needing a decision before Phase 1 starts

1. **Is this worth doing given the measured win so far is unproven at benchmark level?** Fixes 1+2
   are validated end-to-end on the wave-equation benchmark; this one so far is validated only at
   the isolated-call micro-benchmark level (1.91x → target ~1.0x on that specific measurement).
   Worth a small time-boxed Phase 1 spike to get a real `bench_performance.py` number before
   committing to Phases 2-3's broader refactor, or proceed straight to the full design?
2. Does the semantic-keyed lookup replace `operator_spec.py`'s fixed-position convention
   *everywhere* (a codebase-wide change to how `extractStateInfo` communicates its layout), or
   coexist as a parallel path used only by the JVP dispatch machinery, leaving the plain
   forward/backward path's positional convention untouched? The latter is smaller and lower-risk;
   the former removes a second parallel convention from the codebase permanently.
3. Does reverse-mode (`backward()`/`gradcheck`) need an equivalent pruning pass for consistency,
   or is this scoped to forward-mode only? Reverse-mode's cost shape is different — there's no
   torch-side "synthesize a zero tangent" step for `backward()`, so the motivating problem may be
   forward-mode-specific; worth confirming rather than assuming before Phase 1 locks scope.
