# Roadmap: Unified State Interface and Forward-Mode AD Infrastructure

## Motivation

The current Warp backend has gradually evolved towards using semantic state objects (e.g. particle states, correction states, grid states) instead of flattened kernel argument lists. This has significantly improved readability and maintainability. However, the codebase still contains a mixture of legacy interfaces, repeated wrapping logic, and Python-side tensor marshalling that complicate future extensions, particularly forward-mode automatic differentiation.

Rather than implementing forward AD directly on the existing architecture, the goal is to first complete the transition towards a unified execution model. Once this abstraction is in place, forward-mode AD becomes an extension of the state representation rather than a modification of every kernel.

---

# Overall Intent

The long-term objective is to establish a single abstraction for simulation data that is independent of

* the storage backend (Torch vs Warp),
* the differentiation mode (none, reverse, forward),
* and the neighborhood traversal method (grid vs adjacency).

The SPH operators should only operate on semantic state objects and should not need to know how those states are stored or differentiated.

---

# Repository Reality Check (Current Status)

This section captures current implementation status relative to the target architecture. Last verified against the repo on 2026-08-06.

Detailed bug-fix narratives that used to live in this section (AD-bridge reentrancy, grid-path float64/1D bugs, curl's compile failure, renormalization instabilities, the gradcheck rollout, CI wiring) have been distilled into forward-looking rules in [`docs/lessons_learned.md`](docs/lessons_learned.md) and trimmed out of here to keep this plan focused and short. Read that file before touching kernel code, the AD bridge, or CI/test tooling again — it captures the *why*, not just the *what*. Git history (commits around 2026-08-05 to 2026-08-06) has the original investigation narrative if ever needed.

## Notebook Corpus Status

All operation-relevant notebooks are ported and accounted for in root (`warp_density`, `warp_interpolate`, `warp_gradient`, `warp_divergence`, `warp_laplacian`, `warp_curl`, `warp_renorm`, `warp_custom`, `warp_profile`). Grid dispatch and CRK checks don't need their own notebooks — see `docs/lessons_learned.md`'s "Notebook/documentation conventions". `docs/regression/notebook_test_matrix.md` has the notebook-to-test mapping.

## Already in Place

* A high-level semantic interface exists through `warpOperation` and state objects (`ParticleState`, `OperationProperties`).
* State-aware autograd infrastructure exists (`extractStateInfo`, `warpWrapper2`, `StateAwareWarpFunction`).
* Conversion hot paths use caching for non-differentiable data (dummy tensors, dtype caches). Differentiable Warp-array caching (`getCachedWarpArray` and its `wp_autograd.py` mirrors) was deliberately removed, not fixed — see `docs/lessons_learned.md` for why that class of caching is unsafe here.
* A structured kernel ABI is demonstrated by renormalization covariance (`wp_covariance.py`) using state structs rather than fully flattened arguments.

## Gaps Against the Target

* ~~`warpOperation` still routes through `sphOperation_warp`~~ — **done, and inverted**. `warpOperation` now dispatches directly from state objects to each operator's `_computeSPHX_stateBackend`; `sphOperation_warp` (the flat-tensor "manual" entry point) assembles the same state objects and calls `warpOperation` instead of the other way around. See "States as the Primary Path" below.
* All six operators (Density, Interpolate, Gradient, Divergence, Curl, Laplacian) now launch via the structured `warpWrapper2` wrapper; the flat `warpWrapper` path and the `operations_grid/` package it served are gone entirely. See "Working Prototype → Production" below.
* State objects remain torch-native and do not yet own synchronized torch+warp field representations (Phase 3, `Field` abstraction — still not started).
* ~~Adjacency and grid execution paths are still largely duplicated across operation families~~ — **done**. Every operator's traversal now lives in one `_Func_Adjacency`/`wp.kernel` pair that branches on `useAdjacency` at runtime; `operations_grid/` has been deleted.
* ~~CRK and renormalization corrections cannot run on grid-mode traversal~~ — **done for all six operators**. Every operator's `correctionData` struct threads CRK (and, where applicable, grad-h/renorm) through the grid branch for free, since there is only one branch-per-traversal-mode kernel now, not a separate grid kernel that never got the correction paths wired in.
* `LaplacianScheme.Dot` does not support scalar fields in `dim>1` domains — guarded with a `ValueError`, not fixed. See `docs/lessons_learned.md`. (Unrelated to the traversal-unification work above; still open.)

## Largest Remaining Change

Phase 1 (structured kernel ABI) and Phase 5 (traversal consolidation) are now done for every operator — see "Working Prototype → Production" below. What's left is Phase 2/3/4: state objects are still torch-native dataclasses (`ParticleState` et al.), not a `Field` abstraction that owns synchronized torch+warp representations with dirty-tracking, and there's still real per-call Python-side struct construction (`extractStateInfo`) rather than cached/reused state. The key remaining migration step is introducing that `Field` abstraction, while preserving legacy fallback conversion paths for compatibility. Forward-mode AD (Phase 6/7) still hasn't been started — but it now has a much smaller surface to extend, since there is exactly one kernel per operator to add tangent-carrying arguments to, not two.

## Working Prototype → Production: Unified Kernel + Traversal — DONE for all six operators

`wp_grad.py` (repo root) started as a from-scratch prototype reimplementation of the Gradient operator, exercised against the production `warpOperation` path via `warp_gradient.ipynb`. **That prototype has since landed in production, and the same recipe has now been applied to every operator**: Density, Interpolate, Gradient, Divergence, Curl, and Laplacian each have exactly one unified kernel (`operations/wp_<op>.py`) handling both traversal modes. `operations_grid/` — the entire package, all six `wp_<op>_grid.py` files plus `wp_operation_grid.py` and `sphOperation_warp_grid` — has been deleted; `sphOperation_warp` (`operations/wp_operation.py`) is now a single dispatcher with no adjacency-type branch at all. See "Landing Interpolate", "Landing Density", and "Landing Divergence, Curl, and Laplacian" below for what differed operator-to-operator, and "Collapsing `sphOperation_warp`" for how the top-level dispatcher itself simplified once the last operator was exempted.

### What's new here vs. what already existed

The structured kernel ABI (`particleDataSoA_{1,2,3}`, `domainData`, `adjacencyData`, `gridData`, `correctionData_{1,2,3}` and their `getParticle`/`getL_i`/`getGradH_i`/`getVolume_i`/`getCRK_i` accessors, all in `warp_state.py`) and the state-aware autograd bridge (`extractStateInfo`/`warpWrapper2`/`StateAwareWarpFunction`, in `warp_state_util.py`) already existed before `wp_grad.py` — they were built for renormalization covariance (`renorm/wp_covariance.py`). What `wp_grad.py` demonstrates is that this machinery generalizes cleanly to a full differentiable neighbor-sum operator, and — the actually new part — that **one kernel can serve both traversal modes**:

* `computeGradient_Func_i` — the per-neighbor physics (kernel gradient, CRK correction, grad-h, the four `GradientScheme` variants). Line-for-line the same math as the old `computeSPHGradientTensor_Func` in `operations/wp_gradient.py`, just reading from `referenceState`/`domainState`/`correctionData` structs instead of eight parallel flat arrays.
* `computeGradient_Func_Adjacency` — resolves `xi/hi/mi/rhoi` and the four correction flags for query point `i` once, then does `for o in range(numOffsets): ... beginIndex, numIndices = (adjacencyState fields) if useAdjacency else checkOffset(gridState, ...)`, and calls `computeGradient_Func_i` with whichever `(beginIndex, numIndices, offsetArray)` it got. This loop shell is the traversal abstraction Phase 5 asks for — the same function drives both the CSR neighbor-list case and the compact-hash grid-cell case, because the two only differ in how `(beginIndex, numIndices)` are produced, not in what happens once you have them.
* `computeGradient_Kernel` — a trivial `wp.tid()` + bounds check + one call into the `_Func_Adjacency` above. One kernel, not two.
* `computeGradient` (Python) — calls `warpWrapper2` directly with `defaultStateArguments=(queryParticles, operationProperties, domain, queryVolumes, referenceVolumes, adjacency, referenceParticles, crkState, gradHState, renormalizationState)`, i.e. it never goes through `sphOperation_warp`/`sphOperation_warp_grid`'s `isinstance(adjacency, CompactHashMap)` dispatch. There is nothing left to dispatch: `extractStateInfo` already sets `useAdjacency = not isinstance(adjacency, CompactHashMap)` and hands both traversal representations to the kernel every time (the unused one as cheap dummy tensors), and the kernel picks the live one at runtime via the `useAdjacency` bool.

Interface-wise `computeGradient(...)` takes the same arguments as `warpOperation(...)` (`ParticleState`, `OperationProperties`, `DomainDescription`, `queryValues`/`referenceValues`, `queryVolumes`/`referenceVolumes`, `adjacency`, `referenceParticles`, `crkState`, `gradHState`, `renormalizationState`) and drops only `consistentDivergence`, which `warpOperation` exposes as a top-level kwarg but which has no meaning for a pure gradient.

### Validation (`warp_gradient.ipynb`)

* Cells 3–5 run the *same* `CompactHashMap` (from `radiusSearchCompactHashMap`) through both `warpOperation(..., operation=WarpOperation.Gradient, ...)` (legacy path → `sphOperation_warp_grid` → `operations_grid/wp_gradient_grid.py`) and the new `computeGradient(...)`. Both traverse in grid mode (`isinstance(adjacency, CompactHashMap)` is `True` for both), and the recorded MAE against the analytic gradient is identical to the printed digits (`0.006049483548849821` both times); the direct difference `linearGradient_new - linear_gradient_warp` is exactly `0.0`. This is bit-exact parity between the old dedicated grid kernel and the new unified kernel's grid branch.
* Separately, running the same new kernel once with an explicit `AdjacencyList`/`AdjacencyListWarp` neighbor list (`useAdjacency=True` branch) and once with a `CompactHashMap` (`useAdjacency=False` branch) agrees only to within roughly one float epsilon rather than bit-exactly. That's expected, not a bug: the two branches sum the same neighbor contributions in different orders (CSR neighbor order vs. cell-by-cell grid order), and floating-point addition isn't associative. Worth calling out explicitly in the operator's docstring/tests when this migration lands, so nobody mistakes the ~1 ULP drift for a regression.

### Migration recipe (pivoting an existing operator to this style)

This recipe was applied to all six operators; kept here for reference the next time a new SPH operator is added, or if this pattern needs to be reapplied after a larger refactor:

1. Take the existing `compute<Op>Tensor_Func` from `operations/wp_<op>.py` and rewrite its signature to consume `referenceState: Any` (a `particleDataSoA_D`), `domainState: domainData`, and `correctionData: Any` instead of the current flat `wp.array` parameter list, pulling per-neighbor values via `getParticle(referenceState, j)` and per-query correction terms via `getL_i`/`getGradH_i`/`getVolume_i`/`getCRK_i` at the call site rather than as separate flat args. The physics body does not change — compare `computeGradient_Func_i` (`wp_grad.py:29`) with `computeSPHGradientTensor_Func` (`operations/wp_gradient.py:17`) for a worked diff.
2. Wrap it in a `compute<Op>_Func_Adjacency` that resolves query-point state once, then loops `for o in range(numOffsets)` picking `(beginIndex, numIndices, offsetArray)` from `adjacencyState` or from `checkOffset(...)` against `gridState` depending on `useAdjacency` — copy `computeGradient_Func_Adjacency` (`wp_grad.py:128`) verbatim except for the inner call. `numOffsets` is `1` when `useAdjacency` and `gridState.numOffsets` otherwise (see `computeGradient_Kernel`, `wp_grad.py:226`).
3. Reduce the `wp.kernel` to `wp.tid()` + bounds check + one call into step 2's function. This single kernel replaces both `compute<Op>Tensor_Kernel` (`operations/wp_<op>.py`) and `compute<Op>Tensor_grid_Kernel` (`operations_grid/wp_<op>_grid.py`).
4. Replace the two Python backends (`compute<Op>_warpBackend` and `compute<Op>_grid_warpBackend`) with one function that calls `warpWrapper2` the way `computeGradient` does — `defaultStateArguments` matching `parseArguments`'s ten-argument order, plus operator-specific tensors (e.g. `queryValues`/`referenceValues`) in `additionalArguments`.
5. Point `sphOperation_warp`'s dispatch for that operation at the new unified function and delete the `isinstance(adjacency, CompactHashMap)` branch for it; once every operator is migrated, `wp_operation.py` and `wp_operation_grid.py` collapse into one dispatcher.
6. Delete `operations_grid/wp_<op>_grid.py` only after the `operation-matrix` and `gradcheck` skills both pass for the migrated operator across adjacency and grid modes (and CRK/renorm, once plumbed through — the unified `correctionData` struct threads CRK/renorm through the grid branch for free, which is what closes the "CRK and renormalization cannot run on grid-mode traversal" gap noted above).

This recipe *is* Phase 1 (structured ABI) and Phase 5 (traversal consolidation) for one operator at a time, using infrastructure (`warpWrapper2`, `warp_state.py` structs) that Phase 2/3 already built for the covariance kernel — so the remaining phases mostly collapse into "repeat this six times and delete the old files," rather than requiring new infrastructure design. Gradient was migrated first here (not Density, as the original Step 4 below suggests) because it has the most correction paths (CRK, grad-h, renormalization, volume) to prove out; Density is the better *second* target precisely because it's the simplest operator.

### Landing Gradient in production: what changed, and two bugs the port surfaced

Following the recipe above turned up two real bugs — one pre-existing in the `wp_grad.py` prototype (invisible there because the prototype was only ever checked forward-value, never gradchecked), one introduced while porting. Both are exactly the class of bug the `docs/lessons_learned.md` testing-methodology lessons warn about: forward-only checks (the notebook, `operation_matrix.py`) cannot see either one, only `torch.autograd.gradcheck` can.

* **`zero_like_warp(outputValues)` on the *array itself*, not an element, silently breaks for any output longer than 3 components.** `zero_like`/`zero_like_warp` (`utils/wp_util.py`) is overloaded per concrete type, but its `wp.array(dtype=vector(length=N,...))` overloads only go up to `N=3` and its `matrix` overloads only up to `3x3`. A scalar-field gradient in 2D flattens to `flatOutputShape=2` — covered, which is why the prototype's notebook (scalar fields only) never saw a problem. A vector- or matrix-field gradient flattens to 4, 6, 8, or 9 components — uncovered, and Warp's overload resolution falls back to the fully generic `type(input)() * scalar_t(0.0)` overload, which fails to compile at all for an array argument (`Could not find function array<...> as a built-in`). Fixed by indexing the array first: `zero_like_warp(outputValues[i])`, matching the pattern the pre-migration kernel already used (`type(outputValues[i])(scalar_t(0.0))`). Fixed in both `operations/wp_gradient.py` and the `wp_grad.py` prototype.
* **A ternary reintroduced during the port silently zeroed `d(output)/d(referenceValues)`.** The pre-migration kernel wrote `fj`'s grad-h correction as an explicit `if useGradHTerms: fj = referenceValues[j] / referenceOmegas[j] else: fj = referenceValues[j]` (and the `wp_grad.py` prototype kept that explicit form too). While rewriting it into the new kernel, this became a ternary — `fj = referenceValues[j] / referenceOmegas[j] if useGradHTerms else referenceValues[j]` — which is exactly the "ternary assigned to a local, both branches index the same array" shape that `docs/lessons_learned.md` documents as having already broken Interpolate once (silently-zero adjoint, correct forward value, no error). `useGradHTerms=False` in every failing case, so the runtime-taken branch was always the innocuous one — the mere presence of the ternary was enough to zero the adjoint. Caught by `scripts/gradcheck_gradient_native.py`'s "line of 7 particles" case (any case with a real neighbor loop; the "single particle" no-neighbor case can't exercise it, which is why it passed). Reverted to the explicit `if/else` form.

Other changes made while landing this:

* `operations_grid/grid_util.py` (the `checkOffset`/`iterateCell`/`wrapCellComponentPeriodic` grid-traversal primitives) moved to `radiusSearch/grid_util.py`. The unified Gradient kernel needs `checkOffset`, but `operations/wp_gradient.py` importing anything under `operations_grid.*` would force `operations_grid/__init__.py` to execute while `operations/wp_operation.py` (which imports `operations_grid` itself, later in the same file) is still mid-import — a real circular import, not a hypothetical one. `radiusSearch` has no dependency on `operations`/`operations_grid` in either direction, so it's a safe home for traversal primitives that both sides need; this is also the right conceptual home per Phase 5 (traversal shouldn't be owned by the grid-specific operator package). All six `operations_grid/wp_*_grid.py` importers and `renorm/wp_covariance.py` were repointed to the new path.
* `sphOperation_warp`'s top-level dispatch (`operations/wp_operation.py`) now reads `if operation != WarpOperation.Gradient and (adjacency is None or isinstance(adjacency, CompactHashMap)): return sphOperation_warp_grid(...)` — Gradient is exempted because `computeSPHGradient_warpBackend` now handles `None`/`CompactHashMap`/`AdjacencyList` itself (via `extractStateInfo`). The `queryKinds`/`referenceKinds` AllToAll-dummy fallback just below it used to read `adjacency.numNeighbors` unconditionally, which assumed grid-mode Gradient calls would never reach it (they used to redirect away above); since Gradient can now reach that line with `adjacency=None`, it falls back to `getCachedDummyTensor(...)` in that case instead of crashing on `None.numNeighbors`.
* `computeSPHGradient_warpBackend`'s public flat-tensor signature is unchanged (so `wp_operation.py`'s call site needed no changes beyond the dispatch-branch condition above), but it no longer supports `scatteredQuantities`/pre-scattered quantities — raises `NotImplementedError` if a caller passes one. Nothing in this repo's tests, scripts, or notebooks does (confirmed by grep); the prototype had already dropped this for the same stated reason ("wasn't ever used and caused issues with autograd").
* A dead import (`from ..operations.wp_gradient import computeSPHGradientTensor_Func`, never actually called) in `diffusion/viscosity.py` was removed rather than preserved under the new kernel's different signature.

Validation: all 63 pytest cases in `tests/operations/` pass (including `gradcheck_gradient_native.py` and the grid/adjacency dispatch tests), and `scripts/run_operation_matrix_sweep.sh --full` was run before considering this done, per that script's own guidance to use `--full` (not `--quick`) for a change that touches a shared traversal path and the AD bridge.

### Landing Interpolate in production: simpler operator, one real correctness question resolved

Interpolate was the second operator migrated, following the recipe above exactly (`computeSPHInterpolation_Func_i` / `_Func_Adjacency` / `_Kernel` in `operations/wp_interpolate.py`, `operations_grid/wp_interpolate_grid.py` deleted, `sphOperation_warp`'s dispatch condition extended to `operation not in (WarpOperation.Gradient, WarpOperation.Interpolate)`). It's a simpler operator than Gradient — no `outerTensorProduct`, no grad-h, no renormalization, no extra output dimension — so this migration mostly confirmed the recipe generalizes rather than surfacing new infrastructure gaps. Two things worth recording:

* **Position wrapping for grid traversal turned out to be unnecessary, not just for Gradient but for Interpolate too — resolving an inconsistency in the pre-migration code.** The pre-migration `sphOperation_warp_grid` unconditionally computed periodicity-wrapped positions (`x`, `y`, via `torch.remainder`) for every operation, but only Interpolate's grid backend actually used them (`computeSPHInterpolant_grid_warpBackend(y, x, ...)`) — Gradient's grid backend already took raw, unwrapped positions and was already proven correct (bit-exact vs. the unified kernel, which also uses raw positions). `checkOffset`'s cell-index computation wraps the *integer cell index* via modulo (`wrapCellComponentPeriodic`), which is mathematically equivalent to wrapping the position first, as long as the domain's cell width evenly divides its extent (true by construction for a uniform hash grid) — so pre-wrapping was redundant, not load-bearing. The unified Interpolate kernel uses raw positions like Gradient's, and both the 63-case pytest suite and the `--full` operation-matrix sweep (including jittered/periodic configurations) pass at `MAE=0.000`, confirming this empirically rather than just on paper. The dead `x`/`y`/`minD`/`maxD` computation (and the now-unused `getDomainExtents` import) was removed from `operations_grid/wp_operation_grid.py` since Interpolate was its only consumer.
* **`CRKState` requires `gradA`/`gradB` even though Interpolate never reads them.** `getCRK_i`/`correctionData` always carry all four CRK fields (`A`, `B`, `gradA`, `gradB`) because Gradient/Divergence/Curl need the gradient-correction terms, but `CRKState` the dataclass has no defaults, and Interpolate's flat backend signature only ever received `crk_A`/`crk_B` (no `crk_gradA`/`crk_gradB` — it doesn't need the kernel-gradient correction, only the kernel-value correction). Reusing `crk_B` as a stand-in for `gradA`/`gradB` (as a first draft of the adapter did) is a real bug, not just an inelegance: `gradB` is a `[N,D,D]` matrix field, and `crk_B` is `[N,D]`, so this is a shape mismatch that `extractStateInfo`'s struct-building would choke on. Fixed by building correctly-shaped dummy tensors (`getCachedDummyTensor((1,dim), ...)` / `((1,dim,dim), ...)`) instead, matching how `wp_operation.py`'s own dummy-filling already does this for other unused optional corrections.

Validation: all 63 pytest cases pass, `scripts/gradcheck_interpolate_native.py` passes standalone, and `scripts/run_operation_matrix_sweep.sh --full` is clean (all 20 gated configurations `HIGH=0 ERR=0 NAN=0`, including the `Interpolate[matrix]` case that exercises the rank>3 flatten/reshape path).

### Landing Density in production: the trivial case, confirming the floor of the recipe

Density is the simplest operator in the family — no `queryValues`/`referenceValues` at all (it computes the density field, it doesn't consume one), and none of the four correction paths (no CRK, no volume, no grad-h, no renormalization) apply to it. `computeSPHDensity_Func_i` is just `out += mj * sphKernel(...)` summed over neighbors; `_Func_Adjacency` and `_Kernel` are the same traversal shell as Gradient/Interpolate, just with an empty `additionalArguments=()` in the `warpWrapper2` call (there's no per-operator tensor beyond the particle state itself). This migration surfaced no new bugs — it's here mainly as confirmation that the recipe's traversal-shell part is genuinely operator-agnostic even at the minimum end of the correction-path spectrum, and as the reason `sphOperation_warp`'s grid-redirect exemption condition is now `operation not in (WarpOperation.Gradient, WarpOperation.Interpolate, WarpOperation.Density)`.

One structural note carried over from Interpolate: `sphOperation_warp` and `sphOperation_warp_grid` both special-cased Density as their very first branch (before the `queryValues`/`preScatteredQuantities` validation that doesn't apply to it), unlike Gradient/Interpolate which flowed through that shared validation. Exempting Density from the top-level grid redirect preserves that ordering — its dispatch in `sphOperation_warp` still happens before the `queryValues`/`preScatteredQuantities` checks, unchanged from before this migration. The corresponding `if operation == WarpOperation.Density: return computeSPHDensity_grid_warpBackend(...)` branch in `sphOperation_warp_grid` was removed since Density can no longer reach it, along with its now-dead `computeSPHDensity_grid_warpBackend` import.

Validation: all 63 pytest cases pass, both `scripts/gradcheck_density.py` (closed-form self-term check) and `scripts/gradcheck_density_native.py` pass, and `scripts/run_operation_matrix_sweep.sh --full` is clean.

### Landing Divergence, Curl, and Laplacian in production: the rest of the Gradient family

These three share nearly all of Gradient's correction-path machinery (CRK, grad-h, volume, renormalization) and its `computeKernelGradientCRK`-based per-neighbor loop; each differs from Gradient only in how the per-neighbor term is *contracted* into the output, and in a couple of operator-specific scalars that don't fit the fixed 14-argument struct prefix (`queryState, referenceState, domainState, useAdjacency, adjacencyState, gridState, correctionData, mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt`) that `extractStateInfo`/`warpWrapper2` always build:

* **Divergence** uses `divergenceProduct` (contracts the input's last/first axis against the kernel gradient) instead of Gradient's `outerTensorProduct` (which appends a new axis). `dotMode` reuses the canonical ABI's `divergenceMode_int` slot directly (`OperationProperties.divergenceDotMode` already flows through `extractStateInfo` into that field — no new plumbing needed), but `consistentDivergence` has no home in the canonical struct (it's a `sphOperation_warp`-level kwarg, not an `OperationProperties` field), so it travels as an extra `wp.bool` in `warpWrapper2`'s `additionalArguments`, the same mechanism Gradient already uses for `queryValues`/`referenceValues`.
* **Curl** uses `curlProduct` (Levi-Civita / cross-product contraction, with separate 1D/2D/3D overloads) and has Curl-specific output-shape logic (full input shape in 3D, one axis dropped in 2D, always scalar in 1D) in place of Gradient's "append a spatial axis" rule. No extra non-struct scalars needed. While copying `curlProduct` over, also found and dropped `getStride` — a dead helper defined in both the old adjacency and grid files but never actually called from either.
* **Laplacian** is the one case where `positiveDivergence_int` (already in the canonical struct prefix, but ignored/pass-through-only in Gradient/Divergence/Curl) is genuinely read and used. Its per-neighbor term (`q_ij`, reusing `GradientScheme` to pick a differencing form — see the long comment in `computeSPHLaplacianTensor_Func_i` on why all four schemes collapse to a `(fj - fi)`-based difference here specifically) is combined with the kernel gradient via `computeDotLaplacian`/`computeLaplacianDot2`/a direct kernel-Laplacian evaluation, selected by `laplacianMode_int` (also already in the struct prefix). No extra non-struct scalars needed either. `LaplacianScheme.Dot`'s existing scalar-field-in-`dim>1` guard (`docs/lessons_learned.md`) was preserved verbatim in the new `_computeSPHLaplacian_stateBackend`.

The now-familiar ternary-adjoint-zeroing pattern (`fj = referenceValues[j] / referenceOmegas[j] if useGradHTerms else referenceValues[j]`) was avoided from the start in all three by writing the explicit `if/else` form directly, rather than being caught by gradcheck after the fact as it was for Gradient.

Validation: all 63 pytest cases pass for each operator's migration individually and cumulatively, `gradcheck_divergence_native.py`/`gradcheck_curl_native.py`/`gradcheck_laplacian_native.py` all pass standalone, and a final `scripts/run_operation_matrix_sweep.sh --full` run after all three (and the `sphOperation_warp` collapse below) landed together is clean — all 20 gated configurations `HIGH=0 ERR=0 NAN=0`, with adjacency/grid MAE identical per scheme/correction combination for every operator (e.g. `Divergence[Naive] [adjacency/base]` and `[grid/base]` both `MAE=0.7625`).

### Collapsing `sphOperation_warp`: `operations_grid/` deleted entirely

Once Laplacian — the last operator still using the old split — was migrated, `sphOperation_warp`'s top-level branch (`if operation not in (...) and (adjacency is None or isinstance(adjacency, CompactHashMap)): return sphOperation_warp_grid(...)`) had every `WarpOperation` value in its exemption tuple, making the branch permanently unreachable: no operation could ever take it. Rather than leave a dead branch (and a dead `operations_grid` package behind it) in place, both were removed:

* The redirect branch and the `from ..operations_grid import sphOperation_warp_grid` import were deleted from `operations/wp_operation.py`. `sphOperation_warp` now goes straight from argument validation/defaulting to the per-operation dispatch (`if operation == WarpOperation.Density: ... elif operation == WarpOperation.Interpolate: ...` etc.) for every operator, with `adjacency=None` and grid-vs-list traversal handled inside each operator's own backend via `extractStateInfo`, exactly as documented above for each operator individually.
* `operations_grid/wp_laplacian_grid.py`, `operations_grid/wp_operation_grid.py`, and `operations_grid/__init__.py` were deleted, along with the directory itself — nothing in the codebase imports `operations_grid` anymore (confirmed by grep; the only remaining references are historical comments explaining *why* the old split existed, in the operator files and two test docstrings, which were reworded to stop describing a dispatch path that no longer exists).
* Two now-dead `sphOperation_warp_grid`-only kwargs (`consistentDivergence`, `divergenceDotMode`) were dropped from the (now-deleted) `sphOperation_warp_grid` signature and from `sphOperation_warp`'s internal call to it — moot now that the whole function is gone, but recorded here since it was a small independent cleanup made in passing.

This is Phase 1 and Phase 5 fully realized for the SPH operator layer: one structured kernel ABI, one traversal-branching kernel per operator, no adjacency-type-based dispatch tree left anywhere in `sphOperation_warp`.

### States as the Primary Path: `warpOperation` dispatches directly, `sphOperation_warp` adapts

Collapsing `sphOperation_warp` (above) removed the adjacency-type dispatch tree, but left a different piece of redundancy in the call graph: every call, including the common case of a caller who already has `ParticleState`/`OperationProperties` objects in hand, still went `warpOperation` (state objects) → disassembles into ~25 flat positional/keyword tensors → `sphOperation_warp` (flat) → dispatches to `compute<Op>_warpBackend` (flat) → reassembles the exact same tensors back into `ParticleState`/`CRKState`/`GradHState`/`RenormalizationState` → `_compute<Op>_stateBackend` (state objects) → `warpWrapper2`. Two full disassemble/reassemble round trips per call, on the path every operator call actually takes.

This has been inverted so states are the primary path, matching the target architecture's framing (state objects "independent of the storage backend... differentiation mode... traversal method"):

* `warpOperation` (`operations/wp_operation.py`) now does the dispatching itself: it takes `queryParticles`/`referenceParticles`/`crkState`/`gradHState`/`renormalizationState`/`operationProperties` as before, normalizes `gradHState`/`renormalizationState` if given as a bare tensor or tuple (unchanged from before), runs the same validation that used to live in `sphOperation_warp` (queryValues/referenceValues presence, the preScatteredQuantities combo checks, the CRK-gradA/gradB-required-for-Gradient/Divergence/Curl check — now checked as `crkState.gradA is None` rather than a separate `crk_gradA` flat arg), and calls the appropriate `_computeSPHX_stateBackend` directly. No flattening, no reassembly.
* `sphOperation_warp` is now the thin adapter: it keeps its exact pre-existing flat-tensor signature (so no caller-visible break), does flat-API-only sanity checks that can't occur through the state API by construction (e.g. `useCRK=True` but `crk_A=None` — structurally impossible if you're building a `CRKState` object instead of independent flags-plus-tensors), builds `ParticleState`/`OperationProperties`/`CRKState`/`GradHState`/`RenormalizationState` from its flat arguments, and calls `warpOperation`. It no longer dispatches per-operation itself — that's `warpOperation`'s job now, exercised by both entry points.
* The five `compute<Op>_warpBackend` flat-tensor adapter functions (Interpolate/Gradient/Divergence/Curl/Laplacian; Density's equivalent was folded directly into its `_stateBackend`) are deleted entirely — nothing called them except `sphOperation_warp`'s old per-operation dispatch, and that dispatch is gone. Every operator file now exposes exactly one public backend, `_computeSPHX_stateBackend`, taking state objects.
* One real (non-mechanical) piece of logic had to move, not just get deleted: Interpolate's CRK dummy-`gradA`/`gradB` fill (`CRKState` requires `gradA`/`gradB` even though Interpolate never reads them — see "Landing Interpolate" above). This used to live in the now-deleted `computeSPHInterpolant_warpBackend`, reached from both entry points because both funneled through it. It now lives directly in `_computeSPHInterpolant_stateBackend` (`operations/wp_interpolate.py`), gated on `crkState.gradA is None or crkState.gradB is None` rather than always overwriting — so a caller who *does* supply real `gradA`/`gradB` on a shared `CRKState` (e.g. reusing one `CRKState` across an Interpolate call and a Gradient call) now gets those real tensors passed through instead of unconditionally discarded, which is harmless either way since Interpolate's kernel never reads them, but is the more honest behavior for a state-first API.
* A handful of `sphOperation_warp`-level dummy-tensor fills (`renormalizationMatrices`, `queryOmegas`/`referenceOmegas`, `queryVolumes`/`referenceVolumes`, `crk_A`/`crk_B`/`crk_gradA`/`crk_gradB` all defaulting to `getCachedDummyTensor(...)` when `None`) were dropped rather than carried over: tracing them showed every one was immediately discarded a few lines later by each `compute<Op>_warpBackend`'s own `X if useX else None` before ever reaching a state object, i.e. they were dead code left over from an earlier fully-flat design, not load-bearing. Likewise the `queryKinds`/`referenceKinds` AllToAll-dummy fallback that used to run inside `sphOperation_warp` is gone — `checkKinds` (`utils/arg_check.py`, called from `extractStateInfo`) already does the identical `None` → dummy substitution, so `ParticleState.kinds=None` now flows through cleanly without a redundant fill upstream.

Validation: all 63 pytest cases pass, all seven gradcheck scripts (`tests/operations/test_gradcheck_scripts.py`) pass, and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

---

# Phase 0 - Build Regression Ground Truth From Notebooks (First Step)

## Status: Done (acceptance gate for Phase 1+ is met, with deliberate deferrals)

A reproducible regression baseline exists: 63 pytest cases passing (`tests/operations/`, forward-value + grid-path + gradcheck coverage), `scripts/operation_matrix.py` clean across every configuration CI gates on, and `torch.autograd.gradcheck` coverage landed for all six operators. See `docs/lessons_learned.md` for the technical rules this baseline-building work surfaced — several real bugs in kernels and the AD bridge were found and fixed along the way, and that file is what carries forward, not this status log.

## Goal

Create a reproducible regression suite and documentation baseline before refactoring execution interfaces (Phase 1+).

## Deliverables

* `tests/operations/{conftest,test_operations_core,test_operations_consistency,test_operations_crk_analytic,test_grid_modes,test_gradcheck_scripts}.py` — consolidated pytest coverage (63 passed) spanning forward-value analytic checks, grid-path (`adjacency=None`) checks, and per-operator gradcheck-script coverage, for density/interpolate/gradient/divergence/curl/laplacian.
* `docs/regression/notebook_test_matrix.md` — notebook-to-test mapping.
* `scripts/operation_matrix.py` — forward-value diagnostic matrix, configurable device/precision/dim/jitter; `scripts/run_operation_matrix_sweep.sh` wraps it (`--quick` for routine use, `--full` for a broader sweep). See the `operation-matrix` skill.
* `scripts/gradcheck_{density,density_native,interpolate_native,gradient_native,divergence_native,curl_native,laplacian_native}.py` — per-operator `torch.autograd.gradcheck` coverage, run via `tests/operations/test_gradcheck_scripts.py` or directly. See the `gradcheck` skill.
* `.github/workflows/tests.yml` — CI: full pytest suite + `operation_matrix.py --ci` across five targeted configurations (2D float32, 2D float64, 1D, jittered, CUDA-gated 3D), push/PR triggered.
* `.claude/skills/gradcheck/`, `.claude/skills/operation-matrix/` — reusable skills for re-running this coverage during the Phase 1+ migration instead of re-deriving it each time.
* Deliberately deferred, not scheduled (revisit only after Phases 1-6 settle — see `docs/lessons_learned.md` for why sound thresholds need real investigation first): `tests/data/` golden-data baseline fixtures, a nightly CI sweep of the full precision×dim×jitter product, full CUDA CI coverage beyond the 3D step, per-test behavior-spec docs.

## Exit Criteria

* Every operation has at least one notebook-derived regression case. — **Met**, 6/6 operations.
* Baselines reproduce on repeated runs for the same backend/device. — **Met** for the deterministic lattice case; golden-output verification is deferred (see above).
* CI can block regressions before and during migration. — **Largely met**: pytest + `operation_matrix.py --ci` across five configurations gate every push. Full CUDA CI and the full jittered/precision/dim sweep are both deliberately deferred, not unstarted gaps.

---

# Phase 1 – Standardize Kernel Interfaces

## Status: Done

Every SPH operator (Density, Interpolate, Gradient, Divergence, Curl, Laplacian) now exposes exactly the kernel ABI described below — see "Working Prototype → Production" in the Repository Reality Check section for how this landed operator-by-operator.

## Goal

Ensure every SPH operator exposes a common kernel interface.

A typical kernel should follow a common structure similar to

```python
queryState
referenceState
domainState

useAdjacency
adjacencyState
gridState

correctionData

... operator parameters ...

output
```

rather than each operator defining its own unique collection of arrays.

## Tasks

* Audit all existing kernels.
* Identify kernels that still use flattened argument lists.
* Convert legacy kernels to the unified state interface.
* Keep argument ordering consistent across all operators.
* Document the standard kernel ABI.

## Notes

The objective is consistency rather than minimizing the number of arguments.

Different operators may ignore parts of the state, but they should still expose the same conceptual interface whenever practical.

This also makes generic dispatch, testing, profiling and AD wrappers substantially simpler.

---

# Phase 2 – Consolidate State Abstractions

## Goal

Make state objects the canonical representation of simulation data.

Instead of treating Torch tensors and Warp arrays as primary objects, the state should own both representations.

Conceptually,

```
SimulationState
    ParticleState
    BoundaryState
    GridState
    CorrectionState
```

Each state contains semantic fields rather than implementation-specific arrays.

## Tasks

* Review all existing state structures.
* Remove remaining duplicated representations.
* Define clear ownership of every field.
* Standardize naming across states.
* Minimize operator-specific state layouts.

## Notes

Operators should consume semantic information ("positions", "densities", "neighbor list") rather than implementation details.

---

# Phase 3 – Introduce a Field Abstraction

## Goal

Represent every simulation quantity through a common field abstraction.

Conceptually,

```
Field

    Torch tensor

    Warp array

    metadata

    synchronization state
```

instead of manually converting between Torch and Warp whenever a kernel is launched.

## Tasks

* Design a lightweight Field class.
* Store both Torch and Warp representations.
* Cache Warp views whenever possible.
* Introduce synchronization/dirty flags.
* Eliminate repeated wp.from_torch() calls.
* Eliminate repeated Python-side marshalling.

## Notes

The intention is to reuse existing memory rather than recreate Warp arrays repeatedly.

This should reduce launch overhead while simplifying wrapper code.

---

# Phase 4 – Improve State Construction

## Goal

Reduce handwritten boilerplate when constructing Warp state objects.

## Tasks

* Identify repetitive state construction.
* Introduce helper utilities or builders.
* Automatically populate state fields where possible.
* Centralize validation logic.

## Notes

Some device-side helper functions (e.g. neighborhood traversal or correction loading) are algorithmic abstractions and should remain explicit.

The focus is on removing mechanical Python-side boilerplate.

---

# Phase 5 – Consolidate Traversal Abstractions

## Status: Done

Every operator's `_Func_Adjacency` now branches on `useAdjacency` at runtime between the CSR neighbor-list case and the compact-hash grid-cell case (via `checkOffset`, moved to `radiusSearch/grid_util.py` for exactly this sharing); `operations_grid/` — the duplicated-per-operator grid implementation this phase was meant to eliminate — has been deleted entirely. See "Working Prototype → Production" in the Repository Reality Check section.

## Goal

Ensure every operator performs neighborhood traversal through the same abstraction.

Current traversal methods include

* neighbor lists
* hashed grids

Both should expose a common conceptual interface.

## Tasks

* Review duplicated traversal code.
* Factor repeated traversal setup into reusable Warp helper functions.
* Keep traversal-specific logic isolated.

## Notes

The runtime traversal decision is algorithmic and should remain explicit.

The objective is to avoid rewriting the same dispatch logic across many operators.

---

# Phase 6 – Extend States for Forward-Mode AD

## Goal

Introduce tangent information as part of the state representation instead of extending every kernel interface.

Conceptually,

```
Field

    primal

    tangent
```

or

```
Field

    Torch
    Warp

    Torch tangent
    Warp tangent
```

depending on implementation.

## Tasks

* Design tangent storage.
* Decide ownership and lifetime.
* Extend state builders.
* Extend helper functions (e.g. particle loading).
* Avoid modifying kernel interfaces where possible.

## Notes

Forward-mode should become a property of the execution context rather than individual kernels.

Most kernels should continue operating on the same semantic state objects.

---

# Phase 7 – Revisit AD Wrappers

## Goal

Simplify the Python AD wrappers once the new abstractions are available.

## Tasks

* Reduce argument bookkeeping.
* Remove duplicated state construction.
* Share infrastructure between reverse and forward mode.
* Centralize synchronization.

## Notes

Ideally, reverse mode and forward mode should differ primarily in how the execution context is constructed rather than how kernels are launched.

---

# Concrete Migration Plan (Execution Order)

## Step 0 - Establish Regression Baseline

Complete Phase 0 before interface-level refactoring. This is the acceptance gate for all later steps.

## Step 1 - Canonical Structured Kernel ABI

Adopt the covariance-style kernel ABI as canonical for all operations:

```python
queryState
referenceState
domainState

useAdjacency
adjacencyState
gridState

correctionData

... operation scalars ...

output
```

Enforce argument ordering and naming consistency across all operators.

## Step 2 - Introduce Minimal Field Type (No API Break)

Implement a lightweight `Field` object with:

* torch view
* cached warp view
* dtype/device/shape metadata
* synchronization ownership flags
* fallback conversion for legacy paths

Keep public APIs torch-compatible while wrapping lazily under the hood.

## Step 3 - Centralize State Normalization

Consolidate duplicated state extraction/default logic into a single authoritative path that builds all kernel structs and scalar config.

## Step 4/5 - Migrate All Six Operators — Done (actual order differed from plan)

Done, but Gradient was the first production migration template, not Density as originally planned — it has the most correction paths (CRK, grad-h, renormalization, volume) to prove out, so migrating it first validated the recipe against the hardest case rather than the easiest. Order actually used: Gradient, Interpolate, Density, Divergence, Curl, Laplacian. Each move went from a flat-wrapper adjacency-only kernel plus a separate flat-wrapper grid-only kernel to one structured-wrapper kernel handling both traversal modes, validated against the Phase 0 baselines (pytest + gradcheck + operation-matrix) at each step — see "Working Prototype → Production" in the Repository Reality Check section for the per-operator details and bugs found. "Maintain legacy shim compatibility" held throughout: each operator's flat-tensor `compute<Op>_warpBackend` function kept its exact pre-migration signature, so `sphOperation_warp`'s call sites needed no changes beyond the dispatch-branch condition.

## Step 6 - Consolidate Traversal and Close Capability Gaps — Done

Adjacency/grid duplication in orchestration is gone (`operations_grid/` deleted, `sphOperation_warp` is a single dispatcher with no adjacency-type branch). CRK/renormalization now run on grid-mode traversal for every operator that supports them, for free — the unified `correctionData` struct is threaded through both traversal branches identically, so there was no separate "grid CRK support" to add.

## Step 7 - Extend Fields for Forward-Mode AD

Add tangent storage to `Field` and extend state builders so forward mode becomes an execution-context property rather than a kernel-signature expansion.

## Step 8 - Retire Legacy Internals Behind Compatibility Shim

Keep legacy entrypoints callable, but route internals through the unified state+field path and remove duplicated flat internals after parity/performance targets are met.

## Cross-Cutting Gates (Every Step After Step 0)

* numerical parity against baseline outputs
* gradient parity where differentiable
* launch/conversion overhead checks
* documentation updates synchronized with behavior changes

---

# Design Considerations

## Semantic Interfaces

Kernel interfaces should describe *what* data is required rather than *how* that data is stored.

---

## Stable APIs

Adding a new particle attribute should require updating the state definition rather than every kernel.

---

## Separation of Concerns

Algorithms should not manage

* Torch conversion,
* Warp conversion,
* tangent allocation,
* synchronization,
* or ownership.

These belong to the state infrastructure.

---

## Minimize Duplication

Repeated state construction and repeated traversal setup should be centralized wherever possible.

Mechanical code should be generated or shared.

Algorithmic code should remain explicit.

---

## Backend Independence

The SPH implementation should gradually become independent of Warp-specific storage conventions.

A future backend should primarily require replacing the storage layer rather than rewriting SPH operators.

---

# Expected Benefits

* Consistent kernel interfaces.
* Simpler kernel dispatch.
* Reduced Python bookkeeping.
* Reduced tensor marshalling.
* Lower launch overhead.
* Cleaner separation between algorithms and storage.
* Easier implementation of forward-mode AD.
* Shared infrastructure between reverse and forward differentiation.
* Improved maintainability as additional SPH operators are added.

---

# Summary

Forward-mode AD should not be implemented as an additional layer of wrappers around the existing architecture.

Instead, the existing transition towards semantic state objects should be completed first. The state abstraction should become the single source of truth for simulation data, caching both Torch and Warp representations while managing synchronization internally.

Once this infrastructure exists, forward-mode differentiation naturally becomes an extension of the state representation (through tangent fields) rather than a change to every kernel or wrapper. This approach minimizes future maintenance, reduces Python-side overhead, and creates a cleaner separation between SPH algorithms, storage backends, and differentiation modes.
