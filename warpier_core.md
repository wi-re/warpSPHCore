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

This section captures current implementation status relative to the target architecture. Last verified against the repo on 2026-08-05 (commit `027c58f`).

## Notebook Corpus Status

* The root-level notebooks (`warp_density.ipynb`, `warp_divergence.ipynb`, `warp_gradient.ipynb`, `warp_interpolate.ipynb`, `warp_laplacian.ipynb`, `warp_custom.ipynb`) are mid-port to the current API. They previously used older calling conventions and compared against `diffSPH` as a reference backend; `diffSPH` no longer works in the current environment (see the commented-out imports in `demo_util.py`), so those comparisons have been dropped in favor of Warp-only runnable examples with analytic reference fields.
* ~~Curl has not been ported yet...~~ **Done.** `warp_curl.ipynb` exists in root (landed in commit `cc85bfb`), uses the current API (`ParticleState`/`warpOperation`/`OperationProperties(operation=WarpOperation.Curl, ...)`, no `diffSPH`), and was re-verified 2026-08-05 by executing every cell via `nbclient` end-to-end with no errors.
* ~~Grid, CRK, and renormalization notebooks... still live under `old/`~~ **Resolved — not a porting gap after all, per user clarification (2026-08-06).** `warp_renorm.ipynb` exists in root, uses the current API (`ParticleState`/`warpOperation`/`computeRenormalizationMatrices`), and was re-verified 2026-08-05 (all cells execute cleanly). `old/warp_grid.ipynb`, `old/warp_crk.ipynb`, and `old/warp_crk_test.ipynb` do **not** need root ports:
  * Grid dispatch is just `adjacency=None` on the same operator call every other root notebook already makes — there is no separate "grid" example to port, and direct pytest coverage already exists (`tests/operations/test_grid_modes.py`).
  * CRK checks were pulled directly into the relevant operator notebooks (interpolate, gradient, etc.) instead of being split into standalone `warp_crk`/`warp_crk_test` notebooks — so those two are redundant with what already exists in root, not unported.
  * `warp_profile.ipynb` (benchmarking) already exists in root (landed in commit `027c58f`, current API — `from sphWarpCore import *` / `generateNeighborTestData` / `profile_util.py`) — the `old/warp_profile.ipynb` copy is simply superseded, not a gap.
  `docs/regression/notebook_test_matrix.md`'s "not yet ported" notes for these three have been corrected accordingly — see that file.

## Already in Place

* A high-level semantic interface exists through `warpOperation` and state objects (`ParticleState`, `OperationProperties`).
* State-aware autograd infrastructure exists (`extractStateInfo`, `warpWrapper2`, `StateAwareWarpFunction`).
* Conversion hot paths already use caching (`getCachedWarpArray`, cached dummy tensors, dtype caches).
* A structured kernel ABI is demonstrated by renormalization covariance (`wp_covariance.py`) using state structs rather than fully flattened arguments.

## Gaps Against the Target

* `warpOperation` still routes through `sphOperation_warp`, so legacy internals remain dominant.
* Most operators still launch via flat tensor argument wrappers (`warpWrapper`) instead of structured wrappers.
* State objects remain torch-native and do not yet own synchronized torch+warp field representations.
* Adjacency and grid execution paths are still largely duplicated across operation families.
* Some modules (CRK / renormalization) still require adjacency-list mode and do not support compact-hash traversal paths — confirmed: `computeCRKFactors` and `computeRenormalizationMatrices` both raise `NotImplementedError` when `adjacency` is `None` or a raw `CompactHashMap`, even though every operator dispatch (`sphOperation_warp`) happily builds a grid datastructure on the fly for the operation itself. This means grid-mode traversal cannot currently be combined with CRK or renormalization corrections at all — not a performance gap, a hard capability gap.
* ~~The grid dispatch path... it is simply never tested. No code changes are required...~~ **Partially superseded — it *was* tested (indirectly, via Stage 2's Interpolate gradcheck) and needed real code changes.** `tests/operations/test_grid_modes.py`'s forward-only coverage still needed no code changes, but exercising the grid path under `SPHWARPCORE_PRECISION=float64` (needed for tight gradcheck tolerances) surfaced two real, previously-latent bugs, both now fixed:
  * `datastructure.hCell` (a `CompactHashMap` field, `hashMap_t.py`) is stored as a plain Python `float`. Every grid operator's kernel launch (`wp_density_grid.py`, `wp_interpolate_grid.py`, `wp_gradient_grid.py`, `wp_divergence_grid.py`, `wp_curl_grid.py`, `wp_laplacian_grid.py`) passed it straight into `wp.launch` unwrapped, where Warp's scalar-argument type inference defaults a bare Python `float` to `wp.float32` regardless of the kernel's declared `scalar_t` type — so *any* grid-path operation under `SPHWARPCORE_PRECISION=float64` failed outright with a kernel argument type-mismatch error (`expected float64, got float32`), even for plain forward Density. Not float64-specific in principle, just never exercised under float64 before now. Fixed by wrapping with `scalar_t(datastructure.hCell)` at each of the 6 call sites, matching the existing `wp.int32(...)`/`wp.uint32(...)`/`wp.bool(...)` cast convention already used right next to it in every one of those files.
  * `sphOperation_warp_grid` (`wp_operation_grid.py`) builds periodicity-wrapped query/reference positions via `x = torch.vstack([...]).mT` (and same for `y`) — but only `WarpOperation.Interpolate` actually consumes `x`/`y`; every other operator (Density, Gradient, Divergence, Curl, Laplacian) uses the raw, unwrapped positions and lets the kernel's own `domainMin`/`domainMax`/`periodicity` args handle it. For a 1D domain specifically, the `vstack(...).mT` construction produces a `(n,1)` tensor with a degenerate `(1,n)` stride; PyTorch's `is_contiguous()` reports `True` regardless (a size-1 axis can't violate contiguity by its definition), so the already-present defensive `.contiguous()` calls in `castTorchToWarpAsBuiltins`/`getCachedWarpArray` are no-ops, and Warp's stricter `wp.from_torch` stride check then rejects it — breaking grid-path Interpolate specifically (and only) on 1D domains. Fixed by building `x`/`y` via `torch.stack(..., dim=1)` instead of `vstack(...).mT` — equivalent values, canonical strides for any dimensionality, no transpose. Verified on a realistic 2D non-periodic domain (worked even before this fix — the stride issue is 1D-specific) and on the 1D gradcheck domain (failed before, passes after).
  * Both confirmed via a direct 1D and 2D reproduction script (ad hoc, not committed) and the full regression sweep below; not yet added as their own pytest cases (grid-path + float64 has no CI coverage at all currently — see the CUDA/nightly-CI gaps above).
* ~~The top-level `OperationProperties`/`warpOperation` API does not expose `dotMode` for divergence...~~ **Fixed.** `OperationProperties` now has a `divergenceDotMode` field (default `False`, preserving old behavior), threaded through both `sphOperation_warp` and `sphOperation_warp_grid` down to `computeSPHDivergence_warpBackend`'s `dotMode` argument. `scripts/operation_matrix.py`'s `Divergence-Matrix[...]` rows now verify both conventions directly: the matrix field as-is with `divergenceDotMode=True`, and the pre-transposed (`.mT`) field with `divergenceDotMode=False` (the old default/workaround) — both now match the analytic tensor divergence. The `warp_divergence.ipynb` `.mT` workaround is no longer required going forward, though the notebook itself hasn't been updated to use the new flag yet.
* **Fixed alongside the above:** `GradientScheme.Symmetric` had a duplicated `* apparentVolume` factor in the Symmetric branch of `wp_divergence.py`, `wp_divergence_grid.py`, `wp_gradient_grid.py`, `wp_curl.py`, and `wp_curl_grid.py`, which was corrupting output for Divergence/Curl (both traversal modes) and for grid-dispatched Gradient specifically. This was caught by the new operation matrix (see below) and has been fixed in all five files; confirmed via a fresh matrix run on both `cpu` and `cuda`.
* ~~The curl operator has a known kernel compile/launch failure...~~ **Fixed.** Root cause: `computeSPHCurlTensor_Func`'s directionality-mask early-out used a bare `return` inside a Warp `@wp.func` declared with a non-`void` return type. `nvcc` tolerates this (undefined behavior, but happened to compile and run correctly on `cuda`); the CPU/LLVM backend correctly rejects it as invalid IR (`"non-void function ... should return a value"`), which is what the `xfail` was catching. Fixed in both `wp_curl.py` and `wp_curl_grid.py` by returning `outputValue * scalar_t(0.0)` instead. Also carried the `GradientScheme.Symmetric` double-`apparentVolume` fix (see above) into `wp_curl.py`/`wp_curl_grid.py`, which weren't part of the original batch. Confirmed via `scripts/operation_matrix.py`: all 4 `GradientScheme` variants now pass on both `cpu`/`cuda`, both traversal modes.
* ~~Renormalization has two known instabilities...~~ **Both fixed.**
  * The "mixed backend launch" CPU instability was a missing `device=` argument on `wp.launch(kernel=pinv2x2_warp, ...)` in `wp_covariance.py` — without it, Warp launches on its default device context regardless of which device the input tensors actually live on. One-line fix: pass `device=inv_warp.device` explicitly.
  * A separate, more serious correctness bug (not device-specific) was found in `pinv2x2_warp`'s pseudo-inverse math itself: it used a general 2x2-SVD closed form that computes two *independent* rotation angles (`theta` for `U`, `phi` for `V`) via two separate `atan2` calls. SPH renormalization/covariance matrices are symmetric by construction (a sum of `x_ij ⊗ x_ij`-type terms for any isotropic kernel), so `theta` and `phi` are mathematically required to coincide — but for a near-isotropic matrix (the common case: a locally regular, well-resolved particle neighborhood), both `atan2` calls' arguments round to ~0 independently, and the two angles can land on unrelated values instead of staying synchronized. Reproduced directly against production covariance matrices: a matrix that should invert to `~diag(1.0016, 1.0016)` instead produced a matrix rotated by ~43°, with the correct singular values but garbage off-diagonal structure. Root-caused and fixed by replacing the general asymmetric-SVD formula with the correct, numerically robust closed-form symmetric-matrix eigendecomposition (a single `atan2` call, so there is nothing left to desync) in `wp_covariance.py`'s `pinv2x2_warp` kernel and its dead-code Python twin. `pinv/twod.py`'s `pseudoInverse2x2` has the identical latent bug but is a generic (currently unused, not necessarily-symmetric-input) utility, so it was intentionally left as-is rather than force-symmetrized. Verified via `scripts/operation_matrix.py`: renormalization matches CRK-level accuracy across every operator, both devices, both traversal modes.

## Additional Findings from `scripts/operation_matrix.py` (2026-08-05)

The diagnostic matrix (see Phase 0 follow-up tasks below), run repeatedly across both `cpu`/`cuda`, both traversal modes, and all three correction paths as fixes landed same-day, surfaced every issue recorded above. **As of the latest run, the full matrix is clean: 186/186 applicable cells `OK`, 0 `HIGH`, 0 `ERR`, 0 `NAN`, on both `cpu` and `cuda`.** The pytest suite reflects the same: 34 passed, 0 xfailed (previously 30 passed, 4 xfailed with two of those `xfail`s masking forced pre-emptive skips, not "ran and tolerated failure"). The stale `xfail`/`try-except-xfail` scaffolding in `test_operations_core.py` and `test_operations_crk_analytic.py` has been removed now that the underlying bugs are fixed, so these paths get real regression coverage instead of a permanent skip.

## Backward-Mode (Reverse AD) Findings (2026-08-05)

Started building `torch.autograd.gradcheck`-style coverage for the reverse-mode AD path (Phase 0's "gradient/finite checks" task, previously entirely unstarted) and immediately found two real bugs in the AD bridge, plus one apparent upstream limitation:

* **Fixed.** The `getCachedWarpArray` identity cache (`wp_util.py`, keyed on tensor `data_ptr`/shape/stride/dtype) and its mirrors in `wp_autograd.py` (`_WRAPPER_ARGS_CACHE`, `_KERNEL_ARGS_CACHE`) reused the *same* `wp.array` object — including its persistent `.grad` buffer — whenever a tensor's underlying storage recurred. For the common self-interaction case (`referenceParticles=None`, so `referencePositions is queryPositions`), this meant the "query" and "reference" roles of the same tensor shared one Warp array, and Warp's tape correctly summed both roles' contributions into that one buffer — but PyTorch *also* sums gradients across an argument used in two slots of the same `autograd.Function` call, so the already-summed total was added to itself. Every position/support/mass gradient for self-interaction operations was silently 2x too large. Root-caused via a direct reproduction (density on 3 particles: caching gave `12.6`, removing it gave the correct `6.3`, confirmed against both a central-difference check and an independent pure-PyTorch reference implementation). All three caches ripped out (not just worked around) per explicit steer — this caching layer has a history of exactly this kind of subtle correctness bug. `getCachedWarpArray`/`clearWarpArrayCache`/`clearKernelArgsCache` kept as no-ops for API compatibility. No regression: pytest suite and `operation_matrix.py --ci` both still fully green after removal.
* ~~Open, and not fixable from this repo alone...~~ **Fixed and confirmed upstream — turned out to be TWO independent bugs in `wp_autograd.py`, not `warp-lang` itself, and not one bug.** Reported to the `warp-lang` team via `scripts/repro_warp_grad_reentrancy.py`; they reproduced both on current Warp (CPU and CUDA) and confirmed the root cause and intended fix pattern. Both `WarpFunctionWrapper.backward` and `StateAwareWarpFunction.backward` needed two separate, non-overlapping fixes; applying only one does not fix the other's failure mode (confirmed via a 2×2 isolation matrix in the repro script):
  * **Bug 1 — reading the gradient back off Warp.** `wp.to_torch(wa.grad)` returned a view into Warp-owned memory, unreset between calls. This is specifically what breaks `torch.autograd.gradcheck`'s own internal reentrancy self-check (it calls `backward()` twice against one retained graph to verify determinism). Fix: `.clone()` the gradient when reading it out of Warp, and call `ctx.tape.zero()` after collecting all gradients so the buffer starts clean for whatever reuses that memory next.
  * **Bug 2 — seeding the output gradient before `tape.backward()`.** The original code did `out_warp.grad = castTorchToWarpAsBuiltins(grad_out.contiguous())`. Root cause per the `warp-lang` team: `wp.from_torch()` is zero-copy, so this direct assignment makes the incoming torch tensor itself the *live* output-adjoint buffer. Warp's backward pass consumes output adjoints by reading **and zeroing** them — mutating that torch tensor in place. Any caller reusing the same `grad_outputs` tensor object across separate `torch.autograd.grad(...)` calls (fresh forward pass and fresh `wp.Tape()` each time — a realistic pattern, e.g. a preallocated gradient-seed buffer reused every training step) sees a correct gradient on the first call and silently `0` after, because its own tensor was zeroed out from under it. The Bug-1 fix alone does **not** fix this (confirmed both by us and by the `warp-lang` team's own repro). **Confirmed fix (the intended Warp usage pattern, not just a workaround):** seed gradients via `Tape.backward(grads={array: seed})` instead of assigning `array.grad` directly. When the array's `.grad` buffer already exists (true here — `wp.zeros(..., requires_grad=True)` allocates it up front), this internally does `array.grad.assign(seed)` — a value copy into the array's own persistent buffer — rather than aliasing the caller's tensor as the live adjoint buffer. This fixes Bug 2 outright; no `.clone()` of the incoming gradient is needed at all with this pattern (a plain `.clone()` on the incoming tensor was floated as a smaller, alternative workaround, but `Tape.backward(grads=...)` is what upstream recommends).
  Both fixes now applied to both `backward()` methods. Verified: `scripts/gradcheck_density_native.py` (calls `gradcheck` directly, no workarounds) passes; a reused-`grad_outputs` stress test against the real Density op (fresh-forward-per-call loop, and repeated backward on one retained graph) passes, and the caller's `grad_outputs` tensor is confirmed to no longer be mutated in place; no regression in the forward-mode pytest suite or `operation_matrix.py --ci`. No `warp-lang` upgrade needed — the fix lives entirely in `sphWarpCore`'s own AD bridge. The `warp-lang` team noted this is a documentation/example gap on their side (the zero-copy interop + output-adjoint-consumption interaction isn't currently spelled out clearly in their Torch interop docs) and said they'll improve it.
* `scripts/gradcheck_density.py` — first gradcheck script, covering `WarpOperation.Density`. Validates `d(density)/d(position|support|mass)` via hand-rolled analytical vs. central-difference Jacobians on two deliberately small, controllable cases: a single particle (`h=1`, checked against an exact closed form — `d(rho)/d(position)` must be exactly `0` since the kernel gradient vanishes at `r=0`) and a line of particles from `-1` to `1` (checks self- vs. non-self-particle gradient terms separately, and cross-validates against an independent pure-PyTorch re-implementation of the same kernel formula, with an optional `--plot` comparison). Currently passing. Other operators (interpolate, gradient, divergence, curl, laplacian) still need their own scripts following the same pattern — not yet started.
* `scripts/gradcheck_density_native.py` — a companion canary script: the same two cases (single particle, line of 7), but calling `torch.autograd.gradcheck` directly against `warpOperation` with no workarounds at all (no manual Jacobian, no per-call cloning). Originally written expecting it to fail (and it did, with `GradcheckError: Backward is not reentrant`) until the AD-bridge fix above landed — **it now passes**, confirming the reentrancy issue is fixed at the source. `scripts/gradcheck_density.py`'s manual-Jacobian workaround is no longer strictly necessary for Density (this native script supersedes it) but has been left in place since it's still-passing, still-useful coverage (closed-form single-particle check, self/non-self breakdown, plot) — not a redundant risk.
* `scripts/_gradcheck_common.py` — shared, non-entrypoint helper module extracted from the two Density scripts once the pattern proved out: `DEVICE`/`DTYPE`/`KERNEL` constants, `make_domain`, `single_particle_case`, `line_case`, `build_adjacency` (frozen/non-differentiable, built once from detached positions — see its docstring for why), and the pure-PyTorch `wendland2_kernel_1d` reference kernel. Both Density scripts now import from it instead of duplicating the setup; new gradcheck scripts should too. Stage 4 added `grid_case_2d` (a small `n_per_side × n_per_side` regular 2D grid) alongside the original 1D `line_case`, for operators whose behavior only diverges in ≥2D (Divergence's `divergenceDotMode`; Curl will need it too).
* `scripts/gradcheck_interpolate_native.py` — Stage 2 of the rollout plan below. Found and confirmed a real bug: `computeSPHInterpolation_Func`'s `fv = referenceValues[jj] if preScatteredQuantities else referenceValues[j]` ternary silently zeroed the adjoint for `referenceValues` regardless of dtype or branch taken; fixed by converting to an explicit if/else block (the form every other operator already uses for this branch). See the Gradcheck Script Rollout Plan entry below for the full root-cause writeup.
* `scripts/gradcheck_gradient_native.py` — Stage 3 of the rollout plan below. Clean: all 20 cases (4 `GradientScheme` variants × 2 field ranks × 2 particle setups, plus shared-tensor regression guard) pass, no bug found. See the Gradcheck Script Rollout Plan entry below for why a structurally-similar ternary in this operator did *not* reproduce the Stage 2 bug.
* `scripts/gradcheck_divergence_native.py` — Stage 4 of the rollout plan below. Clean: all 28 cases (vector- and matrix-rank fields × both `divergenceDotMode` conventions × 4 `GradientScheme` variants, on a new small-2D-grid case, plus shared-tensor regression guard) pass, no bug found.
* `scripts/gradcheck_curl_native.py` — Stage 5 of the rollout plan below. Clean: all 10 cases (4 `GradientScheme` variants on the 2D grid case, plus shared-tensor regression guard) pass, no bug found.
* `scripts/gradcheck_laplacian_native.py` — Stage 6 of the rollout plan below. Found and fixed two real bugs: a raw-float-literal type-promotion compile failure in `computeSPHLaplacianTensor_Func`'s `eps = 1e-8` (`wp_laplacian.py`), and a genuine out-of-bounds read in `LaplacianScheme.Dot`'s `computeLaplacianDot2` for scalar fields in ≥2D domains (now guarded off with a clear `ValueError` rather than silently returning wrong gradients — the correct scalar-field formula, if one exists, is left open). See the Gradcheck Script Rollout Plan entry below for the full writeup.
* `scripts/repro_warp_grad_reentrancy.py` — the minimal, sphWarpCore-independent repro underlying the two-bugs finding above. Was reported upstream and confirmed by the `warp-lang` team (status noted in the script's own docstring); now kept as a local regression guard — re-run it if `wp_autograd.py`'s `backward()` methods change, it should always print `PASS`/`PASS` on the last row. Builds a toy `y = x**2` `torch.autograd.Function` with the two candidate fixes (`use_tape_grads` — the confirmed `Tape.backward(grads={...})` pattern — and `read_clone`+`tape_zero`) independently toggleable, and prints a 4-row × 2-column PASS/FAIL matrix (all 4 fix combinations × both bug checks) — only "both fixes" clears both columns; each fix alone clears exactly one. Confirmed on both `cpu` and `cuda`.

### Gradcheck Script Rollout Plan (Stages 1-6 all done)

All six stages are now complete (`scripts/gradcheck_{density,interpolate,gradient,divergence,curl,laplacian}_native.py`), following `gradcheck_density_native.py`'s pattern throughout — call `torch.autograd.gradcheck` directly against `warpOperation`, no manual Jacobian, no per-call cloning (`gradcheck_density.py`'s manual-Jacobian workaround is legacy, kept only as extra Density coverage). Stages 2 and 6 found and fixed real bugs that the forward-only test suite (`operation_matrix.py`, pytest) had never caught, because gradients were entirely untested territory before this rollout; Stages 3, 4, 5 gradchecked clean. The *forward*-value check the original Gradient stage note called for (per-`GradientScheme` plot vs. a hand-coded reference) was never implemented — still open if wanted.

* **Stage 2 — Interpolate. Done, and found a real bug (fixed).** `scripts/gradcheck_interpolate_native.py` checks `positions`, `supports`, `masses`, `densities`, `queryValues`, `referenceValues`, both scalar- and vector-rank fields, and both "distinct query/reference tensors" and "shared (same object) query/reference tensor" variants (the latter as a standing regression guard for the AD-bridge shared-tensor class of bug found in Stage 1 — confirmed it is *not* recurring here). `densities` are computed once via the (already-verified) Density op for realistic magnitudes, then detached and re-leafed as an independent gradcheck input rather than chained through Density's own backward (that's Stage 1's job; chaining would make this a second-derivative check instead).
  * **Bug found and fixed:** `d(interpolate_output)/d(referenceValues)` was silently always zero — a real, load-bearing bug (the operator is unusable for any loss that backprops through the interpolated *values*, e.g. training a field via interpolation). Root-caused with a minimal, sphWarpCore-independent Warp repro (bisected concrete-vs-generic `Any` dtype × ternary-vs-if/else array read, 4 combinations): a Python ternary expression assigned to a local var (`fv = referenceValues[jj] if preScatteredQuantities else referenceValues[j]` in `computeSPHInterpolation_Func`, `wp_interpolate.py`) compiles fine but silently produces a **zero adjoint** for the array read, regardless of dtype (concrete `float64` and generic `Any` both fail identically) and regardless of which branch is actually taken at runtime. The equivalent explicit `if: ... else: ...` block form does not have this problem. `wp_interpolate.py` was the *only* operator file using the ternary form for this `preScatteredQuantities` branch — every other operator (`wp_gradient.py`, `wp_divergence.py`, `wp_curl.py`, `wp_laplacian.py`, and their `_grid` counterparts) already uses the if/else block, so this was isolated to Interpolate, not a systemic pattern. Fixed by rewriting to the if/else block form. Verified: all 8 gradcheck cases now pass, full pytest suite (56 passed) and `scripts/operation_matrix.py --ci` (186/186 OK) both still green — no regression. Not yet reported upstream to `warp-lang` (unlike the reentrancy bug, this one has a trivial local fix, so low urgency, but the underlying "ternary array-read adjoint" gap is worth flagging to them at some point since it fails silently rather than erroring).
  * **Follow-up check on the grid counterpart, two more real bugs found and fixed.** `wp_interpolate_grid.py`'s `computeSPHInterpolation_grid_Func` was checked directly and does *not* have the ternary (it reads `referenceValues[j]` unconditionally, `preScatteredQuantities` isn't even wired through on the grid path) — but actually exercising it end-to-end (grid traversal, `adjacency=None`, under `SPHWARPCORE_PRECISION=float64`) hit two unrelated, pre-existing bugs blocking the grid path generally, not just Interpolate: the `hCell`-dtype bug and the 1D-domain stride bug, both described in the Reality Check section above under "The grid dispatch path... it is simply never tested." Once both were fixed, grid-path Interpolate gradients matched the non-grid path on both 1D and a realistic 2D domain.
* **Stage 3 — Gradient. Backward-mode gradcheck done, clean — no bug found.** `scripts/gradcheck_gradient_native.py` checks `positions`, `supports`, `masses`, `densities`, `queryValues`, `referenceValues` across all four `GradientScheme` variants (Naive/Symmetric/Difference/Summation) and both scalar- and vector-rank input fields (output rank is always input rank + 1, for the spatial dimension), plus the shared query==reference-tensor regression guard (once per field rank, Naive only — Stage 2 already established this class of bug isn't scheme/operator-specific). All 20 cases pass; full pytest suite (56 passed) unaffected since no production code changed this stage. Notably, `computeSPHGradientTensor_Func`'s `apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]` *is* a ternary reading an array in one branch (`referenceVolumes[j]`) — structurally similar to the Stage 2 bug — but it gradchecked clean; the failure mode there was specifically two branches both indexing the *same* array (`referenceValues[jj]`/`referenceValues[j]`), not a ternary in general, so this one one wasn't expected to fail and didn't. The *forward*-value check the rollout plan also called for (does `WarpOperation.Gradient`'s output match a hand-coded reference SPH gradient sum, per `GradientScheme`, via a per-particle plot) was **not done** — out of scope for this pass, which focused on the backward-mode gradcheck family; still open if wanted later.
* **Stage 4 — Divergence. Backward-mode gradcheck done, clean — no bug found.** `scripts/gradcheck_divergence_native.py` checks `positions`, `supports`, `masses`, `densities`, `queryValues`, `referenceValues` on a small 2D grid case (new `grid_case_2d` helper in `_gradcheck_common.py` — 1D domains can't meaningfully exercise `divergenceDotMode`, see below). Two field ranks: a plain vector field (shape `(n,D)` → scalar output, where `divergenceDotMode`'s two index formulas are algebraically identical since `outputElements==1` — this only sanity-checks the flag is threaded through, all 4 `GradientScheme` variants with `dotMode=False` plus one with `dotMode=True`) and a matrix field (shape `(n,D,D)` → vector output, where `dotMode` genuinely changes which axis is contracted — both conventions checked across all 4 schemes), plus the shared-tensor regression guard. 28 cases total, all pass; full pytest suite (56 passed) unaffected, no production code changed this stage.
* **Stage 5 — Curl. Backward-mode gradcheck done, clean — no bug found.** `scripts/gradcheck_curl_native.py` checks `positions`, `supports`, `masses`, `densities`, `queryValues`, `referenceValues` on the 2D grid case, all 4 `GradientScheme` variants, plus the shared-tensor regression guard. 10 cases, all pass. Same `apparentVolume` ternary pattern as Gradient/Divergence, already established safe. No compile-history repeat of the bare-`return`-in-`@wp.func` bug either — this stage exercises the backward path specifically, which the original bug (a *forward* compile/launch failure) never got a regression guard for until now.
* **Stage 6 — Laplacian. Done — found and fixed two real bugs, one is a guarded-off structural limitation, not a fix.**
  * **Bug 1, fixed: raw Python `float` literal type-promotion.** `computeSPHLaplacianTensor_Func`'s `eps = 1e-8` (used as `r_ij + eps * h_ij` in two places) assigns an unwrapped Python float inside a `@wp.func`; Warp doesn't auto-promote it to match `h_ij`'s `float64`, causing `computeSPHLaplacianTensor_Kernel` to fail compilation outright under `SPHWARPCORE_PRECISION=float64` (`RuntimeError: Input types must be the same, got ['float32', 'float64']`, plus a confusing cascade failure on the *next* compile attempt reporting an unrelated-looking `get_dim_4` undeclared-identifier error — both vanished together once this was fixed, so the second was fallout from the first, not a separate bug). Fixed by wrapping: `eps = scalar_t(1e-8)`. `wp_laplacian_grid.py` had the identical bug at the identical line and was already fixed (by the user) before this stage started; this fix brings the non-grid file in line. Root-caused via 3x isolated, deterministic repro (same failure every time in isolation, ruling out compiler-cache flakiness) after the user pointed out the general anti-pattern: any bare numeric literal multiplied against a `scalar_t` value inside a `@wp.func`/`@wp.kernel` needs `scalar_t(...)` around it, since Warp does not auto-promote.
  * **Bug 2, fixed via a guard (not a formula fix — see below): `LaplacianScheme.Dot` silently reads out of bounds on scalar fields in ≥2D domains.** `computeLaplacianDot2` (the `Dot` scheme's implementation) indexes `q_ij[block*dim + k]` for `k in range(dim)`, which assumes the field's flattened size is a multiple of the spatial dimension `dim` -- true for a vector field (`flatInputShape == dim`) but false for a genuine scalar field (`flatInputShape == 1`): with `dim=2`, `q_ij[1]` reads past the end of a length-1 Warp vector. A `leading_dim = inputLength // dim` variable is computed but never used anywhere in the function -- looks like an unfinished substitution, not intentional. This was invisible to every check that existed before this stage: the 1D domains earlier gradcheck stages used have `dim=1`, where the indexing is always in-bounds (never exercised); `operation_matrix.py`'s existing forward-only Laplacian check uses a *linear* scalar field, whose true Laplacian is exactly zero everywhere, so a garbage read that happens to be small doesn't necessarily fail that check's tolerance. `scripts/gradcheck_laplacian_native.py`, deliberately built on the 2D `grid_case_2d` case specifically to exercise this, caught it immediately: real, large Jacobian mismatches (e.g. numerical `24.2` vs. analytical `0.0`) for every `GradientScheme` × `Dot` combination once compilation was unblocked by Bug 1's fix.
    * Asked the user how to handle it (physics-correctness call, not an AD-bridge fix): fix the formula, guard it, or just document it. **Chose to guard.** Rather than guess at the correct scalar-field generalization of a formula whose only reference (a commented-out `torch.einsum` block citing DJ Price's SPH/MHD review, eq. 96) is for a vector/tensor Laplacian, `computeSPHLaplacian_warpBackend` (and its grid counterpart `computeSPHLaplacian_grid_warpBackend`) now raise `ValueError` up front when `laplacianMode == LaplacianScheme.Dot` and `flatInputShape % spatialDim != 0` -- converting a silent wrong-gradient bug into a loud, immediate failure with a message naming the working alternatives (`Naive`/`Brookshaw`/`Default`). The actual formula fix (if `Dot` should support scalar fields at all) is left for someone with the SPH domain expertise to redo properly.
    * `operation_matrix.py`'s Laplacian matrix (which only ever tests a scalar field) previously called `evaluate()` for every `GradientScheme` × `LaplacianScheme` cell unconditionally; that now hit the new guard and turned 24 cells red (`ERR`) under `--ci`, which would have broken the CI gate. Fixed by having the matrix generator return `Cell("NA", None, "Dot scheme doesn't support scalar fields")` directly for `Dot` cells instead of routing them through `evaluate()` -- consistent with how the existing 18 structurally-inapplicable NA cells (CRK/renorm on grid traversal) are already handled. Matrix is back to clean: `Summary: OK=162, HIGH=0, ERR=0, NAN=0, NA=42` (up from `NA=18` before this stage), `--ci` exits 0.
  * All non-`Dot` combinations (`Naive`/`Brookshaw`/`Default` × all 4 `GradientScheme` variants, both particle setups, plus the shared-tensor regression guard) gradcheck clean -- 26 real passes + 8 confirmed-guard-fires = 34 total in `scripts/gradcheck_laplacian_native.py`. Full pytest suite (56 passed) and `operation_matrix.py --ci` (162 OK / 0 ERR / 42 NA) both green.

## CI Wiring for the 64-bit/1D/3D Operation Matrix and Gradcheck Scripts (2026-08-06)

`scripts/operation_matrix.py` had already grown `--precision`/`--dim` support and the six `gradcheck_*_native.py` scripts existed, but neither was reachable from `pytest tests/` or `.github/workflows/tests.yml` — both were still purely manual. Wired both in:

* `tests/operations/test_gradcheck_scripts.py` — new file, parametrized over all seven `scripts/gradcheck_*.py` scripts (six `_native` ones + the legacy `gradcheck_density.py`), each run via `subprocess.run([sys.executable, script_path], ...)` and asserted to exit `0`. Subprocess isolation is required, not a style choice: `SPHWARPCORE_PRECISION` is baked into every compiled kernel at first `sphWarpCore` import per-process (same constraint `operation_matrix.py`'s `_configure()` documents), so importing these scripts' modules directly into one pytest process would corrupt each other's precision. All 7 now run as part of `pytest tests/ -v` (63 passed total, ~40s added). `repro_warp_grad_reentrancy.py` was deliberately left out — it always exits 0 and prints an illustrative PASS/FAIL matrix (three of its four rows are *expected* to fail), so it isn't a pass/fail gate; stays a manual regression check.
* `.github/workflows/tests.yml` — the single `operation_matrix.py --ci` step became five: the original 2D/float32 gate, plus `--precision float64 --nx 24`, `--dim 1 --nx 64`, `--jitter 0.01` (confirmed clean at this jitter level — all cells `OK`, matching the notebooks' jittered examples), and a `--dim 3` step gated behind a runtime `torch.cuda.is_available()` check that no-ops with a log message on the CPU-only hosted runner (Warp's CPU backend is single-core/unoptimized, so 3D is only run on CUDA). Each added step runs in well under 20s locally, so the total push/PR cost stays small. This is a deliberately curated subset chosen to hit the axes that have each hidden a real bug before (float64: raw-literal type promotion; 1D: the grid-path stride bug; jitter: CRK/renorm's actual code path) — not the full precision×dim×jitter×device product, which is left for a still-deferred nightly sweep (see the CI task note below).
* `scripts/run_operation_matrix_sweep.sh` — new wrapper script with `--quick` (the same single CI-gate invocation above, safe to run every iteration) and `--full` (loops `--precision × --dim{1,2} × --jitter{0.0,0.01} × --device{cpu, cuda if available}`, all `--ci`-gated since those are the confirmed-clean axes, plus the `--dim 3` CUDA-only cases at the same precision/jitter values, plus one final `--jitter 0.3` pass **without** `--ci` for human inspection only). The `--full` mode was run once end-to-end to validate it: every `--ci`-gated combination came back clean (`HIGH=0, ERR=0, NAN=0` on every invocation, both `cpu` and `cuda`). Confirmed the hard way that jitter above ~0.01 is *not* safe to gate on yet — an earlier version of this script swept `--jitter {0.15, 0.3}` under `--ci` and immediately hit 184 real `HIGH` cells (mostly Gradient/Divergence/Laplacian at `--dim 1`), which is expected per the still-open "sound jittered thresholds need investigation" item above, not a regression — the script was corrected to only gate on `{0.0, 0.01}` and treat heavier jitter as diagnostic-only.
* `.claude/skills/gradcheck/SKILL.md` and `.claude/skills/operation-matrix/SKILL.md` — new project skills documenting both of the above (all-ops vs. single-op gradcheck; quick vs. full operation-matrix sweep) so they're quick to reuse during the upcoming interface migration (Phases 1+) instead of being re-derived from scratch each time a kernel gets rewritten.

## Largest Remaining Change

The key migration step is introducing a `Field` abstraction that carries both torch and warp representations with synchronization ownership, while preserving legacy fallback conversion paths for compatibility.

---

# Phase 0 - Build Regression Ground Truth From Notebooks (First Step)

## Status: In Progress (Partial)

`tests/operations/{conftest,test_operations_core,test_operations_consistency,test_operations_crk_analytic}.py` and `docs/regression/notebook_test_matrix.md` landed in commit `027c58f`. The suite initially ran at 30 passed / 4 xfailed (curl compile failure on both devices, CPU renormalization instability on 2 tests); all four underlying bugs have since been found and fixed (see Reality Check), the `xfail` scaffolding removed, and the suite now runs clean at **34 passed, 0 xfailed**. This is real, currently-green coverage and should not be treated as the unstarted first step anymore — but several parts of the original scope are still open, listed below.

## Goal

Create a reproducible regression suite and documentation baseline before refactoring execution interfaces.

## Why This Comes First

The repository already has a broad notebook corpus in the root directory (density, interpolate, gradient, divergence, curl, laplacian, grid, CRK, renormalization, profiling utilities). These notebooks encode expected behaviors and should be converted into stable regression tests before architectural changes.

## Tasks

* ~~Inventory all operation-relevant notebooks in the repository root.~~ Done. All operation-relevant notebooks are now accounted for in root — grid/CRK don't need their own notebooks (see Notebook Corpus Status above), and curl/renorm/profile are all ported.
* ~~Extract deterministic scenarios from each notebook (fixed seeds, fixed particle counts, fixed modes).~~ Done for density/interpolate/gradient/divergence/laplacian; curl scenario exists but is unverified (compile failure).
* ~~Convert scenarios into `pytest` parameterized tests for all operation families.~~ Done, consolidated into 3 files rather than one-per-operation (see Deliverables note).
* Capture baseline outputs (golden data or compact summaries) with documented tolerances. **Deliberately deferred, not scheduled.** Current tests check analytic properties (finite, positive, MAE vs. closed-form linear-field derivatives) rather than stored golden snapshots; no `tests/data/` exists. Deferred because picking sound tolerances/thresholds for randomized (jittered) data needs real investigation and isn't a quick follow-up — revisit only after the core migration (Steps 1-6) has settled, not before.
* Add gradient/finite checks where applicable. **Done, and wired into the test suite.** `scripts/gradcheck_{density,interpolate,gradient,divergence,curl,laplacian}_native.py` (plus the legacy `gradcheck_density.py`) cover all six operators — see the Gradcheck Script Rollout Plan below. `tests/operations/test_gradcheck_scripts.py` now runs all seven as parametrized pytest cases, each shelled out via `subprocess` (required, not a style choice: `SPHWARPCORE_PRECISION` is baked into compiled kernels at first `sphWarpCore` import per-process and can't change mid-process, so importing several of these scripts' modules into one pytest process would have the later ones silently reuse the first script's precision — see `operation_matrix.py`'s `_configure()` for the identical constraint). This makes them part of `pytest tests/ -v`, which CI already runs, so no separate CI step was needed. `repro_warp_grad_reentrancy.py` is intentionally excluded — it always exits 0 and prints an illustrative PASS/FAIL matrix rather than gating on a single result, so it stays a manual regression check (re-run by hand after `wp_autograd.py` changes).
* Generate matching markdown scenario docs so each test is also an executable behavior spec. **Partially done** — one summary mapping doc exists (`docs/regression/notebook_test_matrix.md`); no per-test behavior-spec docs.
* Add CI entrypoints for a deterministic subset and an extended nightly matrix. **Expanded.** `.github/workflows/tests.yml` runs on push/PR (Python 3.12, `ubuntu-latest`, CPU-only torch — GitHub-hosted runners have no CUDA so `cuda`-parametrized pytest cases self-skip): the full pytest suite (now including the gradcheck scripts above), then `scripts/operation_matrix.py --ci` as a hard gate across several targeted configurations rather than just the single default cell, chosen to cover the axes that have each hidden a real bug (see the Reality Check / Gradcheck Rollout sections) while staying fast enough to run on every push:
  * `--device cpu --ci` — 2D, float32, non-jittered (the original gate).
  * `--device cpu --precision float64 --nx 24 --ci` — float64 catches the raw-float-literal type-promotion class of bug (Stage 6) that float32 silently tolerates.
  * `--device cpu --dim 1 --nx 64 --ci` — 1D catches bugs the 2D-only case can't reach (the grid-path stride bug in the Reality Check section was 1D-specific).
  * `--device cpu --jitter 0.01 --ci` — a small jitter actually exercises CRK/renorm instead of the near-no-op perfect lattice, while staying within threshold (confirmed: all cells `OK` at `--jitter 0.01`, matching the notebooks' jittered examples).
  * A conditional 3D step (`--device cuda --dim 3 --nx 8 --ci`) that only runs if `torch.cuda.is_available()` at runtime — `dim=3` is prohibitively slow on Warp's CPU backend (single-core, unoptimized), so it's guarded off entirely on the CPU-only hosted runner rather than run small-and-slow; it'll start running automatically the day a GPU runner is added, no workflow change needed.
  Running the *full* precision×dim×jitter×device product on every push would take too long to be worth it as a per-push gate, so this is a deliberately curated subset, not full coverage — an extended nightly run sweeping more of that space (larger `--nx`, more `--jitter` values, `--precision float64 --dim 3` together, etc.) on a schedule trigger is still **deliberately deferred, not scheduled**, for the same "sound thresholds need real investigation" reason as the golden-data deferral above. Full CUDA CI coverage (2D/1D on GPU, not just the 3D step) is separately still open — would need a self-hosted/GPU runner.

### Newly identified follow-up tasks

* ~~Add direct grid-path coverage...~~ Done. `tests/operations/test_grid_modes.py` mirrors the base-path cases from `test_operations_core.py` (density, interpolate, gradient, divergence, curl, laplacian) with `traversal="grid"` (`adjacency=None`), plus one direct grid-vs-adjacency agreement check on the gradient operator. Required a small, backward-compatible addition to the shared `op()` test helper in `conftest.py` (a `traversal` kwarg), no production code changes. 56 passed (34 prior + 22 new) on both `cpu` and `cuda`.
* ~~Add a dedicated divergence layout check...~~ Done, and the underlying `dotMode` gap it was built to expose is fixed (see Reality Check). `scripts/operation_matrix.py` now verifies both `divergenceDotMode` conventions against the analytic tensor divergence as a standing regression check; this should still be ported into the pytest suite so it runs in CI once CI exists.
* ~~Track the curl compile failure and the two renormalization instabilities as real defects...~~ Done — all three (curl's bare-`return` UB, renorm's missing `wp.launch(device=...)`, and the separate `pinv2x2` symmetric-SVD desync bug it uncovered) are fixed and verified; see Reality Check.
* A first cut of a cross-cutting diagnostic exists: `scripts/operation_matrix.py` runs every operation against its gradient/laplacian-scheme variants, both traversal modes, and all three correction paths (none/CRK/renorm) on deterministic linear fields (optionally jittered via `--jitter`, since a perfect lattice barely exercises CRK/renorm), and prints a pass/fail/error matrix with MAE to the console. It is not a replacement for the pytest suite (no assertions, just reporting) but has directly found 5 real defects so far (Symmetric-scheme double-volume bug in Divergence/Curl/grid-Gradient, curl's CPU compile UB, Laplacian's gradient-scheme incompatibility, renorm's mixed-backend launch, and the `pinv2x2` desync).

## Deliverables

* `tests/operations/conftest.py`, `test_operations_core.py`, `test_operations_consistency.py`, `test_operations_crk_analytic.py` — consolidated coverage for density/interpolate/gradient/divergence/curl/laplacian, in place of the originally planned one-file-per-operation layout. Revisit whether per-operation splitting is still wanted once grid/3D/golden-data coverage is added and files grow.
* `docs/regression/notebook_test_matrix.md` — notebook-to-test mapping (needs the root-vs-`old/` correction noted above).
* `scripts/operation_matrix.py` — manual console diagnostic (see follow-up tasks).
* `tests/operations/test_grid_modes.py` — grid-path (`adjacency=None`) coverage for the base operator set.
* `tests/operations/test_gradcheck_scripts.py` — subprocess-driven pytest coverage for all six `scripts/gradcheck_*_native.py` canaries (plus the legacy `gradcheck_density.py`), part of the regular `pytest tests/` run.
* `.github/workflows/tests.yml` — CI: pytest suite (including the gradcheck scripts) + `scripts/operation_matrix.py --ci` on CPU across 2D/float32, 2D/float64, 1D, and jittered configurations, plus a CUDA-gated 3D step, all push/PR triggered.
* Still missing (see deferral notes above for the first two): `tests/data/` golden-data baseline fixtures (**deferred**), nightly extended CI run sweeping the full precision×dim×jitter product (**deferred**), per-test behavior-spec docs, full CUDA CI coverage (2D/1D on GPU — the 3D step alone is covered when a GPU runner is present).

## Exit Criteria

* Every operation has at least one notebook-derived regression case. — **Met for 6/6 operations.** Curl is ported (`warp_curl.ipynb` in root) and passing.
* Baselines reproduce on repeated runs for the same backend/device. — Met for the deterministic lattice case; verification against stored golden output is **deferred** (see above), not scheduled near-term.
* CI can block regressions before and during migration. — **Largely met.** `.github/workflows/tests.yml` blocks on pytest failures (including per-operator gradcheck scripts) and on any non-`OK` cell in `scripts/operation_matrix.py --ci` across five targeted configurations (2D float32, 2D float64, 1D, jittered, and CUDA-gated 3D). Not met for full CUDA CI coverage or for a nightly sweep of the full precision×dim×jitter product — both **deliberately deferred**, not just unstarted gaps.

---

# Phase 1 – Standardize Kernel Interfaces

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

## Step 4 - Migrate Density End-to-End First

Use density as the first production migration template:

* move launch path from flat wrapper to structured wrapper
* maintain legacy shim compatibility
* validate parity against Phase 0 baselines

## Step 5 - Migrate Remaining Core Operators

Migrate interpolate, gradient, divergence, curl, and laplacian one-by-one using the same structured ABI and parity gates.

## Step 6 - Consolidate Traversal and Close Capability Gaps

Reduce adjacency/grid duplication in orchestration and add missing compact-hash support where currently blocked (especially CRK/renormalization paths).

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
