# Lessons Learned

Distilled, forward-looking rules extracted from Phase 0 (regression baseline
+ AD-bridge hardening). This is not a changelog — every entry here is a
generalizable fact worth knowing *before* touching kernel code, the AD
bridge, or the test/CI setup again, e.g. during the Phase 1+ interface
migration in `warpier_core.md`. For the historical "what broke, how it was
found, what commit fixed it" narrative, see git history / commit messages
around 2026-08-05 to 2026-08-06 instead — it has been trimmed out of
`warpier_core.md` to keep that file focused on the forward plan.

## Warp kernel authoring gotchas

* **A bare `return` inside a `@wp.func` declared with a non-`void` return
  type is undefined behavior, not an error — and the two backends disagree.**
  `nvcc` (CUDA) tolerates it and happens to compile/run correctly; the
  CPU/LLVM backend correctly rejects it (`"non-void function ... should
  return a value"`). If an early-out is needed in a value-returning
  `@wp.func`, return an explicit value (e.g. `x * scalar_t(0.0)`), never a
  bare `return`. A CUDA-only test pass will not catch this — the CPU
  backend must be exercised too.

* **Raw Python numeric literals inside a `@wp.func`/`@wp.kernel` do not
  auto-promote to match a `scalar_t` value.** `eps = 1e-8` used against a
  `float64` value fails kernel compilation outright under
  `SPHWARPCORE_PRECISION=float64` (and can produce a confusing, seemingly
  unrelated cascade error on the *next* compile attempt). Always wrap:
  `eps = scalar_t(1e-8)`. This only surfaces under non-default precision, so
  a float32-only test pass will not catch it — see the "always sweep
  precision" testing lesson below.

* **A ternary expression assigned to a local variable, where each branch
  indexes the same Warp array, can compile fine, run the correct branch at
  runtime, and still produce a silently-zero adjoint for that array read.**
  `fv = arr[jj] if cond else arr[j]` is the exact pattern that broke
  Interpolate's `d(output)/d(referenceValues)` — always zero, no error, no
  warning. The equivalent explicit `if: ... else:` block does not have this
  problem and is what the rest of the codebase already uses. Forward-only
  checks cannot see this at all; it only shows up in a gradcheck. Prefer
  explicit `if/else` over a ternary for any array read inside kernel code
  that needs to stay differentiable, and treat a structurally similar
  ternary elsewhere as worth a gradcheck, not an automatic red flag — a
  ternary is only dangerous when both branches index the *same* array
  (confirmed non-issue for the `apparentVolume = mj/rhoj if ... else
  referenceVolumes[j]` pattern, which indexes different arrays per branch).

* **A plain Python scalar (e.g. a `float` field on a Python-side struct)
  passed straight into `wp.launch` gets Warp's default type inference —
  `wp.float32` — regardless of the kernel's declared `scalar_t`.** This
  silently breaks any kernel argument comparison under
  `SPHWARPCORE_PRECISION=float64` (`expected float64, got float32`). Cast
  explicitly at the call site (`scalar_t(some_python_float)`), matching the
  existing `wp.int32(...)`/`wp.bool(...)` convention used for other scalar
  launch arguments.

* **`wp.launch(...)` without an explicit `device=` argument launches on
  Warp's default device context, not the device of the input tensors.** Any
  code path that might run with CPU-resident tensors while CUDA is also
  available needs `device=<tensor>.device` passed explicitly — omitting it
  is a real, device-dependent correctness bug, not just a style nit.

* **Symmetric-matrix inverse/eigendecomposition math needs a single shared
  rotation angle, not two independently-computed ones.** A general
  asymmetric 2x2-SVD closed form computes `U`'s rotation via one `atan2`
  and `V`'s via a second, independent `atan2`. For a genuinely symmetric
  input (true by construction for any covariance/renormalization-style
  matrix — a sum of `x_ij ⊗ x_ij` terms), those two angles are
  mathematically required to coincide, but near-isotropic inputs (the
  common, well-resolved-neighborhood case) can round both `atan2` calls to
  arguments near zero independently, letting them desync into unrelated
  values — correct singular values, garbage off-diagonal/rotation
  structure. Use the closed-form symmetric eigendecomposition (one
  `atan2`) for anything guaranteed symmetric; don't reuse a generic
  asymmetric formula "because it's more general."

* **Build periodicity-wrapped position tensors with
  `torch.stack([...], dim=1)`, not `torch.vstack([...]).mT`.** In a 1D
  domain, `vstack(...).mT` produces a `(n, 1)` tensor with a degenerate
  `(1, n)` stride. PyTorch's `is_contiguous()` reports `True` regardless (a
  size-1 axis can't violate contiguity by definition), so defensive
  `.contiguous()` calls upstream are no-ops — and Warp's stricter
  `wp.from_torch` stride check then rejects it. `torch.stack(dim=1)`
  produces canonical strides at any dimensionality and has no transpose to
  go wrong. This is 1D-specific — a 2D-only test pass will not catch it.

## AD-bridge / autograd gotchas

* **`SPHWARPCORE_PRECISION` is baked into every compiled kernel at first
  `sphWarpCore` import, per-process, and cannot change afterward.** Any
  tooling that needs to test more than one precision in the same run must
  isolate each precision in its own subprocess (`os.environ` mutation +
  re-import inside one process does nothing once a kernel has already
  compiled). This is why `scripts/operation_matrix.py` imports
  `sphWarpCore` lazily inside a `_configure()` called after arg parsing, and
  why `tests/operations/test_gradcheck_scripts.py` runs each gradcheck
  script via `subprocess.run` rather than importing its module directly.

* **Never cache/reuse a `wp.array` (and its persistent `.grad` buffer) by
  tensor identity across the "query" and "reference" roles of the same
  self-interaction call.** When `referenceParticles=None` (so
  `referencePositions is queryPositions`), an identity-keyed cache hands
  both roles the *same* underlying Warp array. Warp's tape correctly sums
  both roles' contributions into that one buffer — but PyTorch *also* sums
  gradients across a tensor argument used in two slots of the same
  `autograd.Function` call, so the already-summed total gets added to
  itself: every position/support/mass gradient for self-interaction ops
  comes out silently 2x too large. There is no safe way to cache a
  differentiable Warp array by storage identity across independent
  "roles" in one call — this class of bug is why the caching layer was
  removed outright rather than patched (see `feedback_ripout_warp_array_caching`
  in project memory for the "why remove, not patch" reasoning).

* **When reading a gradient back out of a Warp array
  (`wp.to_torch(wa.grad)`), `.clone()` it, and call `tape.zero()`
  afterward.** The raw tensor from `wp.to_torch` is a live, unreset view
  into Warp-owned memory; the next thing that reuses that same underlying
  buffer will silently corrupt or read stale data through it. This is
  specifically what breaks `torch.autograd.gradcheck`'s own internal
  reentrancy self-check (it calls `backward()` twice against one retained
  graph to verify determinism).

* **To seed an output gradient before `tape.backward()`, use
  `Tape.backward(grads={array: seed})` — never `array.grad = seed` /
  direct assignment.** `wp.from_torch()` is zero-copy, so a direct
  assignment makes the caller's own torch tensor the *live* output-adjoint
  buffer. Warp's backward pass reads **and zeros** output adjoints as it
  consumes them, mutating that tensor in place. Any caller that reuses the
  same `grad_outputs` tensor object across separate
  `torch.autograd.grad(...)` calls (a realistic pattern — a preallocated
  gradient-seed buffer reused every training step) gets a correct gradient
  on the first call and silent zeros after, because its own tensor was
  zeroed out from under it. `Tape.backward(grads={...})` internally does a
  *value copy* into the array's own persistent `.grad` buffer instead of
  aliasing — this is the confirmed-correct upstream pattern (`warp-lang`
  team reproduced and confirmed this exact root cause), not a workaround.
  Both this fix and the clone+zero fix above are independently necessary —
  applying only one does not fix the other's failure mode.

* **Call `torch.autograd.gradcheck` directly against the real op
  (`gradcheck(f, inputs)`), not through a hand-rolled Jacobian / manual
  per-call cloning wrapper.** The direct form is what actually exercises
  reentrancy (gradcheck's internal double-backward self-check) — a manual
  Jacobian loop can pass while the underlying reentrancy bug is still live.

## Testing/CI methodology lessons

* **Forward-value checks and backward-mode gradchecks catch structurally
  different bug classes — neither substitutes for the other.** The
  forward-only suite (`operation_matrix.py`, the core pytest suite) never
  caught the ternary-adjoint-zeroing bug or a `LaplacianScheme.Dot`
  out-of-bounds read; both needed dedicated `torch.autograd.gradcheck`
  coverage to surface at all. Any new operator or kernel rewrite needs
  both kinds of coverage, not just one — see the `gradcheck` and
  `operation-matrix` skills for how to run each.

* **Always sweep precision and dimension when validating a kernel-level
  change, not just the default float32/2D case.** The raw-literal
  type-promotion bug only broke under `float64`; the periodicity-stride bug
  only broke in `dim=1`; the `LaplacianScheme.Dot` out-of-bounds read only
  triggered for `dim>1`. Each of these was invisible at the *other* setting
  they didn't specifically test. `scripts/operation_matrix.py --precision`
  / `--dim` and the CI matrix built on top of it exist specifically to make
  this cheap to check.

* **Jitter beyond ~0.01 is not currently validated to stay under the MAE
  threshold — do not gate CI on it without first investigating sound
  thresholds.** Heavier jitter (0.15-0.3, the range that actually stresses
  CRK/renormalization the way the notebooks' jittered examples do)
  produces many real `HIGH` cells today on a first attempt (confirmed:
  184 `HIGH` cells at `--dim 1 --jitter 0.15`). That's expected diagnostic
  behavior at that jitter level, not a regression — but it means any
  future CI/tooling work involving jitter above ~0.01 needs its own
  threshold investigation first, it can't just reuse the existing
  `--threshold 0.4` default.

* **Warp's CPU backend is single-core and unoptimized — 3D workloads are
  prohibitively slow on CPU, not just moderately slower.** Only run `dim=3`
  checks on CUDA; don't substitute a smaller `--nx` on CPU as a stand-in,
  it isn't a like-for-like check at any particle count. CI's 3D step is
  gated behind a runtime `torch.cuda.is_available()` check for this reason
  and simply no-ops on CPU-only runners.

* **A flat, dimension-agnostic "target neighbor count" is not comparable
  across spatial dimensions** — the same count implies a wildly different
  support radius per dimension, and silently produces an oversized support
  (badly wrong operator output) for low-dimension domains. Use `n_h`
  (particles per smoothing length, per axis) converted via `n_h_to_nH`,
  which already accounts for `dim`, instead of picking a raw neighbor
  count when sweeping dimension.

## Architectural facts still true (open capability gaps, not yet closed)

* **CRK and renormalization corrections only work with a precomputed
  adjacency list.** `computeCRKFactors` and `computeRenormalizationMatrices`
  both raise `NotImplementedError` when handed `adjacency=None` or a raw
  `CompactHashMap`, even though every base operator dispatch
  (`sphOperation_warp`) happily builds a grid datastructure on the fly for
  the operation itself. Grid-mode traversal cannot currently be combined
  with CRK or renormalization at all — a real capability gap, not a
  performance one. This is Phase 6's ("Consolidate Traversal and Close
  Capability Gaps") target, not yet started.

* **`LaplacianScheme.Dot` does not support scalar fields in domains with
  `dim>1`**, and is guarded with an explicit `ValueError` rather than
  fixed — `computeLaplacianDot2`'s indexing assumes the field's flattened
  size is a multiple of `dim`, true for vector fields but false for scalar
  ones. The correct scalar-field generalization (if one exists) needs SPH
  domain expertise to redo properly; this is a deliberate stop-gap, not a
  resolved bug.

## Notebook/documentation conventions

* **"Grid mode" doesn't need its own notebook.** Grid dispatch is just
  `adjacency=None` on the same operator call every other notebook already
  makes — there's no separate example to write. Direct grid-path coverage
  belongs in pytest (`tests/operations/test_grid_modes.py`), not a
  notebook.
* **CRK checks belong inside each operator's own notebook** (interpolate,
  gradient, etc.), not split into standalone CRK-only notebooks — that's
  the pattern the current root notebooks already follow.
