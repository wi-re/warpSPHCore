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
  `warpSPHCore_PRECISION=float64` (and can produce a confusing, seemingly
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
  that needs to stay differentiable.
  **Update 2026-08-11: fixed upstream in warp-lang 1.17.0.dev3.**
  `scripts/repro_ternary_adjoint_zeroing.py` now passes for both the ternary
  and if/else forms under the `warp_dev` conda env (1.17.0.dev3), while still
  failing the ternary case under the pinned/installed `warp` env (1.12.0).
  Verified by temporarily restoring the ternary in `wp_gradient.py`'s
  `computeSPHGradientTensor_Func_i` (the `useGradHTerms` branch) and rerunning
  `scripts/gradcheck_gradient_native.py`: PASSED under `warp_dev`, FAILED
  (zeroed analytical Jacobian) under `warp` 1.12.0 — then reverted back to
  the explicit `if/else` since 1.17 isn't on PyPI yet and `pyproject.toml`
  doesn't pin a `warp-lang` floor. Once a 1.17+ release is published and the
  project's Warp dependency is bumped to it, the `if/else` workarounds in
  `wp_gradient.py`, `wp_divergence.py`, `wp_curl.py`, and `wp_laplacian.py`
  (all four share this exact `useGradHTerms` pattern) can be converted back
  to ternaries.
  **Correction 2026-08-21 (`warpier_tier2_correction_jvp_plan.md` phase b):
  the "only dangerous when both branches index the same array" caveat above
  was wrong.** This entry previously claimed the `apparentVolume = mj/rhoj
  if not correctionData.useVolume else correctionData.referenceVolumes[j]`
  pattern was a "confirmed non-issue" because its two branches index
  *different* arrays. Directly tested under the installed warp 1.16.0 while
  validating phase (b)'s apparent-volume JVP tangent (a spike comparing the
  production JVP against `torch.autograd.functional.jacobian` on the
  primal operator kept failing by a few percent on every value-having
  operator): `torch.autograd.gradcheck` on plain primal
  `warpOperation(Gradient, ..., referenceVolumes=rv)` w.r.t. `rv` fails
  outright — the analytical Jacobian is *exactly* zero while the numerical
  one is not, and `rv.grad` after a bare `.backward()` is all zeros too.
  Converting this exact ternary to an explicit `if/else` (no other change)
  fixes it. Root cause not fully isolated (not necessarily the same
  same-array mechanism as the `referenceValues` case above), but the
  practical rule is simpler and stricter than the old caveat: **treat every
  ternary that reads a Warp array inside kernel code needing to stay
  differentiable as suspect, regardless of whether the two branches touch
  the same array or different ones** — verify with a gradcheck rather than
  reasoning about which arrays are indexed. Fixed in `wp_gradient.py`,
  `wp_divergence.py`, `wp_curl.py`, `wp_laplacian.py`, `wp_interpolate.py`,
  and `wp_covariance.py` (every primal kernel with an `apparentVolume`/`vj`
  ternary of this shape) by converting to explicit `if/else`, same pattern
  as the `useGradHTerms` fix above. This is a **primal** bug, not a JVP one
  — it silently broke `d(output)/d(referenceVolumes)` for any caller of
  `warpOperation(..., referenceVolumes=...)` with `requires_grad=True` on
  that tensor, predating and unrelated to the Tier-2 JVP work.

* **A manual nested `for row / for col: acc[row] += x[col] * mat[row, col]`
  accumulation into a Warp vector/matrix-indexed local can produce a
  finite-but-wrong adjoint, with the *forward* value completely correct.**
  Found in `crk/kernel.py`'s `correctGradientCRK` (`term4`, the CRK
  product-rule term contracting `gradBi` against `x_ij`): a downstream
  user's `torch.autograd.gradcheck` against a real CRK-corrected force
  computation failed with a finite (not NaN/Inf) but numerically wrong
  Jacobian, reproducing even in 1D (where any index-order/transpose choice
  is a no-op on a 1x1 matrix — ruling that out as the cause) and in a
  from-scratch single-neighbor-pair repro with no dependency on
  `crk_moments.py`/`crk_terms.py` at all. Routing the exact same
  contraction through the existing `matmul(...)` `@wp.func` (already used
  elsewhere, e.g. `wp_gradient.py`'s renormalization path) instead of the
  manual loop fixed it outright — despite `matmul`'s own implementation
  using a structurally similar `res[i] += mat[i, j] * vec[j]` loop, so this
  is not simply "loops are broken", something about the *inlined,
  hand-written* loop specifically (as opposed to the same shape inside a
  separately-compiled `@wp.func` call) triggered a bad adjoint. Forward-only
  checks and even the existing `gradcheck_crk_native.py` (which only checks
  `d(A,B,gradA,gradB)/d(position)`, not that a downstream consumer of
  `gradB` differentiates correctly) both missed this —
  `gradcheck_crk_correction_native.py` was added specifically to close this
  gap, exercising `correctGradientCRK` through a real operator call with a
  genuinely nonzero, position-dependent `B`/`gradB`. Treat any new manual
  index-accumulation loop over a Warp vector/matrix as suspect until it has
  its own gradcheck coverage; prefer an existing, already-gradchecked
  helper (`matmul`, `wp.dot`, etc.) over a fresh hand-rolled loop when one
  fits.

* **A plain Python scalar (e.g. a `float` field on a Python-side struct)
  passed straight into `wp.launch` gets Warp's default type inference —
  `wp.float32` — regardless of the kernel's declared `scalar_t`.** This
  silently breaks any kernel argument comparison under
  `warpSPHCore_PRECISION=float64` (`expected float64, got float32`). Cast
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

* **A kernel output declared `wp.array(dtype=vec_t)`, where `vec_t = vector(length=dim_t,
  dtype=scalar_t)`, fails at launch time whenever `dim_t` isn't pinned via env var** (the default in
  this repo's test suite) — Warp resolves `vec_t` to a length-`Any`, ungrounded vector type in that
  case and rejects it (`TypeError: ... cannot be generic, got array(ndim=1, dtype=vec0f)`). Declare
  such outputs `wp.array(dtype=Any)` instead (matching every other dimension-generic production
  kernel, e.g. `wp_gradient.py`'s `outputValues`), and allocate the concrete `(numPairs, dim)` torch
  tensor separately, cast via `castTorchToWarpAsBuiltins` (which resolves the concrete vector width
  from the tensor's own shape, the same way `buildParticleSoA` already does for `positions`). Found
  while porting the shared Tier-2 JVP `∇W_ij` kernel (`wp_kernelGradientJVP.py`).

* **A loop-accumulated local value (e.g. `proj = dot(q[block], n)`, built via a runtime `for k in
  range(dim):` loop) that is then consumed by a further non-linear op *in the same function body*
  (e.g. `out = proj * n`, reusing `n` again) silently drops part of its reverse-mode gradient
  contribution — forward value is unaffected; the analytical adjoint comes out at roughly half its
  correct value, or worse once summed over multiple neighbors with varying signs. Moving the
  accumulation loop into its own separate `@wp.func` that *returns* the accumulated value, called
  from the original function instead of inlining the loop, fixes it.** This is a recurring Warp
  code-generation limitation in this repo (has surfaced more than once) — version 1.16.0 at time of
  writing. Latest occurrence: `math/wp_laplaciandot.py`'s `computeLaplacianDot2`
  (`LaplacianScheme.Dot`'s forward kernel) and independently `coreOperations/wp_laplacianJVP.py`'s
  own `computeSPHLaplacianDotJVP_Func_i` (Tier-2 JVP) — same root pattern, two unrelated call sites,
  confirmed the same bug rather than two separate ones, both fixed the same way (`computeLaplacianDot2`'s
  own `computeDotLaplacian(q_ij, n_ij, dim, base)` overload / `wp_laplacianJVP.py`'s
  `_laplacianDotProjJVP`). Minimal from-scratch repro (no dependency on this codebase's kernel
  structure): a `@wp.func` computing `proj = q[0]*n[0]` inside a `for k in range(dim):` loop (`dim` a
  runtime `wp.int32` kernel argument, not a Python literal), then `out[i] += proj * n[d]` in the
  *same* function — `wp.Tape`'s backward gives `d(out)/dn` at half the finite-difference value;
  moving just the loop into a second `@wp.func` that returns `proj`, called from the first, gives the
  correct value with no other change. Two things that do **not** fix it, confirmed by trying both
  before finding the real fix: fully avoiding the bug by unrolling everything (works, but not always
  practical) is not the same as restructuring the multiply within one function (`leftVec = proj * n`,
  then read `leftVec[d]` instead of `n[d]` a second time — bug survives identically); the loop itself
  has to move to a separate function, not just the multiply that consumes its result. Confirmed fixed
  via `torch.autograd.gradcheck` (`scripts/gradcheck_tier2_jvp_laplacian.py`, all four
  `LaplacianScheme`s now pass) and finite differences agreeing with the jacobian-based test reference.

* **A quantity that divides by the same regularized pairwise distance twice
  (once inside building a unit vector `n_ij = x_ij/D_ij`, once again
  dividing the final `dot(vector, n_ij)` by `D_ij`) can produce a wrong
  reverse-mode adjoint at an exact self-pair (`r_ij == 0`) when `vector` is
  CRK-corrected — even though the *forward* contribution there is always
  exactly `0` (n_ij vanishes identically at `x_ij == 0`, forcing the dot
  product to `0` regardless of `vector`).** Found in `wp_laplacian.py`'s
  (primal) and `wp_laplacianJVP.py`'s (Tier-2 JVP) `LaplacianScheme.Brookshaw`
  formula (`-2*q_ij*dot(kernelGradient, n_ij)/D_ij`) once CRK tangent
  support was extended to it (`warpier_tier2_correction_jvp_plan.md` phase
  (e)). Without CRK, `kernelGradient` is itself exactly `0` at a self-pair
  (a symmetric kernel's gradient vanishes at its own peak, with a *correct*
  adjoint there too, thanks to `sphGradient_`'s existing custom
  `@wp.func_grad` — see this doc's self-pair entry above,
  `project_tier2_jvp_distinct_role_adjoint_bug` in Claude's memory). With
  CRK enabled, `correctGradientCRK`'s own value at `x_ij == 0` is
  generically **nonzero** (its `Ai*W_ij*Bi`/`W_ij*gradAi` terms don't vanish
  at the kernel's own peak the way the plain gradient does) — and Warp's
  reverse-mode through "a nonzero vector dotted against an exactly-zero
  `n_ij`, itself then divided by `D_ij` again" at that exact point produces
  a wrong adjoint (confirmed via a from-scratch minimal repro with no
  dependency on this codebase's kernel structure: a single query particle
  summing over several neighbors, `computeKernelGradientCRKJVP` feeding
  `dot(G, x_ij/D_ij)/D_ij`, reproduced the bad adjoint at `n=2`
  mutually-neighboring particles; the same pattern with no *second*
  division — i.e. `dot(G, x_ij/D_ij)` alone, Divergence/Curl's own shape —
  did **not** reproduce it). **Fixed** by guarding the Brookshaw
  contribution with an explicit `if r_ij > 0:` in both kernels — the true
  contribution at `r_ij == 0` is always exactly `0` (CRK or not), so
  skipping it outright changes no forward value anywhere (confirmed:
  `operation_matrix.py` and every existing non-CRK gradcheck script stayed
  bit-identical) while sidestepping the bad adjoint. This mirrors
  `wp_densityHVP.py`'s own explicit self-pair `pairMask` precedent (a
  different bug, same "don't rely on `0` falling out of the math naturally"
  discipline) more than it mirrors the loop-accumulated-local entry above —
  no loop or accumulation is involved here, just a double division through
  an exactly-zero intermediate. Divergence/Curl needed no such guard: their
  own combination formulas (`dot(coeff,G)`, the 2D cross product) have no
  *second* division by a quantity that itself vanishes at `r_ij == 0` the
  way Brookshaw's `n_ij` does. Full writeup:
  `scripts/spike_forward_mode_tier2_crk_extension.py`'s module docstring.

## AD-bridge / autograd gotchas

* **`warpSPHCore_PRECISION` is baked into every compiled kernel at first
  `warpSPHCore` import, per-process, and cannot change afterward.** Any
  tooling that needs to test more than one precision in the same run must
  isolate each precision in its own subprocess (`os.environ` mutation +
  re-import inside one process does nothing once a kernel has already
  compiled). This is why `scripts/operation_matrix.py` imports
  `warpSPHCore` lazily inside a `_configure()` called after arg parsing, and
  why `tests/operations/test_gradcheck_scripts.py` runs each gradcheck
  script via `subprocess.run` rather than importing its module directly.
  **This is a deliberate design tradeoff, not an oversight worth
  "fixing"** (confirmed with the repo owner 2026-08-07, after this kept
  getting flagged as a design smell in reviews without the rationale being
  written down anywhere): some Warp versions don't fully recompile kernels
  when the backing global type changes mid-process, and using a fully
  generic `dim_t=Any`/dtype-agnostic kernel would make casting inside
  genuinely generic kernels very difficult given current Warp limitations.
  A single precision/dimension resolved once at import time was judged the
  reasonable compromise for now. The eventual `Field` abstraction
  (`warpier_core.md` Phase 3) can still track per-field dtype metadata
  (including richer Warp dtypes like `mat33f` that don't fit `scalar_t` at
  all) — it just can't use that metadata to change the underlying compiled
  kernel type within one process run.

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

* **Removing a `@torch.jit.script` decorator can expose a latent eager-mode
  bug that TorchScript was silently absorbing — always smoke-test the
  undecorated function, not just re-import it.** `util/support.py`'s
  `volumeToSupport(volume: float, targetNeighbors: int, dim: int)` calls
  `torch.sqrt`/`torch.pow` on plain Python `float`/`int` arguments (the
  function is pure scalar math — no tensors anywhere in its signature).
  Under `@torch.jit.script`, TorchScript's compiled scalar-op dispatch
  handles `torch.sqrt`/`torch.pow` on a bare `float` without complaint; the
  identical code called eagerly crashes immediately (`TypeError: sqrt():
  argument 'input' must be Tensor, not float`). Because this function sits
  underneath `generateNeighborTestData`, which every operator's shared
  `particle_case` pytest fixture depends on, removing one decorator (as
  part of a Python-3.14-driven jit.script cleanup — `torch.jit.script` is
  unsupported there and has no `torch.compile` substitute for SPH's
  non-static tensor shapes) broke the *entire* test suite (67/67 →
  11/67, all `ERROR` not `FAIL`) instantly and silently until someone
  actually ran the tests. Fixed by switching to `math.sqrt`/`**(1/3)` —
  genuine scalar math, matching what the function's own type annotations
  already promised. Every other `@torch.jit.script` decorator already
  removed from this codebase happened to decorate pure-tensor functions
  (no bare-scalar-into-torch-op calls), so they were unaffected — but that
  was luck, not a property checked in advance. Any future jit.script
  removal needs an actual run of the affected function (or its test
  coverage), not just a successful `import`.

* **A dispatch branch that mirrors an existing one but omits a kwarg, or a scope-boundary guard that
  isn't updated when a new scheme lands, is invisible to a single-scheme smoke test.** Two real bugs
  shipped this way in the Tier-2 JVP dispatch wiring: `gradientMode` was forwarded to every Tier-2
  operator's dispatch call except Laplacian's (silently defaulted to `GradientScheme.Symmetric`
  regardless of what was requested — caught only because the Jacobian-reference test parametrizes
  over all four `GradientScheme`s: 15/20 Laplacian cases failed, `Symmetric` alone passed); and a
  centralized scope-boundary check written *before* `LaplacianScheme.Naive` was implemented kept
  letting it through to Brookshaw's dispatch (would have silently returned Brookshaw's answer for a
  `Naive` request) until a guard was added inside the function itself, caught by re-reading the
  scope-boundary logic against what was actually implemented, not by a failing test. Any new dispatch
  branch added by mirroring an existing one needs the same kwarg-forwarding and scope-guard review,
  and its test file needs to sweep every scheme/mode value the operator supports, not just the
  default one.

## Architectural facts still true (open capability gaps, not yet closed)

* **CLOSED (2026-08-06):** ~~CRK and renormalization corrections only work
  with a precomputed adjacency list~~ — both now run under grid-mode
  traversal (`adjacency=None` or an explicit `CompactHashMap`), not just an
  explicit `AdjacencyList`. `computeCRKFactors`'s `NotImplementedError` was
  removed when CRK was ported to the dual-path kernel style (see
  warpier_core.md's "Landing CRK's dual-path rework"). Renormalization's
  guard turned out to already be dead code — `computeRenormalizationMatrices_`
  has read its neighbor count from Covariance's own per-particle kernel
  output (`covarianceReturnNumNeighbors=True`), not `adjacency.numNeighbors`,
  since the very first restructure commit, so the capability was live but
  untested; `scripts/gradcheck_renorm_native.py` now covers it (forward
  parity + gradcheck across all three traversal inputs). Finding that gap
  also surfaced two independent bugs, now fixed: `pinv/wp_pinv1x1.py`
  referenced `wp.mat11f`/`wp.vec1f` (don't exist on the `warp` module —
  only as warpSPHCore's own precision-specific subclasses in
  `math/wp_vec1.py`) and hardcoded the float32 variant regardless of
  `warpSPHCore_PRECISION`; and `WarpFunctionWrapper.backward`
  (`autograd/stateLessWarpFunction.py`) only checked `isinstance(outputs_warp,
  list)` when seeding gradients for a multi-output warp function, not
  `(list, tuple)` — `launch_kernel` returns a tuple for multi-output, so any
  `warpWrapper`-wrapped (not `warpWrapperStateaware`-wrapped) multi-output
  function's backward pass was unreachable. See warpier_core.md's
  "Renormalization Grid-Mode Coverage" note for the full story.

* **CLOSED (2026-08-07):** ~~`checkKinds` substitutes a length-1 dummy
  tensor for `queryKinds`/`referenceKinds` when they're `None`, even under
  `OperationDirection.AllToAll`~~ — `checkKinds` (`autograd/arg_check.py`)
  now takes `queryNumParticles`/`referenceNumParticles` and builds a
  correctly-sized dummy array for `AllToAll` instead of a length-1 one;
  `extractStateInfo` (`autograd/arg_extract.py`, the live call path) passes
  `qPos.shape[0]`/`rPos.shape[0]` through. `ParticleState(kinds=None)`
  under `AllToAll` is no longer an out-of-bounds read — verified directly
  with a 529-particle 2D Density call. Found while landing CRK (see
  warpier_core.md's "Landing CRK's dual-path rework"); fixed in
  warpier_core.md's "RadiusSearch Package Split, Mechanical `__all__`, and
  the AllToAll-Kinds Fix". One residual: `autograd/arg_parse.py`'s
  `parseArguments` — dead code, superseded by `extractStateInfo`, not
  called anywhere in the repo — still has the old unsized `checkKinds`
  call and would reintroduce this bug if ever revived.

* **CLOSED (2026-08-07):** ~~`pinv2x2_warpBackend` (the dim=2 pseudo-inverse
  `pinv_warp` dispatches to for every 2D renormalization call) has no
  gradcheck coverage~~ — it turned out to have no backward pass at all to
  gradcheck: unlike `pinv1x1` (`warpWrapper`-wrapped `launch_kernel`),
  `pinv2x2_warpBackend` did a raw `wp.launch` directly on tensors converted
  via `castTorchToWarpAsBuiltins`, with no `torch.autograd.Function`
  wrapping it. Any `loss.backward()` through a 2D renormalization call was
  silently treating the pseudo-inverse as contributing no gradient from its
  input covariance matrix, which is wrong. Ported to the same
  `warpWrapper`/`launch_kernel` pattern `pinv1x1` uses (kernel parameter
  order changed to inputs-first/outputs-last to match `launch_kernel`'s
  assembly convention); `scripts/gradcheck_pinv_native.py` now gradchecks
  both `pinv1x1` and `pinv2x2_warpBackend` directly, and a full
  `computeRenormalizationMatrices` call was verified end-to-end to produce
  finite gradients through a genuine 2D case. **Lesson generalized:** a raw
  `wp.launch` call reachable from a differentiable pipeline is a silent
  gradient hole, not merely "untested" — it doesn't raise, it just drops
  that link's contribution. Grep for bare `wp.launch(` outside
  `autograd/launcher.py` itself before trusting gradients through any new
  operator.

* **Not a bug — by design.** `LaplacianScheme.Dot` computes a dot product
  between the field quantity and the kernel gradient, which is only
  mathematically defined for a vector field matching the domain's spatial
  dimension; a scalar field has no such dot product to take. The explicit
  `ValueError` guard (`coreOperations/wp_laplacian.py`, triggered when
  `flatInputShape % spatialDim != 0` under `dim>1`) is the intended,
  permanent behavior for that combination, not a stop-gap awaiting a
  "correct" scalar-field generalization — there isn't one. Callers with
  scalar fields should use `LaplacianScheme.Naive`/`Brookshaw`/`Default`
  instead, per the error message. (Previously logged here as an open gap
  needing SPH domain expertise to fix; reclassified 2026-08-07 — the guard
  itself was always the correct fix, not a placeholder for one.)

* **FIXED (found 2026-08-20, root-caused and fixed same day):**
  reverse-mode differentiation w.r.t. a pairwise kernel's *own primal
  position* was wrong specifically at an exact self-pair (`x_i == x_j`,
  `r == 0`) between a query point and a reference point, for any operator
  whose kernel math differentiates a kernel-*derivative*-shaped quantity a
  further time w.r.t. position (`sphGradient_`'s reverse-mode Jacobian —
  Gradient/Divergence/Curl's own reverse-mode, and every Tier-2 JVP
  operator's `dW`/`dG`/`dL`). **Originally mischaracterized** (see the
  superseded write-up this replaces, and
  `project_tier2_jvp_distinct_role_adjoint_bug` in Claude's memory for the
  full corrected investigation trail) as a generic "query != reference
  tensors" bug — it is not. Genuinely distinct, *non-coincident*
  query/reference position sets differentiate correctly in every
  configuration tried (all `GradientScheme`s/`SupportScheme`s, 1D/2D,
  non-uniform supports, both primal `warpOperation` and the Tier-2 JVP
  bridge). The real trigger is a query point and a reference point occupying
  the *exact same location* — which is what every existing "distinct-role"
  gradcheck script's `positions.detach().clone()` construction happens to
  produce (same coordinates, different tensor objects), and also what every
  `referenceParticles=None` self-referencing script's shared-leaf self-pair
  is, which is why none of them (before this fix) actually exercised the
  bug in a way gradcheck's *combined* per-leaf sensitivity could see: for a
  once-differentiated (`W`-shaped) quantity, the true self-pair contribution
  to `d(output)/dx_i` cancels exactly (`+H(0) - H(0) = 0`) regardless of
  the individual `H(0)` value, so a wrong-but-also-zero per-role value is
  indistinguishable from a correct one there.

  **Root cause:** `math/wp_normalize.py`'s `norm_hess_warp` (the hand-
  derived Hessian-of-`|x|` behind `vectorNormalize_warp`'s custom adjoint,
  which is what Warp's automatic differentiation of `sphGradient_` chains
  through) regularizes with `get_epsilon(x)` (`1e-15` at float64) and blows
  up like `O(1/eps)` exactly at `x=0`. `sphGradient_`'s implicit product
  rule multiplies that against `kernelTerm(q=0)`, which is exactly `0.0` in
  floating point for a smooth kernel's derivative at its symmetric peak —
  `huge * 0.0` silently collapses to `0.0` instead of the true finite
  limit, a removable `0 * infinity` singularity (like `sin(x)/x` at `x=0`).
  Confirmed empirically before the fix: one-sided finite differences from
  both sides of a self-pair converged cleanly to the *same* nonzero value
  as the FD step shrank (1e-3 → 1e-8) — proof the true derivative is smooth
  and well-defined, not a genuine kink the code could defensibly round to
  zero. Contrast with `kernels/hessian.py`'s `sphKernelHessian_` (the
  *closed-form* Hessian, used by e.g. `coreOperations/wp_densityHVP.py` and
  `sphKernelGradientJVP_ij`'s `KernelMeanSymmetric`/`SuperSymmetric`
  branches, never by plain `sphGradient_` differentiation before this fix):
  it already has an explicit `if q < eps: factorA[i,i] = 1.0` near-origin
  branch precisely because the general `outer(x,x)/r²` term is direction-
  dependent at `r=0` in `dim > 1` and needs a hand-derived substitution —
  and per `wp_densityHVP.py`'s own independent notebook check, its `r=0`
  value is correctly finite and physically meaningful (the kernel's own
  peak curvature), not zero. `norm_hess_warp`'s generic, automatically-
  composed route had no equivalent near-origin handling.

  **Not the same as, but related to, `wp_densityHVP.py`'s self-pair
  exclusion** (`pairMask = ii != jj` when assembling Density's HVP): that
  exclusion is for *true* self-interaction (`i == j`, the same particle in
  both argument slots of one shared position variable), where the total
  second derivative is *exactly* zero for any finite `H(0)` by a
  translation-invariance identity (`H_aa + 2H_ab + H_bb = 0`) — a different,
  stronger three-term identity than the simple `+H(0)-H(0)` antisymmetry
  above, and already handled correctly by explicit index-based masking.
  This bug's `x_i == x_j` trigger is about two *independent* variables
  (different tensors, or the same tensor at different indices) that happen
  to coincide numerically — there the correct partial is the nonzero
  `H(0,h)` peak curvature, not zero, and nothing upstream has (or should
  need) `i`/`j` identity information to tell the two cases apart; only the
  kernel math's own `r=0` limit needs to be right.

  **Fix:** `kernels/gradient.py`'s `sphGradient_` (split into `_1D`/`_2D`/
  `_3D` concrete-length variants, mirroring `math/wp_normalize.py`'s own
  workaround for the same "`@wp.func_grad` rejects `dim_t`-generic
  functions" Warp restriction) now carries a custom `@wp.func_grad` that
  returns `sphKernelHessian_`/`sphGradientDkDh_`'s already-validated closed
  forms directly, instead of relying on Warp's automatic composition
  through `vectorNormalize_warp`/`norm_hess_warp`. Verified: the
  raw-`wp.Tape`, r>0 case is unchanged to ~1e-11; the coincident-position
  self-pair gradcheck (previously a ~97%-relative-error mismatch) now
  matches finite differences; `scripts/gradcheck_tier2_jvp_interpolate.py`'s
  previously-known-failing distinct-role case now passes;
  `scripts/kernel_sanity_native.py` all-pass; `scripts/operation_matrix.py
  --ci` unchanged (`OK=258, HIGH=0, ERR=0, NAN=0`); full `pytest tests/`
  unchanged (292 passed, 1 skipped).

## Notebook/documentation conventions

* **"Grid mode" doesn't need its own notebook.** Grid dispatch is just
  `adjacency=None` on the same operator call every other notebook already
  makes — there's no separate example to write. Direct grid-path coverage
  belongs in pytest (`tests/operations/test_grid_modes.py`), not a
  notebook.
* **CRK checks belong inside each operator's own notebook** (interpolate,
  gradient, etc.), not split into standalone CRK-only notebooks — that's
  the pattern the current root notebooks already follow.
