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
* Curl has **not** been ported yet — there is no `warp_curl.ipynb` in root, and no root notebook exercises `WarpOperation.Curl`.
* Grid, CRK, and renormalization notebooks (`warp_grid.ipynb`, `warp_crk.ipynb`, `warp_crk_test.ipynb`, `warp_profile.ipynb`, etc.) still live under `old/` and use the pre-port conventions. `docs/regression/notebook_test_matrix.md` currently cites these as if they were the live source notebooks; that mapping should be corrected once curl/grid/CRK/renorm ports land in root, or the doc should explicitly say it derived those cases from the archived versions.

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
* The grid dispatch path (`sphOperation_warp_grid`, `operations_grid/*`) exists and mirrors every operator, and can already be exercised today by passing `adjacency=None` to `warpOperation`/`sphOperation_warp` — it is simply never tested. No code changes are required to add grid-path coverage, only test cases.
* ~~The top-level `OperationProperties`/`warpOperation` API does not expose `dotMode` for divergence...~~ **Fixed.** `OperationProperties` now has a `divergenceDotMode` field (default `False`, preserving old behavior), threaded through both `sphOperation_warp` and `sphOperation_warp_grid` down to `computeSPHDivergence_warpBackend`'s `dotMode` argument. `scripts/operation_matrix.py`'s `Divergence-Matrix[...]` rows now verify both conventions directly: the matrix field as-is with `divergenceDotMode=True`, and the pre-transposed (`.mT`) field with `divergenceDotMode=False` (the old default/workaround) — both now match the analytic tensor divergence. The `warp_divergence.ipynb` `.mT` workaround is no longer required going forward, though the notebook itself hasn't been updated to use the new flag yet.
* **Fixed alongside the above:** `GradientScheme.Symmetric` had a duplicated `* apparentVolume` factor in the Symmetric branch of `wp_divergence.py`, `wp_divergence_grid.py`, `wp_gradient_grid.py`, `wp_curl.py`, and `wp_curl_grid.py`, which was corrupting output for Divergence/Curl (both traversal modes) and for grid-dispatched Gradient specifically. This was caught by the new operation matrix (see below) and has been fixed in all five files; confirmed via a fresh matrix run on both `cpu` and `cuda`.
* ~~The curl operator has a known kernel compile/launch failure...~~ **Fixed.** Root cause: `computeSPHCurlTensor_Func`'s directionality-mask early-out used a bare `return` inside a Warp `@wp.func` declared with a non-`void` return type. `nvcc` tolerates this (undefined behavior, but happened to compile and run correctly on `cuda`); the CPU/LLVM backend correctly rejects it as invalid IR (`"non-void function ... should return a value"`), which is what the `xfail` was catching. Fixed in both `wp_curl.py` and `wp_curl_grid.py` by returning `outputValue * scalar_t(0.0)` instead. Also carried the `GradientScheme.Symmetric` double-`apparentVolume` fix (see above) into `wp_curl.py`/`wp_curl_grid.py`, which weren't part of the original batch. Confirmed via `scripts/operation_matrix.py`: all 4 `GradientScheme` variants now pass on both `cpu`/`cuda`, both traversal modes.
* ~~Renormalization has two known instabilities...~~ **Both fixed.**
  * The "mixed backend launch" CPU instability was a missing `device=` argument on `wp.launch(kernel=pinv2x2_warp, ...)` in `wp_covariance.py` — without it, Warp launches on its default device context regardless of which device the input tensors actually live on. One-line fix: pass `device=inv_warp.device` explicitly.
  * A separate, more serious correctness bug (not device-specific) was found in `pinv2x2_warp`'s pseudo-inverse math itself: it used a general 2x2-SVD closed form that computes two *independent* rotation angles (`theta` for `U`, `phi` for `V`) via two separate `atan2` calls. SPH renormalization/covariance matrices are symmetric by construction (a sum of `x_ij ⊗ x_ij`-type terms for any isotropic kernel), so `theta` and `phi` are mathematically required to coincide — but for a near-isotropic matrix (the common case: a locally regular, well-resolved particle neighborhood), both `atan2` calls' arguments round to ~0 independently, and the two angles can land on unrelated values instead of staying synchronized. Reproduced directly against production covariance matrices: a matrix that should invert to `~diag(1.0016, 1.0016)` instead produced a matrix rotated by ~43°, with the correct singular values but garbage off-diagonal structure. Root-caused and fixed by replacing the general asymmetric-SVD formula with the correct, numerically robust closed-form symmetric-matrix eigendecomposition (a single `atan2` call, so there is nothing left to desync) in `wp_covariance.py`'s `pinv2x2_warp` kernel and its dead-code Python twin. `pinv/twod.py`'s `pseudoInverse2x2` has the identical latent bug but is a generic (currently unused, not necessarily-symmetric-input) utility, so it was intentionally left as-is rather than force-symmetrized. Verified via `scripts/operation_matrix.py`: renormalization matches CRK-level accuracy across every operator, both devices, both traversal modes.

## Additional Findings from `scripts/operation_matrix.py` (2026-08-05)

The diagnostic matrix (see Phase 0 follow-up tasks below), run repeatedly across both `cpu`/`cuda`, both traversal modes, and all three correction paths as fixes landed same-day, surfaced every issue recorded above. **As of the latest run, the full matrix is clean: 186/186 applicable cells `OK`, 0 `HIGH`, 0 `ERR`, 0 `NAN`, on both `cpu` and `cuda`.** The pytest suite reflects the same: 34 passed, 0 xfailed (previously 30 passed, 4 xfailed with two of those `xfail`s masking forced pre-emptive skips, not "ran and tolerated failure"). The stale `xfail`/`try-except-xfail` scaffolding in `test_operations_core.py` and `test_operations_crk_analytic.py` has been removed now that the underlying bugs are fixed, so these paths get real regression coverage instead of a permanent skip.

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

* ~~Inventory all operation-relevant notebooks in the repository root.~~ Done, with the caveat above that curl/grid/CRK/renorm sources are still archived in `old/`.
* ~~Extract deterministic scenarios from each notebook (fixed seeds, fixed particle counts, fixed modes).~~ Done for density/interpolate/gradient/divergence/laplacian; curl scenario exists but is unverified (compile failure).
* ~~Convert scenarios into `pytest` parameterized tests for all operation families.~~ Done, consolidated into 3 files rather than one-per-operation (see Deliverables note).
* Capture baseline outputs (golden data or compact summaries) with documented tolerances. **Not done** — current tests check analytic properties (finite, positive, MAE vs. closed-form linear-field derivatives) rather than stored golden snapshots. No `tests/data/` exists.
* Add gradient/finite checks where applicable. **Not done** — no `torch.autograd.gradcheck` or finite-difference AD verification exists anywhere in the repo yet.
* Generate matching markdown scenario docs so each test is also an executable behavior spec. **Partially done** — one summary mapping doc exists (`docs/regression/notebook_test_matrix.md`); no per-test behavior-spec docs.
* Add CI entrypoints for a deterministic subset and an extended nightly matrix. **Not done** — no `.github/workflows` or other CI config exists in the repo.

### Newly identified follow-up tasks

* Add direct grid-path coverage: run the existing operation/variant matrix with `adjacency=None` (or an explicit `CompactHashMap`) instead of a precomputed `AdjacencyListWarp`. No implementation work is needed to unlock this — `sphOperation_warp` already routes `adjacency=None` to `sphOperation_warp_grid` for every operator.
* ~~Add a dedicated divergence layout check...~~ Done, and the underlying `dotMode` gap it was built to expose is fixed (see Reality Check). `scripts/operation_matrix.py` now verifies both `divergenceDotMode` conventions against the analytic tensor divergence as a standing regression check; this should still be ported into the pytest suite so it runs in CI once CI exists.
* ~~Track the curl compile failure and the two renormalization instabilities as real defects...~~ Done — all three (curl's bare-`return` UB, renorm's missing `wp.launch(device=...)`, and the separate `pinv2x2` symmetric-SVD desync bug it uncovered) are fixed and verified; see Reality Check.
* A first cut of a cross-cutting diagnostic exists: `scripts/operation_matrix.py` runs every operation against its gradient/laplacian-scheme variants, both traversal modes, and all three correction paths (none/CRK/renorm) on deterministic linear fields (optionally jittered via `--jitter`, since a perfect lattice barely exercises CRK/renorm), and prints a pass/fail/error matrix with MAE to the console. It is not a replacement for the pytest suite (no assertions, just reporting) but has directly found 5 real defects so far (Symmetric-scheme double-volume bug in Divergence/Curl/grid-Gradient, curl's CPU compile UB, Laplacian's gradient-scheme incompatibility, renorm's mixed-backend launch, and the `pinv2x2` desync).

## Deliverables

* `tests/operations/conftest.py`, `test_operations_core.py`, `test_operations_consistency.py`, `test_operations_crk_analytic.py` — consolidated coverage for density/interpolate/gradient/divergence/curl/laplacian, in place of the originally planned one-file-per-operation layout. Revisit whether per-operation splitting is still wanted once grid/3D/golden-data coverage is added and files grow.
* `docs/regression/notebook_test_matrix.md` — notebook-to-test mapping (needs the root-vs-`old/` correction noted above).
* `scripts/operation_matrix.py` — manual console diagnostic (see follow-up tasks).
* Still missing: `tests/operations/test_grid_modes.py` (or equivalent grid-path pytest coverage), `tests/data/` baseline fixtures, CI entrypoints, gradient/finite-difference checks, per-test behavior-spec docs.

## Exit Criteria

* Every operation has at least one notebook-derived regression case. — Met for 5/6 operations; curl exists but is not currently passing.
* Baselines reproduce on repeated runs for the same backend/device. — Met for the deterministic lattice case; not yet verified against stored golden output.
* CI can block regressions before and during migration. — **Not met**, no CI wiring exists yet.

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
