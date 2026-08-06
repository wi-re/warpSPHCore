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

* `warpOperation` still routes through `sphOperation_warp`, so legacy internals remain dominant.
* Most operators still launch via flat tensor argument wrappers (`warpWrapper`) instead of structured wrappers.
* State objects remain torch-native and do not yet own synchronized torch+warp field representations.
* Adjacency and grid execution paths are still largely duplicated across operation families — but a working counter-example now exists for Gradient, see "Working Prototype" below.
* CRK and renormalization corrections cannot run on grid-mode traversal at all (a real capability gap, not a performance one) — see `docs/lessons_learned.md`'s "Architectural facts still true" for specifics. This is Phase 6's target, and the Gradient prototype below plumbs `correctionData` through the grid branch for free, which is the template for closing this gap operator-by-operator.
* `LaplacianScheme.Dot` does not support scalar fields in `dim>1` domains — guarded with a `ValueError`, not fixed. See `docs/lessons_learned.md`.

## Largest Remaining Change

The key migration step is introducing a `Field` abstraction that carries both torch and warp representations with synchronization ownership, while preserving legacy fallback conversion paths for compatibility.

## Working Prototype: Unified Kernel + Traversal for Gradient (`wp_grad.py`, `warp_gradient.ipynb`)

`wp_grad.py` (repo root, not yet moved into `src/sphWarpCore/operations/`) is a from-scratch reimplementation of the Gradient operator that already delivers several things the phases above still list as future work. `warp_gradient.ipynb` exercises it against the production `warpOperation` path and against itself in both traversal modes. This section records how it does that so the same recipe can be repeated for Interpolate, Divergence, Curl, and Laplacian.

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

For each of Interpolate, Divergence, Curl, Laplacian (Density is even simpler — no per-neighbor gradient term to carry):

1. Take the existing `compute<Op>Tensor_Func` from `operations/wp_<op>.py` and rewrite its signature to consume `referenceState: Any` (a `particleDataSoA_D`), `domainState: domainData`, and `correctionData: Any` instead of the current flat `wp.array` parameter list, pulling per-neighbor values via `getParticle(referenceState, j)` and per-query correction terms via `getL_i`/`getGradH_i`/`getVolume_i`/`getCRK_i` at the call site rather than as separate flat args. The physics body does not change — compare `computeGradient_Func_i` (`wp_grad.py:29`) with `computeSPHGradientTensor_Func` (`operations/wp_gradient.py:17`) for a worked diff.
2. Wrap it in a `compute<Op>_Func_Adjacency` that resolves query-point state once, then loops `for o in range(numOffsets)` picking `(beginIndex, numIndices, offsetArray)` from `adjacencyState` or from `checkOffset(...)` against `gridState` depending on `useAdjacency` — copy `computeGradient_Func_Adjacency` (`wp_grad.py:128`) verbatim except for the inner call. `numOffsets` is `1` when `useAdjacency` and `gridState.numOffsets` otherwise (see `computeGradient_Kernel`, `wp_grad.py:226`).
3. Reduce the `wp.kernel` to `wp.tid()` + bounds check + one call into step 2's function. This single kernel replaces both `compute<Op>Tensor_Kernel` (`operations/wp_<op>.py`) and `compute<Op>Tensor_grid_Kernel` (`operations_grid/wp_<op>_grid.py`).
4. Replace the two Python backends (`compute<Op>_warpBackend` and `compute<Op>_grid_warpBackend`) with one function that calls `warpWrapper2` the way `computeGradient` does — `defaultStateArguments` matching `parseArguments`'s ten-argument order, plus operator-specific tensors (e.g. `queryValues`/`referenceValues`) in `additionalArguments`.
5. Point `sphOperation_warp`'s dispatch for that operation at the new unified function and delete the `isinstance(adjacency, CompactHashMap)` branch for it; once every operator is migrated, `wp_operation.py` and `wp_operation_grid.py` collapse into one dispatcher.
6. Delete `operations_grid/wp_<op>_grid.py` only after the `operation-matrix` and `gradcheck` skills both pass for the migrated operator across adjacency and grid modes (and CRK/renorm, once plumbed through — the unified `correctionData` struct threads CRK/renorm through the grid branch for free, which is what closes the "CRK and renormalization cannot run on grid-mode traversal" gap noted above).

This recipe *is* Phase 1 (structured ABI) and Phase 5 (traversal consolidation) for one operator at a time, using infrastructure (`warpWrapper2`, `warp_state.py` structs) that Phase 2/3 already built for the covariance kernel — so the remaining phases mostly collapse into "repeat this six times and delete the old files," rather than requiring new infrastructure design. Gradient was migrated first here (not Density, as the original Step 4 below suggests) because it has the most correction paths (CRK, grad-h, renormalization, volume) to prove out; Density is the better *second* target precisely because it's the simplest operator.

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
