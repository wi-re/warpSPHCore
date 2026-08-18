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

This section captures current implementation status relative to the target architecture. Last verified against the repo on 2026-08-07.

Detailed bug-fix narratives that used to live in this section (AD-bridge reentrancy, grid-path float64/1D bugs, curl's compile failure, renormalization instabilities, the gradcheck rollout, CI wiring) have been distilled into forward-looking rules in [`docs/lessons_learned.md`](docs/lessons_learned.md) and trimmed out of here to keep this plan focused and short. Read that file before touching kernel code, the AD bridge, or CI/test tooling again — it captures the *why*, not just the *what*. Git history (commits around 2026-08-05 to 2026-08-06) has the original investigation narrative if ever needed.

## Module Layout After the Repository Restructuring (2026-08-06)

The narrative sections below ("Working Prototype → Production", "Landing Gradient/Interpolate/Density/...", "Collapsing `sphOperation_warp`", "States as the Primary Path", "Landing Covariance's dual-path rework") were written against the file layout as it existed *at the time each piece of work landed*, one to two commits before `73432b6` ("refactor repo into a clearer structure") reorganized the package. That narrative — the recipe, the bugs found, the reasoning — is still accurate and worth reading; only the file paths it cites are now stale. Read paths in those sections through this table:

| Path as written below | Current path |
| --- | --- |
| `operations/wp_operation.py` (`warpOperation`, `sphOperation_warp`) | `operations.py` (top-level module, not a package) |
| `operations/wp_<op>.py` (Density/Interpolate/Gradient/Divergence/Curl/Laplacian kernels) | `coreOperations/wp_<op>.py` |
| `operations/__init__.py` | gone — `operations.py` is a flat module |
| `operations_grid/` (all `wp_<op>_grid.py`, `wp_operation_grid.py`, `__init__.py`) | still gone — this was already deleted before the restructuring; unaffected |
| `renorm/wp_covariance.py` | `coreOperations/wp_covariance.py` — and see the note at the end of "Landing Covariance's dual-path rework" below: covariance's Python entry point changed shape again, not just location |
| `renorm/wp_pinv2x2.py` | `pinv/wp_pinv2x2.py` |
| `renorm/wp_renormalization.py`, `renorm/__init__.py` | folded into a single top-level `renorm.py` (`computeRenormalizationMatrices_`/`computeRenormalizationMatrices`) — `renorm` is a flat module, not a package, same as `operations.py` |
| `pinv/oned.py` | `pinv/wp_pinv1x1.py` |
| `pinv/twod.py` | deleted, superseded by `pinv/wp_pinv2x2.py` |
| `mathutil/__init__.py`, `mathutil/wp_math.py`, top-level `math.py` | `math/` package (`wp_distance.py`, `wp_eps.py`, `wp_eye.py`, `wp_matmul.py`, `wp_norm.py`, `wp_normalize.py`, `wp_outerTensorProduct.py`, `wp_pow.py`, `wp_sqrt.py`) |
| `kernels/adjoints.py` | `math/wp_normalize.py` |
| top-level `ops.py` | deleted (unused) |
| top-level `state.py` (`ParticleState`, `OperationProperties`, `CRKState`, `GradHState`, `RenormalizationState`) | `dataTypes/particleData.py` (`ParticleState`), `dataTypes/properties_t.py` (`OperationProperties`), `dataTypes/corrections_t.py` (`CRKState`/`GradHState`/`RenormalizationState`) — see "Data Types / State Split" below. Deleted in `e5d7177`, two commits after this table's `73432b6`, not part of this move itself. |
| top-level `warp_state.py` (`particleDataSoA_{1,2,3}`, `adjacencyData`, `gridData`, `domainData`, `correctionData_{1,2,3}`, `getParticle`/`getL_i`/etc.) | `dataTypes/particleData.py`, `dataTypes/adjacency_t.py`, `dataTypes/domain_t.py`, `dataTypes/corrections_t.py` (structs); `util/stateUtil.py` (accessor functions — moved again since, see below) — see "Data Types / State Split" below. Also deleted in `e5d7177`. |
| `radiusSearch/radius_util.py` (`AdjacencyList`, `AdjacencyListWarp`, `DomainDescription`, `PointCloud`) | `dataTypes/adjacency_t.py` / `dataTypes/domain_t.py` / `dataTypes/particleData.py` — deleted in `e5d7177`. |
| `radiusSearch/hashMap_t.py` (`CompactHashMap`) | `dataTypes/hashMap_t.py` — deleted in `e5d7177`. |
| `utils/` package (`wp_util.py`, `wp_autograd.py`, `support.py`, `arg_check.py`, `stateUtil.py`) and top-level `util.py` | merged into one `util/` package, split further by concern — see "Utils → Util Package Rename" below. Renamed/deleted in `bf5ba8b`, three commits after this table's `73432b6`. |
| `diffusion/` (`viscosity.py`, `util.py`) | deleted outright, not moved — see "Diffusion Removal" below. Deleted in `8aa79fb`. |
| top-level `sph.py` (`from .diffusion import *` re-export shim) | deleted — was already broken before `8aa79fb` (re-exported three names `diffusion/viscosity.py` never defined) and unreferenced anywhere in the repo. |
| top-level `warp_state_util.py` (`parseArguments`, `extractStateInfo`, `warpWrapper2`) | `autograd/arg_parse.py` (`parseArguments`), `autograd/arg_extract.py` (`extractStateInfo`), `autograd/wrapper.py` (`warpWrapper2`) — see "Autograd Package Consolidation" below. Deleted in `4adfd27`. |
| `util/wp_autograd.py` (`WarpFunctionWrapper`, `warpWrapper`, `StateAwareWarpFunction`, `launch_kernel`, `clearKernelArgsCache`) | `autograd/stateLessWarpFunction.py` (`WarpFunctionWrapper`/`warpWrapper`), `autograd/stateAwareWarpFunction.py` (`StateAwareWarpFunction`, plus new `warpWrapperStateaware` alias), `autograd/launcher.py` (`launch_kernel`) — see "Autograd Package Consolidation" below. Deleted in `4adfd27`. |
| `util/wp_util.py`'s `getCachedDummyTensor`/`getCachedIdentityMatrices`/`clearDummyTensorCache`/`getCachedWarpArray`/`clearWarpArrayCache` | `autograd/cache.py` — see "Autograd Package Consolidation" below. Moved in `4adfd27`; `util/wp_util.py` itself survives, trimmed to just `generateNeighborTestData` plus the removed-cache module note. |
| `util/arg_check.py` | `autograd/arg_check.py` (renamed path, unchanged contents) — moved in `4adfd27`. |
| `pinv/threed.py` (fully-commented WIP 3×3 SVD/pseudo-inverse sketch, never wired up) | `pinv/wp_pinv3x3.py` (rename only, still fully commented out) — moved in `4adfd27`. |
| top-level `autograd.py` (pre-existing `WarpFunctionWrapper`/`warpWrapper`/`StateAwareWarpFunction`/`extractStateInfo`/`warpWrapper2` re-export shim, predates the whole `73432b6`→ restructuring streak) | deleted — collided by name with the new `autograd/` package `4adfd27` introduced (Python resolves the package over the same-named module, so this file was silently shadowed, not erroring) and was independently broken anyway, since both of its own imports pointed at files `4adfd27` deleted. See "Autograd Package Consolidation" below. |

`operations_grid/grid_util.py` → `radiusSearch/grid_util.py` (noted inline below) predates this restructuring and is unaffected. `warp_state.py`, `warp_state_util.py`, `utils/`, `kernels/`, `crk/`, `radiusSearch/`, `diffusion/`, `state.py`, `radius.py`, `type_config.py`, `types.py`, `util.py`, `autograd.py`, and `enumTypes.py` all kept their existing top-level/package paths through this restructuring — but see the next section for `kernels/`'s own internal split, which landed one commit later (`a1628a4`, same day) and is not part of the `73432b6` move this table describes. `state.py` and `warp_state.py` themselves were deleted two commits later still (`e5d7177`, same day) — see "Data Types / State Split" below; `utils/`/`util.py` were merged into `util/` and `diffusion/` was deleted outright two commits after that still (`8aa79fb`, `bf5ba8b`, same day) — see "Diffusion Removal" and "Utils → Util Package Rename" below; `warp_state_util.py`, `util/wp_autograd.py`, and top-level `autograd.py` were all retired one commit later still (`4adfd27`, same day) — see "Autograd Package Consolidation" below. All are listed here only because they were unaffected *by this specific `73432b6` commit*.

## Kernel Function Internals Split (2026-08-06, `a1628a4` "Cleanup Kernel Functions")

`kernels/wp_kernel.py` (713 lines, every kernel-shape function plus every kernel-property/derivative wrapper in one file) has been deleted and split by function, not by kernel shape:

| What | Now lives in |
| --- | --- |
| `sphKernel`/`sphKernel_ij` | `kernels/kernel.py` |
| `sphKernelGradient`/`sphKernelGradient_ij` | `kernels/gradient.py` |
| `sphKernelDerivative`/`sphKernelDerivative_` (new — 1D scalar derivative, not previously a separate entry point) | `kernels/derivative.py` |
| `sphKernelDkDh` | `kernels/gradH.py` |
| `sphKernelHessian` | `kernels/hessian.py` |
| `sphKernelLaplacian` | `kernels/laplacian.py` |
| `sphKernelScale`/`sphKernelC_d`/`sphKernelN_H`/`sphKernel_xi` | `kernels/properties.py` |
| `eval_k`/`eval_dkdq`/`eval_d2kdq2`/`eval_d3kdq3`/`eval_C_d`/`eval_kernelScale`/`eval_packing` (the `kernel: wp.int32` dispatch switches over `KernelFunctions`) | `kernels/eval_kernel.py` — deliberately **not** re-exported from `kernels/__init__.py` (commented out there with "should not be used directly"); everything above calls into these instead of duplicating the dispatch |
| The 11 per-shape implementations (`cubicSpline`, `poly6`, `spiky`, `wendland2/4/6`, `quarticSpline`, `quinticSpline`, `viscosityKernel`, `cohesionKernel`, `adhesionKernel`, `B7`), each exposing `<name>_k/_dkdq/_d2kdq2/_d3kdq3/_C_d/_kernelScale/_packingRatio` | new `kernels/kernelFunctions/` subpackage, one file per shape; `kernels/kernelFunctions/__init__.py` builds each shape's `__all__` entries mechanically from a shared `suffixes` list |
| `computeKernelCRK`/`computeKernelGradientCRK` | moved out of `kernels/` entirely into `crk/kernel.py` — these are CRK-correction functions (they consume `Ai`/`Bi`/`gradAi`/`gradBi`), not kernel-shape functions, so they moved to sit next to the rest of `crk/` and are now exported from `crk/__init__.py`, not `kernels/__init__.py` |
| `computePairwiseSupport`, `iPow`, `cpow_warp` | dropped from `kernels/__init__.py`'s exports (kernels/wp_kernel.py used to re-export them as a convenience). `computePairwiseSupport` already lived in `utils/support.py`; `iPow`/`cpow_warp` already lived in `math/`. `radiusSearch/verlet.py`'s import was repointed from `warpSPHCore.kernels` to `warpSPHCore.utils` to match. `kernels/utils.py`, a near-empty import-only shim, was deleted as dead weight. |

New in this pass, not a move: `math/wp_dim.py`'s `get_dim(...)` — a `wp.func` overloaded per vector/matrix length (1/2/3, plain and `wp.array`-wrapped) that returns the dimension as a plain int. `crk/kernel.py`'s `correctGradientCRK` uses it to get `dim` from a vector argument at the call site instead of threading a separate `dim` scalar through every CRK call.

`warpSPHCore/__init__.py`'s import block was rewritten from a hand-maintained explicit symbol list (`from .kernels.wp_kernel import eval_kernelScale, computeKernelCRK, ...`) to `from .kernels import *` / `from .math import *` / `from .crk import *` plus `__all__.extend(kernels.__all__)` (and the same for `math`/`crk`) — so each subpackage's own `__all__` is now the single source of truth for what it re-exports at the top level, rather than a second list to keep in sync by hand. One casualty of that rewrite: `eval_kernelScale`/`eval_k`/`eval_C_d` were dropped from the imports (correctly — they're intentionally not in `kernels.__all__` anymore, see the table above) but three stale references to them were left behind in `__init__.py`'s own hand-written `__all__` list, which broke `from warpSPHCore import *` with `AttributeError: module 'warpSPHCore' has no attribute 'eval_kernelScale'`. Fixed by removing the three stale names from that list; verified with `from warpSPHCore import *`.

Also found and fixed in passing while verifying this split, **pre-existing, not introduced here**: `kernels/hessian.py`'s `sphKernelHessian_` computed `hessian = factorA * k2 + factorB * k1` and never returned it (same in the old `wp_kernel.py`, per git history) — `sphKernelHessian` was unused anywhere in the codebase (confirmed by grep), so nothing had exercised the bug, but it's a real one the moment this function gets wired up. Added the missing `return hessian`.

Validation after this split: all 64 pytest cases pass, all 7 `gradcheck_*_native.py` scripts plus `gradcheck_density.py` pass, and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

## Data Types / State Split (2026-08-06, `e5d7177` "More refactor (focused on state moving)")

Top-level `state.py` (the torch-side `ParticleState`/`OperationProperties`/`CRKState`/`GradHState`/`RenormalizationState` dataclasses) and top-level `warp_state.py` (the warp-side `particleDataSoA_{1,2,3}`/`adjacencyData`/`gridData`/`domainData`/`correctionData_{1,2,3}` structs plus the `getParticle`/`getL_i`/`getVolume_i`/`getVolume_j`/`getGradH_i`/`getGradH_j`/`getCRK_i` accessor `wp.func`s) have both been deleted. Their contents, plus the torch-side dataclasses that used to live in `radiusSearch/radius_util.py` and `radiusSearch/hashMap_t.py` (also now deleted), were merged into a new `dataTypes/` package — grouped by *concept* (adjacency, domain, particle, correction) rather than by torch-vs-warp representation, so each file pairs a torch dataclass with its warp-struct counterpart:

| What | Now lives in |
| --- | --- |
| `AdjacencyList`/`AdjacencyListWarp` dataclasses, `adjacencyData`/`gridData` `wp.struct`s | `dataTypes/adjacency_t.py` |
| `CompactHashMap` dataclass | `dataTypes/hashMap_t.py` |
| `DomainDescription` dataclass, `domainData` `wp.struct` | `dataTypes/domain_t.py` |
| `PointCloud`/`ParticleState` dataclasses, `particleDataSoA_{1,2,3}` `wp.struct`s | `dataTypes/particleData.py` |
| `CRKState`/`GradHState`/`RenormalizationState` dataclasses, `correctionData_{1,2,3}` `wp.struct`s | `dataTypes/corrections_t.py` |
| `OperationProperties` dataclass | `dataTypes/properties_t.py` |
| `getParticle`/`getL_i`/`getVolume_i`/`getVolume_j`/`getGradH_i`/`getGradH_j`/`getCRK_i` accessor `wp.func`s | `utils/stateUtil.py`, re-exported from `utils/__init__.py` |

Every call site that used to do `from ..radiusSearch.radius_util import AdjacencyList, DomainDescription, PointCloud`, `from ..warp_state import (domainData, adjacencyData, ...)`, or `from ..state import ParticleState, OperationProperties, ...` (all six `coreOperations/wp_<op>.py` files, `operations.py`, `radius.py`, `radiusSearch/__init__.py`, `radiusSearch/verlet.py`, `warp_state_util.py`) now does a single `from ..dataTypes import *` instead — one import surface for every semantic-state type, matching the target architecture's framing of state objects as independent of storage backend. `enumTypes.py` also gained an explicit `__all__` (previously every enum was hand-listed in `warpSPHCore/__init__.py`'s import block instead), so `warpSPHCore/__init__.py` now does `from .enumTypes import *` / `from .dataTypes import *` plus `__all__.extend(enumTypes.__all__)` / `__all__.extend(dataTypes.__all__)`, the same pattern `a1628a4` established for `kernels`/`math`/`crk` above.

**This move shipped with real breakage, found and fixed while verifying it here, not in the original commit:**

* **Same failure class as the `a1628a4` `eval_kernelScale` incident above: a hand-written `__all__` entry survived the deletion of the import that used to populate it.** `warpSPHCore/__init__.py`'s `__all__` still listed `getParticle`/`getL_i`/`getVolume_i`/`getVolume_j`/`getGradH_i`/`getGradH_j`/`getCRK_i`, but the `from .warp_state import (...)` that used to bind those names into the module's namespace had been commented out (not deleted) and never replaced with an import from their new home. `from warpSPHCore import *` failed with `AttributeError: module 'warpSPHCore' has no attribute 'getParticle'`. Fixed by adding `from .utils.stateUtil import (getParticle, getL_i, getVolume_i, getVolume_j, getGradH_i, getGradH_j, getCRK_i)` to `__init__.py`. This class of bug (a re-export list that outlives the import feeding it) is now two-for-two across these two refactoring passes — worth a moment's grep of `__all__` against actual bound names after any future mechanical import rewrite.
* **Five files outside the gated pytest suite still imported the deleted modules verbatim** — nothing caught these because nothing in `tests/` exercises them, but they're real breakage for anyone running them directly: `wp_grad.py` (repo-root historical prototype, see "Working Prototype → Production" above; also had its own copy-paste casualty — a relative `from ..dataTypes import *`, invalid because `wp_grad.py` is a standalone script, not a package member, left behind by whatever mechanical replace handled the in-package files), `src/warpSPHCore/diffusion/viscosity.py`, `src/warpSPHCore/diffusion/util.py`, `tests/operations/conftest.py` (imported at collection time — this one *did* break `pytest tests/` outright, `ModuleNotFoundError: No module named 'warpSPHCore.state'`), and `scripts/_gradcheck_common.py`. All five repointed to `dataTypes`.
* A trailing round of dead `# from ...`/`# "Name",` fragments left behind by the same mechanical replace (in `__init__.py`, `radius.py`, `radiusSearch/__init__.py`, `warp_state_util.py`) was deleted outright rather than left commented — nothing referenced them, and the codebase's convention is to delete dead code rather than comment it out.

Validation: all 64 pytest cases pass, `from warpSPHCore import *` succeeds and exposes every name in `__all__` (re-checked after the fix above, since that's exactly the check the previous incident of this kind was missing), and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

## Diffusion Removal (2026-08-06, `8aa79fb` "fix broken things and remove viscosity")

`diffusion/` (`diffusion/viscosity.py`, `diffusion/util.py`) is deleted entirely — not moved, not folded into another package. This was a deliberate product decision, not a refactor-driven relocation: the artificial-viscosity/dissipation formulations it implemented were not a good fit for this backend, and the frontend that consumes `warpSPHCore` already has its own dissipation functions, so this was dead weight the SPH-operator layer doesn't need to own. Every reference the two fixes above (in `warp_grad.py` — corrected here — and the "Landing Gradient" narrative) make to `diffusion/viscosity.py` is now purely historical; there is no current replacement path to point to, unlike every other module move in this document.

Also removed in the same commit, found orphaned while verifying this pass: top-level `sph.py`, a `from .diffusion import *` re-export shim listing `computeDiffusionWarp`/`sphDiffusion_warp`/`computeDiffusion_warpBackend` in its `__all__`. It was already broken *before* this commit — `diffusion/viscosity.py` never actually defined those three names (it defined `computePiViscosity` and friends instead) — and nothing in the codebase imported `warpSPHCore.sph` (confirmed by grep across `.py`/`.ipynb`), so it had been silent dead code for some time. Deleting `diffusion/` just turned one kind of breakage (a stale re-export list) into another (a missing submodule); since neither was reachable, this was a clean deletion, not a functional change.

## Utils → Util Package Rename and Further Splitting (2026-08-06, `bf5ba8b` "another round!")

The `utils/` package (created by the `a1628a4`/`e5d7177` passes above) and the old flat top-level `util.py` (the original home of `castTorchToWarp`/`castWarpToTorch`/`castTorchToWarpAsBuiltins`/`getNextPrime`/`generateNeighborTestData`, all thin re-exports from `utils/wp_util.py`) have been merged into one `util/` package, and `wp_util.py` itself split further by concern rather than left as a catch-all:

| What | Now lives in |
| --- | --- |
| `castTorchToWarp`/`castWarpToTorch`/`castTorchToWarpAsBuiltins`/`_torch_scalar_to_warp_dtype`/`_get_warp_vector_dtype`/`_get_warp_matrix_dtype` | `util/cast.py` (new — previously in `utils/wp_util.py`) |
| `checkDirectionality_i`/`checkDirectionality_j`/`checkDirectionality_Func` | `util/directionality.py` (new — previously in `utils/wp_util.py`) |
| `zero_like`/`zero_like_warp` | `math/wp_zero.py` (new) — moved out of `util/` entirely into `math`, re-exported from `math/__init__.py`. Everywhere that used to do `from .utils.wp_util import zero_like_warp` (or the package-level `warpSPHCore.zero_like_warp`) now gets it via `from ..math import zero_like_warp` instead (see `util/stateUtil.py`'s own import for the pattern) |
| `getNextPrime` | `math/prime.py` (new) — also moved into `math`, not `util`, despite being a plain-Python helper with no SPH-specific math in it |
| `getCachedDummyTensor`/`getCachedIdentityMatrices`/`clearDummyTensorCache`/`getCachedWarpArray`/`clearWarpArrayCache`/`generateNeighborTestData` | stay in `util/wp_util.py` (trimmed down from its pre-split size now that cast/directionality/zero/prime have moved out) |
| `getParticle`/`getL_i`/`getVolume_i`/`getVolume_j`/`getGradH_i`/`getGradH_j`/`getCRK_i`(/`getCRK_j`) | `util/stateUtil.py` (renamed path, same file as `utils/stateUtil.py`, now reads `zero_like_warp` from `..math` instead of `.wp_util`) |
| `WarpFunctionWrapper`/`warpWrapper`/`launch_kernel`/`clearKernelArgsCache`/`StateAwareWarpFunction` | `util/wp_autograd.py` (renamed path) |
| `volumeToSupport`/`n_h_to_nH`/`volumeToSupport_warp`/`computePairwiseSupport` | `util/support.py` (renamed path) |
| `checkInputRenormalization`/`checkInputGradHTerms`/`checkInputVolume`/`checkInputCRK`/`checkQV`/`checkKinds` | `util/arg_check.py` (renamed path) |
| New, not a move: `vec1f`/`vec1h`/`vec1d`/`mat11f`/`mat11h`/`mat11d` — `wp.vector`/`wp.matrix` subclasses for the length-1/1×1 case | `math/wp_vec1.py` |

`util/__init__.py` re-exports all of the above (from `wp_util`, `wp_autograd`, `support`, `stateUtil`, `directionality`, `cast`) so external call sites doing `from .util import *` / `warpSPHCore.util.X` see one flat surface, same as `utils/__init__.py` did before it. `warpSPHCore/__init__.py` now does `from .util import *` (plus `from .radiusSearch import *`, replacing the old explicit `from .radius import (...)` re-export list) instead of the individual `from .utils.wp_util import (...)` / `from .utils.stateUtil import (...)` imports the previous section added — one line instead of several, at the cost of the hand-written `__all__` list in `__init__.py` now having a long run of commented-out entries (`# "radiusSearchCompactHashMap"`, `# 'zero_like_warp'`, etc.) that look stale but aren't: every one of those names still reaches `__all__` via `__all__.extend(radiusSearch.__all__)` / `__all__.extend(util.__all__)` at the bottom of the file. Verified directly rather than assumed: `[n for n in warpSPHCore.__all__ if not hasattr(warpSPHCore, n)]` returns empty.

No regressions found in this pass (unlike the two split/move commits above) — the mechanical rename kept every downstream `from ..util import *` / `from ..dataTypes import *` style import working because callers already went through package-level re-exports rather than reaching into `utils/wp_util.py` directly, and `math/__init__.py` picking up `zero_like_warp`/`getNextPrime` alongside the packages that already re-exported them meant no call site needed its own import changed.

Validation: all 64 pytest cases pass, `from warpSPHCore import *` succeeds with zero missing `__all__` names, and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

## Autograd Package Consolidation (2026-08-06, `4adfd27` "move autograd out")

The torch↔Warp launcher/autograd-bridge machinery — previously split across top-level `warp_state_util.py` (687 lines: `parseArguments`, `extractStateInfo`, `warpWrapper2`) and `util/wp_autograd.py` (330 lines: `WarpFunctionWrapper`, `StateAwareWarpFunction`, `launch_kernel`, `clearKernelArgsCache`) — is now one `autograd/` package, both source files deleted, split one-function(-group)-per-file:

| What | Now lives in |
| --- | --- |
| `extractStateInfo` | `autograd/arg_extract.py` |
| `parseArguments` | `autograd/arg_parse.py` |
| `warpWrapper2` | `autograd/wrapper.py` |
| `launch_kernel` | `autograd/launcher.py` |
| `StateAwareWarpFunction` (+ new `warpWrapperStateaware = StateAwareWarpFunction.apply` alias, mirroring the existing `warpWrapper = WarpFunctionWrapper.apply` pattern) | `autograd/stateAwareWarpFunction.py` |
| `WarpFunctionWrapper`/`warpWrapper` | `autograd/stateLessWarpFunction.py` |
| `getCachedDummyTensor`/`getCachedIdentityMatrices`/`clearDummyTensorCache`/`getCachedWarpArray`/`clearWarpArrayCache` (previously in `util/wp_util.py`) | `autograd/cache.py` |
| `checkInputRenormalization`/`checkInputGradHTerms`/`checkInputVolume`/`checkInputCRK`/`checkQV`/`checkKinds` (previously `util/arg_check.py`) | `autograd/arg_check.py` (renamed path, same contents — moved here because `arg_extract.py`/`arg_parse.py` are its only callers) |

`autograd/__init__.py` re-exports all of the above; `warpSPHCore/__init__.py` picks it up via `from .autograd import *` / `__all__.extend(autograd.__all__)`, appended after the `util`/`radiusSearch` extends rather than folded into the same explicit list — by this point in the file's history the hand-written `__all__` list is mostly commented-out placeholders (see the "Utils → Util" section above for why that's not actually broken) plus a handful of live entries, so a fifth `__all__.extend(...)` line was the path of least resistance rather than reconciling the list.

Also new, not a move: `pinv/wrapper.py`'s `pinv_warp(C, numNbrs)`, a single dimension-dispatching pseudo-inverse entry point (1×1 → `pinv1x1`, 2×2 → `pinv2x2_warpBackend`, else → the `torch.linalg.eigvals`/`torch.linalg.pinv` fallback with the eigenvalue-reordering logic). `renorm.py`'s `computeRenormalizationMatrices_` now calls this instead of inlining the same `if dim == 2: ... else: ...` branch itself — the branch itself is unchanged (commented out in place, not deleted, in `renorm.py`), just relocated to be reusable by any future caller that needs a plain pinv without going through the full renormalization pipeline. `pinv/threed.py` (a fully-commented, never-wired-up 3×3 SVD/pseudo-inverse sketch) was renamed to `pinv/wp_pinv3x3.py` in the same pass, still inert.

**This move shipped with one real, if low-impact, bug — a naming collision, not an import-list omission this time:**

* **A pre-existing top-level `autograd.py`** (a thin re-export shim for `WarpFunctionWrapper`/`warpWrapper`/`StateAwareWarpFunction`/`extractStateInfo`/`warpWrapper2`, present since before this whole restructuring streak started — the very first module-layout table in this document lists it as untouched by `73432b6`) **now shares its name with the new `autograd/` package this commit introduced.** A file and a same-named package can coexist on disk in the same directory; Python's default `FileFinder` resolves the package first, so `import warpSPHCore.autograd` silently returned the new package and the old file became permanently unreachable rather than raising an `ImportError` or a `RuntimeWarning`. That would have been enough on its own to make the file dead weight, but it was also independently broken: both of its own imports (`from .util.wp_autograd import (...)` and `from .warp_state_util import (...)`) pointed at files this same commit deleted. Found by grepping for the old module paths after the move (per the standing rule from the previous update) rather than by any test failing — nothing imported `warpSPHCore.autograd` as the *file* specifically (everything either used the package's new re-exports or the untouched `from .autograd import WarpFunctionWrapper` top-level convenience import, which already resolved to the package). Deleted.
* `wp_grad.py` (repo-root historical prototype) had the same two now-dead import paths as the deleted `autograd.py` (`from warpSPHCore.util.wp_autograd import *` and `from warpSPHCore.warp_state_util import warpWrapper2`) — repointed to `from warpSPHCore.autograd import *` / `from warpSPHCore.autograd import warpWrapper2`. A docstring in `coreOperations/wp_density.py` referencing `warp_state_util.py` for where `extractStateInfo` dispatches was updated to point at `autograd/arg_extract.py`.

Validation: all 64 pytest cases pass, `from warpSPHCore import *` succeeds with zero missing `__all__` names, and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

## Landing CRK's dual-path rework (2026-08-06)

`crk/crk_moments.py`, `crk/crk_volume.py`, and `crk/crk_density.py` were the last kernels in the repo still written in the pre-Phase-1 neighbor-list-only style (flat `(neighborList, neighborOffset, numNeighs)` arguments, no `useAdjacency` branch, no grid path). They compute the raw geometric moments / apparent-volume estimate / CRK-corrected consistency density that `crk/crk_terms.py`'s `computeCRKTermsWarp` solves into the CRK correction terms (A, B, gradA, gradB) — a separate call path from `warpOperation`/`WarpOperation` (CRK is not itself a `WarpOperation` member; callers precompute a `CRKState` via `computeCRKFactors` and pass it into `warpOperation(..., crkState=...)`), but otherwise a full instance of the same operator-kernel shape everything else in `coreOperations/` already migrated to. Ported using the exact recipe in "Migration recipe (pivoting an existing operator to this style)" above: `_Func_i` (per-neighbor physics, now reading `referenceState`/`domainState`/`correctionData` via `getParticle`/`getVolume_j`/`getCRK_i`/`getCRK_j` instead of flat arrays) → `_Func_Adjacency` (resolves query-point state once, branches `useAdjacency` per offset via `checkOffset` for the grid case) → one `wp.kernel` → a `_computeCRKX_stateBackend` Python entry point calling `warpWrapper2` directly, mirroring `coreOperations/wp_covariance.py` (the closest precedent: also solves for correction terms from a neighbor-summed matrix, also has a "divide once at the end, not per-offset" postprocessing step).

Two things worth recording:

* **The per-particle neighbor count needed by `computeCRKTermsWarp`'s low-neighbor-count fallback (`num_nbrs = adjacency.numNeighbors`) doesn't exist for grid/`CompactHashMap` traversal**, same gap `renorm.py`'s `computeRenormalizationMatrices` still has (see "Landing Covariance's dual-path rework" above, "Still requires an explicit `AdjacencyList`" bullet) — except here it's been closed rather than left open. `computeCRKMoments_Kernel` now returns an `output_numNeighbors` array (same pattern as `computeCovariance_Kernel`'s `outputNeighbors`: an `_Func_i`-level counter incremented once per neighbor that passes the directionality check, summed across offsets in `_Func_Adjacency`), and `crk_wrapper.py`'s `computeCRKFactors` passes that instead of `adjacency.numNeighbors` into `computeCRKTermsWarp`. Verified exactly equal (`torch.equal`, all 1024 particles) against `adjacency.numNeighbors` for the neighbor-list traversal case, so this is not a behavior change there — it just also works when `adjacency=None` or a `CompactHashMap`, which `adjacency.numNeighbors` never could. `computeCRKFactors`'s `NotImplementedError` for `adjacency is None or isinstance(adjacency, CompactHashMap)` is gone as a result — CRK factors can now be computed under grid traversal, not just neighbor-list.
* **Found while writing the neighbor-count verification above, not a regression from this migration:** `checkKinds` (`autograd/arg_check.py`) substitutes a length-1 dummy tensor for `queryKinds`/`referenceKinds` when they're `None` *even for `OperationDirection.AllToAll`* — but `checkDirectionality_j`'s `opInt == 9` (AllToAll's value) branch is `return queryKind != 2`, i.e. it genuinely reads kind values rather than trivially passing, unlike `opInt == 0`. Every migrated kernel's `if opInt != 0: if not checkDirectionality_j(referenceKinds[j], opInt): continue` guard therefore indexes that length-1 dummy array at `j` up to `N-1` whenever `ParticleState.kinds=None` under `AllToAll` — an out-of-bounds read, not a real "no kinds" no-op. Every fixture in this repo's actual test suite sets `kinds` explicitly (`conftest.py`'s `particle_case`, `scripts/operation_matrix.py`'s cases), so this has never been exercised by a real test; it only surfaced here because a hand-rolled verification script omitted `kinds` and produced a neighbor count silently off by a few per particle (root-caused by comparing against the `kinds`-supplied fixture pattern before concluding this, not by a crash — CPU/GPU array indexing here doesn't bounds-check by default). Not fixed here — out of scope for a traversal-style port, and every real call site already avoids it — but worth flagging before anyone builds a `ParticleState` with `kinds=None` under `AllToAll` and wonders why results are nondeterministic. **Fixed 2026-08-07** — see "RadiusSearch Package Split, Mechanical `__all__`, and the AllToAll-Kinds Fix" below.
* **Separately, found while gradient-checking this migration: CRK's A/B/gradA/gradB were never actually differentiable end-to-end, dual-path or not — two independent bugs, both now fixed.** `computeCRKFactors`'s only callers before this (`conftest.py`'s `crk_state()`, the notebooks, `scripts/operation_matrix.py`) never call `.backward()` on anything upstream of the CRK computation itself, which is why neither bug had ever been caught:
  1. `crk_terms.py`'s `computeCRKTermsWarp` used in-place `A[mask] = 1.0` / `gradA[mask, i] = 0.0` / etc. on tensors (`A`, `B`, `gradA`, `gradB`) that are themselves `StateAwareWarpFunction` autograd outputs. `Tensor.__setitem__` bumps the version counter unconditionally, independent of whether `mask` actually selects any elements, so `loss.backward()` through any pipeline touching `computeCRKFactors` raised `RuntimeError: ... modified by an inplace operation ... expected version 0 instead`, regardless of whether the fallback branch was materially exercised. Fixed by switching to `torch.where`.
  2. **The actual primary cause**, found by bisecting with a minimal standalone repro that isolated `computeCRKVolume_Func_Adjacency` from `crk_terms.py`, masking, and `torch.where` entirely (a raw `wp.Tape().backward()` on just that one `@wp.func`'s output was already `nan`, with `torch.autograd.set_detect_anomaly` pinning the failure to `StateAwareWarpFunctionBackward` for the volume kernel specifically): **Warp's adjoint for a *dynamic* `for` loop (`for o in range(numOffsets)`, `numOffsets` a runtime value, not a compile-time constant) that accumulates into a local via `+=` and then feeds that local into a nonlinear op (division) *inside the same `@wp.func`* produces NaN gradients.** `computeCRKVolume_Func_Adjacency` and `computeCRKDensity_Func_Adjacency` (`crk_volume.py` / `crk_density.py`) both used to return the already-divided value (`1/wsum`, `mDensity/vol1`) directly from the looped function — the exact shape that triggers it. No other migrated operator hit this: every other `_Func_Adjacency` (Gradient, Covariance, Density, ...) returns the raw loop-accumulated value with no further transform inside the same function: confirmed empirically by testing every structural piece in isolation (flat-array version of the same math: clean; struct/`getParticle`-based version with no loop: clean; the real function with the loop and the postprocessing together: NaN; the real function with the loop but returning the raw sum: clean) — it is a narrow, specific trigger, not a general "dynamic loops can't be differentiated" limitation. Fixed by returning `(wsum, masked)` / `(mDensity, vol1, masked)` from the looped function and applying the division one level up, in the `@wp.kernel`, outside the loop.

  Both fixes were independently necessary — fixing only one still left the other's failure mode able to surface. `scripts/debug_crk_backward.py` (a plain, non-gradcheck `.backward()` harness with `torch.autograd.set_detect_anomaly` on by default, printing every intermediate CRK-terms quantity) was built while chasing this and is kept as a regression-debugging tool/template. `scripts/gradcheck_crk_native.py` now passes both the forward dual-path parity check and `torch.autograd.gradcheck` (both traversal modes, both the well-conditioned line-of-7 case and the singular single-particle case) and is registered in `tests/operations/test_gradcheck_scripts.py`'s `GRADCHECK_SCRIPTS`, same as every other operator's script.

  Bug 2 is isolated to a ~20-line, warpSPHCore-independent minimal repro in `scripts/repro_warp_dynamic_loop_division.py` (same status/style as `scripts/repro_warp_grad_reentrancy.py` for the earlier AD-bridge reentrancy bugs — kept for an upstream warp-lang report and as a standing regression guard, not currently CI-gated). It follows Warp's own documented dynamic-loop workaround (["Limitations and Workarounds in Differentiability"](https://nvidia.github.io/warp/stable/user_guide/differentiability.html): move the loop body into its own `@wp.func`, consume the *result* from the caller) to the letter, then varies only *where* a post-loop op on the loop's accumulated value happens. Findings, precise enough to matter for any future dual-path kernel: a **linear** read of the accumulator (`total * constant`) is safe even inside the same `@wp.func` as the loop; a **nonlinear** read (division, squaring) inside that same function is not — division comes back `-inf`, squaring comes back a silently-wrong `0.0`, both consistent with the backward pass evaluating the nonlinear op's local derivative at an accumulator value of `0` (its pre-loop initial value) instead of its true final loop-accumulated value. Moving the nonlinear op to the *caller* (a different `@wp.func`/`@wp.kernel` scope) fixes it, which is exactly the `computeCRKVolume_Func_Adjacency`/`computeCRKDensity_Func_Adjacency` fix above. Status: not yet reported upstream.

Validation: all 65 pytest cases pass (the pre-existing 64 plus `gradcheck_crk_native.py`, now registered — including all 10 in `tests/operations/test_operations_crk_analytic.py`, unchanged), `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0, NA=42`), and a standalone forward-parity check (adjacency-list vs. `adjacency=None` vs. an explicit `CompactHashMap`, 1024 particles) shows the auto-built-`CompactHashMap` and explicit-`CompactHashMap` paths bit-exact and the neighbor-list-vs-grid paths agreeing to ~1e-6 (summation-order float noise, same ~1 ULP-class drift documented for Gradient's grid/adjacency parity above) across apparent-area, CRK-density, A, B, gradA, and gradB.

## Renormalization Grid-Mode Coverage (2026-08-06): a documented gap that was already closed in code, just untested — plus two real bugs found closing it

While auditing this plan against the repo, found that "Landing Covariance's dual-path rework" below (and `docs/lessons_learned.md`) both claimed `computeRenormalizationMatrices` (`renorm.py`) still required an explicit `AdjacencyList` and couldn't run under grid-mode traversal, same restriction as CRK before the port above. That claim was **stale as of the very first restructure commit (`73432b6`)**, not just outdated by the CRK work: `computeRenormalizationMatrices_` has always read its neighbor count from Covariance's own per-particle kernel output (`covarianceReturnNumNeighbors=True`), not `adjacency.numNeighbors` — the one thing that would have actually required an `AdjacencyList` — and the `NotImplementedError` guard in `computeRenormalizationMatrices` has been commented out (dead, not active) the whole time. Nobody had ever exercised it, though: `tests/operations/conftest.py`'s `renorm_state()` fixture always passes an explicit `AdjacencyList`, never `None`/`CompactHashMap`.

Writing `scripts/gradcheck_renorm_native.py` (forward-value parity + `torch.autograd.gradcheck`, all three traversal inputs, following the same recipe as `gradcheck_covariance_native.py`/`gradcheck_crk_native.py`) to actually exercise this surfaced two independent, real bugs — neither related to renorm's own logic, both in shared infrastructure nothing 1D-pinv-shaped had hit before:

* `pinv/wp_pinv1x1.py` (the dim=1 pseudo-inverse path `pinv_warp` dispatches to) referenced `wp.mat11f`/`wp.vec1f` — these don't exist on the `warp` module itself, only as warpSPHCore's own precision-specific subclasses (`math/wp_vec1.py`), and the reference was hardcoded to the float32 variant regardless of `warpSPHCore_PRECISION`. Every gradcheck script runs under forced `float64`, so this was a guaranteed `AttributeError` the instant anything called `pinv1x1` under test — it just so happened nothing had, since Covariance's own gradcheck script never routes through `pinv_warp` at all. Fixed by selecting the matching named subclass (`mat11f`/`mat11d`/`mat11h`) from `scalar_t` at import time, mirroring `wp_pinv2x2.py`'s dtype-generic pattern.
* `WarpFunctionWrapper.backward` (`autograd/stateLessWarpFunction.py`) only checked `isinstance(outputs_warp, list)` when seeding gradients for a multi-output warp function — not `(list, tuple)`, unlike its own `forward` method and unlike `StateAwareWarpFunction`'s equivalent check, both of which handle either. `launch_kernel` (`autograd/launcher.py`) returns a `tuple` for multi-output kernels, so `pinv1x1`'s backward pass (two outputs: the inverse and the eigenvalue) fell into the single-output branch and crashed with `AttributeError: 'tuple' object has no attribute 'grad'`. This bug is not specific to renorm or pinv1x1 — it's latent for *any* `warpWrapper`-wrapped (not `warpWrapperStateaware`-wrapped) multi-output function; `pinv1x1` is simply the only such function anything currently gradchecks. Fixed by widening the `isinstance` check to `(list, tuple)`.

With both fixed, `computeRenormalizationMatrices` passes gradcheck and bit-exact forward parity across `AdjacencyList`/`CompactHashMap`/`adjacency=None` for both the single-particle (singular-covariance fallback) and line-of-7 cases. `scripts/gradcheck_renorm_native.py` is registered in `GRADCHECK_SCRIPTS`. The stale `NotImplementedError`-restriction bullet in "Landing Covariance's dual-path rework" below and the corresponding claim in `docs/lessons_learned.md`'s "Architectural facts still true" section have both been corrected/struck through to point here. `renorm.py`'s own module docstring and the two dead commented-out lines (the stale `NotImplementedError` and the stale `# num_nbrs = adjacency.numNeighbors`) were also cleaned up in the same pass.

Not covered here, left open at the time: `pinv2x2_warpBackend` (the dim=2 path) is still not gradchecked at all — it doesn't go through `warpWrapper`/`launch_kernel`, just a raw `wp.launch` on cast tensors, so it isn't obviously even wired for autograd. That's a separate, pre-existing gap (noted in "Landing Covariance's dual-path rework" below) this pass didn't touch. **Closed 2026-08-07** — see "`pinv2x2_warpBackend` Ported to the Autograd Bridge, Gradchecked" below.

Validation: 66 pytest cases pass (the pre-existing 65 plus `gradcheck_renorm_native.py`), `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`).

## RadiusSearch Package Split, Mechanical `__all__`, and the AllToAll-Kinds Fix (2026-08-07, `6f0209b` "refactor round, also fixes the latent querykinds bug")

Another pure-restructuring pass, same shape as the six above, plus one real bug fix. Three module splits, one file merge, one systemic `__init__.py` rewrite:

| What | Now lives in |
| --- | --- |
| `radiusSearch/wp_compactHash.py` (1145 lines: hashing, sorting, indexing, neighbor counting/collection, the `radiusSearchCompactHashMap`/`buildCompactHashMap` entry points) | `radiusSearch/compactHash/` package, split by concern: `buildHashmap.py` (`buildCompactHashMap`), `wrapper.py` (`radiusSearchCompactHashMap`/`radiusSearchCompactHashMap_`), `search.py` (`radiusSearchOnCompactHashMap`), `wp_collectNeighbors.py`/`wp_countNeighbors.py` (the two neighbor-enumeration kernels), `wp_hashCells.py`/`wp_index.py` (cell hashing/indexing `wp.func`s), `sort.py` (`sortReferenceParticles`), `grid.py` (`computeGridSupport`/`getDomainExtents`/`compute_h`), `morton.py`/`indexing.py` (Z-order and linear-index helpers). `compactHash/__init__.py` re-exports `buildCompactHashMap`/`radiusSearchCompactHashMap`/`radiusSearchCompactHashMap_`, same public surface as before. |
| `radiusSearch/verlet.py` (436 lines) | `radiusSearch/verlet/` package: `build.py` (`buildVerletList`), `filter.py` (`filterVerletList`/`updateNeighborsVerlet`), `wp_countVerlet.py`/`wp_updateVerlet.py` (the two Verlet-list kernels), `validCheck.py`/`util.py` (validity-metric and minimum-image-delta helpers). `verlet/__init__.py` re-exports the same three public names as before. |
| `radiusSearch/radius_naive.py` | `radiusSearch/naive/radius_naive.py` (rename only, same contents; `radiusSearch/__init__.py`'s import path updated to match) |
| `radiusSearch/wp_radius_small.py` | `radiusSearch/small/wp_radius_small.py` (rename only) — **but unlike the other three moves, nothing re-exports it: `radiusSearch/__init__.py` never imports `small/`, and `radiusSearch.__all__` doesn't mention `warp_radius_search_small`.** Its only caller, top-level `demo_util.py`, was rewritten in this same commit from an explicit `from warpSPHCore.radiusSearch.wp_radius_small import warp_radius_search_small` to a blanket `from warpSPHCore import *` and no longer calls the function at all (confirmed by grep — zero remaining references anywhere in the repo). Not broken, just orphaned: the module still imports cleanly on its own, it's simply unreachable from `warpSPHCore.*`. Worth a decision next time this area is touched — wire it back into `radiusSearch/__init__.py`'s exports, or delete it as dead weight the way `diffusion/` was. |
| Top-level `types.py` (`vec_t`, `mat_t`, `int_t`, `vecArray_t`, `matArray_t`, `intArray_t`, `scalarArray_t`, `scalar()`) | merged into `type_config.py`, which already held `scalar_t`/`dim_t`/`get_type_config`/`get_precision`/`get_dim`/`get_torch_precision`/`to_torch`/`to_numpy` — one module for "what type are we running as" instead of two. `types.py` deleted; every `from ..types import *` across `coreOperations/`/`crk/`/etc. repointed to `from ..type_config import *`. |

**`warpSPHCore/__init__.py`'s hand-maintained `__all__` list — the thing that caused three separate stale-re-export incidents across the last four restructuring passes (`a1628a4`'s `eval_kernelScale`, `e5d7177`'s `getParticle`-family, and the general pattern called out at the end of "Data Types / State Split" above) — is gone, replaced with a mechanical loop:**

```python
submodules = []
from .type_config import *
submodules.append(type_config)
from .dataTypes import *
submodules.append(dataTypes)
# ... one pair per subpackage ...
__all__ = []
for submodule in submodules:
    __all__.extend(getattr(submodule, '__all__', []))
```

Every subpackage/module re-exported at the top level (`type_config`, `dataTypes`, `enumTypes`, `util`, `math`, `autograd`, `kernels`, `radiusSearch`, `operations`, `crk`, `renorm`, `pinv`) now needs only its own correct `__all__` — `operations.py` and `renorm.py` gained explicit `__all__` lists in this same commit for exactly this reason, since neither had one before (they used to be pulled in by name in the old hand-written list). There is no longer a second list to keep in sync by hand, which closes this entire bug class going forward rather than just fixing the latest instance of it. Verified directly, same check as every prior pass: `[n for n in warpSPHCore.__all__ if not hasattr(warpSPHCore, n)]` returns empty (145 names total).

**The fix referenced in this commit's own message — `ParticleState(kinds=None)` under `OperationDirection.AllToAll` being an out-of-bounds read (documented as a "smaller open item" above and in "Landing CRK's dual-path rework") is now closed.** `checkKinds` (`autograd/arg_check.py`) used to substitute a length-1 dummy tensor for a missing `queryKinds`/`referenceKinds` unconditionally; `checkDirectionality_j`'s `AllToAll` branch genuinely indexes that array by neighbor index (`referenceKinds[j]` for `j` up to `N-1`), so the length-1 stand-in was read out of bounds the moment `AllToAll` was combined with `kinds=None`. Fixed by sizing the dummy tensor to the real particle count instead: `checkKinds` now takes `queryNumParticles`/`referenceNumParticles` and, for `AllToAll`, builds `getCachedDummyTensor((queryNumParticles,), ...)` / `((referenceNumParticles,), ...)` rather than `((1,), ...)`. `extractStateInfo` (`autograd/arg_extract.py`, the only caller on the live call path) already had `qPos.shape[0]`/`rPos.shape[0]` in scope and now passes them through. Verified directly here (not just by re-reading the diff): a 529-particle 2D `ParticleState(kinds=None)` Density call under `OperationDirection.AllToAll` now runs to completion with uniform, non-NaN output, where before this fix it would have read past the end of a length-1 array.

One loose end found while verifying this, not fixed here because it's unreachable: `autograd/arg_parse.py`'s `parseArguments` — the pre-`extractStateInfo` single-shot version of this same logic, kept as a public re-export (`autograd/__init__.py`) but not called anywhere in the codebase (confirmed by grep; `extractStateInfo`/`arg_extract.py` fully superseded it, see "Autograd Package Consolidation" above) — still calls `checkKinds(operationMode, device, queryKinds, referenceKinds)` without the new `queryNumParticles`/`referenceNumParticles` arguments, so it would still build a length-1 dummy under `AllToAll` if anything ever called it. Not a live bug today, but worth fixing (or deleting `parseArguments` outright, since nothing uses it) before anyone reaches for it.

**Bug found and fixed while verifying this pass, unrelated to the restructuring's own diff:** `scripts/_gradcheck_common.py` (imported by all ten `gradcheck_*.py` scripts) still did `from warpSPHCore.radiusSearch.compactHash.wp_compactHash import buildCompactHashMap` — a path that never existed under the new layout (`wp_compactHash.py`'s contents landed in `compactHash/buildHashmap.py`, not a same-named file one level deeper). This broke `tests/operations/test_gradcheck_scripts.py`'s entire parametrized suite (10/10 cases) with `ModuleNotFoundError`, even though nothing in this commit's own diff touched that script — it's the same failure class flagged after every prior restructuring pass ("grep for the old module paths after the move," per the standing rule from "Autograd Package Consolidation" above), just this time in `scripts/`, which isn't covered by the mechanical `from ..dataTypes import *`-style package re-exports that protected everything under `src/`. Fixed by importing `buildCompactHashMap` from top-level `warpSPHCore` instead (already re-exported there via `radiusSearch.__all__`), matching how the same file already imports `radiusSearchCompactHashMap`.

Validation: all 66 pytest cases pass (the 56 non-gradcheck cases were unaffected by the `_gradcheck_common.py` bug; all 10 gradcheck cases pass once that import was fixed), `from warpSPHCore import *` succeeds with zero missing `__all__` names, and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0, NA=42` — the 42 `NA` cells are exactly the expected `Laplacian[.../Dot]` scalar-field cases in `dim>1`, see the reclassified item below, not a regression).

## `pinv2x2_warpBackend` Ported to the Autograd Bridge, Gradchecked (2026-08-07)

Closes the last "smaller open item" from "What's Next" above: `pinv2x2_warpBackend` (`pinv/wp_pinv2x2.py`, the dim=2 pseudo-inverse `pinv_warp` dispatches to for every 2D renormalization call) used to do a raw `wp.launch` directly on tensors converted via `castTorchToWarpAsBuiltins` — no `torch.autograd.Function` wrapping it at all, unlike `pinv1x1` (`warpWrapper`-wrapped `launch_kernel` call, since "Renormalization Grid-Mode Coverage" above). That meant it had no backward pass to gradcheck in the first place, not just untested — any `loss.backward()` through a 2D renormalization call was silently relying on `pinv2x2_warpBackend`'s output being treated as a graph leaf with no gradient contribution from `C`, which is wrong (the pseudo-inverse's Jacobian w.r.t. its input is nonzero and needed).

Ported to the same pattern `pinv1x1` uses:

* `pinv2x2_warp`'s `wp.kernel` parameter order changed from `(C, L, EV, num_nbrs)` to `(C, num_nbrs, L, EV)` — inputs first, outputs last — to match `launch_kernel`'s `kernel_inputs = inputs + outputs` assembly convention (`autograd/launcher.py`). Kernel body unchanged.
* `pinv2x2_warpBackend` now calls `warpWrapper(launch_kernel, pinv2x2_warp, outputSize, (_mat22_t, _vec2_t), C, num_nbrs)` instead of building `wp.array`s by hand and calling `wp.launch` directly. `_mat22_t`/`_vec2_t` are picked from `scalar_t` at import time (`wp.mat22f`/`wp.mat22d`/`wp.mat22h`, `wp.vec2f`/`wp.vec2d`/`wp.vec2h`) — unlike the 1×1 case, Warp already ships named 2×2/vec2 subclasses per precision, so no new `math/wp_vec1.py`-style type file was needed, just the same dispatch-by-`scalar_t` pattern `wp_pinv1x1.py` established.
* Public signature and return value (`pinv2x2_warpBackend(C, num_nbrs) -> (inv, evs)`) unchanged, so `pinv/wrapper.py`'s `pinv_warp` and `renorm.py`'s only caller needed no changes.

`scripts/gradcheck_pinv_native.py` (new) gradchecks `pinv1x1` and `pinv2x2_warpBackend` directly as pure matrix→(inverse, eigenvalues) functions, independent of the renormalization pipeline — `pinv1x1` had also never been gradchecked in isolation before this (only indirectly, via `gradcheck_renorm_native.py`'s dim=1 cases, which exercise it but don't gradcheck it as its own function). Three cases: `pinv1x1` on three 1×1 matrices away from its zero-fallback threshold; `pinv2x2_warpBackend` on two well-conditioned, deliberately *non*-symmetric 2×2 matrices (`num_nbrs>=4`, so the eigendecomposition branch runs, not the low-neighbor identity fallback) — non-symmetric on purpose, since the kernel symmetrizes `b = 0.5*(C01+C10)` internally and gradcheck's numerical Jacobian should confirm both `C01` and `C10` each get a correctly-weighted gradient contribution back, not just the symmetric-input case; and a direct check that the `num_nbrs<4` fallback branch produces an exact identity output with an exact zero gradient (not just small), confirming the branch doesn't leak a spurious adjoint. Registered in `GRADCHECK_SCRIPTS` (`tests/operations/test_gradcheck_scripts.py`).

Also verified end-to-end (beyond the isolated gradcheck): a full `computeRenormalizationMatrices` call on a 256-particle regular 2D grid, with `positions` and `densities` as leaves and `loss.backward()` run through `renormalizationMatrices`, produces finite gradients (`positions.grad` has no NaNs) and a sane near-identity `L` for a locally regular neighborhood — confirming the fix holds through the full pipeline, not just the direct-call gradcheck.

Validation: all 67 pytest cases pass (the pre-existing 66 plus `gradcheck_pinv_native.py`, now registered), and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0, NA=42`, unchanged — this fix only affects the backward pass, which the forward-value matrix sweep doesn't exercise).

## Kernel-Properties Struct ABI: `kernelState` Replaces the Seven Flat Scalars (2026-08-07)

The canonical structured kernel ABI's trailing seven scalars (`mode_uint`, `kernel_int`, `gradientMode_int`, `laplacianMode_int`, `positiveDivergence_int`, `divergenceMode_int`, `opInt`) — present in every `@wp.kernel`/`@wp.func` signature described throughout this file up to this point — are now collapsed into a single `kernelState` `wp.struct` (`dataTypes/kernelState_t.py`: `kernelFunction`, `supportMode`, `gradientMode`, `laplacianMode`, `positiveDivergenceMode` (`wp.bool`), `divergenceMode` (`wp.bool`), `operationMode`), passed as one `kernelProperties: kernelState` argument in place of that septet. This is groundwork for the same reason every other struct-ification in this document happened: argument order stops being load-bearing for same-typed scalar parameters, ahead of the larger `warpier_core` architecture this file tracks. Landed across every low-level kernel-shape `@wp.func` (`kernels/{kernel,gradient,derivative,gradH,hessian,laplacian}.py`), all three CRK kernels (`crk/{crk_density,crk_volume,crk_moments}.py`), and all seven `coreOperations/wp_*.py` operators; `autograd/arg_extract.py`'s `build_fn` now constructs one `kernelState()` per launch (same per-call-rebuild pattern already used for `domState`/`qPart`/`corrState`) instead of returning the seven scalars separately in the kernel-args tuple. `checkDirectionality_i`/`_j` (`util/directionality.py`) intentionally still take a raw `opInt: wp.int32` parameter — callers now pass `kernelProperties.operationMode` at the call site instead of threading a separate `opInt` alongside the struct. `autograd/arg_parse.py` (`parseArguments`) was deliberately left untouched: it's dead code, confirmed unreferenced anywhere in the repo, already missing three of the seven scalars before this change (see "RadiusSearch Package Split" above).

Also fixed in the same pass, found finishing (not committed) work already sitting in the tree, not part of the struct-ABI change's own design: `math/wp_distance.py`'s `mod_distance` had been wrongly converted to take a whole `domainData` struct with a hardcoded `[0]` index — reverted to its original per-dimension-scalar signature, since it operates on one already-indexed dimension, not a domain-wide array trio, so struct-wrapping doesn't apply to it. `minimumImageDistance` was calling itself with a missing argument after its own signature had changed — fixed by re-adding an explicit `D: wp.int32` parameter (sourced from `domainState.dim` at its two call sites), deliberately *not* `x.length` read inside the loop bound: older Warp versions have silently treated vector `.length`/matrix `.size` as zero when used to drive a dynamic for-loop's range, the same *class* of dynamic-loop adjoint fragility as "Landing CRK's dual-path rework"'s NaN-gradient bug above, even though a quick isolated test against the installed Warp 1.15.0 didn't reproduce it for this specific shape. Several `radiusSearch`/`util` call sites outside the operator/kernel scope (`verlet/wp_updateVerlet.py`, `verlet/wp_countVerlet.py`, `util/support.py`'s `isInSupport`, `compactHash/wp_countNeighbors.py`, `compactHash/wp_collectNeighbors.py`, `radiusSearch/small/wp_radius_small.py`) broke because `computeDistanceVec`/`computeCartesianDistance`'s signatures now require a `domainData` struct — fixed by constructing a local `domainData` struct inline from each call site's existing raw `periodicity`/`domainMin`/`domainMax` array parameters, without changing those functions' own outer signatures (out of scope for this pass).

~~One loose end left open, not fixed~~ — **resolved 2026-08-07, confirmed by the repo owner**: `kernels/gradient.py`'s `sphKernelGradient` (the `xi`/`xj` variant) picked up a `/scalar_t(2.0)` in its `SuperSymmetric` branch that the pre-refactor code didn't have, bringing it into agreement with its twin `sphKernelGradient_ij` (the precomputed-`xij` variant), which already had that division beforehand. This was an intentional correctness fix, not an accidental regression, and has been reported upstream as warp-lang issue #1740.

Validation: all 67 pytest cases pass (including every `gradcheck_*_native.py` script), `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`), `from warpSPHCore import *` exposes every `__all__` name.

## `OperationDirection.AllToAll` vs `TrueAllToToAll`: Ghost Particles Were Leaking Into Every Directional Op (2026-08-07)

Every migrated operator's `_Func_i`/`_Func_Adjacency` guards directionality with `if kernelProperties.operationMode != <all-to-all sentinel>: checkDirectionality_i/_j(...)` — i.e. the check is *skipped entirely* when `operationMode` equals the sentinel value, on the assumption that "all to all" means no masking at all. That sentinel used to be `OperationDirection.AllToAll = 0`, which is also `OperationProperties.operationMode`'s default (`dataTypes/properties_t.py`) — so by default, every operator's directionality check was a no-op.

That's a real bug, not just a naming inconsistency: in an SPH simulation, "all to all" means fluid and boundary particles interact with each other and with themselves, but ghost particles are *not* supposed to influence either — i.e. `AllToAll` should still exclude ghosts, not skip masking outright. `checkDirectionality_j`'s own `elif opInt == 9: # All to all -> return queryKind != 2` branch already implemented exactly that ghost-exclusion semantic — but it was unreachable, since nothing ever set `operationMode` to the enum value 9 that would route into it (every caller used the value-0 `AllToAll` sentinel instead, which skipped the check before it could reach that branch).

Fixed by splitting the two concepts `enumTypes.py`'s `OperationDirection` had conflated under one name: `TrueAllToToAll = 0` is now the "skip masking entirely, ghosts included" sentinel (kept at value 0 so every `!= 0`-shaped guard across the codebase — now written as `!= wp.static(OperationDirection.TrueAllToToAll.value)` — keeps working unchanged), and `AllToAll = 9` is the real "fluid + boundary interact, ghosts excluded" mode, now reachable and routing into `checkDirectionality_j`'s pre-existing `queryKind != 2` branch as intended. Every kernel-level guard in `kernels/`, `crk/`, and `coreOperations/` — previously written against the old `AllToAll` name — was repointed to `TrueAllToToAll`; `OperationProperties.operationMode`'s default (`dataTypes/properties_t.py`), `operations.py`, `crk/crk_wrapper.py`, and `autograd/arg_check.py`'s `checkKinds` still reference `OperationDirection.AllToAll` (now value 9) unchanged, since those are the genuine user-facing default and are unaffected by the sentinel-value split.

Validation: all 67 pytest cases pass with this change applied on top of the `kernelState` struct-ABI work above, confirming the two are compatible.

## Per-Particle State Structs (`WarpParticle_D`/`ParticleCorrectionData_D`) and `getIndexRange`: Landing Across `coreOperations` and `crk` (2026-08-07)

One layer under the `kernelState`/`domainData` struct-ification above: every operator's `_Func_i` used to take the query particle's `xi`/`hi`/`mi`/`rhoi` (from `getParticle`, a plain tuple) and its correction terms (`useGradientRenormalization`/`Li`/`useGradHTerms`/`omega_i`/`useVolume`/`Vi`/`useCRK`/`Ai`/`Bi`/`gradAi`/`gradBi`, individually extracted in `_Func_Adjacency` via `getL_i`/`getGradH_i`/`getVolume_i`/`getCRK_i`) as roughly fifteen separate positional arguments. Two new per-particle `wp.struct` families close that gap, dimension-specialized the same way `particleDataSoA_D`/`correctionData_D` already are:

* `WarpParticle_1/2/3` (`dataTypes/particleData.py`: `position`, `support`, `mass`, `density`, `kind`) and `getParticleData(SoA, i)` (`util/stateUtil.py`, overloaded per `particleDataSoA_D`) — a single-particle view over the existing SoA structs, replacing `getParticle`'s tuple return. Call sites read `iPtcl.position`/`.support`/`.mass`/`.density`/`.kind` (and `jPtcl.*` for the per-neighbor particle inside the loop) instead of unpacking `xi, hi, mi, rhoi, ki = getParticle(...)`.
* `ParticleCorrectionData_1/2/3` (`dataTypes/corrections_t.py`: `useGradientRenormalization`, `renormalizationMatrix`, `useVolume`, `volume`, `useGradHTerms`, `omega`, `useCRK`, `A`, `B`, `gradA`, `gradB`) and `getParticleCorrectionData_i`/`_j` (`util/stateUtil.py`, overloaded per `correctionData_D`) — bundles the query particle's correction terms (built once per `_Func_Adjacency` call via the existing `getL_i`/`getVolume_i`/`getGradH_i`/`getCRK_i` accessors) into one struct threaded down to `_Func_i` as `iCorrectionData`, instead of five separate `useX`/value pairs.
* `getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)` (`radiusSearch/grid_util.py`) collapses the `if useAdjacency: ... else: checkOffset(...)` branch every `_Func_Adjacency` duplicated into one call, returning `(beginIndex, numIndices)`; the caller still does its own `if beginIndex < 0: continue`.

Landed first as a pilot on `coreOperations/wp_gradient.py` alone, verified with `scripts/gradcheck_gradient_native.py` and the `gradient`-filtered pytest subset, then propagated mechanically to the remaining six `coreOperations/wp_*.py` files (density, interpolate, divergence, curl, covariance, laplacian) and finally to all three CRK kernels (`crk/{crk_density,crk_volume,crk_moments}.py`) — the same `iPtcl`/`jPtcl`/`iCorrectionData`/`correctionData`/`getIndexRange` shape now covers every per-neighbor kernel in the repo, not just the SPH operator family.

Two things worth a closer look before extending this pattern further:

* ~~`getParticleCorrectionData_j` is unused~~ — **not a gap, by design (clarified 2026-08-07).** Nothing in-tree calls it, but it's kept as part of the public per-particle accessor surface for downstream library users who want the bundled `_j`-side struct rather than the individual `getVolume_j`/`getGradH_j`/`getCRK_j` accessors. No change needed.
* ~~`ParticleCorrectionData_D` duplicates the four `useX` flags~~ — **stale claim, corrected 2026-08-07.** Re-checked `dataTypes/corrections_t.py` directly: `ParticleCorrectionData_1/2/3` only ever had `renormalizationMatrix`/`volume`/`omega`/`A`/`B`/`gradA`/`gradB` — there are no `useX` fields on the per-particle struct to duplicate `correctionData`'s broadcast flags. This bullet kept getting repeated across reviews (including a fresh one) without anyone re-reading the current struct definition; it's struck here rather than left to propagate again.

## Pre-Phase-3 Cleanup Pass (2026-08-07)

A full-repo survey ahead of starting Phase 3 (`Field`) turned up several loose ends, some cosmetic, two genuinely load-bearing. Verified against the actual repo state (imports, `pytest`, direct function calls), not just against this document's own narrative — which is itself part of what this pass found.

* **`diffusion/viscosity.py` had come back from the dead, broken.** "Diffusion Removal" above documents it being deleted outright in `8aa79fb`. It reappeared at 344 lines in `6ccc665` ("refactor kernel ints into a combined kernelstate") — not a real restoration, an accidental one: a stale editor tab for the file, still open from before the delete, got saved back into the working tree during unrelated work. The resurrected version doesn't even parse as valid Python as a kernel signature (`kernelProperties.kernelFunction : wp.int32` as a parameter name) and imports three modules that no longer exist (`..utils.wp_autograd`, `..types`, `.util`). Nothing imports the file, so it never broke `import warpSPHCore` or any test — it just sat there. Deleted again, for good this time (nothing else references it besides this document's own history and a stale `.egg-info/SOURCES.txt` build artifact, both harmless).
* **`autograd/arg_parse.py` (`parseArguments`) deleted.** Confirmed zero in-tree callers (superseded by `extractStateInfo`/`arg_extract.py` since "Autograd Package Consolidation" above) and confirmed still-broken (its `checkKinds` call was never updated for the `queryNumParticles`/`referenceNumParticles` args added by the AllToAll-kinds fix in "RadiusSearch Package Split" above, so calling it would silently reintroduce that exact out-of-bounds read). A dead function that's also broken is worse than no function; removed rather than fixed. The two docstrings that referenced it by name (`arg_extract.py`, `autograd/wrapper.py`) were reworded instead of left dangling.
* **`radiusSearch/small/wp_radius_small.py` (`warp_radius_search_small`) re-exported from `radiusSearch/__init__.py` — and two real bugs fixed to make that safe.** This module was orphaned by the `radiusSearch` package split ("RadiusSearch Package Split" above), which claimed at the time that it "still imports cleanly on its own" — also stale, it turns out: it was missing `import torch` (used by `warp_radius_search_small` itself) and `scalar_t` (used inside both `@wp.kernel` signatures, which made the kernels fail to compile at all, `TypeError: unsupported callable`). Beyond the import gaps, its neighbor-threshold mode dispatch inside both kernels checked `mode == 1/2/4/5` with an inline comment claiming `0=gather,1=scatter,2=symmetric,3=superSymmetric` — but `SupportScheme`'s actual values are `Gather=11, Scatter=12, MeanSymmetric=13, KernelMeanSymmetric=14, SuperSymmetric=15, PartialSymmetric=16` (`enumTypes.py`), and `supportSchemeToUint` passes those values straight through. None of the `if/elif` branches could ever match, so `threshold` silently stayed at its `0.0` initializer for every call regardless of chosen scheme — the function returned only self-pairs (`dist <= 0.0`) and nothing else, for as long as it's existed since `SupportScheme` was renumbered into the 11–16 range (`a5d4bcb`, well before the restructuring streak). Confirmed empirically: before the fix, a 32-particle 2D self-search at support radius 0.2 returned exactly 32 edges (self only); after, ~156 (≈4.9 neighbors/particle, in line with the expected density). Fixed by replacing the duplicated, wrong dispatch in both kernels with a single call to the existing, correct `computePairwiseSupport(hx[i], hy[j], mode)` (`util/support.py`) — already used for exactly this purpose elsewhere, and now the single source of truth for support-scheme-to-threshold mapping instead of a second, drifted copy of it.
* **Removing `@torch.jit.script` from `util/support.py`'s `volumeToSupport` exposed a real, previously-latent bug — not caused by this pass, but surfaced by it.** Every other `@torch.jit.script` decorator in the repo was already commented out (from earlier work, all confirmed still-working plain functions). `volumeToSupport` was the one exception, commented out in this same session as part of a broader jit.script cleanup (Python 3.14 support for `torch.jit.script`/`script_method` is gone; no `torch.compile` alternative works here since SPH's tensor shapes aren't static, which is one of the reasons this library is built on Warp rather than TorchScript/`torch.compile` in the first place). Its `dim==2`/`dim==3` branches called `torch.sqrt`/`torch.pow` on plain Python `float`/`int` arguments (the function's own signature is `volume: float, targetNeighbors: int, dim: int` — no tensors involved anywhere) — under TorchScript's compiled scalar-op dispatch this works, but the identical code called eagerly crashes immediately (`TypeError: sqrt(): argument 'input' must be Tensor, not float`). Since `volumeToSupport` sits underneath `generateNeighborTestData`, which every operator's shared `particle_case` pytest fixture depends on, this one-line decorator removal broke effectively the entire test suite (67/67 → 11/67, all `ERROR` not `FAIL`) the moment it landed, before this pass caught and fixed it. Fixed by switching to `math.sqrt`/`**(1/3)` — genuinely scalar math, matching what the function's own type annotations already promised. Re-verified all three `dim` branches directly and the full 67-case pytest suite (CPU and CUDA parametrizations both) passes clean again. **Worth internalizing as a rule, not just a one-off fix: `@torch.jit.script` silently tolerates some scalar/Tensor mixing that eager PyTorch does not, so every remaining/future decorator removal in this codebase needs the same eager-mode smoke check this one got, not just "does it still import."**
* **The `kernels/gradient.py` `SuperSymmetric`/`2.0` discrepancy flagged as unconfirmed in "Kernel-Properties Struct ABI" above is resolved** — confirmed by the repo owner as an intentional correctness fix (bringing `sphKernelGradient` into agreement with `sphKernelGradient_ij`, which already had the division), and the underlying Warp behavior that motivated it has been filed upstream as warp-lang issue #1740.
* **The global `scalar_t`/`dim_t` precision singleton (`type_config.py`) is an intentional, permanent design tradeoff, not an oversight** — worth recording here since it kept getting flagged as a design smell without the rationale being written down anywhere. Per the repo owner: some Warp versions don't fully recompile kernels when the backing global type changes mid-process, and using `dim_t=Any`/generic dtypes throughout would make casting inside fully-generic kernels very difficult given current Warp limitations. A single precision/dimension resolved once at import time was judged a reasonable compromise for now, revisitable if Warp's own constraints change. `Field` (Phase 3) can still track per-field dtype metadata (including richer Warp dtypes like `mat33f` that don't fit `scalar_t` at all) — it just shouldn't assume that metadata can *change* the underlying kernel type within a single process run.
* **Static analysis remains an open, separate follow-up — not attempted in this pass.** Flagged during the survey (no ruff/mypy/pyright anywhere in the repo, despite a long history of exactly the mechanically-catchable bugs a linter would flag, several of them found again in this very pass). Not a quick add: Warp's parenthesized generic types (`vector(dtype=..., length=...)`, `matrix(shape=..., dtype=...)`) trip a real type checker's "call expression not allowed in type expression," and `scalar_t` is simultaneously a runtime value and a type annotation throughout the codebase, which type checkers don't tolerate well either. A plain non-type-checking linter (`ruff` in lint-only mode) or even just `python -m py_compile` across `scripts/`/root-level scripts in CI would sidestep the Warp-typing problem entirely and still catch the import-path-rot and orphaned-module bugs that have bitten this repo repeatedly — but picking and configuring that tool is real, distinct work, tracked here as a to-do rather than folded into this cleanup.

Validation for this pass: all 67 pytest cases pass (`pytest tests/`, CPU+CUDA parametrizations), `from warpSPHCore import *` exposes every `__all__` name with zero missing, and `warp_radius_search_small`/`radiusNaive`/`radiusNaiveFixed`/`mod` were each smoke-tested directly (not just import-checked) after their respective fixes.

## Notebook Corpus Status

All operation-relevant notebooks are ported and accounted for in root (`warp_density`, `warp_interpolate`, `warp_gradient`, `warp_divergence`, `warp_laplacian`, `warp_curl`, `warp_renorm`, `warp_custom`, `warp_profile`). Grid dispatch and CRK checks don't need their own notebooks — see `docs/lessons_learned.md`'s "Notebook/documentation conventions". `docs/regression/notebook_test_matrix.md` has the notebook-to-test mapping.

## Already in Place

* A high-level semantic interface exists through `warpOperation` and state objects (`ParticleState`, `OperationProperties`).
* State-aware autograd infrastructure exists (`extractStateInfo`, `warpWrapper2`, `StateAwareWarpFunction`).
* Conversion hot paths use caching for non-differentiable data (dummy tensors, dtype caches). Differentiable Warp-array caching (`getCachedWarpArray` and its `wp_autograd.py` mirrors) was deliberately removed, not fixed — see `docs/lessons_learned.md` for why that class of caching is unsafe here.
* A structured kernel ABI is demonstrated by every operator, Covariance included: Covariance is now the seventh operation dispatched through `warpOperation`/`operations.py` (`operationProperties.operation = WarpOperation.Covariance`), exactly like Density/Interpolate/Gradient/Divergence/Curl/Laplacian — not a standalone `computeCovarianceMatrix` function anymore (see the update at the end of "Landing Covariance's dual-path rework" below). Its kernel is `coreOperations/wp_covariance.py`, dual-path (grid + adjacency) and gradchecked (`scripts/gradcheck_covariance_native.py`) the same as the six migrated operators. The pseudo-inverse (`pinv/wp_pinv2x2.py`) and the covariance-to-renormalization-matrix postprocessing (top-level `renorm.py`) remain separate from the covariance kernel file.

## Gaps Against the Target (closed items, kept for history)

* ~~`warpOperation` still routes through `sphOperation_warp`~~ — **done, and inverted**. `warpOperation` now dispatches directly from state objects to each operator's `_computeSPHX_stateBackend`; `sphOperation_warp` (the flat-tensor "manual" entry point) assembles the same state objects and calls `warpOperation` instead of the other way around. See "States as the Primary Path" below.
* ~~All six operators launch via a structured wrapper; the flat path and `operations_grid/` are gone~~ — **done**. See "Working Prototype → Production" below.
* ~~Adjacency and grid execution paths are still largely duplicated across operation families~~ — **done**. Every operator's traversal now lives in one `_Func_Adjacency`/`wp.kernel` pair that branches on `useAdjacency` at runtime; `operations_grid/` has been deleted.
* ~~CRK and renormalization corrections cannot run on grid-mode traversal~~ — **done, both at the kernel level and the Python entry-point level, as of 2026-08-06.** Every operator's `correctionData` struct has threaded CRK/grad-h/renorm through the grid traversal branch for free since Phase 1/5 landed (no separate grid kernel to wire the correction paths into). What was still gated at the *entry-point* level — `computeCRKFactors` and `computeRenormalizationMatrices` each raising/appearing to require an explicit `AdjacencyList` — is now also closed: CRK by the port in "Landing CRK's dual-path rework", renormalization by the test-coverage pass in "Renormalization Grid-Mode Coverage" (its restriction turned out to already be dead code, just never exercised).

## What's Next

The big remaining piece, in priority order — no re-investigation needed, this is the actual next step:

1. **Phase 3: the `Field` abstraction.** Phases 1 and 5 (structured kernel ABI, traversal consolidation) are done for all 7 operators (Density/Interpolate/Gradient/Divergence/Curl/Laplacian/Covariance) plus CRK. Phase 2 (state consolidation) is informally done — state objects are already semantic (`ParticleState`, `CRKState`, etc.), just torch-native rather than owning a synchronized torch+warp pair. What's not started: `Field` (torch view + cached warp view + dtype/device/shape metadata + dirty/sync flags), which is the prerequisite for both Phase 4 (stop rebuilding kernel structs from scratch in `extractStateInfo` on every call) and Phase 6/7 (forward-mode AD, which needs somewhere to hang tangent storage). Concrete starting point per the plan's own Step 2: a minimal `Field` class with a torch-compatible public surface and a legacy fallback conversion path, wrapping lazily — start by profiling how much of `extractStateInfo`'s per-call cost (`autograd/arg_extract.py`) is actually the repeated `wp.from_torch()`/struct-population work `Field` is meant to eliminate, since that's the concrete win Phase 3 buys and it's worth confirming the size of the win before committing to the abstraction's design.
2. Forward-mode AD (Phase 6/7) — do not start before (1). It's designed to be "extend `Field` with a tangent slot," not a kernel-by-kernel rewrite; starting it against the current torch-native state objects would mean redoing the work once `Field` lands. Note: Tier 1 (the value-only JVP slice, `warpOperationJVP`) needed no `Field`/struct work at all and has already landed against the current torch-native state objects (`warpier_forward_mode_plan.md` Phase 2, 2026-08-18) — this item is about the Tier-2 (position/support/mass/density tangent) dual-struct work, which is the piece that does need `Field`.

Smaller open items, independent of the above, each already root-caused (no re-investigation needed, just implementation):

* ~~`ParticleState(kinds=None)` under `OperationDirection.AllToAll` is an out-of-bounds read~~ — **fixed 2026-08-07.** `checkKinds` (`autograd/arg_check.py`) now sizes its `AllToAll` dummy array to the real particle count (`queryNumParticles`/`referenceNumParticles`, threaded in from `extractStateInfo`) instead of a length-1 stand-in. See "RadiusSearch Package Split, Mechanical `__all__`, and the AllToAll-Kinds Fix" below and `docs/lessons_learned.md`. The loose end this bullet used to flag — `autograd/arg_parse.py`'s dead, never-updated `parseArguments` reintroducing the same bug if ever called — is moot as of "Pre-Phase-3 Cleanup Pass" below: `parseArguments` is deleted, not just unused.
* ~~`LaplacianScheme.Dot` doesn't support scalar fields in `dim>1` domains~~ — **reclassified 2026-08-07: by design, not a bug.** `LaplacianScheme.Dot` computes a dot product between the field quantity and the kernel gradient, which is only mathematically defined for a vector field matching the domain's spatial dimension — a scalar field has no such dot product to take. The explicit `ValueError` guard in `coreOperations/wp_laplacian.py` is therefore the intended, permanent behavior for that combination, not a stop-gap awaiting a "correct" scalar-field generalization; callers with scalar fields should use `LaplacianScheme.Naive`/`Brookshaw`/`Default` instead, per the error message. See `docs/lessons_learned.md`.
* ~~`pinv2x2_warpBackend` (the dim=2 pseudo-inverse) has no gradcheck coverage~~ — **fixed 2026-08-07.** It didn't go through the `warpWrapper` autograd bridge at all (raw `wp.launch` on cast tensors), so it had no backward pass to gradcheck in the first place. Ported to the same `warpWrapper`/`launch_kernel` pattern `pinv1x1` uses; `scripts/gradcheck_pinv_native.py` now gradchecks both `pinv1x1` and `pinv2x2_warpBackend` directly. See "`pinv2x2_warpBackend` Ported to the Autograd Bridge, Gradchecked" below.
* **Report the Warp dynamic-loop + nonlinear-op NaN-gradient bug upstream.** Root-caused and isolated in `scripts/repro_warp_dynamic_loop_division.py` (see "Landing CRK's dual-path rework" below for the exact trigger condition); worked around locally in `crk_volume.py`/`crk_density.py`, but not yet filed against warp-lang.
* **Add static analysis to the repo — genuinely new work, not cleanup.** No linter or type checker runs anywhere today (not in CI, not locally), despite a long history of exactly the bugs a linter catches mechanically (stale imports after a file move, missing `import`s, dead re-export lists) — several more found in "Pre-Phase-3 Cleanup Pass" below, in files nothing had touched in months. Needs a real tool decision, not just flipping a switch on: Warp's parenthesized generic types (`vector(dtype=..., length=...)`) and `scalar_t` being both a runtime value and a type annotation both trip up real type checkers, so `mypy`/`pyright` aren't a drop-in fit. A non-type-checking linter (`ruff` in lint-only mode) or `python -m py_compile` across `scripts/`/root scripts in CI sidesteps that and would still catch the import-rot class of bug. Tracked here as its own follow-up, deliberately not bundled into the cleanup pass below.

## Working Prototype → Production: Unified Kernel + Traversal — DONE for all six operators

**Historical, kept short.** `wp_grad.py` (repo root) started as a from-scratch prototype reimplementation of the Gradient operator, used to prove out one kernel serving both traversal modes (neighbor-list and grid) before that recipe was applied to every operator. Density, Interpolate, Gradient, Divergence, Curl, and Laplacian each ended up with exactly one unified kernel (`coreOperations/wp_<op>.py`) handling both traversal modes; `operations_grid/` (the old split-by-traversal package) was deleted entirely once every operator had migrated. The prototype itself and its migration-recipe steps are no longer needed as a reference — the migration is done everywhere, including downstream library users — so the step-by-step recipe and worked-example line numbers that used to live here have been removed rather than left to rot. `wp_grad.py` was deleted at commit `dd19188` ("finish refactor"); see that commit (and the ones just before it) in git history if the prototype's exact contents are ever needed again. See "Landing Interpolate", "Landing Density", and "Landing Divergence, Curl, and Laplacian" below for what differed operator-to-operator, and "Collapsing `sphOperation_warp`" for how the top-level dispatcher itself simplified once the last operator was exempted.

### Landing Gradient in production: what changed, and two bugs the port surfaced

Porting Gradient to this style turned up two real bugs — one pre-existing in the original prototype (invisible there because the prototype was only ever checked forward-value, never gradchecked), one introduced while porting. Both are exactly the class of bug the `docs/lessons_learned.md` testing-methodology lessons warn about: forward-only checks (the notebook, `operation_matrix.py`) cannot see either one, only `torch.autograd.gradcheck` can.

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

### Landing Covariance's dual-path rework: closing the renorm grid-traversal gap, and splitting the file three ways

Covariance (`renorm/wp_covariance.py`) was the *original* demonstration of the structured kernel ABI — the six operators above all copied its `queryState`/`referenceState`/`domainState`/`useAdjacency`/`adjacencyState`/`gridState`/`correctionData` shape, not the other way around — and its kernel (`computeCovariance_Func_i`/`_Func_Adjacency`/`_Kernel`) already branched on `useAdjacency` internally, exactly like every migrated operator's. What it *didn't* have was a Python entry point that actually let that branch be exercised both ways: `computeRenormalizationMatrices`, the only caller, unconditionally `raise NotImplementedError`'d unless given an explicit `AdjacencyList`, rejecting both `adjacency=None` and a `CompactHashMap`. So the kernel-level dual path had been sitting unused since it was first written — nothing forward-value (`operation_matrix.py`) or backward-mode (no gradcheck existed for covariance at all) had ever exercised the grid branch.

The fix was almost entirely deletion, not new logic: `computeCovarianceMatrix` (`renorm/wp_covariance.py`, new name for the covariance-only half of what used to be `computeRenormalizationMatrices_`) drops that restriction and calls `warpWrapper2` directly, the same as every migrated operator's `_computeSPHX_stateBackend` — `extractStateInfo` already auto-builds a `CompactHashMap` for `adjacency=None` and dispatches on `isinstance(adjacency, CompactHashMap)` for you, so there was no dispatch logic to write. Verified (see `scripts/gradcheck_covariance_native.py`, the first gradcheck script to deliberately exercise the grid branch rather than only the default neighbor-list one — every other `gradcheck_*_native.py` script's `build_adjacency` helper calls `radiusSearchCompactHashMap(..., returnCompactHashMap=False)`, which despite the name returns an `AdjacencyList`, not a `CompactHashMap`; `_gradcheck_common.py` gained a `build_grid_adjacency` helper alongside it for scripts that specifically need the grid branch):

* Forward-value parity: `adjacency` (neighbor list), a real `CompactHashMap`, and `adjacency=None` (auto-built `CompactHashMap`) all produce bit-identical covariance matrices for the same particle configuration.
* `torch.autograd.gradcheck` passes for both the neighbor-list and grid traversal branches, checked against positions/supports/masses/densities.

Splitting the file three ways (asked for separately from the dual-path fix, done in the same pass since both touch the same file):

* `renorm/wp_covariance.py` — the covariance kernel trio plus `computeCovarianceMatrix`. Nothing else.
* `renorm/wp_pinv2x2.py` (new) — `pinv2x2_warp`/`pinv2x2_warpBackend` (the production closed-form symmetric-2x2 pseudo-inverse) and the pure-PyTorch reference `pinv2x2` it's kept in step with. This is a genuinely separate Warp operation (turns *any* symmetric 2x2 matrix into its pseudo-inverse + eigenvalues; has no idea what a covariance matrix or a neighbor loop is) that happened to be defined inside the covariance file historically.
* `renorm/wp_renormalization.py` (new) — `computeRenormalizationMatrices_`/`computeRenormalizationMatrices`, unchanged in behavior: calls `computeCovarianceMatrix`, applies the low-neighbor-count fallback, then a pseudo-inverse. ~~**Still requires an explicit `AdjacencyList`**~~ — **stale, see "Renormalization Grid-Mode Coverage" above**: the low-neighbor fallback was already reading its neighbor count from the covariance kernel's own per-particle output at the time this paragraph was written, not `adjacency.numNeighbors`, so the `NotImplementedError` this paragraph describes as active was already dead code, just untested and undiscovered until the pass documented above. Gradchecking `pinv2x2_warp` (the dim=2 pseudo-inverse) itself is still open, unrelated to this correction — see the note at the end of "Renormalization Grid-Mode Coverage" above.

`renorm/__init__.py` and the top-level `warpSPHCore/__init__.py` both now export `computeCovarianceMatrix` alongside `computeRenormalizationMatrices`; `crk/crk_terms.py`'s dead `pinv2x2` import and `warp_renorm.ipynb`'s import cell were repointed to `wp_pinv2x2.py`.

Validation: all 64 pytest cases pass (the pre-existing 63 plus `gradcheck_covariance_native.py`, now registered in `GRADCHECK_SCRIPTS`), and `scripts/run_operation_matrix_sweep.sh --quick` is clean (`OK=258, HIGH=0, ERR=0, NAN=0`) — unchanged from before this rework, since `computeRenormalizationMatrices`'s external behavior didn't change.

**Update (2026-08-06, folded into the "refactor repo into a clearer structure" commit): the standalone `computeCovarianceMatrix` function described above never shipped as such.** By the time this dual-path work actually landed in code, Covariance was taken one step further and folded directly into the same state-primary dispatch pattern the "States as the Primary Path" section above describes for the other six operators, rather than staying a bespoke top-level function:

* The covariance kernel trio (`computeCovariance_Func_i`/`_Func_Adjacency`/`_Kernel`) plus a private `_computeSPHCovariance_stateBackend` now live in `coreOperations/wp_covariance.py`, alongside the other six operators' kernel files (not under `renorm/`, which no longer exists as a package — see the module-layout table above). There is no public `computeCovarianceMatrix` function anymore.
* `operations.py`'s `warpOperation` dispatches to it directly: `operation == WarpOperation.Covariance` is a top-level `elif` branch, exactly like `WarpOperation.Density`/`Gradient`/etc., taking a `covarianceReturnNumNeighbors` kwarg that maps to `_computeSPHCovariance_stateBackend`'s `returnNumNeighbors`. Callers that used to call `computeCovarianceMatrix(p, operationProperties, domain, adjacency=...)` now call `warpOperation(p, operationProperties, domain, adjacency=..., ...)` with `operationProperties.operation = WarpOperation.Covariance` set — this is what `scripts/gradcheck_covariance_native.py` and `renorm.py`'s `computeRenormalizationMatrices_` both do.
* `pinv/wp_pinv2x2.py` and top-level `renorm.py` (`computeRenormalizationMatrices_`/`computeRenormalizationMatrices`) are unaffected by this — the split into three files described above still holds, only the covariance file's own public surface changed shape.
* `warpSPHCore/__init__.py` does **not** re-export a `computeCovarianceMatrix` symbol (the commented-out `# from .renorm.wp_covariance import computeCovarianceMatrix` / `# "computeCovarianceMatrix"` lines left behind by this transition have been removed, not just left commented). Covariance is only reachable through `warpOperation`, same as every other operation — there is deliberately no operator-specific top-level convenience function for it, unlike before.

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

## Status: planned in detail — see `warpier_fields.md`

`warpier_fields.md` is the executable plan for Phases 3 and 4, written against the repo
after `3cdd4e6` and cross-checked against the `warpSPH` frontend. It carries the
measured baseline (~900 us of N-independent CPU overhead per operator call, of which
~33% is 38 `wp.from_torch` calls), the prototyped savings, the `Field`/`StateBundle`
design, an eight-step execution order with per-step gates, and the argument for why
reintroducing wrapper reuse does not reintroduce the gradient-accumulation bug that got
the previous cache deleted.

It also carries three findings that reach beyond Phase 3: `kinds` can be made a
required member with zero call-site churn in either repo, which closes the open
`kinds=None` + `AllToAll` out-of-bounds read (its Step A); Warp 1.15 has no
forward-mode AD of any kind, so Phase 6 is entirely hand-written, but every SPH
operator is *measurably linear in the field values*, which makes tangents w.r.t.
values a re-launch of the existing kernel rather than a new one (its §3.6); and the
`warpSPHIntegrators` buffer-pool work in that repo's NOTES.md §2.1 should share this
plan's `ExecutionMode` switch rather than introduce its own (its §7).

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

## Status: readiness audited 2026-08-17 (`warpier_fields.md` Step G); Tier 1 landed in production 2026-08-18 (`warpier_forward_mode_plan.md` Phase 2)

**Update 2026-08-18.** `warpier_forward_mode_plan.md` Phase 2 promoted finding 1 below from a
throwaway spike into a supported API: `structFor`'s `FORWARD` rows now alias `REVERSE`'s
struct classes instead of raising (`util/fieldRegistry.py`), `getStateBundle(dim, FORWARD)`
hands back the same dim-keyed bundle `REVERSE` uses (`util/stateBundle.py`), and
`warpOperationJVP` (`operations.py`, next to `warpOperation`) is a real, tested entry
point for the Tier-1 value-tangent JVP — restricted to the five operators that actually
take `queryValues`/`referenceValues` (Interpolate/Gradient/Divergence/Curl/Laplacian;
Density and Covariance raise, since they'd otherwise silently ignore the tangent and hand
back the primal result), and shaped in its signature for the full Tier-2 tangent surface
(positions/supports/masses/densities) even though every Tier-2 argument still raises
`NotImplementedError` naming Phase 4. Gated by
`tests/operations/test_forward_mode_tier1.py`, in addition to the `spike_forward_mode_tier1.py`
gate below. `launchOperator`'s own `ExecutionMode.FORWARD` rejection (`autograd/operator_spec.py`)
is deliberately unchanged — `warpOperationJVP` never sets `ctx.mode = FORWARD`, it just
calls `warpOperation` (mode `AUTO`/`REVERSE` as usual) with tangent arrays substituted for
value arrays, so any caller reaching `launchOperator` with `FORWARD` set is still an error.
This also meant two tests asserting the old "FORWARD is rejected everywhere" behavior
(`test_struct_for_forward_mode_rejected`, `test_forward_mode_rejected_regardless_of_cache_warmth`)
had to be rewritten to assert the new alias-not-raise behavior instead of deleted outright,
since the underlying finding they pin (a cache-warmth-dependent inconsistency) is still
worth a regression test — now checked as "cold and warm cache agree" rather than "both raise".

Step G was an audit against landed code, not an implementation. Its four checks and what
they actually found, so Phase 6 starts from tested facts rather than the plan's
expectations:

**1. Tier 1 works, and it is cheaper than requirement 3 assumed.** The spike is
`scripts/spike_forward_mode_tier1.py` (kept as a standing gate in
`tests/operations/test_gradcheck_scripts.py`, not thrown away — see below). It checks the
JVP identity `JVP_v[f](qval, rval)·(dq, dr) == f(dq, dr)` for Interpolate, Gradient
(Naive/Difference/Summation/Symmetric), Divergence, Curl and Laplacian, in 1D and 2D,
scalar and vector fields, with the renormalisation correction both off and on: **14/14
cases agree to float64 round-off (rel_err 7e-17 … 2.7e-16)**, and `f(0) == 0` exactly in
every case, so each operator is genuinely linear and not merely affine. `warpier_fields.md`
§3.6's linearity measurement therefore does carry over to the JVP itself.

The stronger finding: **Tier 1 needs no new struct type at all**, so it needs no
`ExecutionMode.FORWARD` row in `structFor`'s table and no widening of the `StateBundle`
cache key. A Tier-1 tangent is the *existing* kernel, on the *existing* struct types,
launched with the tangent array in place of the value array. `structFor`'s mode axis and
the bundle key's mode dimension are insurance for Tier 2 (or for a fused dual-struct
design that computes primal and tangent in one launch), not prerequisites for Tier 1.

**2. Neither of torch's own forward-mode entry points can validate that bridge, and one
of them fails dangerously.** Probed directly in the spike:

* `torch.autograd.forward_ad` + `make_dual` raises `NotImplementedError: You must
  implement the jvp function for custom autograd.Function to use it with forward mode AD`.
  That is honest, and it names Phase 6's actual deliverable: a `jvp` staticmethod on the
  bridge (or a separate non-`autograd.Function` bridge, per `warpier_fields.md` §3.6
  requirement 6).
* `torch.autograd.functional.jvp` **returns without raising and hands back a silently
  zero tangent** (rel_err 1.0 against the verified Tier-1 answer). Its double-backward
  trick needs a differentiable backward, which reading gradients out of a `wp.Tape` is
  not, and `strict=False` (the default) degrades to zeros rather than erroring;
  `strict=True` does raise. `warpier_fields.md`'s Step G text said to validate the spike
  against this function — **do not**. A Phase 6 bridge that also produced zero tangents
  would agree with it perfectly. The spike uses a reverse-mode Jacobian
  (`torch.autograd.functional.jacobian`, which needs only the first-order backward the
  gradcheck suite already validates) contracted with the tangent, which is exact on cases
  this small.

**3. Warp-side helper overloads are the Phase-6 cost requirement 3 does not mention.**
No Python-side `dim == 1/2/3` struct ternary survives outside `structFor` (verified:
`arg_extract.py` has no non-comment reference to a dim-suffixed struct class). But
`util/stateUtil.py` carries three concretely-typed `@wp.func` overload sets
(`getParticleData`, `getParticleCorrectionData_i`, `getParticleCorrectionData_j`), one per
dim, resolved by *warp's* own overload dispatch on struct type rather than by any Python
branch. So if Phase 6 ever does introduce dual struct types, registering them in
`structFor` is **not** the whole change — each one needs its own set of these overloads
too. Tier 1 as spiked avoids this entirely, which is the main argument for doing Tier 1
first. (`castTorchToWarpAsBuiltins`'s `ndim == 1/2/3` branch is shape-driven, not
dim-driven, and needs no change: a tangent array has the same shape as its primal.)

**4. Two bugs found, both fixed, both of the "runs fine, plausible number" class this
plan keeps producing:**

* `getStateBundle(dim, ExecutionMode.FORWARD)` only raised on a **cold** cache. Once any
  bundle existed for that dim, the FORWARD request hit the dict before any validation and
  got the REVERSE-shaped bundle back. Fixed by validating the mode ahead of the lookup;
  pinned by `test_forward_mode_rejected_regardless_of_cache_warmth`.
* `computeRenormalizationMatrices` mutated the **caller's** `OperationProperties`
  (`operationProperties.operation = WarpOperation.Covariance`). Any caller reusing that
  object afterwards silently launched Covariance where it asked for a Gradient, and the
  resulting `(N, D, D)` tensor is plausible enough to pass unnoticed. Every call site in
  both repos happens to pass a freshly-constructed properties object, so it never bit --
  but `warpier_fields.md` §3.5's suggested follow-up (hoist those constructions out of the
  hot path so a reusable, hashable properties object can key the `StateBundle`) would have
  *introduced* the bug rather than found it. Fixed with `dataclasses.replace`; pinned by
  `tests/operations/test_renorm_no_caller_mutation.py`. Found by the Tier-1 spike, which
  reuses one properties object across the renorm call and the gradient consuming its output.

Requirement 6's separation (bundle construction independent of the AD bridge) holds at the
level that matters — `stateBundle.py` imports no torch at all, and constructing/refreshing
a bundle still works with `torch.autograd.Function` replaced by a poison class — but it is
currently *cosmetic in one respect*: the module lives in the `autograd/` package, so
`import warpSPHCore.autograd.stateBundle` runs that package's `__init__.py` and pulls
`StateAwareWarpFunction` in as a side effect. Moving the file out of `autograd/` is the
one-line-per-importer fix; it is deferred to `warpier_fields.md` Step I, which reworks
these imports anyway. Nothing today imports the bundle without the bridge, so the
distinction is latent.

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
