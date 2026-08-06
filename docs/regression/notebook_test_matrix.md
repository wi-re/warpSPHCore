# Notebook to Regression-Test Matrix

This document defines the initial mapping from root notebooks to the Phase-0 regression test suite.

## Purpose

* Preserve behavior during interface migration (`sphOperation_warp` to `warpOperation` + structured wrappers).
* Provide deterministic reference scenarios before refactoring.
* Keep notebook examples and automated tests aligned.

## Mapping

* `warp_density.ipynb` -> `tests/operations/test_operations_core.py`
  * density positivity/finite checks
* `warp_interpolate.ipynb` and `warp_interpolate copy.ipynb` ->
  * `tests/operations/test_operations_core.py`
  * `tests/operations/test_operations_consistency.py`
  * scalar/vector/matrix interpolation paths
* `warp_gradient.ipynb` ->
  * `tests/operations/test_operations_core.py`
  * `tests/operations/test_operations_crk_analytic.py`
  * linear analytic gradient checks, correction-path checks
* `warp_divergence.ipynb` -> `tests/operations/test_operations_core.py`
  * divergence of linear vector field vs analytic constant
* `warp_laplacian.ipynb` -> `tests/operations/test_operations_core.py`
  * linear scalar laplacian near zero
* `warp_curl.ipynb` -> `tests/operations/test_operations_core.py`
  * vector-field curl vs. analytic reference (ported to the current API, ports/passes as of 2026-08-05)
* `warp_renorm.ipynb` -> `tests/operations/test_operations_crk_analytic.py`
  * renormalization-corrected gradient checks (ported to the current API, passes as of 2026-08-05)
* `old/warp_grid.ipynb` -> **not a porting gap** (clarified 2026-08-06): grid dispatch is just `adjacency=None` on the same operator calls every root notebook already makes, not a distinct example -> `tests/operations/test_grid_modes.py` (landed) mirrors the base-path cases with `traversal="grid"` for direct coverage.
* `old/warp_crk.ipynb` and `old/warp_crk_test.ipynb` -> **not a porting gap** (clarified 2026-08-06): CRK checks were folded directly into the relevant operator notebooks (interpolate, gradient, etc.) instead of being split into standalone CRK notebooks -> `tests/operations/test_operations_crk_analytic.py`
  * CRK-assisted gradient analytic checks and baseline comparison
* `warp_profile.ipynb` (root, current API, benchmarking only) -> not part of the regression suite by design; see `warpier_core.md`'s Notebook Corpus Status.

## Current Phase-0 Scope

* Deterministic 2D periodic particle setup.
* Core operations:
  * density
  * interpolate
  * gradient
  * divergence
  * curl
  * laplacian
* Field-rank coverage:
  * scalar fields
  * vector fields
  * matrix fields
* Correction-path coverage:
  * baseline
  * CRK
  * renormalization

## Notebook Ergonomics Update

The notebooks are being rewritten to be easier to run and review interactively.

Notable changes already applied to `warp_interpolate.ipynb`:

* separate cells per operation example instead of one dense comparison block
* Warp-only examples instead of diffSPH comparisons
* added jitter in the CRK example to exercise the correction path more realistically
* added an error plot for the CRK comparison cell

Notable changes applied to `warp_gradient.ipynb`:

* separate cells for scalar gradient, vector gradient, and jittered CRK checks
* colorbars added to every plot so the visual scale is explicit
* top-level field mode switch for either linear or periodic sinusoidal examples
* optional edge masking for linear fields to hide the periodic jump at the boundary
* jittered CRK gradient example includes an error plot for plain vs corrected output

The gradient notebook should follow the same style:

* separate cells for each gradient example
* Warp-only runnable examples
* CRK gradient check with jittered input and error visualization

## Follow-up Expansion

* Add explicit operation-direction mode tests (`FluidToBoundary`, `BoundaryToFluid`, etc.) where reference fixtures are available.
* Add non-periodic boundary regression scenarios.
* Add 3D scenarios and curl shape/analytic checks in 3D.
* Add fixture snapshots (golden outputs) for strict regression lock-in across releases.
