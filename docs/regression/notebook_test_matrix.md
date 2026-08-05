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
* `warp_grid.ipynb` ->
  * currently covered indirectly via `sphOperation_warp` dispatch with compact-hash data
  * explicit grid-only direct tests to be added in a follow-up file
* `warp_crk.ipynb` and `warp_crk_test.ipynb` -> `tests/operations/test_operations_crk_analytic.py`
  * CRK-assisted gradient analytic checks and baseline comparison

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
