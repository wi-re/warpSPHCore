# Core development records

## 18 November 2026 maintenance handoff

Validation is local to the working tree on `codex/analytic-boundary-ad` at
base revision `dcb14a79b5cd574fc7764c18e4529f77839bf41a`.
The hosted failures recorded on main at `2f9fc64` concerned Warp test
collection and native wheel import provenance. Both now pass locally.

| Check | Local result |
| --- | --- |
| Warp CPU suite | 3,036 passed and 511 skipped in a Core-only editable install with CUDA hidden and Gloo restricted to loopback. |
| Integer-width diagnostic | All eight FP32/FP64 and ID/offset width combinations passed on CPU. |
| Warp operation matrices | All four CI matrices passed on CPU. The diagnostic now uses the documented first- and last-axis divergence contractions. |
| Typed slots, periodic displacement and support conversion | 23 tests passed per precision on CPU and CUDA. The parent checkpoint failed the two local-coordinate regressions and the support-conversion regression in each configured precision on CPU. |
| Python and native Torch | 46 Python tests and 816 focused native operator/AD tests passed on CPU. |
| Extracted wheels | CPU native example and stable-ABI audit passed against the same fresh wheel. The native checker blocked Warp and compiler/JIT imports. The source wheel passed all 14 public configurations per precision on CPU. Both checks ran outside the repository with an editable install present. |
| Docs and repository layout | Production docs build and layout check passed. |

The runtime was Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0 and NumPy 2.5.2.
CUDA regressions ran on an NVIDIA RTX PRO 6000 Blackwell. Hosted CPU CI uses
Python 3.12. A hosted rerun and a hosted GPU runner remain open.

The validation logs, commands, wheel hashes and working-tree patch are under
`core/experiments/repository-maintenance/20261118-r1/` in the shared
[evidence workspace](../docs/development-artifacts.md). The retirement manifest
and the pre-sweep working copy of the plan are under
`core/archives/repository-maintenance-20261118/`. The
[repository plan](../../REPOSITORY_PLAN.md) owns current Core priorities.
The operator and autodiff notes still cite the historical architecture plans
in `docs/historic_plans/`.

## Pre-feature reliability follow-up

The CI selector now routes shared fixtures, harness sources, examples, and
native build inputs to their consumers. It also checks both paths of a move.
The layout guard reads notebook blobs from the index. The standalone CMake
library and package metadata use its project version, and the Python suite
checks that it agrees with the Python package. Warp import overrides now take
precedence before validation.

The follow-up passed 3,036 Warp CPU tests, 52 Python tests, 9 standalone C++
checks and 906 standalone Python tests. The source distribution rebuilt into a
working wheel, the separate CPU native wheel passed the stable-ABI audit, and
the installed CMake example passed. CUDA smoke checks passed both in a Core-only
environment and with the optional frontend for the PDE case. The original
frontend sampler and the self-contained disorder fixture gave identical CPU
positions and masses.

Logs, commands and source hashes are in
`core/experiments/repository-maintenance/20261118-r2/`. The five retired raster
plots can be recovered from Core revision `dcb14a7`. Hosted CI remains the
integration gate.

## Earlier development records

The shared `warpSPHDevelopment/core/` workspace holds raw results, historical
reports and profiler captures. Its README and
`manifests/migration-2026-10-16.json` describe the previous migration.
Exploratory source belongs in the separate Git repository
`core/research/native-backend/`. Normal tests, builds and examples run from a
standalone Core checkout.
