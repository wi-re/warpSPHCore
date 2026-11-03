# Native core merge qualification

This checkpoint closes the agreed core order on top of the coordinated 0.7.1
upstream checkpoint. Frontend force/work kernels, shifting and full native solver
integration remain a separate frontend worktree. Warp stays the default backend.

| Ordered core stage | State |
|---|---|
| Verlet/capacity lifecycle and standard options | Complete |
| Renormalization, small matrices and solver dependencies | Complete |
| Polynomial reconstruction and meshless primals | Complete |
| Compact owned-row search and dependent native traversal | Complete; eager/no-AD contract shared with upstream |
| Derivative and public API parity | Complete within explicit first-order, legacy density-block and nonsmooth contracts |
| Local distribution and merge qualification | Complete; local scope below |

The native execution path covers all 15 radial families, seven standard
operations, applicable correction/preparation passes, polynomial sums, dense and
Krylov updates and MFM/MFV primals. Shared Torch control/solve algebra remains
Torch. Warp-specific inline functions, array/struct factories and arbitrary user
Warp kernels are backend-specific APIs, with numerical native equivalents in C++
headers. No simulated Warp types or implicit Warp fallback are supplied.

## Distribution checks

The final CPU wheel uses portable x86_64 code; the CUDA wheel uses the local CPU
and sm_120 SASS target. Both pin Torch 2.13.0+cu130 and target stable LibTorch ABI
2.10. Installed-wheel checks block Warp and `torch.utils.cpp_extension`, proving
that compatible wheels load through the precompiled path. The universal source
wheel contains the complete manifests/headers and no developer binary; its
isolated JIT run uses a fresh extension cache and blocks Warp.

| Check | Result |
|---|---|
| Final installed CPU wheel, full native suite | 3,625 passed, 470 existing skips |
| Final installed CUDA wheel, full native suite | 4,132 passed, 488 existing skips |
| CUDA wheel host-native CPU path | 3,625 passed, 470 existing skips |
| Installed source wheel, fresh CPU JIT | Core example and seven bounded derivative/owned-row contracts pass |
| Python build/loader/scaffolding | 24 passed |
| Final CPU wheel/Warp double-precision product parity | Pass |
| Full bounded Warp/native parity | 976 passed, 20 existing skips on each of CPU and CUDA |
| Standalone CMake compatibility checks | 4/4 CPU; 6/6 CUDA pass, including compact search oracles |
| Offline HIP gfx942 | All 33 primary GPU translation units compile; final compact adapter rebuilt after its compatibility fix |

Final qualification caught and repaired a stale standalone collection adapter
following CSR32/CSR64 output generalization. CPU and CUDA compact compatibility
oracles now pass. Scalar Laplacian templates no longer instantiate unsupported
vector reads; admitted shapes/results retain their existing behavior. The
source-wheel check also caught an ignored development library admitted by a broad
source include rule. Source archives now respect that exclusion and use universal
tags; the native wheel builder explicitly selects platform tags. The native import
subprocess test now follows the active installed package instead of forcing the
source checkout into an installed-wheel run.

No throughput claims or simulation timings were added. The final CPU wheel uses
CMake Release optimization; local CUDA qualification used `-O2 -DNDEBUG`.
Previous stage reports retain correction, AD, capture and two-rank Gloo evidence.
No NCCL/multi-node, AMD runtime, ROCm-Torch wheel or GH200 hardware qualification is
claimed. AMD compilation suffices for this experimental checkpoint; building a
GH200 wheel requires the target aarch64/Torch/toolchain environment.

## Merge and handoff

The main checkout and paired frontend/integrator worktrees remain untouched. The
native core branch includes upstream 0.7.1. All code changes are saved as ordered
checkpoints, with the permanent native worktree fast-forwarded only when clean.
Research, probes, raw logs, source snapshots and wheels are retained outside core;
ordinary builds/tests do not depend on that evidence workspace. Repository layout
and whitespace checks pass.

Retained evidence (including failures and superseded builds):
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-core-merge-20261103`.
