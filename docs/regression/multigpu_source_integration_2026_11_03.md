# Owned-row baseline integration into native development — 2026-11-03

Merge core/main a2c85b5 (qualified multi-GPU core179df45) into the checkpointed
codex/native-cluster-pairs tree eaa75fb (backend430aef7 plus preserved source
survey). Frontend/dev85ae913 integrates the paired frontend with fixed-count
particle ordering; Integrators/maincdbbc52 is pinned for the combined controls.

The seven coreOperations/wp_* compatibility aliases are retained. Row-chunk
eligibility and query/reference extra roles are applied to the actual Warp
implementations under coreOperations/native/wp_*, including paired CRK
interpolation inputs. Chunked search/checkpoint exports remain in the Warp
backend branch, preserving pure-native imports. Added tests and process workers
follow tests/warp/{operations,radiusSearch,distributed}; no duplicate suite or
monolithic J/W fallback is introduced. Native compiled sources are unchanged.

This is a shared-source baseline merge. Owned-row multi-GPU operations still
execute through Warp. C++ chunked traversal/communication, distributed adjoints
and complete native frontend/JFNK integration are separate ports.

## Local gates

- Native tree Warp planner/search/recipe/operators:334 passed,2 skipped,203
  device deselections in24.23s.
- Frontend against this tree's Warp backend:14 CPU combined/recovery/budget
  controls in46.16s,13 CUDA controls in31.29s. Covers all8 admitted layout
  families, two-rank empty→active ownership/restart and post-sort allocation
  recovery for CompSPH/deltaSPH/DFSPH.
- Native C++ primal/corrected/field-linearization CPU:1751 passed,97 skips,
  20.38s; CUDA:1752 passed,96 skips,179.15s including JIT compilation.
- Native Python/package/import isolation:21 passed in0.74s.
- Paired frontend/main core also pass the54-case FP64 CPU runner/history gate,
  21-case initial CUDA graph/combined selection,9-case FP64 CUDA combined gate,
  all8 shared-GPU NCCL23203 empty-owner/restart controls on both ranks, and
  1→2→3-rank Morton launcher save/resume for CompSPH/CRKSPH.

All numerical fixtures are short, with no video/viewer campaign or performance
claim. Use the normal Warp Python, explicit paired source paths, and the
existing /tmp/warpsph-build-tools/bin Ninja plus /tmp/native-corrected-ad-jit
cache for native JIT. The missing-Ninja initial run was an environment setup
failure; the repeated run with the existing toolchain passes.

Full frontend report:docs/particle_layout/MULTIGPU_INTEGRATION_2026-11-03.md.
Raw evidence is retained in the frontend .tmp/layout-multigpu-integration-20261103.
The all-repository source/test manifest SHA256 is
d5a2db8b92a6eef5e07f824c456470c0a4d315cf6c5f916e4000e8f5fcd487c0.
Accepted Isambard image/profile remains frozen; no remote jobs or package
installation changes occurred.
