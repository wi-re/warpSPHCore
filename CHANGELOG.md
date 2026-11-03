# Changelog

## Unreleased — native development branch

- Native core: bounded CSR32/CSR64 owned-row compact search, Verlet filtering and shared distributed operator dispatch, including CRK, polynomial and meshless dependent passes.

- Native core: polynomial reconstruction assembly with CSR/CSC derivatives and Warp-free MFM/MFV primal, closure and multigrid execution.

- Native core: Warp-free renormalization and linear algebra, shared CPU/CUDA/HIP dense and Krylov updates, and reusable captured CG/BiCGStab workspaces.

- Add native Verlet validity, filtering, snapshot/capacity lifecycle and deferred
  device flags; qualify CPU/CUDA and experimental gfx942 compilation.
- Support native normalization calibration and positive-divergence Laplacians,
  including their applicable first-order derivatives.

- Move exploratory studies to an independent research repository with a verified
  migration manifest; enforce the external-artifact policy in agent rules and CI.

- Integrate the coordinated 0.7.1 core, frontend and integrator interfaces.
- Add opt-in native core/CRK accumulation dispatch within the Warp frontend,
  preserving its Morton ordering and CSR buffers. Native owned-row traversal now shares the upstream
  communication and eager primal contract.
- Add complete-simulation correctness comparisons and CPU/CUDA scaling drivers;
  retain failed FP32 projection controls explicitly in the qualification report.
- Add benchmark-only exclusive frontend stage profiling and stacked CPU/CUDA
  charts, distinguishing remaining Warp kernels from native operators and scaffolding.

## 0.7.1 — 2026-11-03

Coordinated GitHub checkpoint with warpSPH and warpSPHIntegrators 0.7.1.

- Add explicit eager whole-row CSR chunk storage with owned J/pair work,
  replicated particle fields, Gloo/NCCL transport and coordinated rebuilds.
- Preserve deterministic row identity, native search-history reconstruction,
  empty owners, rank/partition changes and allocation-failure propagation.
- Declare query/reference roles in prepared operators and port the admitted
  operators and meshless exchanges without global pair-buffer fallbacks.
- Retain ordinary flat search/operator defaults and existing AD/graph contracts.

The separate native C++ backend branch is excluded from this checkpoint.
Chunked AD/graphs and configurations outside the documented eager support matrix
remain unsupported. See CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md for qualification.
