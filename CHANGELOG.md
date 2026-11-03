# Changelog

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
