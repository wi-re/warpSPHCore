# Core capability plan

Updated 16 October 2026. Capabilities, numerical correctness and composition take
priority. Performance work resumes after the consumer stack has had its first
capability pass.

## Completed core pass

Source commit `2034a4e` completes the bounded core pass: constrained and singular
linear systems, solver/preconditioner composition, native workspaces, search and
interaction masks, correction/derivative contracts, polynomial and TENO/WENO
reconstruction, and MFM/MFV geometry/rate guards. Fourteen runnable example
configurations exercise these pieces together.

The full CPU suite, real-CUDA gates in both precisions, five operation matrices,
source/wheel examples and documentation build passed. See
[qualification and artifact policy](docs/docs/development-artifacts.md),
[examples](docs/docs/examples.md), [reconstruction](docs/docs/reconstruction.md),
[MFM/MFV](docs/docs/mfm.md) and [linear algebra](docs/docs/linalg.md).
These are bounded contracts, not qualification of every downstream scheme.

## Next capability pass

Work through consumer compositions in this order, using a small independent
numerical reference and a reproducible end-to-end example for each:

1. **Pressure solves:** connect boundary constraints, nullspaces and compatibility
   policies to actual pressure operators. Check the original-system residual,
   prescribed values, gauge and explicit failure behavior.
2. **Compressible SPH and CRK:** check physical boundaries and ghost populations,
   reference-volume policies, state validity and rebuild/invalidation across a
   short evolving trajectory. Establish conservation and correction behavior.
3. **Complete MFM/MFV updates:** combine moving geometry, support/volume updates,
   face closure, reconstruction, timestep policy and boundaries. Track raw
   positivity and conservation alongside any repairs or fallbacks.
4. **Implicit/vector consumers:** compose vector constraints and block
   preconditioners, verify residuals and adjoint equations, then establish the
   scope of implicit differentiation.

For each consumer, document its supported precision/device/AD combinations and
lifecycle assumptions. Keep missing capability cases explicit; a passing static
kernel test alone does not qualify a time-dependent scheme.

## Open core boundaries

- First-moment MFM face consistency and a complete differentiable Warp
  flux/closure path remain open.
- Nonlinear reconstruction in 3D, sector coverage and geometry derivatives at
  rank/fallback transitions require separate qualification.
- Grad-h geometry derivatives and non-density higher derivatives need their own
  stated scope and independent checks.
- Hosted GPU CI still needs appropriate runner infrastructure.

Address these when a consumer requires them. Maintain the documented guards
until the corresponding numerical evidence exists. Defer timing optimization,
layout experiments and new performance defaults during the consumer pass.

## Development history

Detailed plans, handoffs, qualification reports, raw runs and historical figures
are archived outside Git in `../warpSPHDevelopment/core/`. The original detailed
version of this plan is preserved there as `REPOSITORY_PLAN.md`; the latest
handoff is `docs/development/continuation-2026-10-16.md`. The migration manifest
records original paths and SHA-256 checksums.

Keep public contracts, examples, reproduction tools, small regression fixtures
and curated teaching figures in this repository. Use the
[artifact policy](docs/docs/development-artifacts.md) for future outputs.

### Compressible consumer cleanup — 21 October 2026

The user requested a performance and architecture pass over MFM/MFV. The generic
prepared structured-operator launcher now lives in `autograd.prepared`, with the
old `mfm.fastLaunch` import retained. It exposes host traversal metadata so a
consumer can compile separate grid/list entry points sharing pair physics.
Grid/list flux qualification found and fixed the loss of incoming momentum
between grid cells in `mfmFlux_Func_i`; the single-range neighbour path is
unchanged. Companion evidence is in warpSPH's
`docs/meshless_gizmo_validation/PERFORMANCE_2026-10-21.md`.
Prepared traversal metadata is independent of no-tape eligibility. Native
meshless entry kernels skip unused backward generation, matching the existing
public forward-only guard; shared functions and the differentiable Torch
reference remain available. Final focused core contracts: 125 passed, 12 skipped.

### Shared SPH register evaluation — 21 October 2026

At the user’s request, temporarily prioritize shared core register pressure over solver-specific register tuning. Use the existing operator evaluation suite for bounded, source-matched experiments on static dimensions, traversal specialization and accumulator lifetimes. Preserve forward and derivative contracts; adopt changes only with numerical and measured throughput evidence. Track results in `docs/regression/core_register_pass_2026_10_21.md`. This explicitly reopens timing optimization for this pass.

The shared register screen retained static tensor extents in outer products, divergence and raw CRK moments. Repeated CSR moment timings improve 17–33% in 2D FP32 and 12–16% in 3D FP64; NCU identifies reduced local tensor traffic despite higher register allocation. Blanket register caps, traversal/policy specialization and static distance changes remain experimental because some workloads regress. See `docs/regression/core_register_pass_2026_10_21.md`; MFM/MFV staged consumer qualification is complete; the companion report records saved-state and daemon/video controls, modest MFV improvements and neutral MFM timing. Focused tuning remains deferred.
