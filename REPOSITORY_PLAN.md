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
