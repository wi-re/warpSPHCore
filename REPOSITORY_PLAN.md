# Core capability plan

Updated 18 November 2026. Maintain the existing Core contracts and qualify
consumer compositions before extending the methods or resuming performance work.

## Maintenance and handoff

The maintenance sweep starts from Core `dcb14a7` on
`codex/analytic-boundary-ad`. The two failures recorded for GitHub main
`2f9fc64` concerned Warp test collection and native wheel import provenance.
[Failing CI run](https://github.com/wi-re/warpSPHCore/actions/runs/41727480692)

- Warp integer-role tests use the public facade for configured dtypes and
  check invalid values at import. The dispatch test checks private bindings
  at their owning module. The operation matrix uses the documented tensor
  divergence contraction axes.
- Both wheel checkers verify the resolved package path in a process with
  site startup hooks disabled. The native check also blocks Warp imports
  and compiler/JIT fallback.
- The existing periodic-grid patch is tracked independently of the CI repairs.
  Its SHA-256 is
  `2490e061bfd40858e12bb1bb6a4f94a7bf76ae4b825e62c27a6915a05ed2cd60`.
  CPU FP32 and FP64 checks cover local bits, both seams and the configured
  fixed-image pullbacks. The same focused checks passed on CUDA in both
  precisions.
- The Warp 3D support-radius helper now uses the correct sphere-volume
  coefficient. The regression checks the public conversion in all three
  dimensions.
- CI now selects the consumers of harness code, shared test helpers, examples,
  and native build inputs. CPU runs exclude GPU-only tests. Convergence smoke
  tests write to temporary directories and the PDE case checks for its optional
  frontend dependency. The repository guard checks notebook blobs in the
  Git index and rejects force-added build output.
- Use the [development handoff](docs/development/README.md) for validation
  status and the [artifact policy](docs/docs/development-artifacts.md) for
  retained evidence. Hosted CUDA CI still needs a runner. Review static
  findings in a dedicated change before adding a lint gate.

The former adjacency plan is replaced by the
[neighbor-search contract](docs/docs/neighbor-search.md). The Core
[owned-row record](docs/regression/multigpu_source_integration_2026_11_03.md)
and the frontend's `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md` carry qualification
and scheme ownership. Keep remaining compatibility, physical multi-GPU and
long-flow qualifications with those owners. The
[higher-order research backlog](higherOrderSPH/README.md) owns the remaining
matched scientific comparisons.

## Completed core pass

Source commit `2034a4e` completes the bounded pass for constrained and singular
linear systems, solver composition, native workspaces, neighbor search and
interaction masks, correction/derivative contracts, polynomial and TENO/WENO
reconstruction, and MFM/MFV geometry and rate guards. Qualification at that
revision includes CPU/CUDA in both precisions and fourteen runnable example
configurations. See the [qualification record](docs/docs/development-artifacts.md)
and [public examples](docs/docs/examples.md).

Later checkpoints cover shared prepared traversal and MFM grid-cell momentum
accumulation, fixed-image periodic arithmetic (`e16abb4`) and the local-main
native meshless and solve AD integration (`2054c77`). The
[register evaluation](docs/regression/core_register_pass_2026_10_21.md) preserves
the measured results of the October pass. The paired frontend's
`NATIVE_MESHLESS_SOLVE_AD_HANDOFF.md` and `NATIVE_INVERSE_VALIDATION_HANDOFF.md`
record the consumer coverage and limitations. These results apply to their
recorded source revisions and configurations.

## Next capability pass

Work through consumer compositions in this order, using independent numerical
references and reproducible end-to-end examples.

1. **Pressure solves.** Check constrained pressure operators, original-system
   residuals, prescribed values, nullspace compatibility, gauge and failures.
   Coupled incompressible Newton–Krylov evaluation belongs with the frontend
   and integrator owners.
2. **Compressible SPH and CRK.** Check physical boundaries and ghosts,
   reference volumes, state validity and neighborhood rebuilds across a short
   trajectory. Track conservation and correction behavior.
3. **Complete MFM/MFV updates.** Qualify moving geometry, support/volume updates,
   face closure, reconstruction, timestep policy and boundaries. Track raw
   positivity and conservation through repairs and fallbacks.
4. **Implicit/vector consumers.** Compose vector constraints and block
   preconditioners. Verify residuals and adjoint equations before widening the
   scope of implicit differentiation.

For each consumer, document device, precision, AD and neighborhood-lifecycle
assumptions. Keep performance tuning tied to a measured consumer workload.

## Open core boundaries

- First-moment MFM face consistency and complete Warp flux/closure AD.
- Nonlinear reconstruction in 3D, sector coverage and derivatives at rank or
  fallback transitions.
- Grad-h geometry tangents, non-density HVPs and AD beyond the documented
  operator and preparation contracts.
- CUDA CI, heavier-jitter reference thresholds and wider nightly test matrices
  remain separate test infrastructure work. Revalidate the archived Warp
  dynamic-loop issue and its repro before any upstream submission.

The [public contracts](docs/docs/index.md) describe supported behaviors. The
former plan and completed session notes are preserved at the revisions
listed in the development handoff.
