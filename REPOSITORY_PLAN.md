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

### Periodic geometry arithmetic — 8 November 2026

An isolated correction from the ACSPH FP32 investigation is prepared on
`codex/stable-periodic-distance`. Shared Warp displacement preserves local bits;
native direct/prepared traversal uses the same image rule. Tensor motion checks
and search preparation also preserve local differences/in-domain coordinates.
A new negative control proves old preparation can drop a pair inside its support
cutoff. Focused CPU, Warp CUDA graph/derivative and standalone native CPU checks
pass. Installed Core is unchanged. Native CUDA and evolving consumer qualification,
frontend copies and pinned historical-control integration remain before rollout.
Stable round-based reconstruction helpers retain their existing half-box policy.

### Native periodic geometry integration — 9 November 2026

The uncommitted `codex/stable-periodic-distance` correction is reconciled on
`codex/native-periodic-integration` with the current native runtime and import
selection. The source worktree remains unchanged. Tensor search preparation and
Verlet motion retain local bits; native direct/prepared geometry also retains
large-image coordinate remainders. Frontend wrapping/coverage and the CRK AD
caller are reconciled in the paired checkout. Explicit Warp fixed-image
pullbacks restore box-bound derivatives omitted by `wp.mod`; unwrapped pairs
retain no bound dependency. CPU tensor/Warp/operator/import/whole-run gates pass;
CUDA and final installed gates are being completed in the paired plan. Keep
existing half-box conventions, reference floors, rank/cutoff policies and the
FP32 determinant repair. See the paired frontend native port plan and handoff.


### Native periodic geometry combined qualification — 9 November 2026

Core e16abb4 and frontend `2737857` pass the declared CPU/CUDA geometry/runtime
matrix. CUDA Warp geometry/search/motion and fixed-image bounds:49 passes per
precision; native standalone:10 passes plus expanded direct/prepared geometry.
Source launch/import rejection and actual isolated installed-without-Warp runs
pass separately. Cold/final installed CUDA:49 pass/one skip/strict inherited
checkpoint xfail FP32,50 pass/one skip FP64; both Scene precisions pass.
Combined owners retain three exact inherited CUDA FP32 CompSPH failures;
a broader follow-up reproduces four additional CPU FP32 Inutsuka comparisons
identically before distance integration. No failure waiver or main merge.
Continue the bounded shared quotient/acoustic/native-spectrum FP32 range repairs
in the paired frontend M3 plan. CPU receipt is finalized; CUDA receipt is the
single-use frontend-native-periodic-integration-cuda-20261109-r1 archive.


### Qualified bounded FP32 scalar ranges — 9 November 2026

Generic Number/Directional quotient AD avoids the squared-denominator range
failure for finite FP32 quotients, preserving normal/FP64 replay and trace layout.
Only exceptional JVP scalar intermediates widen locally; tensor/Jacobian storage
stays FP32. Native general symmetric-spectrum primals scale large finite FP32
matrices before symmetrization/Jacobi, preserving cutoffs and sweep count.
Paired frontend normalizes exceptional acoustic impedance products before tracing.
Fresh complete CPU/CUDA libraries, Core/frontend standalone scalar paths 12/2,
Core dense/geometry products 44/45 plus four public large-matrix cases per device,
and installed no-Warp matrices pass. CPU CompSPH profiles180 each precision;
CUDA retains the exact three inherited FP32 failures (177/180 passes).
Four broader CPU Inutsuka failures also reproduce identically before integration.
No tolerance waiver or main merge: strict Warp/independent references conflict
for one CompSPH component. Torch spectral AD and manual derivative quotient range
limits remain separate. See paired frontend M3 report and immutable
frontend-native-fp32-ranges-20261109-r1 evidence. The preserved native pinv facade
patch is separately checkpointed at `2268211` with CPU/CUDA32/64 dispatch tests.

## Native meshless and solve AD follow-up — 9 November 2026

### Local-main integration — 10 November 2026

User authorized merging this work into local main. Main fast-forwarded from
c2b8aaf to2054c77, incorporating all six qualified implementation commits.
This closeout supersedes the earlier A1/A2 working and no-merge entries below.
The paired frontend integrates its AD branch with the independent ACSPH main
history; current plans/handoffs are under /home/lu26029/dev/warpSPH, particularly
NATIVE_MESHLESS_SOLVE_AD_HANDOFF.md and NATIVE_INVERSE_VALIDATION_HANDOFF.md.
CPU/CUDA FP32/FP64 qualification scope and separate launch/import/real-no-Warp
installation evidence are recorded there. Fresh main-source integration checks
are recorded in the frontend local-merge closeout; no new capability or widened
scope is inferred from merging. Large pressure solves, physical obstacle design,
long chaotic gradients and statistical-response methods retain their restrictions.
Research/evidence remain external, old sealed archives unchanged. No push.


Paired frontend plan: /home/lu26029/.codex/worktrees/native-meshless-solve-ad/warpSPH/NATIVE_MESHLESS_SOLVE_AD_PLAN.md.
This explicitly authorized follow-up starts from merged Core c2b8aaf. A1 exposed
a Warp reference bridge issue: mutable scalar parameter structs remain live in a
tape, changing prior backward branch decisions during composed Godunov passes.
First bounded Core change: snapshot scalar-only additional struct arguments for
AD calls, with a deterministic composed scalar-map regression. Native execution
and numerical values are unchanged. Meshless/solve phases follow the frontend
phase table; no performance campaign, merge or push. Core artifact rules apply.

A1 CPU checkpoint: scalar-only struct snapshots are taken only for AD launches;
array-bearing records keep their existing tensor/field contract and inference
launches avoid the copy. The composed map regression fails on c2b8aaf and passes
CPU FP32/FP64. Actual frontend Godunov/Inutsuka composition also passes. Frontend
full CPU owner qualification and independent equations pass after the capture
repair; CUDA remains queued under the shared lock. No native C++ source changed.
Required repository layout check passes. Evidence is /tmp/native-ad-core-capture-*
pending the paired follow-up receipt. Next: CUDA capture qualification with A1,
then dependency-ordered actual MFM/MFV AD and solve residual/transpose foundations.

A1 differential checkpoint: the paired independent Godunov equations expose
1D cancellation in the viscosity mixed support partial and in the regularized
Hessian's B coefficient. Stable equivalent evaluation is confined to AD helpers;
public derivative-valued Warp forward functions retain their original bodies.
Native kernel-derivative metadata supports the original scalar regularization
when a wider pair trace is used. The standalone conditioned-kernel regression
passes; both paired frontend native/Warp CPU independent products pass. Focused
CUDA Godunov owners now pass both precisions, including exact/rounded clamp
contacts (frontend owns the bounded near-contact differential convention).
Complete current-library GPU qualification is active under the shared lock.

A2 working checkpoint: native first-order products now cover actual moments,
least-squares gradients, selected-extrema limiter and material-aware Riemann
flux/conserved rates. Original primal passes are retained. CSR membership and
branch/extrema choices are controls; x/h/domain/volume/inverse/condition inputs
remain continuous. Current complete CPU library passes77 meshless tests across
FP32/FP64, independent equations, numerical products, conservation, lifetimes
and explicit unsupported scopes. Source import rejection passes77; separate
Warp-runtime launch rejection passes36 FP32/40 FP64 checks. Native-supported
initial public scope is full-rank closure-none CSR, six smooth kernels, fixed
external timestep; projected/chunked/distributed geometry and automatic timestep
AD remain explicit. Frontend prepareStage and two-step package-owned temporal
maps pass CPU numerical checks. CUDA/graphs and real installed gates remain
pending; this is not a full meshless-AD closeout. Native source/library evidence
is `/tmp/native-ad-current-full-*` and `/tmp/native-ad-a2-*`, pending a new receipt.
