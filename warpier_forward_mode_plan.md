# Forward-mode evaluation: from toy operators to automatic implicit SPH solves

Companion to `warpier_core.md` Phase 6 ("Extend States for Forward-Mode AD", audited-but-not-started
as of 2026-08-17/18) and `warpier_adjoint.md` (Tiers 2.0-2.5 JVP derivations, all done as of
2026-08-18). Those two documents established that the *math* for forward-mode SPH is finished and
validated; this plan is about actually running it, using testbeds in the sibling `warpSPH` repo
(`~/dev/warpSPH`) at increasing difficulty, and about where that connects to `warpSPHIntegrators`'s
(`~/dev/warpSPHIntegrators`) own already-scoped implicit-integrator plan.

## Status (as of 2026-08-19)

**Phases 1-3 done. Phase 4 in progress** (steps 1-3 done, Density-operator scope only -- see its
own section below for the full breakdown; steps 4-6 not started). Most of the work landed in the
sibling `warpSPH` repo (new test files there, not this one), which is easy to lose track of from
this document alone -- recorded here explicitly for that reason.

* **Phase 1 -- done.** `warpSPH/tests/test_forwardModeWave.py` (new), commit `ab28fc9` in
  `warpSPH`. Seeds `du0`/`dv0` via `torch.func.jvp` on the plain-torch Wendland bump formula,
  rolls a primal and a tangent system through the identical explicit-Euler `f_wave_equation`
  step sequence, and cross-validates against an independent reverse-mode directional derivative
  (a weighted probe of `u(T)`, backpropagated to `position`/`magnitude`, dotted with the
  perturbation direction). Deliberately uses explicit Euler on *both* sides of the comparison
  rather than the `rungeKutta2` integrator `test_waveEquation.py`'s other tests use --
  the JVP/VJP identity only holds exactly when both sides differentiate the identical graph, and
  reusing the already-tested Euler pattern from `test_gradientsReachSourcePositionAndMagnitude`
  for both sides was simpler than proving RK2 forward/reverse consistency separately. Parametrized
  over 1D/2D and two rollout lengths (3 and 6 steps); all 4 cases pass. No `warpSPHCore` changes
  were needed or made in this phase.
* **Phase 2 -- done.** Commit `7a3041b` in `warpSPHCore`. `structFor`
  (`src/warpSPHCore/util/fieldRegistry.py`) now aliases `ExecutionMode.FORWARD` onto `REVERSE`'s
  struct rows instead of raising, and `getStateBundle` (`util/stateBundle.py`) hands back the same
  dim-keyed bundle for both modes, since Tier 1 needs no new struct shape. `warpOperationJVP`
  (`src/warpSPHCore/operations.py`, next to `warpOperation`) is the new supported entry point --
  it delegates to `warpOperation` on tangent arrays for the five operators that actually take
  `queryValues`/`referenceValues` (Interpolate/Gradient/Divergence/Curl/Laplacian), raises
  `NotImplementedError` for Density/Covariance (no value input -- routing them through would
  silently ignore the tangent and hand back the primal result) and for any Tier-2 tangent argument
  (positions/supports/masses/densities, already in the signature, reserved for Phase 4).
  `launchOperator`'s own `FORWARD` rejection is untouched, since `warpOperationJVP` never sets
  `ctx.mode = FORWARD` -- it just calls `warpOperation` normally with tangents substituted for
  values. New gate: `tests/operations/test_forward_mode_tier1.py` (8 cases, in-process, checks
  the JVP identity against a reverse-mode Jacobian reference). Two pre-existing tests that
  asserted the old "FORWARD always raises" behavior
  (`test_struct_for_forward_mode_rejected`/`test_forward_mode_rejected_regardless_of_cache_warmth`)
  were rewritten, not deleted, to assert the new alias behavior. Verified unaffected:
  `operation_matrix.py --device cpu` (`OK=258, HIGH=0, ERR=0, NAN=0`, matching the baseline), all
  six `gradcheck_*_native.py` scripts, `spike_forward_mode_tier1.py`, and the full `tests/` suite
  (127 passed, 1 skipped).
* **Phase 3 -- done.** `warpSPH/tests/test_implicitWaveEquation.py` (new), commit `d882bf9` in
  `warpSPH`. Backward Euler eliminates `v^{n+1}` algebraically into a single stage equation
  `u^{n+1} - alpha * Laplacian(u^{n+1}) = rhs`; the matvec is exactly
  `p -> p - alpha * warpOperationJVP(Laplacian, tangentQueryValues=p)`, no hand-derived Jacobian.
  A from-scratch matrix-free CG was written (no CG solver existed in `warpSPH` to reuse --
  `bicgstab.py`'s BiCGStab is the documented drop-in for a non-constant-`alpha` case, which would
  make the stage operator non-symmetric). The symmetry the plan asked to verify empirically
  (rather than assume) does hold on the tested case (constant `c`, zero `damping`,
  `<matvec(a),b> == <a,matvec(b)>` to float32 tolerance), so CG was used directly rather than
  falling back to BiCGStab. Validated both ways the plan asked for: agreement with the explicit
  rollout at small `dt`, and monotonic error shrinkage against the closed-form standing wave
  across resolutions. One bug found and fixed along the way: the first draft built "explicit" and
  "implicit" comparison systems via `explicitSystem = system` / `implicitSystem = system` --
  two names for the *same* mutable object, so the explicit loop's in-place state mutation silently
  became the implicit loop's initial condition. Fixed by building two independent systems.

**A finding along the way, outside this plan's scope but relevant to Phase 4's baseline.** While
verifying the `warpSPH` suite was unaffected, `tests/test_implicitShifting.py::test_shiftingConvergesToUniformDensity[ShiftingScheme.implicit]`
(pre-existing, unrelated to any of the above -- confirmed by stashing all Phase 1-3 changes and
re-running, same failure) turned out to be flaky. Two separate issues, only the first fixed:

1. **Fixed, committed (`warpSPH` commit `64c0bde`).** The test's domain is fully periodic with no
   real free surface, but `WeaklyCompressibleSPHConfig`'s `surfaceDetectionConfig.active` defaults
   to `True`, so `solveShifting`'s ColorField-based free-surface heuristic still runs and
   occasionally flags false-positive "surface" particles as the jittered lattice relaxes; its
   post-hoc shift zeroing/projection for those particles then destabilizes the implicit Newton
   solve (density std observed jumping from ~0.008 back up to ~0.25 partway through a 25-step
   relaxation). `deltaSPH`'s smaller, CFL-clamped per-step corrections happened to tolerate the
   same interference, which is why only the `implicit` variant of the test was failing. Fix: set
   `schemeConfig.surfaceDetectionConfig.active = False` in the test, since surface detection isn't
   what it's meant to exercise.
2. **Found, not fixed.** Even with (1) fixed, the test remains flaky specifically on GPU: passes
   reliably and deterministically over repeated runs on CPU (`CUDA_VISIBLE_DEVICES=""`), but on
   GPU occasionally still diverges (density std climbing from ~0.004 back to ~0.17 over the same
   25 steps, at a different point each run). This points at CUDA's nondeterministic scatter-add
   ordering in `_multiplyLaplacianBlock`'s `scatter_sum` calls interacting with an
   already-marginal relaxation factor -- `implicitShifting.py`'s own docstring documents
   `implicitRelaxation=0.1` as swept against instability, with `0.15` "already occasionally
   unstable". This is a robustness gap in `computeImplicitShift`'s undamped-per-step Newton
   iteration itself (e.g. no step-rejection or line search when a step increases the residual),
   pre-existing and unrelated to forward-mode AD -- flagged rather than fixed, since Phase 4
   explicitly uses this same hand-built solver as a comparison baseline and will need to account
   for it (at minimum, prefer CPU or a lower relaxation factor for that comparison's own
   stability, independent of whatever the automatic-JVP path needs).

## Context and motivation

`warpSPHCore`'s adjoint work has two layers:

1. **Reverse-mode (VJP)** is production-complete and battle-tested (`gradcheck_*.py`,
   `operation_matrix.py`), across both the six core operators and custom frontend kernels built on
   the same building blocks.
2. **Forward-mode (JVP)** has had its *math* fully derived and validated in isolated spike
   scripts, but never wired into anything runnable:
   - Tier 1 (tangent w.r.t. field *values*) -- proven trivial: every SPH operator is exactly
     linear and homogeneous in its value arguments, so a Tier-1 JVP is *the same kernel,
     relaunched on the tangent array in place of the value array*. No new code needed.
     (`scripts/spike_forward_mode_tier1.py`, gated in `tests/operations/test_gradcheck_scripts.py`.)
   - Tier 2 (tangent w.r.t. positions/supports/masses/densities) -- all six operators' JVP
     formulas are hand-derived and validated to float64 round-off against the production
     reverse-mode Jacobian (`warpier_adjoint.md`, Tiers 2.0-2.5, all "done" as of 2026-08-18).
     This is the tier that matters once particles actually move.
   - What's explicitly **not started**: `warpier_core.md` Phase 6 -- actually wiring any of
     this into `warpOperation`/`ExecutionMode.FORWARD`. Today `launchOperator` raises
     `NotImplementedError` the instant `ctx.mode is ExecutionMode.FORWARD`
     (`autograd/operator_spec.py:190`), and `getStateBundle`/`structFor` reject `FORWARD`
     the same way (`util/stateBundle.py:172`, `util/fieldRegistry.py:227`).

**Why this matters -- and, just as importantly, why *speed* is not the point.** `warpSPH`
already has two working implicit solvers, both hand-built with no AD of any kind:

- **IISPH** (`modules/incompressible/incompressible.py`): relaxed-Jacobi iteration solving for a
  pressure field that drives predicted density back to rest density.
- **Implicit particle shifting** (`modules/shifting/implicitShifting.py`,
  `wp_implicitShifting.py`, `bicgstab.py`, ported from diffSPH): one Newton step on `grad_x C = 0`
  (the SPH concentration field), with a matrix-free Hessian action fed to a Jacobi-preconditioned
  BiCGStab. Its Hessian comes from `sphKernel`/`sphKernelGradient`/`sphKernelHessian` called
  *directly*, per-pair, bypassing `launchOperator`/`OperatorSpec` entirely -- a hand-rolled
  `wp.kernel`, not a `warpOperation` call.

Both work, and both were genuinely hard to get right: `implicitShifting.py`'s own docstring
documents two non-obvious fixes needed to port diffSPH's version correctly -- a self-pair term
that is analytically zero but numerically unstable in the Hessian's near-origin regularization,
and a block-symmetry subtlety (`Hess(C)`'s off-diagonal block is `-omega_k H_ik`, not the naive
`omega_j H_ij` placement) that silently solved in the wrong direction until diagnosed against a
dense finite-difference Hessian. That is exactly the kind of hand-derivation effort a bidirectional
(forward *and* reverse) adjoint capability should be able to remove: if `warpOperationJVP` (Phase
2) can hand back an exact position-tangent and, composed with itself, an exact Hessian-vector
product for any operator, then standing up a new Newton-Krylov SPH solve stops being a
per-problem derivation exercise and becomes "call the existing operator, ask for its JVP."

**So the goal of every "implicit" phase below is a side-by-side comparison against the existing
hand-built solvers, graded on correctness and on how little bespoke reasoning the automatic path
needs -- not on wall-clock speed.** An automatic solve that is 3x slower than `bicgstab.py`'s
hand-tuned matvec but needs no per-operator Hessian derivation, and doesn't require someone to
independently rediscover the self-pair/block-symmetry pitfalls above, is a *win* for this plan's
purposes. Matching or beating the hand-built solvers' performance is explicitly not a goal or a
gate for any phase here.

## Phase 1 -- Tangent rollout on the wave-equation testbed (`warpSPH`, no `warpSPHCore` changes) [DONE 2026-08-18]

Goal: a working, tested forward-mode sensitivity `d(u(x,T))/d(source position, magnitude)`,
cross-checked against the existing backward-mode result. The cheap starting case -- it proves the
plumbing (seed a tangent IC, propagate it, compare to reverse-mode) without yet needing any new
`warpSPHCore` math, since Tier 1 alone covers it (positions are frozen in this scheme).

1. **Seed the tangent initial condition.** `sampleSmoothPointSourceWaveSystem`
   (`src/warpSPH/sample/waveSystem.py:116`) builds `u0` from `_wendlandKernelBump(distances,
   radius)`, a plain-torch function of `position`/`magnitude` -- already autograd-differentiable.
   Get its JVP via `torch.func.jvp` (or an explicit product-rule expansion) w.r.t.
   `position`/`magnitude`, producing `du0` (and `dv0 = 0`).
2. **Run the tangent trajectory.** Add a small driver (new `tests/test_forwardModeWave.py`,
   modeled on `_buildStandingWaveSystem`/`_standingWaveError` in the existing
   `tests/test_waveEquation.py`) that builds two `WaveSystemv3` instances sharing the same
   `adjacency`/`domain`/`dt`/`schemeConfig`/integrator: one on `(u0, v0)`, one on `(du0, dv0)`.
   Step both through the identical integrator/`f_wave_equation` call sequence -- no new
   `warpSPHCore` code involved, since `f_wave_equation` is already linear in `(u, v)` under these
   settings, so applying it to the tangent state *is* the JVP.
3. **Cross-validate against reverse-mode.** Reuse the existing gradient-flow pattern (step 5 of
   `WAVE_EQUATION_PLAN.md`, already in `tests/test_waveEquation.py`): build one more system with
   `position.requires_grad_()`, run `f_wave_equation`, sum a scalar probe of `u(T)`, `.backward()`,
   and compare `probe.grad`-direction-dot-product against `du(T)` at the same probe points (a
   directional-derivative check, contracting the reverse gradient with the same perturbation
   direction used to seed `du0`). Assert agreement to float32/float64 tolerance across a couple of
   probe points/times and at least two source positions (1D and 2D).
4. **Document the linearity precondition**: this "tangent = rerun on perturbed IC" trick is valid
   only because positions/support/adjacency are frozen and CRK/renorm are off -- a Tier-1-only
   special case, not the general mechanism.

Files touched: `warpSPH/tests/test_forwardModeWave.py` (new).

## Phase 2 -- Promote Tier 1 into a supported `warpSPHCore` API, shaped for Tier 2 [DONE 2026-08-18]

Goal: stop requiring every caller to hand-apply the "relaunch on the tangent array" trick; give it
a name, a test, and an `ExecutionMode.FORWARD` path that isn't a blanket `NotImplementedError`.
**Design the entry point's signature for the general case now** (tangent
positions/supports/masses/densities as optional arguments) even though only the value-tangent path
is implemented here -- Phase 4 extends an existing API instead of replacing it.

1. **Unblock `FORWARD` at the struct layer**, per `warpier_core.md` Phase 6 Step G finding #1:
   Tier 1 needs no new struct type, so `structFor`'s `FORWARD` rows
   (`util/fieldRegistry.py:210-222`) can alias the existing `REVERSE` rows instead of raising, and
   `getStateBundle(dim, ExecutionMode.FORWARD)` (`util/stateBundle.py:172`) can hand back the same
   bundle `REVERSE` uses.
2. **Add a thin, documented JVP entry point** next to `warpOperation`
   (`src/warpSPHCore/operations.py`): `warpOperationJVP(..., tangentQueryValues=None,
   tangentReferenceValues=None, tangentQueryPositions=None, tangentQuerySupports=None, ...)` --
   accepting the full Tier-2 tangent surface in its signature from the start. Value-tangent-only
   calls just re-invoke `warpOperation` with the tangents substituted in (Tier 1's entire
   implementation). Any position/support/mass/density tangent argument raises `NotImplementedError`
   naming Tier 2 explicitly, until Phase 4. Keep `launchOperator`'s explicit `FORWARD` rejection
   (`autograd/operator_spec.py:190`) for anything routed outside this entry point.
3. **Promote the spike into a standing test.** `tests/operations/test_forward_mode_tier1.py`
   asserting `warpOperationJVP` reproduces `spike_forward_mode_tier1.py`'s JVP identity, the same
   way `test_gradcheck_scripts.py` gates the spike script. Include a test that Tier-2 arguments
   raise cleanly.
4. **Update `warpier_core.md`'s Phase 6 status** to record Tier 1's production landing.

Files touched: `src/warpSPHCore/operations.py`, `src/warpSPHCore/util/fieldRegistry.py`,
`src/warpSPHCore/util/stateBundle.py`, `src/warpSPHCore/autograd/operator_spec.py`,
`tests/operations/test_forward_mode_tier1.py` (new), `warpier_core.md`.

## Phase 3 -- Goal 1: an implicit wave-equation step, powered entirely by Tier 1 [DONE 2026-08-18]

Goal: the smallest possible demonstration that a forward-mode JVP bridge is enough to stand up an
implicit SPH solve with *no per-problem Hessian derivation at all* -- "a single SPH operation and
its forward mode," as the toy problem this whole plan is meant to motivate.

Because positions stay frozen in the wave-equation scheme, an implicit step's residual is *linear*
in the unknown `u^{n+1}` (the PDE itself is linear) -- Newton's method degenerates to exactly one
linear solve, and the "Jacobian" the solve needs is just the Laplacian operator itself acting on a
trial field.

1. **Pick a scheme.** Backward Euler first (simplest, unconditionally stable, matches
   `warpSPHIntegrators/NOTES.md` §3.6's "tableau only" DIRK entries); implicit midpoint as a
   follow-up once backward Euler works, since `NOTES.md` §3.3 flags it as strictly better for a
   velocity-dependent force (this scheme's `-damping*v` term) and it's the same shape of stage
   equation.
2. **Matvec = `warpOperationJVP`, nothing else.** The stage system `(I - a*dt^2*c^2*Laplacian) *
   u = rhs` is solved with CG (the operator should be symmetric for a fixed neighbourhood under
   `SupportScheme.SuperSymmetric` -- verify this empirically on the test case rather than assume
   it, and fall back to BiCGStab, reusing `warpSPH/modules/shifting/bicgstab.py`'s existing
   generic solver, if it isn't). Each CG matvec is exactly one
   `warpOperationJVP(Laplacian, tangentQueryValues=p)` call from Phase 2 -- no hand-derived
   Jacobian, no finite differences.
3. **Validate two ways**: (a) against the existing explicit `f_wave_equation` rollout at a small
   `dt` where both should agree closely, and (b) a convergence check mirroring
   `tests/test_waveEquation.py`'s standing-wave closed-form comparison, confirming the implicit
   scheme's own expected order.
4. **Write up what this did and didn't need.** The point worth recording explicitly: no Hessian
   was hand-derived, no per-operator matvec was written -- the existing `warpOperationJVP` from
   Phase 2 *is* the matvec. Contrast this in one paragraph with `wp_implicitShifting.py`'s
   from-scratch Hessian derivation (Phase 4 makes the contrast direct on a harder case).

Files touched: new `warpSPH/tests/test_implicitWaveEquation.py` (or a script promoted to a test
once stable).

## Phase 4 -- Goal 2: automatic vs. hand-built implicit particle shifting [IN PROGRESS: steps 1 (Density only)/2/3/4 done 2026-08-18/19]

**Progress.** Before starting, found and corrected a stale-documentation blocker: `warpier_core.md`
had claimed the `Field` abstraction was "not started", which would have made a generic Tier-2 API
depend on unscoped prerequisite work. It was already done (`warpier_fields.md`'s own "Status as of
2026-08-18": Steps 0/A-J complete, `Field` with a pair-capable `tangent` slot, `StateBundle`, the
declared `OperatorSpec`/`launchOperator` ABI) -- `warpier_core.md` corrected in `warpSPHCore`
commit `fe238d5`.

* **Step 1 -- Density only, done.** `sphKernelJVP`/`sphKernelJVP_ij`
  (`src/warpSPHCore/kernels/kernelJVP.py`) and `computePairwiseSupportJVP`
  (`util/support.py`) promote Tier 2.1's spike math into real `kernels/` functions.
  `computeSPHDensityPositionJVP` (`coreOperations/wp_densityJVP.py`) is a production evaluator --
  one thread per real adjacency-list pair (mirroring `wp_implicitShifting.computeShiftingPairTerms`'s
  established pattern, since `launchOperator` doesn't support pair-indexed threading), not the
  spike's dense-all-pairs shortcut -- giving Density's position/support/reference-mass tangent.
  `warpOperationJVP` dispatches to it for exactly that combination; every other Tier-2 combination
  (the other five operators; any value or density tangent) still raises `NotImplementedError`.
  Gated by `tests/operations/test_forward_mode_tier2_density.py` (9 cases, reverse-mode-Jacobian
  reference, ~1e-6 float32 agreement) plus the usual unaffected-baseline checks (full suite 136
  passed/1 skipped, `operation_matrix.py --device cpu` OK=258 unchanged, all six
  `gradcheck_*_native.py` scripts green). `warpSPHCore` commit `b1d798f`.
  Interpolate/Gradient/Divergence/Curl/Laplacian/renorm/CRK Tier-2 JVPs are **not** done --
  scoped down to Density because that's what step 2 needed, not attempted for all six operators
  yet as step 1's own text originally asked.
* **Step 2 -- done.** Validated standalone against `wp_implicitShifting.py`'s own `J`/`Jw`, exactly
  as asked: `warpOperationJVP(Density, tangentQueryPositions=<coordinate basis>)` with `omega`
  passed as the mass channel, one call per spatial dimension, reconstructs `Jw` bit-for-bit
  identically to the hand-built pairwise-`sphKernelGradient`+`scatter_sum` computation, on a
  jittered lattice (`warpSPH/tests/test_implicitShiftingGradientJVP.py`, `warpSPH` commit
  `91b92fb`). Confirms the claim literally: no new derivation was needed for the gradient itself,
  only for wiring the existing Tier-2.0 building blocks (`sphKernelGradient`'s ingredients) through
  a JVP-shaped production entry point.
* **Step 3 -- done.** `Hess(Density) @ v` ("the actual experiment") is now a production entry
  point: `computeSPHDensityPositionHVP` (`coreOperations/wp_densityHVP.py`), dispatched to by a
  new `warpOperationHVP` sibling to `warpOperationJVP` (`operations.py`). **Finding: composing
  this generically through torch does not work, confirmed by trying it first, not assumed.**
  `torch.func.jvp` applied twice over `computeSPHDensityPositionJVP` errors immediately
  (`RuntimeError: Cannot access data pointer of Tensor that doesn't have storage` -- functorch's
  dual tensors have no real storage for `wp.from_torch` to view), and nested
  `torch.autograd.forward_ad.make_dual`/`dual_level` runs but silently returns `tangent=None` --
  the same failure mode `spike_forward_mode_tier1.py` already found one order lower for
  `StateAwareWarpFunction` (no `jvp()` ever registered), sharper here since
  `computeSPHDensityPositionJVP` isn't wrapped in a `torch.autograd.Function` at all, forward or
  reverse. `torch.autograd.functional.hessian`'s usual double-backward HVP trick is equally out --
  `StateAwareWarpFunction.backward()` reads a `wp.Tape`, which isn't itself differentiable (same
  root cause `spike_forward_mode_tier1.py` flagged for `torch.autograd.functional.jvp`'s
  double-backward variant). So Step 3 needed the "small explicit second-order helper" the plan
  flagged as the fallback -- but that helper turned out to need **zero new kernel math**: the
  closed form `HVP_i = sum_j m_j * H_ij @ (v_i - v_j)` falls out of differentiating
  `computeSPHDensityPositionJVP`'s own `dW_ij = ∇W_ij · dx_ij` formula once more by hand, and
  `H_ij` is exactly `kernels.hessian.sphKernelHessian` -- an existing, already-validated Tier-2.0
  building block (`warpier_adjoint.md`'s own building-blocks table), previously only consumed by
  `warpSPH`'s hand-rolled `wp_implicitShifting.py`. Validated two independent ways: (a)
  `scripts/spike_forward_mode_tier2_density_hvp.py` (float64 subprocess, gated in
  `test_gradcheck_scripts.py`'s `SPIKE_SCRIPTS`) checks it against
  finite-differencing `computeSPHDensityPositionJVP` itself along `v` (a different formula, no
  `sphKernelHessian` involved), agreeing to ~1e-9 relative on Gather/Scatter/MeanSymmetric/
  SuperSymmetric in 1D and 2D; (b) `warpSPH/tests/test_implicitShiftingHessianJVP.py` (new, `warpSPH`
  commit pending) checks it bit-close against `wp_implicitShifting.py`'s own hand-built `H`/
  `_multiplyLaplacianBlock` matvec on the same jittered-lattice case Step 2 used, passing at
  `rtol=1e-4` on first run. Two scope notes carried forward, not fixed here: `sphKernelHessian`
  itself only special-cases `SuperSymmetric`'s two-term-average branch, not
  `KernelMeanSymmetric`'s (both get it in `sphKernelJVP_ij`'s first-order dispatch) -- a
  pre-existing gap in that Tier-2.0 building block, silently inherited by both
  `wp_implicitShifting.py` (sidesteps it by always using `Gather`) and this new function, flagged
  in the spike script rather than fixed; and self-pairs (`i==j`) are dropped by hand before
  assembly in `computeSPHDensityPositionHVP`, the same way `computeImplicitShift` needs to for its
  own Hessian -- i.e. the composed-JVP route did **not** turn out to need this hand-holding any
  less than the hand-built version does, a first data point for step 5's own question, not yet
  step 5 itself (this was needed just to get step 3's own numbers to agree, not a deliberate test
  of the pitfall).
* **Step 4 -- done.** `warpSPH/modules/shifting/implicitShiftingAutomatic.py`'s new
  `computeImplicitShiftAutomatic` is a drop-in replacement for `computeImplicitShift` (same
  signature, same `bicgstabSolve` call, same boundary/relaxation/initializer handling) with `grad
  C`/`Hess C . v` sourced entirely from `warpOperationJVP`/`warpOperationHVP` (one call per
  coordinate direction for the RHS and the Jacobi-preconditioner diagonal, one general call per
  matvec) instead of `wp_implicitShifting.py`'s hand-rolled per-pair kernel +
  `torch.einsum`/`scatter_sum` assembly -- no `sphKernelGradient`/`sphKernelHessian` call and no
  hand-derived block-symmetry sign anywhere in the new file. Validated two ways
  (`warpSPH/tests/test_implicitShiftingComparison.py`, new): a single Newton step from the same
  starting state agrees with the hand-built solve to `rtol=1e-3` (both solves are handed an
  equivalent linear system, so this mostly exercises `bicgstabSolve` itself, not new math); 8
  outer relaxation iterations (rebuilding the adjacency each step, mirroring
  `wrapper.solveShifting`'s own loop) drive both to matching density-uniformity equilibria,
  tracking within ~6e-5 of each other at every step. **Not glossed over**: pushing past 8
  iterations on this same seed/case, the two histories start diverging by iteration ~12 (automatic
  jumps back up while hand-built keeps relaxing smoothly) -- consistent with, and further evidence
  for, this plan's own already-documented finding (Status section above, and
  `implicitShifting.py`'s own docstring) that `implicitRelaxation=0.1`'s undamped-per-step Newton
  iteration is only marginally stable regardless of which matvec drives it; not chased further
  here since fixing that robustness gap is explicitly out of this plan's scope, but it's the first
  concrete data point for step 5's "does the automatic path reproduce the same pitfalls" question,
  worth carrying into step 5/6's own writeup rather than re-discovering there.
* **Steps 5-6 -- not started.** Step 5 (deliberately probe the self-pair/block-symmetry pitfalls --
  step 3 already found the self-pair drop was still needed by hand, and step 4 found a matching
  marginal-stability sensitivity; formalize both plus the block-symmetry check) and step 6
  (three-way comparison against `computeDeltaShift` too, plus the effort/robustness writeup) are
  next.
* **A finding while validating step 2, worth carrying into step 6's comparison**: the jittered-
  lattice implicit-shifting baseline (`warpSPH/tests/test_implicitShifting.py`) this phase's
  step 4/6 will run against has its own pre-existing GPU-only marginal-stability issue (see this
  plan's "Status" section above, item 2 under the `test_implicitShifting.py` finding) --
  unfixed, flagged for whoever runs step 6's comparison.

Goal: wire Tier 2 into `warpOperationJVP`, then build an implicit shifting solve whose `grad C`/
`Hess C` come from *composed JVPs* instead of `wp_implicitShifting.py`'s hand-rolled per-pair
kernel, and compare three ways: explicit shift (`computeDeltaShift`), the existing hand-built
implicit shift (`computeImplicitShift`), and this new automatic one. This is the goal-2 comparison
the user asked for, and the one place in this plan where "how much derivation did this need"
is the headline result, not just a number matching to float64 round-off.

1. **Extend `warpOperationJVP`** for the six core operators using Tier 2.1/2.2/2.4/2.5's already
   assembled JVPs (Density/Interpolate, Gradient/Divergence/Curl/Laplacian-Brookshaw,
   renormalization, CRK). Gate against the corresponding `gradcheck_*_native.py` scripts and
   `operation_matrix.py`, same pattern as every prior tier in `warpier_adjoint.md`.
2. **`grad C` is exactly Tier 2.1's Density-operator position JVP.** `C_i = sum_j omega_j * W_ij`
   is `Density_i = sum_j m_j * W_ij` with `omega` standing in for mass -- so
   `warpOperationJVP(Density, tangentQueryPositions=v)` (with `omega` passed as the mass channel)
   *is* `grad C . v` with no new derivation. Validate this identity first, standalone, against
   `wp_implicitShifting.py`'s `J` output.
3. **`Hess C . v` is a JVP of that JVP** ("Tier-2-squared"): differentiate
   `warpOperationJVP(Density, tangentQueryPositions=v)` itself w.r.t. positions in the direction
   `v` again. Whether this composes cleanly through the existing bridge or needs a small explicit
   second-order helper is exactly the kind of thing to discover by trying it, not to
   pre-design -- this step is the actual experiment. Validate against `wp_implicitShifting.py`'s
   `H` output on the same case.
4. **Swap the matvec, keep the solver.** `bicgstab.py`'s `bicgstabSolve` already takes a plain
   `matvec` closure -- feed it the composed-JVP Hessian action instead of
   `_multiplyLaplacianBlock`'s hand-assembled one, everything else in `computeImplicitShift`
   (preconditioner, boundary masking, relaxation) unchanged. Run both on the same jittered-lattice
   test case `implicitShifting.py`'s own docstring describes, and confirm the two solves converge
   to the same equilibrium shift.
5. **Specifically test the two pitfalls the hand port hit**, since they're the concrete evidence
   for this phase's actual point: does the composed-JVP Hessian handle the self-pair (`i==j`,
   zero separation) case correctly without a manual drop, the way the hand version needed one? Is
   the resulting operator symmetric without anyone having to work out the `-omega_k H_ik`
   off-diagonal sign by hand? Report both findings explicitly, whichever way they come out -- if
   the automatic path reproduces the same pitfalls, that's worth knowing too, and changes how much
   of a win "automatic" actually is here.
6. **Compare, three ways, on correctness and on effort/robustness, not speed**: `computeDeltaShift`
   (explicit baseline), `computeImplicitShift` (hand Hessian + BiCGStab), and the new
   automatic-JVP + BiCGStab solve. Report equilibrium shift agreement between the two implicit
   solves, and qualitatively how each was built (lines of hand-derived math vs. composed calls to
   an existing bridge) -- that comparison, not a timing table, is the deliverable.

Files touched: `src/warpSPHCore/operations.py` (Tier-2 branch of `warpOperationJVP`), new
`warpSPH/modules/shifting/implicitShiftingAutomatic.py` (or a script, if it doesn't need to be
production code yet), new `warpSPH/tests/test_implicitShiftingComparison.py`.

## Phase 5 -- Goal 3: the incompressible wrapper already exists

Not a task -- a pointer, so it isn't accidentally rebuilt. `systems/incompressible.py` +
`modules/incompressible/incompressible.py` already wrap the compressible/WCSPH momentum equation
with an IISPH pressure solve (relaxed-Jacobi iteration on the density-error residual,
`solveIncompressible`). "Wrap the explicit WCSPH/compressible scheme into an incompressible
wrapper" is already done; what Phase 6 (below) adds is an *alternative* implicit solver to compare
it against, not a rebuild of the wrapping itself.

## Phase 6 -- Goal 4: automatic vs. IISPH for incompressibility (future, separate plan)

Not scoped in detail here -- flagged as the natural next target once Phase 4 proves the
automatic-Hessian pattern out on the smaller shifting case, and deliberately kept out of this
plan's near-term scope because it's a materially bigger lift: coupled pressure/velocity DOFs
across the whole domain (not one scalar field per particle), boundary and free-surface handling,
and EOS coupling that shifting's pure-geometry objective doesn't have.

The shape would mirror Phase 4: an automatic Newton-Krylov pressure-Poisson solve built from
composed `warpOperationJVP` calls (the pressure-gradient/divergence operators already exist as
core operators, so no new operator-level derivation should be needed, only composition), compared
against `solveIncompressible`'s existing IISPH relaxed-Jacobi iteration -- again on correctness and
on how much bespoke pressure-solver machinery the automatic path avoids, not on iteration count or
wall-clock time. `warpSPHIntegrators/NOTES.md` §3.4's solver ladder (Picard -> JFNK-with-FD ->
exact-JVP -> user `solve_linear`) is the right frame for where this sits: IISPH is closer to rung
1 (fixed-point relaxation), the hand-built implicit shift is rung 4 (user-supplied, exact,
hand-derived), and this goal is asking what rung 3 (exact JVP matvecs, general-purpose) actually
costs and buys once it exists. Scope this as its own plan once Phase 4's findings are in.

## `warpSPHIntegrators` context (read before scoping Phase 3 or Phase 6 further)

`warpSPHIntegrators` (`~/dev/warpSPHIntegrators`, the library every scheme's
`integrator.function(...)` already calls) has a detailed, empirically-verified implicit/multistep
plan in its own `NOTES.md` §3 ("Multistep and implicit methods"), independent of anything in this
plan. Read it in full before building Phase 3's implicit wave-equation driver or scoping Phase 6;
the load-bearing points:

- **§3.0-3.1**: because this simulation never re-sorts particles and carries its neighbour list
  through the state (revalidated cheaply, not rebuilt), the two objections that normally make
  implicit SPH expensive don't apply here -- "one RHS evaluation = one neighbour rebuild" is
  false, and a DIRK stage solve needs no new state algebra (a ~60-line probe over the library's
  existing `initializeNewState`/`applyStateUpdate`/`updateStep` primitives already reaches full
  order). Phase 3 could use this driver directly instead of a bespoke CG loop, once it exists as a
  registered scheme -- worth checking at that point rather than assuming a hand-rolled loop is
  the final form.
- **§3.4 "Newton without forward-mode AD"** concludes Newton's stage solve does **not** require
  forward-mode AD at all -- FD directional derivatives are enough for the non-stiff regime this
  codebase mostly lives in. Its solver ladder: (1) fixed-count Picard, (2) JFNK with FD matvecs,
  (3) exact matvecs via `torch.func.jvp`/a native JVP -- "a speed and robustness optimisation,
  **not** a capability gate", (4) a user-supplied `solve_linear`. This plan's Phases 3/4/6 are
  building rung 3 generically and comparing it against existing rung-4 (shifting) and rung-1
  (IISPH) implementations already in production -- the "automatic vs. hand-built, on ease not
  speed" framing above is exactly what makes rung 3 worth having even though the ladder's own
  text says it's optional.
- **§3.8**'s phased effort estimate (Phase 0 groundwork ~5-6 d -> DIRK+Picard driver ~4-5 d ->
  explicit multistep ~2-3 d, "~2 weeks total") is entirely within `warpSPHIntegrators`, gated on
  nothing from `warpSPHCore` -- worth scoping as its own small piece of work if Phase 3 wants the
  registered driver rather than a one-off script.

## Explicitly out of scope for this plan

- **Actually registering an implicit scheme in `warpSPHIntegrators`** (vs. a standalone script
  for Phase 3) -- `NOTES.md` §3.8's own phased plan, pick up separately if Phase 3's findings
  justify it.
- **Phase 6's full implementation** -- explicitly deferred to its own plan.
- **Tier 2 for operators beyond the six core ones and the shifting comparison** (momentum
  equation, mDBC, surface detection, etc.).
- **Performance tuning or optimization of any automatic path** -- correctness and ease of
  construction are this plan's success criteria; speed is not graded anywhere in it.

## Verification

- Phase 1: `pytest tests/test_forwardModeWave.py` in `warpSPH`; existing
  `tests/test_waveEquation.py` / `tests/test_physics.py -k wave` unaffected.
- Phase 2: `python scripts/spike_forward_mode_tier1.py` green; new
  `pytest tests/operations/test_forward_mode_tier1.py` green; `pytest tests/` and
  `python scripts/operation_matrix.py --device cpu` unaffected (baseline `OK=258, HIGH=0, ERR=0,
  NAN=0`); the six `gradcheck_*_native.py` scripts unaffected.
- Phase 3: new implicit-wave test passes its convergence/agreement checks; existing wave-equation
  tests unaffected.
- Phase 4: new shifting-comparison test passes; `python scripts/gradcheck_deltaShift.py` and
  `tests/test_implicitShifting.py` (the existing hand-built solver's own tests) unaffected; the
  six `gradcheck_*_native.py` scripts and `operation_matrix.py --device cpu` in `warpSPHCore`
  unaffected.
- End-to-end: re-run Phase 1's cross-validation after Phase 2 lands, swapping its hand-rolled
  double-rollout for `warpOperationJVP` where convenient, confirming they agree.
