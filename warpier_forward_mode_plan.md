# Forward-mode evaluation: from toy operators to automatic implicit SPH solves

Companion to `warpier_core.md` Phase 6 ("Extend States for Forward-Mode AD", audited-but-not-started
as of 2026-08-17/18) and `warpier_adjoint.md` (Tiers 2.0-2.5 JVP derivations, all done as of
2026-08-18). Those two documents established that the *math* for forward-mode SPH is finished and
validated; this plan is about actually running it, using testbeds in the sibling `warpSPH` repo
(`~/dev/warpSPH`) at increasing difficulty, and about where that connects to `warpSPHIntegrators`'s
(`~/dev/warpSPHIntegrators`) own already-scoped implicit-integrator plan. Not started -- no code
changes have been made yet.

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

## Phase 1 -- Tangent rollout on the wave-equation testbed (`warpSPH`, no `warpSPHCore` changes)

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

## Phase 2 -- Promote Tier 1 into a supported `warpSPHCore` API, shaped for Tier 2

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

## Phase 3 -- Goal 1: an implicit wave-equation step, powered entirely by Tier 1

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

## Phase 4 -- Goal 2: automatic vs. hand-built implicit particle shifting

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
