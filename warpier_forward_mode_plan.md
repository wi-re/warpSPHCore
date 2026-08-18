# Forward-mode evaluation, from the wave equation to moving-particle SPH

Companion to `warpier_core.md` Phase 6 ("Extend States for Forward-Mode AD", audited-but-not-started
as of 2026-08-17/18) and `warpier_adjoint.md` (Tiers 2.0-2.5 JVP derivations, all done as of
2026-08-18). Those two documents established that the *math* for forward-mode SPH is finished and
validated; this plan is about actually running it, using two testbeds in the sibling `warpSPH`
repo (`~/dev/warpSPH`) at genuinely different difficulty, and about where that connects to
`warpSPHIntegrators`'s (`~/dev/warpSPHIntegrators`) own already-scoped implicit-integrator plan.
Not started -- no code changes have been made yet.

## Context

`warpSPHCore`'s adjoint work has two layers:

1. **Reverse-mode (VJP)** is production-complete and battle-tested (`gradcheck_*.py`, `operation_matrix.py`), across both the six core operators and custom frontend kernels built on the same building blocks (e.g. `warpSPH/scripts/gradcheck_deltaShift.py`, see Phase 3 below).
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

**The goal of this plan is a forward-mode mechanism general enough for real SPH, not just for
one linear toy PDE.** Two testbeds already exist in `warpSPH` at genuinely different difficulty:

- **The wave equation** (`schemes/waveEquation.py`, `tests/test_waveEquation.py`, landed by
  `1622720 fix wave equation code to serve as demo code for forward mode development`): positions
  are frozen, adjacency is built once, no CRK/renorm. The PDE
  `d2u/dt2 = c^2 laplacian(u) - damping*v` is therefore *linear*, and a source's `position`/
  `magnitude` are already real leaf tensors with a passing reverse-mode gradient-flow test
  (`WAVE_EQUATION_PLAN.md` step 5). This is the cheap, fast-to-validate starting case -- good for
  proving the mechanism works end-to-end, but it never exercises a position tangent (Tier 2),
  because positions never move.
- **delta-SPH particle shifting** (`modules/shifting/`): a genuinely harder, already-existing
  case where positions *do* move and the result isn't linear or easily hand-proven. `solveShifting`
  (`modules/shifting/wrapper.py:47`) iterates: rebuild adjacency at the *current* (moved)
  positions -> compute a per-particle anti-clustering displacement
  (`computeDeltaShift`/`computeDeltaShiftWarp`, `modules/shifting/delta.py`,
  `sample/wp_deltaShift.py`) -> add it to `positions` -> repeat. It is **already
  reverse-mode gradchecked** w.r.t. positions/supports/masses/densities
  (`scripts/gradcheck_deltaShift.py`), which makes it an ideal Tier-2 validation target: a
  ground-truth reverse-mode Jacobian already exists to check a new forward-mode JVP against,
  on genuinely position-dependent, iteratively-re-neighbored production code -- not a synthetic
  case built for this plan. There is also an **unimplemented, stubbed-out "implicit" shifting
  variant** already named in the config (`ShiftingScheme.implicit`,
  `configurations/moduleConfigurations/shifting.py:21` -- defined but never dispatched on in
  `wrapper.py`) and a commented-out "alternate implicit-particle-shift path" referenced in
  DFSPH (`schemes/dfsph.py:8`). This is the concrete, already-scoped landing spot for exactly the
  kind of implicit, position-solving step the user has in mind -- see Phase 4.

So the plan below is staged by difficulty, not by repo: Phase 1 proves the mechanism cheaply on
the linear, frozen-position wave equation; Phase 2 promotes that into a real `warpSPHCore` API
*designed* to carry position/support tangents even though it doesn't implement them yet; Phase 3
turns those tangents on and validates them against the harder, moving-particle shifting case;
Phase 4 is the implicit particle-shift step this all points toward, tying into
`warpSPHIntegrators`'s own already-scoped implicit-solver plan.

## Phase 1 -- Tangent rollout on the wave-equation testbed (`warpSPH`, no `warpSPHCore` changes)

Goal: a working, tested forward-mode sensitivity `d(u(x,T))/d(source position, magnitude)`,
cross-checked against the existing backward-mode result. Deliberately the easy case -- it proves
the plumbing (seed a tangent IC, propagate it, compare to reverse-mode) without yet needing any
new `warpSPHCore` math, since Tier 1 alone covers it.

1. **Seed the tangent initial condition.** `sampleSmoothPointSourceWaveSystem`
   (`src/warpSPH/sample/waveSystem.py:116`) builds `u0` from `_wendlandKernelBump(distances,
   radius)`, a plain-torch function of `position`/`magnitude` -- already autograd-differentiable
   (that's what the existing backward test exercises). Get its JVP directly via
   `torch.func.jvp` (or an explicit product-rule expansion, since the function is a two-line
   closed form) w.r.t. `position`/`magnitude`, producing `du0` (and `dv0 = 0`, since `v0` doesn't
   depend on the source in this case).
2. **Run the tangent trajectory.** Add a small driver (new `tests/test_forwardModeWave.py`,
   modeled on `_buildStandingWaveSystem`/`_standingWaveError` in the existing
   `tests/test_waveEquation.py`) that builds two `WaveSystemv3` instances sharing the same
   `adjacency`/`domain`/`dt`/`schemeConfig`/integrator: one on `(u0, v0)`, one on `(du0, dv0)`.
   Step both through the identical integrator/`f_wave_equation` call sequence -- no new
   `warpSPHCore` code involved, since `f_wave_equation` is already linear in `(u, v)` under these
   settings, so applying it to the tangent state *is* the JVP (this is Tier 1's identity, exercised
   live rather than on a synthetic 7-particle case).
3. **Cross-validate against reverse-mode.** Reuse the existing gradient-flow pattern (step 5 of
   `WAVE_EQUATION_PLAN.md`, already in `tests/test_waveEquation.py`): build one more system with
   `position.requires_grad_()`, run `f_wave_equation` for the same number of steps, sum a scalar
   probe of `u(T)` at one or more query points, `.backward()`, and compare
   `probe.grad`-direction-dot-product against `du(T)` at the same probe points from step 2 (a
   directional-derivative check, since reverse-mode gives the full gradient and forward-mode
   gives one directional derivative at a time -- contract the reverse gradient with the same
   perturbation direction used to seed `du0`). Assert agreement to float32/float64 tolerance
   across a couple of probe points/times and at least two source positions (1D and 2D, matching
   the existing test's dimension coverage).
4. **Document the linearity precondition** in the new test/module's docstring: this "tangent =
   rerun on perturbed IC" trick is valid only because positions/support/adjacency are frozen and
   CRK/renorm are off; it is a Tier-1-only special case, not the general mechanism -- Phase 3 is
   where the general (Tier 2) case gets built and proven on a case where this shortcut does not
   apply.

Files touched: `warpSPH/tests/test_forwardModeWave.py` (new), possibly a small shared helper
extracted from `tests/test_waveEquation.py` if the two files end up duplicating system-building
code -- check for that when writing it rather than pre-deciding.

## Phase 2 -- Promote Tier 1 into a supported `warpSPHCore` API, shaped for Tier 2

Goal: stop requiring every caller to know and hand-apply the "relaunch on the tangent array"
trick; give it a name, a test, and an `ExecutionMode.FORWARD` path that isn't a blanket
`NotImplementedError`. **Design the entry point's signature for the general case now** (tangent
positions/supports/masses/densities as optional arguments) even though only the value-tangent
path is implemented in this phase -- so Phase 3 extends an existing API instead of replacing it.

1. **Unblock `FORWARD` at the struct layer**, per `warpier_core.md` Phase 6 Step G finding #1:
   Tier 1 needs no new struct type, so `structFor`'s `FORWARD` rows
   (`util/fieldRegistry.py:210-222`) can simply alias the existing `REVERSE` rows instead of
   raising, and `getStateBundle(dim, ExecutionMode.FORWARD)` (`util/stateBundle.py:172`) can hand
   back the same bundle `REVERSE` uses.
2. **Add a thin, documented JVP entry point** next to `warpOperation`
   (`src/warpSPHCore/operations.py`), e.g. `warpOperationJVP(..., tangentQueryValues=None,
   tangentReferenceValues=None, tangentQueryPositions=None, tangentQuerySupports=None, ...)` --
   accepting the full Tier-2 tangent surface in its signature from the start. When only the
   value-tangent arguments are non-`None`, it calls `warpOperation` again with the tangents
   substituted for `queryValues`/`referenceValues` (Tier 1's entire implementation is "no new
   math"). When any position/support/mass/density tangent argument is non-`None`, raise
   `NotImplementedError` naming Tier 2 explicitly -- a real, narrow gap to fill in Phase 3, not a
   silent wrong answer. Keep `launchOperator`'s explicit `FORWARD` rejection
   (`autograd/operator_spec.py:190`) for anything routed outside this entry point.
3. **Promote the spike into a standing test.** Add `tests/operations/test_forward_mode_tier1.py`
   asserting the new `warpOperationJVP` reproduces `spike_forward_mode_tier1.py`'s JVP identity
   on the same small cases, the same way `test_gradcheck_scripts.py` already gates the spike
   script itself -- this closes the gap between "a script proved it once" and "production code is
   pinned against regressing it." Include a test that the Tier-2 arguments raise cleanly.
4. **Update `warpier_core.md`'s Phase 6 status** section to record Tier 1's production landing,
   and add a backlink from there to Phase 1's wave-equation validation as the first real
   (non-synthetic) consumer.

Files touched: `src/warpSPHCore/operations.py`, `src/warpSPHCore/util/fieldRegistry.py`,
`src/warpSPHCore/util/stateBundle.py`, `src/warpSPHCore/autograd/operator_spec.py`,
`tests/operations/test_forward_mode_tier1.py` (new), `warpier_core.md`.

## Phase 3 -- Wire Tier 2, validated on moving-particle shifting

Goal: turn Phase 2's `NotImplementedError` into real position/support tangents, using the
formulas `warpier_adjoint.md`'s Tiers 2.0-2.5 already derived and gradcheck-matched -- this is
"wire already-proven formulas," not new derivation, for the six core operators. Validate against
`solveShifting`/`computeDeltaShiftWarp` (`warpSPH/modules/shifting/`), which moves particles and
rebuilds adjacency every iteration and already has a reverse-mode reference
(`scripts/gradcheck_deltaShift.py`) to check the new JVP against.

1. **Extend `warpOperationJVP`** for the six core operators using Tier 2.1/2.2/2.4/2.5's
   assembled JVPs directly (Density/Interpolate, Gradient/Divergence/Curl/Laplacian-Brookshaw,
   renormalization, CRK -- Tier 2.3's Laplacian-Naive JVP is lower priority per its own writeup).
   Gate against the corresponding `gradcheck_*_native.py` scripts and `operation_matrix.py`, same
   pattern as every prior tier.
2. **`computeDeltaShiftWarp` is not one of the six core operators** -- it's a custom
   `@wp.func`/`@wp.kernel` in `warpSPH` (`sample/wp_deltaShift.py`) built on the same shared
   building blocks (`computeKernelCRK` for `w_ij`, and implicitly a kernel gradient) but with its
   own accumulated math (`sum_j [m_j/(rho_i+rho_j)] * [1+R*(w_ij/W_0)^n] * gradW_ij`). Its Tier-2
   JVP is therefore its own (mechanical) derivation, following Tier 2.1's playbook exactly:
   chain-rule this specific expression through the already-validated kernel-value/-gradient JVP
   building blocks (Tier 2.0/2.1/2.2), rather than assuming the built-in Density operator's JVP
   applies. Land it as a `warpSPH`-side script mirroring `scripts/spike_forward_mode_tier2_*.py`'s
   shape, validated against `gradcheck_deltaShift.py`'s existing reverse-mode reference exactly
   the way every `warpier_adjoint.md` tier validates against its own `gradcheck_*_native.py`.
3. **Chain one `solveShifting` iteration's tangent, then several.** Unlike Phase 1's linear wave
   equation, this is genuinely nonlinear and re-neighbors every iteration, so the "just rerun the
   same primal code on a tangent IC" shortcut does not apply: each iteration's tangent has to be
   propagated through that iteration's actual JVP (position tangent in, updated position tangent
   out), the way a real tangent-linear model works. Validate first for a single iteration
   (`iters=1`, matching `computeDeltaShift`'s own per-call signature) against
   `gradcheck_deltaShift.py`'s Jacobian, then for a short multi-iteration `solveShifting` run,
   confirming the chained tangent still matches a reverse-mode `.backward()` through the same
   multi-iteration rollout.

Files touched: `src/warpSPHCore/operations.py` (Tier-2 branch of `warpOperationJVP`), new
`warpSPH/scripts/spike_forward_mode_shift_tier2.py`, new
`warpSPH/tests/test_forwardModeShifting.py`.

## Phase 4 -- Toward an implicit particle-shift step

Not implemented in this plan -- the concrete, already-scoped next target once Phase 3 lands.
`ShiftingScheme.implicit` exists as a named-but-unwired enum value
(`configurations/moduleConfigurations/shifting.py:21`; `wrapper.py` never dispatches on it), and
`schemes/dfsph.py:8` references a commented-out "alternate implicit-particle-shift path" -- both
are real, already-recognized gaps for exactly the kind of implicit, position-solving step the
delta-SPH shift currently approximates by explicit fixed-point iteration.

This is *not* a blank slate -- `warpSPHIntegrators` (`~/dev/warpSPHIntegrators`, the library
`WaveSystemv3`'s and the compressible/incompressible schemes' `integrator.function(...)` already
calls) has a detailed, empirically-verified implicit/multistep plan sitting in its own
`NOTES.md` §3 ("Multistep and implicit methods"), independent of anything in this plan. Read it
in full before designing anything here; the load-bearing points:

- **§3.0-3.1**: because this simulation never re-sorts particles and carries its neighbour list
  through the state (revalidated cheaply, not rebuilt), the two objections that normally make
  implicit SPH expensive don't apply here -- "one RHS evaluation = one neighbour rebuild" is
  false, and a DIRK stage solve needs no new state algebra (it reuses the library's existing
  `initializeNewState`/`applyStateUpdate`/`updateStep` primitives as-is; a ~60-line probe already
  reaches full order). Note `solveShifting` *does* rebuild adjacency every iteration
  (`wrapper.py:75`) -- worth checking whether that rebuild-per-iteration is actually load-bearing
  for shifting specifically (particles can move far enough per shift to change neighbors) or
  could adopt the same revalidate-don't-rebuild pattern before assuming §3.0's cost argument
  carries over unchanged.
- **§3.3**: implicit midpoint (Gauss-Legendre, s=1) is flagged as the single highest-value
  addition -- symplectic, A-stable, and holds order 2 for a velocity-dependent force.
- **§3.4 "Newton without forward-mode AD"** is the section most directly relevant to Phase 3's
  bridge: it concludes Newton's stage solve does **not** require forward-mode AD at all -- a
  finite-difference directional derivative `J*v ~= (f(Y+eps*v)-f(Y))/eps` drives it to
  `dt*omega = 1000` with no autodiff of any kind, because only Jacobian-*vector* products are
  ever needed (never a full Jacobian, intractable at particle count). Its recommended solver
  ladder is (1) fixed-count Picard -- non-stiff, no AD, covers the primary use case, (2) JFNK with
  FD matvecs -- stiff, backend-agnostic, (3) exact matvecs via `torch.func.jvp`/a native JVP -- "a
  speed and robustness optimisation, **not** a capability gate", (4) a user-supplied
  `solve_linear`. Phase 2/3's `warpOperationJVP` is exactly rung 3 of that ladder for the parts
  of a residual that route through `warpSPHCore` operators (including, for an implicit particle
  shift, the position tangent Phase 3 lands) -- a real accelerant, not a prerequisite.
- **§3.8** lays out a phased effort estimate entirely within `warpSPHIntegrators`, gated on
  nothing from `warpSPHCore`: Phase 0 (shared groundwork, ~5-6 d) -> Phase 2 (DIRK driver +
  fixed-count Picard solver + implicit midpoint/backward Euler/trapezoidal/SDIRK2/TR-BDF2,
  ~4-5 d) -> Phase 1 (explicit multistep, ~2-3 d) -- "~2 weeks total, then stop and reassess."

The actionable next step here, once picked up, is to scope a dedicated plan against
`warpSPHIntegrators/NOTES.md` §3's roadmap with `ShiftingScheme.implicit` (or DFSPH's shelved
path) as the concrete target -- using this plan's Phase 3 `warpOperationJVP` as an optional matvec
accelerant per §3.4 rung 3, and the fixed-point Picard iteration already at the heart of
`solveShifting` today as evidence the non-stiff case is already halfway implemented in spirit,
just not yet driven by a proper Newton/DIRK stage solve.

## Explicitly out of scope for this plan (flagged, not attempted)

- **Tier 2 for operators/schemes beyond the six core ones and delta-shift** (e.g. the momentum
  equation, pressure solves, mDBC, surface detection) -- Phase 3 proves the mechanism generalizes
  past the frozen-position wave equation, but does not attempt every consumer in the codebase.
- **Actually implementing `ShiftingScheme.implicit` or the DFSPH implicit path** (Phase 4) -- scope
  it as its own plan against `warpSPHIntegrators/NOTES.md` §3 once Phase 3 lands.
- **Adaptive/variable time-stepping interaction with forward mode** -- not touched by any phase
  here; the wave equation and shifting testbeds both use a fixed `dt` per run.

## Verification

- Phase 1: `pytest tests/test_forwardModeWave.py` in `warpSPH` (new test), plus confirm the
  existing `pytest tests/test_waveEquation.py` and `pytest tests/test_physics.py -k wave` are
  unaffected (no production code changed in this phase).
- Phase 2: `python scripts/spike_forward_mode_tier1.py` still green in `warpSPHCore`; new
  `pytest tests/operations/test_forward_mode_tier1.py` green; `pytest tests/` and
  `python scripts/operation_matrix.py --device cpu` unaffected (baseline `OK=258, HIGH=0, ERR=0,
  NAN=0`, per every prior tier's gate in `warpier_adjoint.md`); re-run the six
  `gradcheck_*_native.py` scripts to confirm nothing about the new `FORWARD`-mode struct aliasing
  regresses the reverse-mode path.
- Phase 3: `pytest tests/test_forwardModeShifting.py` in `warpSPH`; `python
  scripts/gradcheck_deltaShift.py` still passes in `warpSPH` (reference untouched); the six
  `gradcheck_*_native.py` scripts and `operation_matrix.py --device cpu` in `warpSPHCore`
  unaffected; re-run Phase 1's wave-equation test to confirm Tier 1's path is unaffected by
  Tier 2 landing alongside it.
- End-to-end: re-run Phase 1's cross-validation test after Phase 2 lands, swapping its hand-rolled
  "call `f_wave_equation` twice" trick for the new `warpOperationJVP` where convenient, to confirm
  the production entry point and the manual trick agree (they should be identical by
  construction, but worth asserting explicitly).
