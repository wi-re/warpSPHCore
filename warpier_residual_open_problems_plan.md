# Residual open problems (consolidated, 2026-08-24)

## Context

Every plan doc that has ever lived at repo root was read in full and cross-checked against the current
tree. The result: almost everything is done. Six of seven root plan/design docs (all four `*_plan.md`
files, plus the `warpier_adjoint.md`/`warpier_fields.md` derivation work they built on) are fully closed
and have been moved to `docs/historic_plans/`, following this repo's own established convention for
closed plans (`warpier_tier2_jvp_remaining_work_plan.md` did the same for its four predecessors).
**Cross-references to a moved doc are left as bare filenames, not rewritten to the new path** — that is
the existing convention throughout this codebase's docstrings and other plan docs (confirmed: none of
the four earlier moved docs' ~40 cross-references elsewhere were ever updated after their move either),
so nothing needed touching for the move itself.

`warpier_core.md`/`warpier_fields.md`/`warpier_adjoint.md` stay at root — they are living
architecture/derivation references (not scoped task lists), heavily cross-referenced from source
docstrings throughout `src/`, and their own content is a mix of done-and-recorded history plus the
still-live "Repository Reality Check" framing the rest of the codebase points back to.

**Two small cleanups already landed this session, not carried forward as action items:** `Field.tangent`
(`dataTypes/field_t.py`) was dead code — never written to by any of the three designs that ended up
delivering forward-mode AD capability — and has been removed; `warpier_core.md`'s "What's Next" section
had a stale bullet claiming Tier-2 forward-mode wiring was still in progress and would route through
`Field.tangent`, corrected in place.

This doc is what's left: **nine open items, no single unifying thread**, so it's organized as a flat
punch list rather than a phased plan. (Items 1 and 3 were investigated/implemented and closed 2026-08-24,
in this same session that also identified them — kept in place, marked closed, rather than renumbering
everything.)

## Open items

### 1. `warpSPH` implicit-shifting solver robustness — CLOSED 2026-08-24 (all three gaps accounted for; one doc bug fixed, one API-drift bug fixed)

Sibling `warpSPH` repo, not this one (`modules/shifting/{bicgstab,implicitShifting,
implicitShiftingAutomatic}.py`). Originally three compounding gaps, found while validating
`docs/historic_plans/warpier_forward_mode_plan.md`'s Phase 4 comparison. **Investigated and resolved
2026-08-24:**

- ~~`computeImplicitShift`/`computeImplicitShiftAutomatic` never check `bicgstabSolve`'s returned
  status~~ **Fixed upstream 2026-08-19** (predates this doc's original write-up; not cross-checked
  against the sibling repo until now). Both now route through
  `modules/shifting/solverDriver.solveImplicitSystem`, which checks `iters < 0` and, when
  `ShiftProperties.implicitFallback` is opted in (or via the new `ShiftingScheme.dynamic`, which
  enables it by default), retries with the other Krylov solver (BiCGStab↔GMRES, keeping the better
  iterate by stamped true residual) and optionally a bounded Richardson polish
  (`krylov_richardson`). Emits a `UserWarning` on any fallback activation. The default
  (`implicitFallback=none`, `ShiftingScheme.implicit`) stays byte-identical to the old
  silent-bailout behavior for existing users. Tested in `tests/test_implicitShiftingFallback.py`.
- ~~`bicgstabSolve`'s own `tol` parameter is dead code~~ **Fixed upstream 2026-08-19.** `bicgstab.py`
  now computes `atol = max(atol, tol, rtol*|b|)`, so `tol` is a real absolute floor.
- ~~The inner BiCGStab solve frequently does not converge at realistic jitter levels
  (`jitter >= 0.01`)~~ **Turned out to be a mischaracterization, root-caused today.** This claim's
  source, `scripts/troubleshoot_implicitShiftingConvergence.py`, was committed at 09:00 on
  2026-08-19 — 86 minutes *before* the very next commit (`61f783e`, 10:26 same day) introduced
  `ShiftingImplicitOperator` and flipped the default from what is now called `exactHessian` to
  `legacyPairwise`, specifically *because* `exactHessian`'s diagonal is configuration-dependent and
  unbounded far from equilibrium. The script's docstring was never updated after that flip, so it
  kept describing the old default's divergence while silently building its test system against the
  new one. Verified empirically today (nx=8/16/32, dim=2/3, seeds 1234/1/42, CPU): with the actual
  shipped default `legacyPairwise`, BiCGStab converges cleanly at every jitter from 0.005 through
  0.2 (`rel_resid` ~5e-5 to 9e-5, under the `1e-4` target) — no divergence bailout. Forcing
  `implicitOperator=exactHessian` on the identical harness reproduces the originally-claimed
  divergence almost exactly (`rho-breakdown` by `jitter=0.01`, `threshold-bailout` with `max|xk|/dx`
  up to ~9.6 from `jitter=0.02` on) — i.e. this is the *same* already-known, already-documented,
  already-accepted `exactHessian` instability the "not the same thing" paragraph below describes,
  not a second, independent bug living in both modes as originally scoped. This is also directly
  corroborated by `test_implicitShiftingConvergesFromFullyRandomPositions`'s own docstring, already
  in the tree: "`exactHessian`'s instability here is the known, documented, accepted tradeoff, not a
  regression to guard." **Fixed the misleading script** (module docstring, help text, and a new
  `--operator {legacyPairwise,exactHessian}` flag defaulting to the production value, so this
  distinction can't get lost again) rather than leaving a stale finding for the next reader.

**Bonus finding, fixed:** while reproducing the above, `pytest tests/` in warpSPH showed 5 failures
(`test_implicitShiftingComparison.py` ×4, `test_implicitShiftingGradientJVP.py` ×1) — an unrelated,
genuine regression: this repo's `warpOperationJVP` signature moved from loose
`tangentQueryPositions=`/`tangentReferencePositions=` tensor kwargs to a bundled
`queryTangentState=`/`referenceTangentState=` `ParticleTangentState` (the Tier-2 correction-JVP work,
`docs/historic_plans/warpier_tier2_correction_jvp_plan.md` phase a1), and warpSPH's
`implicitShiftingAutomatic.py` plus one test were never updated to match (`warpOperationHVP`, used
elsewhere in the same file, kept its old flat-kwarg signature, so this was easy to miss). Fixed both
call sites to build a `ParticleTangentState(positions=..., supports=<zeros>, masses=None)`, matching
the pattern already used in this repo's own `scripts/gradcheck_tier2_jvp_*.py`. Full warpSPH suite:
207 passed, 1 skipped, 0 failed (was 5 failed before this fix).

Not the same thing as, and not fixed by, the already-shipped `legacyPairwise` production default — that
fixes `exactHessian`'s structurally-unbounded-diagonal instability specifically, an orthogonal, already-
closed problem (see `docs/regression/implicit_shifting_operator_choice.md` in warpSPH for the full
derivation) — **which the investigation above confirms is exactly what the third bullet was actually
observing**, not a fourth, still-open problem. Nothing is open in this item for the production default;
`exactHessian` remains an explicit, documented, intentional opt-in for autodiff cross-checking. One
untried lever if anyone ever wants to push on its stability instead of accepting it as-is: MINRES
(`modules/shifting/minres.py`, built for Item 2's IISPH pressure work) is a structural fit for
`exactHessian`'s own documented character ("symmetric, with an exact null space along uniform
translation," per `implicitShifting.py`'s module docstring) the same way it fit the IISPH pressure
operator — untested here, and the fallback chain's alternation is still only BiCGStab↔GMRES, but the
building block already exists.

### 2. Automatic Newton-Krylov vs. IISPH for incompressibility — scoping plan written 2026-08-24; original premise partly corrected

`docs/historic_plans/warpier_forward_mode_plan.md`'s own Phase 6/Goal 4, which explicitly asked to
be "scoped as its own plan" before any implementation. **That plan is now written**:
`warpSPH/COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md` (2026-08-24, scoping only, nothing implemented). Its
central finding corrects an assumption baked into this item's original framing, so it's worth
stating precisely here too.

**What actually landed adjacent to this (recap, unchanged from before):** warpSPH's
`INCOMPRESSIBLE_SOLVER_PLAN.md` (commits 2026-08-19 to 08-21, all 7 phases done) added five opt-in
Krylov solvers — CG, BiCG, BiCGStab, GMRES, MINRES (`modules/incompressible/krylov.py`) — as
alternatives to `solveDivergenceFree`'s relaxed-Jacobi smoother, with a Phase-0 operator probe
showing the IISPH pressure operator is symmetric (`‖A−Aᵀ‖/‖A‖ ≈ 1e-6`), negative-semi-definite with
a gauge null space, ill-conditioned (`κ ≈ 2.4e7`) — MINRES (its exact design domain) is the best
all-round method. Full findings in warpSPH's
`docs/regression/incompressible_pressure_solver_choice.md`.

**The corrected finding (2026-08-24):** Phase 6's own sketch — "an automatic Newton-Krylov
pressure-Poisson solve built from composed `warpOperationJVP` calls... rung 3" — turns out to be
based on a premise that doesn't hold for the sub-problem IISPH actually solves. Reading
`computePressureAccelIISPH`/`computePressureShiftIISPH` directly: the IISPH matvec is
`Divergence(Gradient(p)/rho)`, i.e. two plain `warpOperation` calls (the same generic,
already-general-purpose operator dispatch every explicit scheme uses) applied *directly* to the
trial pressure field `p`. There's no hand-rolled per-pair kernel to replace here the way
`implicitShifting.py`'s Hessian was (a raw `sphKernelHessian` call bypassing `OperatorSpec`
entirely) — `p` **is** the unknown the operator acts on, not a tangent direction through some other
computation, so there was never a differentiation step for `warpOperationJVP` to automate on this
sub-problem. The Krylov work above therefore already *is* "automatic vs. hand-built, on ease not
speed" fully resolved in the automatic path's favor, with zero new derivation needed — it just
didn't need JVP to get there, which is a materially different (and smaller) finding than Phase 6
assumed.

**What's genuinely still missing** (re-examined against the current tree, not assumed): confirmed
real, but bigger than a mechanical follow-on. `schemes/dfsph.py`'s `dfsph_step` is a
splitting/projection scheme — `dvdt` (forces/gravity/diffusion) is computed explicitly once, then
`solveDivergenceFree` solves a **linear** correction against that frozen `dvdt`; nothing in this
codebase re-linearizes momentum and pressure together inside an outer Newton loop, so — unlike
shifting, which had `exactHessian` as pre-existing hand-built ground truth — **there is no existing
coupled hand-built solve to compare an automatic one against.** EOS coupling is confirmed **not
currently exercised on the incompressible path at all**: `weaklyCompressibleEOS` is called from the
*explicit* `schemes/deltaSPH.py` but is dead-commented in `schemes/dfsph.py` — weakly-compressible
and incompressible are two structurally separate scheme families today, not two modes of one
nonlinear system, so "EOS coupling" means a scheme-architecture merge decision, not a solver
derivation.

**Recommendation (in the new plan doc, not acted on):** don't build the full coupled solve
speculatively — there's no reference implementation to validate against and no motivating bug/
request. Instead it proposes a much smaller, concretely bounded candidate Phase 1 with a real
success metric and no new architecture: `solveIncompressible`'s `clamp(pressure, min=0.0)` is
currently only approximated (solve the unconstrained linear system, then clamp — already flagged
as an approximation in `INCOMPRESSIBLE_SOLVER_PLAN.md`'s own Scope section); a projected/
semismooth-Newton treatment of that box constraint is a legitimate, bounded place for
`warpOperationJVP`-style automatic-linearization reasoning to actually do real work (the active set
changes iteration to iteration, so the effective operator isn't fixed the way the plain Poisson
solve's is). Not started; the doc lists three open questions needing a decision from you (is the
full coupled solve still wanted at all; if so does it merge the EOS/incompressible scheme families
or stay pressure-only; is the bounded clamp-Newton candidate worth doing on its own) before either
direction gets an implementation plan.

### 3. Registering an implicit scheme in `warpSPHIntegrators` as a first-class driver — CLOSED 2026-08-24 (all three phases done)

Source: same predecessor plan, explicitly out of scope there. The implicit wave-equation step used a
hand-rolled CG loop in a standalone test rather than `warpSPHIntegrators`'s own DIRK/Picard machinery
(`NOTES.md` §3.0-3.1/§3.8 — the library already carries the state algebra needed; a ~60-line probe over
existing primitives reportedly already reaches full order). `NOTES.md` §3.8 estimates ~2 weeks total for
the phased effort (Phase 0 → 2 → 1), gated on nothing from `warpSPHCore`.

**Phase 0 (groundwork, S1-S5, NOTES.md §3.5) implemented and committed 2026-08-24** (commit `cd2a32f` in
`warpSPHIntegrators`), at the user's direction after confirming scope: `state_norm`/`state_difference`
(`fields.py`), `StepHistory`/`HistoryEntry` with the `dt`/`uid` restart guards (new `history.py`),
`IntegrationScheme` metadata (`implicit`, `steps`, `stiffly_accurate`, `stability`, `startup_order` —
`util.py`), the `NonlinearSolver` protocol + `FixedPointSolver` (new `solvers.py`), and an opt-in
`history=` kwarg threaded through `RungeKuttaB` and `testing.run`. `butcher._error_estimate` was
refactored to use `state_difference` rather than a hand-rolled linear combination of raw stage
derivatives, per NOTES.md S1's "generalise rather than duplicate." 48 new tests.

**One real bug caught and fixed during Phase 0, not just theoretical groundwork risk:** a first draft had
`history=` auto-derive `priorStep` inside `RungeKuttaB` when no explicit `priorStep` was given. That
silently turned on first-stage reuse — and its order cost, per `reuse.py` — for *any* caller that merely
wanted history threaded, on non-FSAL tableaus too (RK4 drifted ~1e-4 per step), with none of
`integration._with_reuse_guard`'s warnings, since those only fire when `priorStep` itself is a kwarg.
Caught by a same-session test, fixed by keeping `history=` strictly bookkeeping-only: it populates
`IntegrationResult.history` but never seeds `priorStep`. A caller that wants reuse still opts in
explicitly, exactly as before: `priorStep=history.as_prior_step()` alongside `history=`.

**Phase 2 (the DIRK driver, NOTES.md §3.6) implemented and committed 2026-08-24** (commit `710ede7`),
continuing directly on Phase 0's primitives at the user's direction ("keep working on the plan"): new
`dirk.py` reuses `butcher.py`'s `_weighted_update`/`_error_estimate`/`finalizeSystem` machinery, closing
each stage's diagonal term through `FixedPointSolver`. Four of the six planned tableaus shipped — Backward
Euler, Implicit Midpoint, Trapezoidal/Crank-Nicolson, SDIRK2 — each verified two ways (by hand against its
own order conditions before coding, and empirically via `testing.convergence` after: all four reach their
claimed order to within 0.01). **TR-BDF2 and ESDIRK3(2)4L[2]SA were deliberately not implemented**: both
are embedded, higher-stage tableaus whose published coefficients are easy to transcribe wrong in a way a
smoke test wouldn't catch (a 2nd/3rd-order convergence measurement looks the same whether the low-order
embedded weights are exactly right or merely close). 34 new tests (`tests/test_dirk.py`).

**Two more real findings from Phase 2, not just design risk:**
- **Implicit midpoint's headline symplectic property needs a caveat the original NOTES.md scoping
  missed.** It is the textbook symplectic Gauss-Legendre s=1 method only when its stage equation is
  solved to convergence. At the shipped 2-iteration Picard default (NOTES.md's own "2 iterations reach
  full order" recommendation), measured long-run energy drift *grows secularly* — the signature of a
  dissipative scheme, not a symplectic one — confirmed by a check that the bound recovers (to ~1e-15,
  flat with `T`) once the solver runs enough iterations (~16) to actually converge. The convergence-order
  claim is unaffected (verified exactly at 2 iterations); only the qualitative long-run energy behavior,
  arguably the scheme's main selling point, needs more iterations than "ship 2" to recover. Registered
  `dissipation=True` to describe the shipped default honestly, and corrected NOTES.md's own headline
  framing of this scheme rather than leaving the overclaim in place.
- **The existing (pre-DIRK) test suite caught a real implementation bug immediately, for free**, the
  moment the four new schemes joined the shared `scheme` fixture every test file already parametrizes
  over: `tests/test_copied_fields.py`'s 12 tests failed because the Picard loop's `step_fn` returned a
  fresh clone as the next iterate, discarding the object that had actually run `preprocess()` — so
  `copied()`-behavior fields (e.g. a summation density, exactly the SPH case this mechanism exists for)
  arrived as `None` at the caller. Fixed by capturing the evaluated stage buffer in the closure instead of
  using the solver's returned iterate for anything but the stage's own derivative. This is the kind of bug
  that stays invisible in an isolated new-feature test file and only surfaces when a broad, pre-existing
  regression suite gets to run against the new code — exactly what happened here.

**Phase 1 (explicit multistep, NOTES.md §3.6) implemented and committed 2026-08-24** (commit `fc5a0ad`),
closing out all three phases in the same session ("keep going"): new `multistep.py` adds Adams-Bashforth
2-5 and Adams-Bashforth-Moulton 2-4 (PECE), reusing `butcher._weighted_update` directly — it never cared
whether its `ks` came from this step's stages or past steps' `StepHistory` entries, exactly the "free
lunch" NOTES.md §3.1 predicted. Bootstraps from Dormand-Prince 5(4) for the first `order-1` steps (order 5,
at or above every shipped order, so none of them lose order to a low-quality cold start). The starter is
hard-coded, not caller-configurable, per NOTES.md's own conclusion that it "must be registered scheme
metadata, not caller policy" — a caller-suppliable lower-order starter would silently cap the whole run at
its own order. 67 new tests (`tests/test_multistep.py`). **Landed with zero comparable bugs to Phase 2's**
— the design leaned entirely on already-battle-tested primitives (`_weighted_update`, `StepHistory`), and
the full suite went green on the first run after widening one pre-existing test's exclusion criteria (not
a bug in the new code).

**A real design question resolved during implementation, not just documented risk:** these multistep
schemes' past derivatives only exist if a caller threads `IntegrationResult.history` forward — unlike
every one-step scheme, where `history=` is pure opt-in bookkeeping. Rather than leave "you must thread
history correctly or get silently wrong results" as a documentation-only warning, made the failure mode
itself safe: without history, a scheme keeps re-running its Dormand-Prince starter forever, verified
bit-for-bit identical to calling `DormandPrince` directly — a caller who forgets gets a correct but more
expensive trajectory (DP5 is higher-order than any of these), never a silently wrong one. Also updated
`warpSPHIntegrators/README.md`'s stale "Known Limitations" section, which had claimed no implicit or
multistep support at all after eleven new schemes (four DIRK + seven multistep) landed this session.

Full `warpSPHIntegrators` suite: 1380 passed, 114 skipped (was 906/54 at the start of this session), no
regressions across all three commits. **Nothing is open in this item** — everything NOTES.md §3.8
recommended (Phase 0 → 2 → 1) is done; what remains (TR-BDF2, ESDIRK3(2), fully implicit RK, BDF, IMEX/ARK,
a stiff `NonlinearSolver`) is each individually scoped and gated on a concrete downstream need, per
NOTES.md's own updated recommendation to stop and reassess rather than build further speculatively.

### 4. Tier-2 JVP / forward-mode AD beyond the six core operators

The momentum equation, mDBC, surface detection, and any other production kernel outside Density/
Interpolate/Gradient/Divergence/Curl/Laplacian have no JVP/geometry-tangent support and were never
attempted — restated unchanged across every forward-mode plan's own "explicitly out of scope" list. No
derivation exists to promote; this would be new math from scratch (`warpier_adjoint.md` Tier-2.0-2.5
style) for whichever operator a future consumer actually needs.

### 5. HVP for the five non-Density operators — closed-form derivation is now the *only* route

`Hess(Interpolate/Gradient/Divergence/Curl/Laplacian) @ v` has no production entry point — only Density
does (`computeSPHDensityPositionHVP`/`warpOperationHVP`, a hand-derived closed form built around
`kernels.hessian.sphKernelHessian`). Two routes to get there generically were tried and are now
confirmed dead ends, not just untried: `torch.func.jvp` composed twice (fails immediately — dual tensors
have no storage for `wp.from_torch`) and nested `torch.autograd.forward_ad.dual_level()` (raises
`RuntimeError: Nested forward mode AD is not supported at the moment` on the installed torch 2.13,
reproduced on a bare `x**3` with zero warp/SPH code involved — a hard PyTorch engine limitation, not
anything this codebase's own bridge could route around; see `docs/historic_plans/
warpier_unified_operator_wrapper_plan.md`'s Phase 4). If a future Newton-style solve needs one of these
five operators' Hessian, the only viable path is the same shape of hand-derived second-order helper
Density's own HVP needed — a genuine per-operator derivation effort, not free composition.

### 6. Grad-H / adaptive-smoothing-length (Omega) tangent support

`GradHState` has no tangent counterpart anywhere (`warpOperationJVP`/`JVPSpec` both reject `gradHState`
outright, unchanged since first written); `kernels/hessian.py`'s `sphKernelDkDh_`/`sphGradientDkDh_`
building blocks are already derived and validated but have **zero production consumers** — nothing in
`warpSPHCore` or `warpSPH` calls them outside spike scripts. Blocked on a concrete consumer landing in
`warpSPH`'s `adaptiveSupport` module first; deriving the tangent speculatively ahead of that consumer has
been declined every time it came up. Not actionable until that consumer exists.

### 7. Warp-lang upstream bug report — drafted, blocked on submission tooling

The dynamic-loop + nonlinear-op NaN-gradient bug (`scripts/repro_warp_dynamic_loop_division.py`: a
**linear** read of a loop accumulator is safe inside the same `@wp.func` as the loop; a **nonlinear**
read — division, squaring — is not, coming back `-inf` or a silently-wrong `0.0`) is isolated to a
~20-line, warpSPHCore-independent minimal repro, worked around locally
(`crk_volume.py`/`crk_density.py`), but was never filed against `warp-lang` (unlike the sibling
`SuperSymmetric`/`2.0` discrepancy, filed as warp-lang issue #1740). **Issue text drafted 2026-08-24**
(title + body below) — **not submitted**: no `gh` CLI and no `GITHUB_TOKEN` available in this
environment, so there's no way to authenticate against `github.com/NVIDIA/warp` from here. Needs the
user to paste the draft in manually, or provide `gh auth login`/a token for a future session.

<details>
<summary>Draft issue text</summary>

**Title:** Nonlinear op on a dynamic-loop accumulator, inside the same `@wp.func` as the loop, reads the
accumulator's pre-loop value in the backward pass

**Body:**

Environment: `warp-lang == 1.15.0`, `torch == 2.13.0+cu130`, Python 3.14.6, Linux (WSL2), reproduced on
both CPU and CUDA.

Per the docs' own ["Limitations and Workarounds in
Differentiability"](https://nvidia.github.io/warp/stable/user_guide/differentiability.html), a dynamic
loop's per-iteration intermediates aren't replayed in the backward pass, and the documented workaround is
to move the loop into its own `@wp.func` and consume the *result* from the caller. Following that
workaround to the letter still produces a wrong gradient if the *nonlinear* op consuming the loop's
result lives in the same `@wp.func` as the loop itself — only moving it to a different function/kernel
scope fixes it. A **linear** post-loop op (`total * 3.0`) in the same function is fine; a **nonlinear**
one (division, squaring) is not.

Minimal repro attached (`repro_warp_dynamic_loop_division.py`, ~20 lines, no external dependencies beyond
`warp`/`torch`): `1.0 / total` where `total` is accumulated over a runtime-valued `for i in range(n):`
loop, computed inside the same `@wp.func` as the loop, gives `-inf`/wrong-value gradients depending on
`total`'s pre-loop initial value (confirmed: the observed gradient matches `d(1/S)/dS` evaluated at the
accumulator's *pre-loop* initial value, not its true post-loop sum — exact match for two different
initial values, both listed in the script). Moving the division one level up (into the calling
`@wp.kernel`, exactly the documented workaround's own example) gives the correct gradient. Squaring
(nonlinear, no division) inside the loop function fails silently with a wrong-but-finite `0.0` rather than
`-inf`, consistent with the same "reads the pre-loop value" mechanism.

Found while gradient-checking a CRK-volume kernel in a downstream project; worked around there by
restructuring around the documented pattern more strictly (nonlinear op moved out of the looped
function), but the underlying behavior seems like a gap in the workaround's own stated guarantee, so
filing it rather than only working around it locally.

</details>

### 8. Static analysis / linting — pending a tool decision from the user

`pyproject.toml` has no `ruff`/`mypy`/`pyright` section; no lint step in `.github/workflows/tests.yml`.
No linter or type checker runs anywhere, despite a track record of exactly the bugs a linter catches
mechanically (stale imports after a file move, dead re-export lists). **User's condition (2026-08-24):**
static-only, no runtime cost. Already satisfiable — `ruff` in lint-only mode and `python -m py_compile`
are both dev-time/CI-only invocations with zero production runtime footprint (neither is imported by, nor
runs inside, any code path a caller executes; the risk case would be a *different* class of tool, e.g. a
runtime type-validation decorator, which isn't what's proposed). Warp's parenthesized generic types
(`vector(dtype=..., length=...)`) and `scalar_t` being both a runtime value and a type annotation both
trip up real static type checkers (`mypy`/`pyright`), which is why `ruff` lint-only or `py_compile` were
the suggested sidestep in the first place. Not implemented — needs the user to confirm one of the two
specifically before it goes into CI, since turning it on will likely surface a first batch of real
findings that need triage.

### 9. CI/testing gaps deliberately deferred since Phase 0, still open

- No `tests/data/` golden-data baseline fixtures.
- No nightly CI sweep of the full precision×dim×jitter product.
- CUDA CI coverage stops at the single 3D step; no broader CUDA matrix.
- No per-test behavior-spec docs.
- **Jitter beyond ~0.01 has no validated MAE threshold.** Heavier jitter (0.15-0.3, the range that
  actually stresses CRK/renormalization) produces many real `HIGH` cells today (184 at `--dim 1
  --jitter 0.15` on a first attempt) — expected diagnostic behavior, not a regression, but nothing gates
  it in CI because no sound threshold has been worked out. Any future CI/tooling work involving jitter
  above ~0.01 needs that investigation first; it cannot reuse the existing `--threshold 0.4` default.

## Fully closed (pointer only)

- `docs/historic_plans/warpier_tier2_correction_jvp_plan.md` — all phases (a1)-(f) plus same-day
  follow-ups (Laplacian Dot/Default CRK+renorm extension, CRK+renorm applied simultaneously, `pinv2x2`
  uniform-grid adjoint root-cause fix).
- `docs/historic_plans/warpier_tier2_jvp_remaining_work_plan.md` — combined Tier-1+Tier-2 JVP done; CSR
  benchmark waived by user decision; embedded-in-backprop demonstration done.
- `docs/historic_plans/warpier_unified_operator_wrapper_plan.md` — Phases 0-3 landed (Tier-1 value JVP,
  Tier-2 geometry JVP + Covariance promoted to public dispatch, per-operator dual-tensor test coverage);
  Phase 4 closed with the negative finding folded into Item 5 above.
- `docs/historic_plans/warpier_forward_mode_plan.md` — Phases 1-4 done (Density-scoped shifting
  comparison); Tier-2 JVP for the remaining five operators done as a follow-on; Phase 5 was never a
  task; Phase 6 and the sibling-repo BiCGStab findings carried forward as Items 1-2 above.
- `warpier_adjoint.md` — Tier 2.0 through 2.5 all done; its three declared out-of-scope items map onto
  Items 6, "the Field/bridge wiring" (superseded, see Context), and periodic-wrap (independently
  verified already correct, not reopened).
- `warpier_core.md` — Phases 0-5 done for the operator-migration narrative; every "smaller open item" in
  its own "What's Next" section is closed except Items 7/8 above.
- `warpier_fields.md` — Steps 0 and A-J all complete and gated.

## Suggested next step

Item 7 is blocked only on submission tooling — hand over `gh` credentials, or paste the draft in, and
it's done in one step. Item 8 is waiting on you to pick `ruff`-lint-only vs. `py_compile` before it goes
into CI. **Item 1 is now CLOSED** (2026-08-24): the two gaps not already fixed upstream turned out, on
investigation, to be one documentation bug (a troubleshooting script's claim, fixed) and one unrelated
API-drift regression (this repo's `warpOperationJVP` signature change breaking a warpSPH consumer,
fixed, 207/207 warpSPH tests now passing) — nothing was actually open in the production path. Items 2-3
remain in the `warpSPH`/`warpSPHIntegrators` incompressible-flow thread — worth deciding whether that's
still a live priority before scoping either further. **Item 2 now has its requested scoping plan**
(`warpSPH/COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md`, 2026-08-24): the linear pressure-Poisson sub-problem
Phase 6 sketched turned out to need no `warpOperationJVP` at all (already closed by the landed Krylov
work, for a different reason than originally assumed) and there's no existing coupled hand-built solve
to build an automatic one against (unlike shifting's `exactHessian`) — so the plan recommends *not*
building the full coupled solve speculatively, and instead flags three open questions plus one small,
concretely bounded candidate (`solveIncompressible`'s `clamp(p, min=0)` as a projected-Newton target)
for you to weigh in on before anything gets implemented. **Item 3 is now CLOSED**: all three phases
(0, 2, 1) implemented and committed 2026-08-24 (`warpSPHIntegrators` commits `cd2a32f`/`710ede7`/`fc5a0ad`),
eleven new schemes registered (four DIRK, seven multistep), plus a real finding that corrects NOTES.md's
own original claim that implicit midpoint's symplectic property comes for free at the recommended
2-iteration solver default (it doesn't — see Item 3). **A genuine follow-on landed the same session**: a
user question about the fixed-Picard-iteration limitation surfaced that `warpSPHIntegrators`' NOTES.md
still claimed "warp has no forward-mode AD" — stale now that this repo's Tier-2 JVP work (Items closed
earlier this session and before) gives `warpSPHCore` a real, narrowly-scoped JVP layer for six operators.
Verified against current source (not assumed) and corrected (`warpSPHIntegrators` commit `34ccc1b`), then
turned into `warpSPHIntegrators/JFNK_PLAN.md` (commit `6e1b0ac`): a phased plan to build the still-missing
Newton-Krylov (JFNK) solver rung, validated first against `warpSPH`'s implicit wave-equation test — which
needs none of the still-unwrapped frontend operators and, found during scoping, already fully implements
this repo's integrator protocol with no adapter work needed — before closing the JVP gap for whichever
specific WCSPH sub-problem becomes the real target, deliberately not speculating across the whole
`deltaSPH_step` RHS at once (same discipline Item 6 already established). Not started; needs you to name
the actual WCSPH implicit target before Phase B can be scoped (see the plan's "Open questions"). Items
4/5/6 aren't backlog items — they're "if a concrete consumer shows up, expect this shape of work," not
something to build speculatively, though Item 4 specifically now has a plausible path to becoming one via
this plan's Phase B. Item 9 is background hygiene, pick up opportunistically.

## Critical files

- `warpSPH/modules/shifting/{bicgstab,implicitShifting,implicitShiftingAutomatic,solverDriver,minres}.py`,
  `warpSPH/scripts/troubleshoot_implicitShiftingConvergence.py` — Item 1 (closed; edited
  2026-08-24: fixed the script's stale docstring/help text and added `--operator`; fixed
  `implicitShiftingAutomatic.py`'s and `tests/test_implicitShiftingGradientJVP.py`'s
  `warpOperationJVP` calls for this repo's `queryTangentState`/`referenceTangentState` signature)
- `src/warpSPHCore/operations.py` — `warpOperationJVP`'s current signature (the source of the
  API-drift regression found and fixed in Item 1; `warpOperationHVP` was unaffected)
- `warpSPH/INCOMPRESSIBLE_SOLVER_PLAN.md`, `warpSPH/modules/incompressible/krylov.py`,
  `warpSPH/docs/regression/incompressible_pressure_solver_choice.md` — Item 2's landed groundwork
- `warpSPH/COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md` (new, 2026-08-24) — Item 2's scoping plan;
  `warpSPH/modules/pressure/iisph.py`, `modules/incompressible/drift.py` — the IISPH matvec that
  plan shows needs no JVP; `warpSPH/schemes/dfsph.py`, `schemes/deltaSPH.py`,
  `modules/eos/weaklyCompressible.py` — where that plan confirms EOS is/isn't wired in
- `warpSPHIntegrators/NOTES.md` §3.5-3.8, `README.md` — Item 3's scope and 2026-08-24 status for all
  three landed phases, and the user-facing scheme docs/limitations list;
  `src/warpSPHIntegrators/{fields,history,solvers,util,butcher,testing}.py`, `tests/test_groundwork.py` —
  the Phase 0 code (new `history.py`/`solvers.py`; `state_difference`/`state_norm` in `fields.py`;
  `IntegrationScheme` metadata in `util.py`; `history=` wiring + the `_error_estimate` refactor in
  `butcher.py`); `src/warpSPHIntegrators/dirk.py`, `tests/test_dirk.py` — the Phase 2 DIRK driver, its
  four registered schemes, and the `dissipation=True` correction for Implicit Midpoint;
  `src/warpSPHIntegrators/multistep.py`, `tests/test_multistep.py` — the Phase 1 Adams-Bashforth/-Moulton
  driver and its seven registered schemes; `src/warpSPHIntegrators/{enums,integration}.py` — all eleven
  new schemes' registry entries
- `warpSPHIntegrators/JFNK_PLAN.md` (new, 2026-08-24) — the JFNK follow-on plan; `warpSPHIntegrators/
  NOTES.md` §3.4 — the corrected forward-mode-AD state; `warpSPH/tests/test_implicitWaveEquation.py`,
  `warpSPH/src/warpSPH/systems/waveSystem.py` — the wave-equation bridge case the plan's Phase A targets;
  `src/warpSPHCore/autograd/{operator_spec,stateAwareWarpFunction}.py` — where the six-operator JVP scope
  and the "unwrapped operator raises `NotImplementedError`" safety property actually live
- `src/warpSPHCore/coreOperations/wp_densityHVP.py` — Item 5's existing reference pattern
- `scripts/repro_warp_dynamic_loop_division.py` — Item 7
- `pyproject.toml`, `.github/workflows/tests.yml` — Items 8/9
