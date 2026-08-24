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
punch list rather than a phased plan. (Item 1 was investigated and closed 2026-08-24, in this same
session that also identified it — kept in place, marked closed, rather than renumbering everything.)

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

### 3. Registering an implicit scheme in `warpSPHIntegrators` as a first-class driver

Source: same predecessor plan, explicitly out of scope there. The implicit wave-equation step used a
hand-rolled CG loop in a standalone test rather than `warpSPHIntegrators`'s own DIRK/Picard machinery
(`NOTES.md` §3.0-3.1/§3.8 — the library already carries the state algebra needed; a ~60-line probe over
existing primitives reportedly already reaches full order). `NOTES.md` §3.8 estimates ~2 weeks total for
the phased effort, gated on nothing from `warpSPHCore`. Not started; worth scoping only if a one-off
script needs to become a reusable, registered scheme.

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
for you to weigh in on before anything gets implemented. Items 4/5/6 aren't backlog items — they're "if
a concrete consumer shows up, expect this shape of work," not something to build speculatively. Item 9
is background hygiene, pick up opportunistically.

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
- `src/warpSPHCore/coreOperations/wp_densityHVP.py` — Item 5's existing reference pattern
- `scripts/repro_warp_dynamic_loop_division.py` — Item 7
- `pyproject.toml`, `.github/workflows/tests.yml` — Items 8/9
