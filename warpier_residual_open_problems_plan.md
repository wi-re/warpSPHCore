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
punch list rather than a phased plan.

## Open items

### 1. `warpSPH` implicit-shifting solver robustness

Sibling `warpSPH` repo, not this one (`modules/shifting/{bicgstab,implicitShifting,
implicitShiftingAutomatic}.py`). Three compounding, still-open gaps, found while validating
`docs/historic_plans/warpier_forward_mode_plan.md`'s Phase 4 comparison but never fixed:

- `computeImplicitShift`/`computeImplicitShiftAutomatic` never check `bicgstabSolve`'s returned status
  (`solverIters`/`convergence` discarded) — a breakdown/divergence bailout still hands back its partial
  `xk` as if converged.
- `bicgstabSolve`'s own `tol` parameter is dead code — only `rtol * |b|` ever sets the convergence
  floor.
- The inner BiCGStab solve for a single Newton step frequently does not converge at all at realistic
  jitter levels (`jitter >= 0.01` on the swept test case hits the divergence bailout almost immediately)
  — a genuine ill-posedness effect of linearizing at a frozen adjacency, not noise. Existing tests pass
  anyway only because `implicitRelaxation=0.1`'s heavy per-step damping averages out many bad/
  non-converged steps over an outer relaxation loop.

Not the same thing as, and not fixed by, the already-shipped `legacyPairwise` production default — that
fixes `exactHessian`'s structurally-unbounded-diagonal instability specifically, an orthogonal, already-
closed problem. These three gaps are live in both modes. Candidate directions (not evaluated further
here): check `solverIters` and fall back on failure; rebuild the adjacency mid-solve or cap the trust
region to the frozen adjacency's valid radius; loosen tolerances; more outer/fewer inner iterations —
real design choices, not a mechanical fix.

### 2. Automatic Newton-Krylov vs. IISPH for incompressibility

`docs/historic_plans/warpier_forward_mode_plan.md`'s own Phase 6/Goal 4 — never scoped in detail,
deliberately deferred as a materially bigger lift than the shifting comparison: coupled pressure/
velocity DOFs across the whole domain (not one scalar field per particle), boundary/free-surface
handling, and EOS coupling that the shifting objective doesn't have. The shape would mirror the shifting
work — an automatic pressure-Poisson solve built from composed `warpOperationJVP` calls, compared
against `solveIncompressible`'s existing IISPH relaxed-Jacobi iteration on correctness and on how much
bespoke pressure-solver machinery the automatic path avoids (`warpSPHIntegrators/NOTES.md` §3.4's
solver-ladder framing: IISPH is rung 1, this would be rung 3). Nothing built. Needs its own plan before
any implementation starts.

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
into CI. Items 1-3 are the only genuinely open design work, all in the `warpSPH`/`warpSPHIntegrators`
incompressible-flow thread — worth deciding whether that's still a live priority before scoping 2 or 3
further; Item 1 is a found-not-fixed bug in currently-shipping code, independent of whether 2/3 ever
happen. Items 4/5/6 aren't backlog items — they're "if a concrete consumer shows up, expect this shape
of work," not something to build speculatively. Item 9 is background hygiene, pick up opportunistically.

## Critical files

- `warpSPH/modules/shifting/{bicgstab,implicitShifting,implicitShiftingAutomatic}.py` — Item 1
- `src/warpSPHCore/coreOperations/wp_densityHVP.py` — Item 5's existing reference pattern
- `scripts/repro_warp_dynamic_loop_division.py` — Item 7
- `pyproject.toml`, `.github/workflows/tests.yml` — Items 8/9
