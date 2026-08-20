# Tier-2 JVP: remaining open work (consolidated)

## Context

This supersedes four prior plan docs, each moved to `docs/historic_plans/` rather than deleted (full
derivation history, status logs, and lessons stay there for reference):

- `docs/historic_plans/warpier_tier2_operators_plan.md` — **fully done** (2026-08-19), all six core
  operators plus both Laplacian schemes wired into `warpOperationJVP`. Nothing open here except two
  deliberate, by-design scope exclusions (HVP for the five non-Density operators; Laplacian
  Dot/Default schemes) — not backlog items, see "Explicitly not planned" below.
- `docs/historic_plans/warpier_tier2_jvp_csr_backend_plan.md` — Steps 1-6 done and cut over
  (2026-08-20); **Step 7 (benchmark) not started** — carried into this doc as Item 2.
- `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md` — bridge (`_jvpCommon.launchGeometryJVP`)
  wired for all six operators (2026-08-20), Steps 0-4 verified; **Step 5 (embedded-in-backprop
  demonstration) not started** — carried into this doc as Item 3. Also surfaced an out-of-scope
  reverse-mode bug, tracked separately (see "Related, tracked elsewhere" below).
- `docs/historic_plans/warpier_tier2_combined_jvp_plan.md` — design finished, empirically verified
  (additivity confirmed against `torch.autograd.functional.jacobian` for all six operators), **but no
  implementation step was started** — carried into this doc as Item 1.

Three genuinely open items remain, listed below in the order most useful to tackle them (Item 1 is
self-contained and highest-value; Items 2-3 are narrower, single-step follow-ups on already-landed
work).

## Item 1 — Combined Tier-1 + Tier-2 JVP (design done, implementation not started)

**Source:** `docs/historic_plans/warpier_tier2_combined_jvp_plan.md` (full design + empirical
verification already in that doc — read it before starting, don't re-derive).

**What's proven:** combining Tier-1 (value-tangent) and Tier-2 (geometry-tangent) JVPs is exact
summation of the two already-shipped paths — verified against `torch.autograd.functional.jacobian`
ground truth for all six operators to float32 round-off (`rel. error` 1e-7 to 1e-6 across the board).
No new derivation needed.

**Remaining steps (renumbered from the source doc, unchanged in substance):**

1. Extend the additivity check to the full case matrix each operator's existing
   `test_forward_mode_tier2_*.py` already sweeps (all `GradientScheme`s, a representative
   `SupportScheme` subset, 1D and 2D).
2. Relax `operations.py`'s current unconditional rejection (Tier-1 tangent + Tier-2 tangent together
   → `NotImplementedError`) into a three-way branch: Tier-2-only (today's path, unchanged),
   Tier-1-only (today's path, unchanged), both-present (call both, sum).
3. Update `warpOperationJVP`'s docstring to state the real contract: supply any subset of value and
   geometry tangents, get their sum back.
4. Update all six `computeSPH<Op>PositionJVP` docstrings (plus `computeSPHDensityPositionJVP`'s) to
   state plainly that they return only the geometry-tangent *partial* contribution, values held at
   primal — not the full derivative on their own.
5. New tests: for each of the five value-having operators, a "combined" case calling
   `warpOperationJVP` with both tangent sets simultaneously, checked against a
   `torch.autograd.functional.jacobian` reference differentiating w.r.t. every input at once. Keep a
   regression case confirming the Tier-1-only and Tier-2-only paths are numerically unchanged.
6. Full verification sweep (see below).

**Naming flag, not a decision:** `computeSPH<Op>PositionJVP` is arguably better named
`computeSPH<Op>GeometryJVP` once its docstring says "frozen values" explicitly — renaming touches five
already-shipped names and their dispatch-table entries; don't do this without confirming the user
wants churn on landed names for a documentation-clarity gain alone.

**Out of scope:** Density (no value input, nothing to combine); fusing the two kernel launches into
one (performance-only, revisit if profiling ever shows it matters); CRK/renormalization/volumes
(unchanged, still rejected).

## Item 2 — CSR backend: benchmark memory/runtime (Step 7)

**Source:** `docs/historic_plans/warpier_tier2_jvp_csr_backend_plan.md`, Step 7 (the only step left).

The CSR port itself is done and cut over — the old COO/pair-indexed kernels are gone, all seven
operators use the per-query-particle-threaded shape, full suite green. What's left is purely
measurement: benchmark memory and runtime on a large-particle-count case, before/after was never
actually measured (the port's payoff — no `O(numPairs)` intermediates, no atomics — was argued from
design, not confirmed with numbers). Report the actual before/after; don't infer the win.

Since the old COO implementation is already deleted, "before" numbers would need to come from git
history (check out the pre-cutover commit, or re-derive from the equivalence-test era) — note this
explicitly when picking the item up rather than assuming a live A/B is still possible.

## Item 3 — Reverse-mode bridge: embedded-in-backprop demonstration (Step 5)

**Source:** `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md`, Step 5 (the only step left).

Steps 0-4 proved each `computeSPH<Op>PositionJVP` call is reverse-mode differentiable *in isolation*
(per-operator `gradcheck` scripts, all passing). The plan's actual motivating goal — embedding a
`warpOperationJVP` call inside a larger differentiable computation (a Newton-style implicit solve) and
backpropagating through the whole thing — was never demonstrated end-to-end. Build a small test that
chains at least two operations through a `warpOperationJVP` call, reduces to a scalar loss, calls
`.backward()`, and asserts the original leaf inputs' `.grad` matches a
`torch.autograd.functional.jacobian`-based reference differentiated one level further than the
existing per-operator gradcheck scripts go. This is the test that actually validates the plan's
stated goal, not just its mechanics.

**Caveat to build the test around, not just note in passing:** the distinct-role adjoint bug (see
below) means any chained demonstration built with genuinely distinct query/reference tensors would
currently fail for an unrelated, already-tracked reason. Build this demonstration using the
self-referencing (`referenceParticles=None`) construction, same as the existing per-operator gradcheck
scripts, and note explicitly in the test that it doesn't cover the distinct-role case for the same
reason those don't.

## Related, tracked elsewhere (not part of this plan)

- **Distinct-role reverse-mode adjoint bug** (found while landing Item 3's predecessor,
  2026-08-20): `d(kernel-derivative-shaped output)/d(primal position)` is wrong when query and
  reference are genuinely distinct tensors, for every operator whose kernel math embeds a derivative
  (Gradient/Divergence/Curl/Laplacian, and every Tier-2 JVP operator). Reproduces identically in
  primal, non-JVP `warpOperation(..., WarpOperation.Gradient, ...)` — predates Tier-2 JVP entirely.
  Fully written up in `docs/lessons_learned.md`'s "Architectural facts still true" section (OPEN,
  2026-08-20 bullet) and tracked as its own project memory
  (`project_tier2_jvp_distinct_role_adjoint_bug`). Root-causing it is a fresh investigation into
  `math/wp_normalize.py`'s `@wp.func_grad` adjoints, not scoped by this plan or any of its
  predecessors.
- **HVP for the five non-Density operators** — explicitly out of scope by original user choice (JVP
  only). Not a backlog item; if a future Newton-style solve needs
  `Hess(Gradient/Divergence/Curl/Laplacian/Interpolate) @ v`, expect the same shape of work Density's
  own HVP needed (a hand-derived second-order helper, not free composition — see
  `docs/lessons_learned.md` / `wp_densityHVP.py`'s docstring for why generic composition doesn't work
  here).
- **Laplacian Dot/Default JVP schemes** — never derived (`warpier_adjoint.md` Tier 2.2's own scope
  note: they need per-spatial-component block indexing that doesn't share structure with
  Brookshaw/Naive). Correctly rejected today; would be a genuinely separate derivation exercise if
  ever needed.

## Verification (run after every item, not just at the end)

```bash
pytest tests/                                                    # full suite, current baseline: 292 passed / 1 skipped
python scripts/operation_matrix.py --device cpu --ci --verbose   # baseline: OK=258, HIGH=0, ERR=0, NAN=0
pytest tests/operations/test_gradcheck_scripts.py                # all registered gradcheck scripts, incl. Tier-2 JVP's six
```

Use this repo's `gradcheck`/`operation-matrix` skills for the last two rather than ad hoc invocation.

## Critical files

- `src/warpSPHCore/operations.py` — `warpOperationJVP`'s dispatch (Item 1)
- `src/warpSPHCore/coreOperations/wp_{interpolate,gradient,divergence,curl,laplacian,density}JVP.py`
  — docstring updates (Item 1), already-landed CSR kernels (Item 2's subject)
- `src/warpSPHCore/coreOperations/_jvpCommon.py` — `launchGeometryJVP`, the reverse-mode bridge
  (Item 3's subject)
- `tests/operations/test_forward_mode_tier2_*.py` — combined-tangent cases land here (Item 1)
- `scripts/gradcheck_tier2_jvp_*.py` — the six per-operator gradcheck scripts Item 3 builds beyond
- `docs/lessons_learned.md` — distinct-role adjoint bug write-up, kept in sync
- `docs/historic_plans/` — the four predecessor docs, full history/derivation detail
