# Tier-2 JVP: remaining open work (consolidated)

## Context

This supersedes four prior plan docs, each moved to `docs/historic_plans/` rather than deleted (full
derivation history, status logs, and lessons stay there for reference):

- `docs/historic_plans/warpier_tier2_operators_plan.md` — **fully done** (2026-08-19), all six core
  operators plus all four Laplacian schemes (Brookshaw/Naive originally, Dot/Default added
  2026-08-20 — see "Related, tracked elsewhere" below) wired into `warpOperationJVP`. Nothing open
  here except one deliberate, by-design scope exclusion (HVP for the five non-Density operators) —
  not a backlog item, see "Explicitly not planned" below.
- `docs/historic_plans/warpier_tier2_jvp_csr_backend_plan.md` — Steps 1-6 done and cut over
  (2026-08-20); **Step 7 (benchmark), carried into this doc as Item 2, waived by user decision
  2026-08-20** — the CSR-over-COO win is an asymptotic storage argument (`O(n * avg neighbors)` vs.
  `O(0)` intermediate storage), not something a runtime benchmark meaningfully adds nuance to; see
  Item 2 below.
- `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md` — bridge (`_jvpCommon.launchGeometryJVP`)
  wired for all six operators (2026-08-20), Steps 0-4 verified; **Step 5 (embedded-in-backprop
  demonstration) not started** — carried into this doc as Item 3. Also surfaced an out-of-scope
  reverse-mode bug, tracked separately (see "Related, tracked elsewhere" below).
- `docs/historic_plans/warpier_tier2_combined_jvp_plan.md` — design finished, empirically verified
  (additivity confirmed against `torch.autograd.functional.jacobian` for all six operators), **but no
  implementation step was started** — carried into this doc as Item 1.

Three genuinely open items were identified. As of 2026-08-20: Items 1 and 3 are done, Item 2 is
waived by user decision (not going to be done — see below). **This closes out every open point on
this plan.**

## Item 1 — Combined Tier-1 + Tier-2 JVP — DONE (2026-08-20)

**Turned out to be already implemented** by the time this was picked up: `operations.py`'s
`warpOperationJVP` dispatch already had the three-way branch (geometry-only / value-only / both
summed) and the "supply any subset, get the sum" docstring, and five of the six value-having
operators' `test_forward_mode_geometry_jvp_*.py` files already had a `_check_combined_jacobian_reference`
+ `test_*_combined_matches_jacobian_reference_2d` case (steps 2-5 below, landed piecemeal alongside
earlier Tier-2 operator work rather than as this plan's own pass) — all six
`computeSPH<Op>PositionJVP`/`computeSPHDensityPositionJVP` docstrings already stated the
frozen-values contract too (step 4). The one genuine gap: `LaplacianScheme.Dot`'s combined test was
missing from `test_forward_mode_geometry_jvp_laplacian_dot.py` (that file itself only landed
2026-08-20, alongside Dot/Default's JVP implementation, and the combined case wasn't added when the
file was created). Added `_check_combined_jacobian_reference`/
`test_laplacianDotGeometryJVP_combined_matches_jacobian_reference_2d` there, mirroring the Default
scheme's version but with a `(n, 2)` vector field (Dot needs dim-sized blocks in 2D, unlike
Brookshaw/Naive/Default). All 4 new cases pass; full verification sweep below re-run clean
(353 passed / 1 skipped, OK=258/HIGH=0/ERR=0/NAN=0, gradcheck 23 passed).

<details>
<summary>Original plan text (for reference — design done, implementation not started)</summary>

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

</details>

## Item 2 — CSR backend: benchmark memory/runtime (Step 7) — WAIVED (2026-08-20), not going to be done

**Source:** `docs/historic_plans/warpier_tier2_jvp_csr_backend_plan.md`, Step 7 (was the only step
left).

**User decision:** the measurement isn't worth doing. CSR's win over COO isn't a runtime-constant
argument that needs a benchmark to confirm — it's an asymptotic storage argument: COO needs
`O(n * avg neighbors)` intermediate storage for the pair-indexed launch, CSR needs `O(0)` (the
per-query-particle kernel never materializes a pairs array at all). That's a complexity-class
difference, not a "how much faster" question a benchmark would meaningfully add nuance to. Separately,
the old COO implementation was already deleted specifically because it shouldn't ship into production
code except when absolutely necessary — resurrecting it from git history to run an A/B, as this item
would have required, cuts against that same judgment. Waived, not deferred: this is not expected to be
picked up later.

## Item 3 — Reverse-mode bridge: embedded-in-backprop demonstration (Step 5) — DONE (2026-08-20)

**Source:** `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md`, Step 5 (was the only step
left).

Added `scripts/gradcheck_tier2_jvp_chained_backprop.py`, registered in `test_gradcheck_scripts.py`'s
`GRADCHECK_SCRIPTS`. Chains `positions/supports/masses` → `warpOperation(Density)` → `densities`
(still attached to the graph) → repacked into a `ParticleState` → `warpOperationJVP(Gradient, ...)`
(consuming that same `densities` tensor) → `(output ** 2).sum()` → `loss.backward()`. Verified with
`torch.autograd.gradcheck` (finite differences — this repo's own established ground truth, not a hand
Jacobian, per `docs/lessons_learned.md`) against the whole chain as one function, not just the JVP
call alone, plus an explicit `loss.backward()` + populated/finite `.grad` check on every leaf
(`positions`/`supports`/`masses`/all four Gradient-JVP tangents/`queryValues`/`referenceValues`).
Passed on the first run — no bugs found; Steps 0-4's per-operator reverse-mode work already covered
the hard part, this just proved composition through an upstream op works too. Self-referencing
construction (`referenceParticles=None`), same convention every sibling `gradcheck_tier2_jvp_*.py`
script uses — the distinct-role caveat below is moot here regardless since it's lifted anyway. Full
verification sweep re-run clean (354 passed / 1 skipped, OK=258/HIGH=0/ERR=0/NAN=0).

**Caveat now lifted (2026-08-20):** this previously warned that a chained demonstration built with
genuinely distinct query/reference tensors would fail, due to the self-pair Hessian bug below. That
bug is now root-caused and fixed (see "Related, tracked elsewhere" below) — the self-referencing
construction is no longer required for this reason. Still fine to build Step 5's demonstration either
way; if built with distinct query/reference tensors, no special-casing around this is needed anymore.

## Related, tracked elsewhere (not part of this plan)

- **Self-pair reverse-mode Hessian bug — FIXED 2026-08-20** (found while landing Item 3's
  predecessor): `d(kernel-derivative-shaped output)/d(primal position)` was wrong specifically at an
  exact self-pair (`x_i == x_j`, `r == 0`) between a query and a reference point, for every operator
  whose kernel math differentiates `sphGradient_`'s output a further time w.r.t. position
  (Gradient/Divergence/Curl, and every Tier-2 JVP operator's `dW`/`dG`/`dL`). **Originally
  mischaracterized as a generic "query != reference tensors" bug — it is not**: genuinely distinct,
  non-coincident query/reference positions were always differentiated correctly; the trigger is
  positional coincidence specifically, which is what every existing "distinct-role" gradcheck script's
  `positions.detach().clone()` construction happened to produce. Root cause: `math/wp_normalize.py`'s
  `norm_hess_warp` blows up like `O(1/eps)` at `x=0` and gets multiplied against an exactly-zero
  `kernelTerm`, silently collapsing a genuine finite nonzero limit (the kernel's own peak curvature,
  same value `kernels/hessian.py`'s `sphKernelHessian_` already computes correctly via its own explicit
  `q < eps` branch) down to `0.0` — a floating-point `0 * infinity` cancellation, not a genuine
  singularity. Fixed by giving `kernels/gradient.py`'s `sphGradient_` a custom `@wp.func_grad` that
  returns `sphKernelHessian_`/`sphGradientDkDh_`'s already-validated closed forms directly, instead of
  Warp's automatic (buggy-at-r=0) composition. Full write-up in `docs/lessons_learned.md`'s
  "Architectural facts still true" section and the `project_tier2_jvp_distinct_role_adjoint_bug`
  project memory (both updated 2026-08-20 with the corrected diagnosis and fix).
- **HVP for the five non-Density operators** — explicitly out of scope by original user choice (JVP
  only). Not a backlog item; if a future Newton-style solve needs
  `Hess(Gradient/Divergence/Curl/Laplacian/Interpolate) @ v`, expect the same shape of work Density's
  own HVP needed (a hand-derived second-order helper, not free composition — see
  `docs/lessons_learned.md` / `wp_densityHVP.py`'s docstring for why generic composition doesn't work
  here).
- **Laplacian Dot/Default JVP schemes — FIXED 2026-08-20** (`warpier_adjoint.md` Tier 2.2's own scope
  note had deferred them: Dot needs per-spatial-component block indexing that doesn't share
  structure with Brookshaw/Naive). Both now implemented (`computeSPHLaplacianDotGeometryJVP`/
  `computeSPHLaplacianDefaultGeometryJVP`, `coreOperations/wp_laplacianJVP.py`) and wired into
  `warpOperationJVP`/`computeSPHLaplacianGeometryJVP` alongside Brookshaw/Naive — Default turned out
  to be Brookshaw's own formula one quotient-rule level deeper (no block indexing at all); Dot's
  block projection was genuinely new math but mechanical once Brookshaw's shared
  `(G, dG, n_ij, dn_ij, D_ij, dD_ij, P, dP)` building blocks were factored out and reused. Both
  schemes' `queryValues`/`referenceValues`/output are generic `Any`-typed (vector-field capable,
  unlike Brookshaw/Naive's fixed `scalar_t`) — Dot re-enforces `wp_laplacian.py`'s own restriction
  that a scalar field is out of scope for it in >1D. Full write-up in `warpier_adjoint.md`'s "Tier
  2.2b — Result" subsection.
- **Laplacian Dot's reverse-mode adjoint — pre-existing bug, found while landing the above, FIXED
  2026-08-20 (same day)** (both the primal `computeLaplacianDot2` and the new Tier-2
  `computeSPHLaplacianDotJVP_Func_i` — same root cause, two call sites): `torch.autograd`/`.backward()`
  through either gave a wrong gradient specifically for `LaplacianScheme.Dot` — confirmed via finite
  differences as independent ground truth. Root cause: a Warp code-generation limitation where a
  loop-accumulated local (`proj = dot(q_block, n_ij)`, built via a runtime loop) consumed by a further
  non-linear op in the *same* function (`proj * n_ij`) silently drops part of its reverse-mode
  gradient. **Fix:** move the accumulation loop into its own separate `@wp.func` that *returns* the
  accumulated value, called from the original function instead of inlined there
  (`math/wp_laplaciandot.py`'s `computeDotLaplacian(q_ij, n_ij, dim, base)` overload /
  `wp_laplacianJVP.py`'s `_laplacianDotProjJVP`) — a recurring pattern in this codebase, not novel to
  this bug. Confirmed fixed via `torch.autograd.gradcheck` (all four `LaplacianScheme`s now pass,
  `scripts/gradcheck_tier2_jvp_laplacian.py`) and finite differences agreeing with the (now-correct)
  jacobian-based test reference. Full write-up, minimal repro, and the two things that looked like
  fixes but weren't, in `docs/lessons_learned.md`'s "Warp kernel authoring gotchas" section. Item 3's
  "embedded-in-backprop" demonstration is no longer blocked for Dot.

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
