# Combined Tier-1 + Tier-2 JVP: one call for the full derivative

## Context

Flagged by the user after reading `wp_interpolateJVP.py` post-hoc: `warpOperationJVP` currently
forces a caller to choose *either* a Tier-1 tangent (`tangentQueryValues`/`tangentReferenceValues`
— the field-value direction) *or* Tier-2 tangents (`tangentQueryPositions`/`tangentQuerySupports`/
etc. — the geometry/mass/density direction), never both at once — `operations.py`'s dispatch raises
`NotImplementedError` if both are supplied (`"fi/fj must be frozen ... combined Tier-1+Tier-2 was
never derived"`). Two concerns, related:

1. **A real caller doesn't think in this taxonomy.** Someone differentiating an SPH operator through
   a larger computation (a Newton solve, a training loop) wants *the* JVP — the total directional
   derivative of `Interpolate`/`Gradient`/etc. in the direction of *however many* of its inputs are
   moving at once. Tier 1 vs. Tier 2 is this codebase's own internal bookkeeping (which building
   block was hard to derive vs. which was free), not a distinction a caller should have to make
   before they can get a straight answer.
2. **It isn't documented what `computeSPH<Op>PositionJVP` actually returns.** Reading
   `wp_interpolateJVP.py` cold, `dInterpolate_i`'s docstring gives a shape but not a plain statement
   that `referenceValues` is held at its **primal** (non-tangent) value in this formula — a reader
   could easily assume this function already *is* the total derivative and get a silently-partial
   answer by calling it in isolation.

## Key finding — verified empirically, not just derived

**Combining Tier 1 and Tier 2 is not new derivation work.** `warpier_adjoint.md`'s own opening
section already establishes Tier 1's premise: *"Every SPH operator is exactly linear in the field
values"* — not approximately, exactly, for every scheme of every operator in this codebase's scope.
Every value-having Tier-2 operator's per-pair coefficient is `coeff_ij = fi*A_ij + fj*B_ij`
(`_jvpCommon.gradientWeights`), where `A_ij`/`B_ij` never depend on `fi`/`fj` themselves — so each
operator is literally `Op(f, geometry) = L(geometry)[f]`, a linear map in `f` whose matrix depends on
geometry. Differentiating a bilinear-in-nothing-but-one-side-linear expression like this along a
path where *both* `f` and `geometry` move has no missing cross term: the total differential is
exactly `dOp = L(g0)[df] + (dL/dgeom · dgeom)[f0]` — the first term is precisely Tier 1's existing
formula (`warpOperation` relaunched with the tangent array in place of the value array, geometry
held at its primal value `g0`), the second is precisely Tier 2's existing formula (`f` held frozen
at its primal value `f0`, geometry tangent applied). **They are simply additive.**

Verified directly (not just argued) against `torch.autograd.functional.jacobian`-based ground truth
— differentiating `warpOperation` w.r.t. *every* input simultaneously (positions, supports, masses,
densities, queryValues, referenceValues) and contracting with all six tangents at once, compared
against `Tier1(dValues) + Tier2(dGeometry)` computed via the existing, already-shipped functions
called independently and summed:

| Operator                  | max abs diff | scale   | rel. error |
|----------------------------|-------------:|--------:|-----------:|
| Interpolate                | 9.5e-7       | 3.99    | 2.4e-7     |
| Gradient (Symmetric)       | 8.6e-6       | 7.66    | 1.1e-6     |
| Divergence (Symmetric)     | 9.5e-6       | 25.44   | 3.8e-7     |
| Curl (Symmetric)           | 9.5e-6       | 29.75   | 3.2e-7     |
| Laplacian, Brookshaw (Symmetric) | 1.5e-5 | 123.24  | 1.2e-7     |
| Laplacian, Naive (Symmetric)     | 9.9e-5 | 94.14   | 1.1e-6     |

All six agree to float32 round-off. This covers the simplest operator (Interpolate, no `A` term),
the most structurally complex remaining shape (Gradient's bilinear `coeff_ij`, shared by
Divergence/Curl/Laplacian-Brookshaw), and Laplacian's structurally-different Naive scheme (Tier 2.3,
different pair kernel entirely) — good coverage of every distinct formula shape in scope. (Density
has no value input at all — Tier 1 doesn't apply to it, already excluded from
`_TIER1_JVP_OPERATIONS`, no change needed there.)

## Design

**At `warpOperationJVP`'s dispatch level only** (`operations.py`) — no changes needed inside any of
the six `wp_<op>JVP.py` files, since this is pure summation of two already-correct, already-tested
pieces:

1. Replace the current unconditional rejection (`tangentQueryValues`/`tangentReferenceValues`
   alongside any Tier-2 tangent → `NotImplementedError`) with a three-way branch:
   - **Tier-2 tangents only** (today's default path) → call `computeSPH<Op>PositionJVP` as today,
     unchanged.
   - **Tier-1 tangents only, no Tier-2 tangents** → the existing Tier-1 path (`warpOperation`
     relaunched on the tangent arrays), unchanged.
   - **Both present** → call both existing paths (Tier-2 with `queryValues`/`referenceValues` at
     their primal value as it already requires; Tier-1 via the existing relaunch with the tangent
     arrays) and return their sum.
2. `queryValues`/`referenceValues` (the primal values) remain required for the Tier-2 half exactly
   as today — nothing about *that* contract changes, only what happens when a value-tangent is
   *also* supplied.
3. Update `warpOperationJVP`'s own docstring to describe this as the actual contract: "supply any
   subset of value tangents and geometry tangents; the return value is their sum, i.e. the operator's
   full JVP in whatever combined direction you asked for" — making `warpOperationJVP` itself the
   "one function that computes the full JVP" the user wants, with Tier 1/Tier 2 demoted to an
   internal implementation detail nobody needs to know about to use it correctly.

## Documentation fix (independent of whether the combined path lands first)

Each `computeSPH<Op>PositionJVP` docstring should state plainly, near the top, not buried:
*"This is the geometry/mass/density-tangent **partial** contribution to the operator's JVP —
`queryValues`/`referenceValues` are held at their **primal** (non-tangent) value here. It is **not**
the full derivative on its own; add the Tier-1 (value-tangent) contribution
(`warpOperation` relaunched with the tangent value array) for that, or call `warpOperationJVP`
directly once it sums both automatically."* Apply this same clarification to all six operators'
functions plus `computeSPHDensityPositionJVP` (whose case is simpler to state — Density has no value
input, so its own JVP is *never* partial in this sense, worth saying explicitly too so a reader
doesn't wonder).

**Naming (flag, don't decide unilaterally):** `computeSPH<Op>PositionJVP` is named for its dominant
tangent (position) but actually covers support/mass/density tangents too, and now — per the
docstring fix above — needs to also communicate "frozen values." `computeSPH<Op>GeometryJVP` would
be more accurate, but renaming touches five already-shipped, already-tested production functions and
their `__all__`/dispatch-table entries; not undertaken as part of this plan without confirming the
user wants churn on already-landed names for a documentation-clarity gain alone.

## Steps

1. Extend the additivity check above (already run for one `GradientScheme`/`SupportScheme` per
   operator during this plan's own drafting) to the full case matrix each operator's existing
   `test_forward_mode_tier2_*.py` file already sweeps (all `GradientScheme`s, a representative
   `SupportScheme` subset, 1D and 2D) — cheap, since it's confirming the same linearity argument
   holds pointwise, not re-deriving anything.
2. Relax `operations.py`'s rejection into the three-way branch (design §1-2 above).
3. Update `warpOperationJVP`'s docstring (design §3).
4. Update all six `computeSPH<Op>PositionJVP` docstrings plus `computeSPHDensityPositionJVP`'s
   (documentation fix above) — do this regardless of how long step 2 takes, it's low-risk and
   addresses the user's second concern independently.
5. New tests: for each of the five value-having operators, extend (or add alongside)
   `test_forward_mode_tier2_<op>.py` with a "combined" case — call `warpOperationJVP` with **both**
   tangent sets simultaneously, compare against a `torch.autograd.functional.jacobian` reference that
   differentiates w.r.t. *every* input at once (positions/supports/masses/densities/queryValues/
   referenceValues), not just geometry as today's tests do. Also keep (don't remove) a case
   confirming the Tier-1-only and Tier-2-only paths are numerically **unchanged** by this refactor —
   a regression guard on the two paths whose behavior must not shift now that they're being summed
   under a third condition.
6. Full verification sweep: `pytest tests/`, `operation_matrix.py --device cpu --ci --verbose`,
   `pytest tests/operations/test_gradcheck_scripts.py` — same discipline as every prior Tier-2 step.

## Explicitly out of scope

- **Density** — no value input, Tier 1 doesn't apply, nothing to combine.
- **HVP, the reverse-mode-through-JVP-bridge plan** (`warpier_tier2_jvp_reverse_mode_plan.md`) —
  independent concerns. Once *that* plan lands, a combined-JVP call would also become
  reverse-mode differentiable for free (it's built from the same two already-existing paths this
  plan sums), but neither plan depends on the other to be useful on its own.
- **CRK/renormalization/volumes** — still out of scope, unchanged; those scope-boundary checks stay
  exactly as they are.
- **Fusing the two kernel launches into one.** The combined path costs two warp kernel launches (the
  existing Tier-2 pair kernel plus a Tier-1 relaunch of the primal per-query-particle kernel) instead
  of a hypothetical single fused kernel computing both contributions in one pass. Worth noting as a
  performance characteristic (not a correctness concern) — a real fusion would be new kernel-writing
  work, not the "just sum two already-correct pieces" shape this plan is scoped to. Revisit only if
  profiling ever shows it matters.

## Verification

```bash
pytest tests/operations/test_forward_mode_tier2_*.py        # existing + new combined cases
pytest tests/                                                # full suite
python scripts/operation_matrix.py --device cpu --ci --verbose   # baseline: OK=258, HIGH=0, ERR=0, NAN=0
pytest tests/operations/test_gradcheck_scripts.py             # unaffected -- no production kernel touched
```

## Critical files

- `src/warpSPHCore/operations.py` — `warpOperationJVP`'s dispatch (steps 2-3)
- `src/warpSPHCore/coreOperations/wp_{interpolate,gradient,divergence,curl,laplacian,density}JVP.py`
  — docstring-only changes (step 4), no logic changes
- `tests/operations/test_forward_mode_tier2_{interpolate,gradient,divergence,curl,
  laplacian_brookshaw,laplacian_naive}.py` — new combined-tangent cases (step 5)
