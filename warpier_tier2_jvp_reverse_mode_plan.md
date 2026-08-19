# Reverse-mode through the Tier-2 JVP bridge

## Context

`warpier_tier2_operators_plan.md` (Steps 0-8, done 2026-08-19) wired forward-mode (JVP) position/
support/mass/density-tangent derivatives into `warpOperationJVP` for all six core SPH operators.
That plan's Step 0 diagnostic (`scripts/diagnostic_tier2_jvp_reverse_mode.py`) empirically confirmed
its own Lookout 2's suspicion: none of the six `computeSPH<Op>PositionJVP` functions are reverse-mode
differentiable w.r.t. their own inputs, because every one launches its pair-indexed warp kernel via a
**bare `wp.launch(...)`** rather than through this repo's autograd bridge
(`StateAwareWarpFunction`/`launchOperator`) — that bridge's convenience layer (`OperatorSpec`/
`extractStateInfo`) only knows how to build the fixed 8-struct, per-query-particle-threaded kernel
ABI every *primal* production operator (`wp_density.py`, `wp_gradient.py`, etc.) shares, and the
Tier-2 JVP kernels need a different ABI entirely (one thread per adjacency **pair**, plus a second
pair of "tangent" particle-state structs alongside the primal ones). `torch.autograd.grad(...,
allow_unused=True)` on `computeSPHDensityPositionJVP`'s output showed `positions`/
`tangentQueryPositions` unreachable — only inputs consumed via plain-torch indexing (e.g.
`referenceParticles.masses[adjacency.j.long()]`) survive.

The user wants this closed: differentiate this repo's existing autograd wrapper (or design a
sibling one for the pair-indexed/dual-code case) so a `warpOperationJVP` call can be embedded
inside an ordinary PyTorch backprop graph — e.g. a Newton-style implicit solve built out of
`warpOperationJVP` calls, where the *solve's own* gradient (w.r.t. whatever produced its inputs)
needs to flow back through every JVP call inside it.

**Sequencing note, added after `warpier_tier2_jvp_csr_backend_plan.md` was written (2026-08-19,
also not started): read that plan before starting this one.** It ports the Tier-2 JVP kernels off
the pair-indexed (COO) launch shape this plan is designing a bridge *for*, onto the same
per-query-particle (CSR) shape every primal operator already uses — which already fits
`extractStateInfo`'s existing convention almost exactly. If that port happens first, this plan's own
"three bespoke pair-kernel `build_fn` closures" shrinks to "extend `extractStateInfo` with two more
struct slots (tangent query/reference state)," reusing its existing adjacency/grid-traversal
machinery entirely rather than duplicating it for a kernel shape about to be replaced. Building the
bespoke pair-indexed bridge this plan describes below is still fully valid and unblocked if the CSR
port hasn't happened yet or won't happen soon — just worth checking which order is intended before
investing in either.

**Scope: JVP only, not HVP.** `wp_densityHVP.py` has the identical bare-`wp.launch` gap and its own
docstring documents *why* composing it generically failed (`torch.func.jvp` errors outright,
`forward_ad.make_dual` silently drops the tangent) — but HVP is a second-order object (a JVP of a
JVP) and stays a hand-derived special case regardless of what this plan does; see "Explicitly out of
scope" below for why this plan doesn't change that calculus.

## Research summary (full detail in the session that wrote this plan; key claims below)

1. **The bridge itself (`autograd/stateAwareWarpFunction.py`'s `StateAwareWarpFunction`) is already
   fully generic** — `forward(ctx, build_fn, launcher, kernel, output_shape, output_dtype,
   *flat_tensors)` takes an arbitrary `build_fn: List[wp.array] -> tuple-of-kernel-args` closure and
   an arbitrary `launcher` (in practice `autograd/launcher.py`'s `launch_kernel`, itself generic:
   `launch_kernel(kernel, output_shape, output_dtype, *args)`). It converts every tensor in
   `flat_tensors` to a `wp.array` via `getCachedWarpArray` (which correctly propagates
   `requires_grad` from the source tensor — confirmed by reading `cache.py`, nothing to fix there),
   opens a `wp.Tape()` around exactly one `launcher(...)` call if anything requires grad, and its
   `backward()` seeds gradients via `tape.backward(grads={output_warp: ...})` before reading each
   distinct `wp.array`'s `.grad` back into a torch tensor (with an `id(wa)`-keyed dedup guard for the
   common case where one tensor fills two roles, e.g. `referenceParticles is queryParticles`).
2. **The actual blocker is `autograd/arg_extract.py`'s `extractStateInfo`**, which is the *only*
   thing that builds a `build_fn` today, and it hard-codes the primal 8-struct ABI
   (`qPart, rPart, domState, useAdjacency, adjState, gState, corrState, kernProps`) — no concept of
   a second "tangent" particle struct, and no way to swap in a different `build_fn` through
   `launchOperator`. This is a convenience-layer limitation, not a `StateAwareWarpFunction`
   limitation: `OperatorSpec`/`launchOperator` were simply never asked to support a different ABI.
3. **`launch_kernel` already supports everything the pair kernels need with no changes**: arbitrary
   `output_shape` (an int, so `output_shape=numPairs` launches exactly one thread per pair, matching
   `_jvpCommon.py`'s existing `(numPairs,)`-shaped `W_t`/`dW_t` outputs) and multi-output
   (`output_dtype=[scalar_t, scalar_t]` for `(W, dW)` or `(G, dG)`/`(L, dL)`, already exercised by
   production multi-output kernels elsewhere).
4. **Feasibility precedent for the one real open question ("can Warp's reverse-mode AD correctly
   differentiate a kernel whose own forward pass already computes a derivative-level quantity?"):**
   this is not new territory. `wp_gradient.py`'s *production* kernel already calls `sphGradient_`
   (`d(kernel value)/dx`) as part of computing `Gradient`'s own forward output, and reverse-mode
   differentiating that (which requires Warp to propagate through an already-once-differentiated
   term, i.e. produce something Hessian-shaped internally) already works today and is
   `gradcheck_gradient_native.py`-validated. The Tier-2 JVP kernels are the same shape one level up
   (their forward pass computes `dW`/`dG`/`dL`, built from `sphKernelHessian_`/`sphGradientDkDh_`/
   `sphKernelLaplacianGradient_` etc.) — reverse-differentiating them asks Warp's adjoint-generation
   for exactly the same *kind* of thing it already does successfully elsewhere, just with one more
   layer of the same building blocks. The custom `@wp.func_grad` adjoints in the dependency chain
   (`safe_sqrt`, `vectorNorm_warp`/`vectorNormalize_warp`, all in `math/wp_*.py`) were deliberately
   hand-regularized at `r=0` specifically so a second differentiation stays finite there instead of
   hitting `0/0` — `wp_densityHVP.py`'s docstring calls this out explicitly as "the concrete
   mechanism a from-scratch double-backward attempt would have to get right." This plan doesn't need
   to get it right *from scratch* — it's already been gotten right, by whoever wrote those adjoints,
   for exactly this kind of second-derivative-level reverse pass. Still: **empirical confirmation
   (Step 0 below) comes before generalizing**, matching this repo's own established discipline
   (every Tier-2.x math derivation in `warpier_adjoint.md` was cross-checked against `wp.Tape` before
   being trusted, never assumed correct from the derivation alone).
5. **`edgeI`/`edgeJ` (the pair-index arrays) and dummy fields (e.g. the tangent SoA's unused
   `kinds`) have an established, uncontroversial way to enter the `flat_tensors`/`build_fn`
   pipeline**: `extractStateInfo` already threads non-differentiable int-dtype tensors through the
   exact same flat-list/`build_fn` machinery today (`qK`/`rK`, the per-particle `kinds` arrays, flat
   positions 8-9) — no special-casing needed, they just go through as ordinary (non-`requires_grad`)
   entries and contribute `None` gradient automatically.

## Approach

Mirror `arg_extract.py`'s own pattern (`extractStateInfo` + its closure-based `build_fn`), but for
the pair-indexed ABI instead of the primal per-query-particle one. Concretely: write a **new,
small, pair-kernel-specific extraction helper** for each of the three distinct pair-kernel ABIs this
plan's predecessor produced, and swap each shared launcher's internals from a bare `wp.launch` to
`StateAwareWarpFunction.apply(build_fn, launch_kernel, kernel, output_shape=numPairs,
output_dtype=[...], *flat_tensors)`.

**Key design property, worth stating explicitly: zero call-site changes.** All three launchers this
plan touches (`_jvpCommon.launchPairKernelJVP`, `wp_kernelGradientJVP.launchPairKernelGradientJVP`,
`wp_laplacianJVP.py`'s local `_launchPairKernelLaplacianJVP`) already return exactly the same
`(out1_t, out2_t)` shape they do today, so none of the six `wp_<op>JVP.py` files, `operations.py`'s
dispatch table, or `warpOperationJVP` itself need to change at all. This is deliberate: it keeps the
blast radius to three internal functions, and it means every one of Tier-2's 254 passing tests stays
a valid regression gate for "did this change break the no-grad path" without modification.

Only **three** pair-kernel ABIs exist across all six operators (per `warpier_tier2_operators_plan.md`
Step 3's own finding that Gradient/Divergence/Curl/Laplacian-Brookshaw share one `(G, dG)` kernel):

1. **`(W_ij, dW_ij)`** — `_jvpCommon._sphKernelJVP_PairKernel`, used by Density and Interpolate.
2. **`(G_ij, dG_ij)`** — `wp_kernelGradientJVP._sphKernelGradientJVP_PairKernel`, used by Gradient,
   Divergence, Curl, and Laplacian(Brookshaw).
3. **`(L_ij, dL_ij)`** — `wp_laplacianJVP._sphKernelLaplacianJVP_PairKernel`, used by
   Laplacian(Naive) only.

Each pair kernel's signature is `(queryState, referenceState, queryTangentState,
referenceTangentState, domainState, kernelProperties, edgeI, edgeJ, out1, out2)` — `build_fn` needs
to reconstruct everything before `edgeI`/`edgeJ` (the outputs are appended automatically by
`launch_kernel`, exactly like `extractStateInfo`'s `build_fn` never returns `outputValues` either).

### Step 0 — single-operator spike + gradcheck (Density's `(W, dW)` kernel only)

Before touching any of the three shared launchers for real, prototype the extraction/`build_fn` for
*just* Density's `(W, dW)` kernel in a throwaway script (mirroring
`scripts/spike_forward_mode_tier2_*.py`'s own role earlier in this plan's predecessor), and validate
with `torch.autograd.gradcheck` directly against `computeSPHDensityPositionJVP` (not a hand Jacobian
— this repo's own `lessons_learned.md` rule, already followed by every `gradcheck_*_native.py`
script). Cases to cover:

- `positions`/`tangentQueryPositions`/`supports`/`tangentQuerySupports` all `requires_grad=True`,
  confirm all four now receive finite, correct gradients (replacing
  `scripts/diagnostic_tier2_jvp_reverse_mode.py`'s `allow_unused=True` `None` result with an actual
  `gradcheck` PASS).
- The self-referencing case (`referenceParticles=None` so `queryParticles is referenceParticles`,
  and/or `tangentReferencePositions is tangentQueryPositions` — both patterns already used
  throughout the Tier-2 test suite) — confirm no double-counted gradient, i.e. that the `id(wa)`
  dedup in `StateAwareWarpFunction.backward` behaves correctly when the *same* tensor is threaded
  into `flat_tensors` at two distinct positions (query role and reference role).
- A case with a self-pair present in `adjacency` (`i == j`, `SupportScheme.KernelMeanSymmetric`
  adjacency construction includes these) — this is exactly where the custom `@wp.func_grad` adjoints
  flagged in the research summary matter most; confirm no NaN/Inf and a value consistent with
  `wp_densityHVP.py`'s own already-proven self-pair identity (self term contributes exactly zero to
  a *first*-order reverse pass here too, for the same translation-invariance reason, though this
  plan does not need to re-derive that — just confirm empirically it doesn't blow up).

**Gate: this step is the actual go/no-go for the rest of the plan.** If `gradcheck` fails here in a
way that traces back to the custom adjoint chain rather than a build_fn bug, that's new information
worth writing up before continuing (not expected, per the feasibility precedent above, but this is
exactly the kind of thing to confirm empirically rather than assume).

### Step 1 — promote the spike into `_jvpCommon.launchPairKernelJVP` (Density + Interpolate)

Once Step 0 passes, fold the extraction/`build_fn` logic into `_jvpCommon.py` for real, replacing
`launchPairKernelJVP`'s bare `wp.launch` body. Density gets re-gradchecked directly. Interpolate
needs **no separate wrapping** — its own downstream math (`dVj*W_ij + Vj*dW_ij`, `index_add_`) is
already ordinary differentiable torch, so once `(W_t, dW_t)` themselves carry a valid `grad_fn`,
`computeSPHInterpolatePositionJVP`'s full chain (including `tangentReferenceMasses`/
`tangentReferenceDensities`, which already worked before this plan for unrelated reasons, plus now
`positions`/`tangentQueryPositions`/`supports`/`tangentQuerySupports`) becomes differentiable "for
free." Re-run `gradcheck` against `computeSPHInterpolatePositionJVP` to confirm this rather than
assume it.

### Step 2 — `wp_kernelGradientJVP.launchPairKernelGradientJVP` (Gradient, Divergence, Curl, Laplacian-Brookshaw)

Same treatment for the `(G, dG)` pair kernel — one wrapping covers all four operators, by the same
"downstream is ordinary torch" argument Step 1 established for Interpolate. `gradcheck` each of the
four operators' Tier-2 JVP functions independently (they assemble `(G, dG)` differently enough --
scalar `coeff` for Gradient, `dot(coeff, G)` for Divergence, the 2D cross product for Curl, the
`n_ij`/`D_ij` regularized-distance chain for Laplacian-Brookshaw -- that a bug in one operator's own
assembly wouldn't necessarily show up in another's).

### Step 3 — `wp_laplacianJVP.py`'s local `_launchPairKernelLaplacianJVP` (Laplacian-Naive)

Same treatment for the `(L, dL)` pair kernel, Naive's sole consumer. `gradcheck` against
`computeSPHLaplacianNaivePositionJVP` directly (through the `computeSPHLaplacianPositionJVP`
dispatcher, `laplacianMode=Naive`).

### Step 4 — full sweep: gradcheck every Tier-2 JVP function, every differentiable input

Extend (or add sibling scripts to) `tests/operations/test_gradcheck_scripts.py`'s pattern with new
`scripts/gradcheck_tier2_jvp_*.py` entries — one per operator, each checking `torch.autograd.
gradcheck` against `computeSPH<Op>PositionJVP` (or `warpOperationJVP` with Tier-2 tangent kwargs,
whichever the existing Tier-2 test files already call) w.r.t. **every** now-differentiable input:
`positions`/`supports`/`masses`/`densities` on both query and reference roles, plus their
`tangent*` counterparts, wherever each operator's own formula actually uses them (mirroring exactly
which tangents each `test_forward_mode_tier2_*.py` file already exercises for the *value* check —
this step is adding the *gradient-of-that-value* check on top, not new cases). Register each new
script in `GRADCHECK_SCRIPTS` (`test_gradcheck_scripts.py`) so it's a permanent CI gate, matching
every other operator's reverse-mode coverage in this repo.

### Step 5 — an actual embedded-in-backprop demonstration

Steps 0-4 prove each `computeSPH<Op>PositionJVP` call is reverse-mode differentiable in isolation.
The user's actual goal is embedding a `warpOperationJVP` call *inside a larger differentiable
computation* (a Newton-style implicit solve, ultimately) — so this step is an end-to-end test that
chains at least two operations through a `warpOperationJVP` call and confirms `.backward()` reaches
all the way to the original leaf inputs. Concretely: build a small scalar loss as a function of, say,
`positions` → `warpOperationJVP(Gradient, ...)` → some reduction → `loss.backward()` → assert
`positions.grad` is populated and matches a `torch.autograd.functional.jacobian`-based reference
(the same reference-construction pattern every Tier-2 *value* test already uses, just differentiated
one level further). This is the test that actually validates the plan's stated goal, not just its
mechanics.

## Explicitly out of scope

- **HVP.** `wp_densityHVP.py`'s bare pair-indexed launch has the identical gap and *could* be
  wrapped the same way this plan wraps JVP's launches (mechanically nothing stops it) — but that
  only makes `computeSPHDensityPositionHVP`'s own *first*-order reverse-mode differentiable, not
  automatically composable into a "true" second-order object the way `torch.func.jvp`-of-`jvp` would
  be. HVP already exists as a hand-derived formula specifically *because* that generic composition
  failed (`wp_densityHVP.py`'s own docstring); this plan doesn't change that calculus and doesn't
  attempt to. If HVP's own reverse-mode gap becomes relevant later, it is a small, separate follow-up
  (same `build_fn` shape, different kernel) — not attempted here.
- **True second-order composition** (differentiating *through* one of these newly-reverse-mode
  JVP calls a second time, forward- or reverse-mode) is not attempted. This plan makes
  `d(JVP output)/d(JVP inputs)` available via ordinary `.backward()` — it does not make `JVP` itself
  composable with a further `torch.func.jvp`/nested nested nested `wp.Tape`. That would require
  genuinely nested taping (a `wp.Tape` recorded while another `wp.Tape` is already open, replaying
  through `StateAwareWarpFunction.backward`'s own tape), which Warp's documented AD model does not
  straightforwardly support and this plan has not researched.
- **`OperatorSpec`/`extractStateInfo`/the primal operators are untouched.** This plan adds new
  extraction/`build_fn` code alongside the existing pattern, it does not generalize
  `extractStateInfo` itself to also handle the pair-indexed ABI (that would couple two independently-
  evolving conventions for no benefit — the primal operators will never need tangent structs).
- **Performance (state-bundle reuse, `warpier_fields.md` Step F/H-style caching)** is deferred.
  `arg_extract.py`'s `use_bundle` path exists purely as a later optimization once correctness is
  established for the primal ABI; this plan's `build_fn`s should always pass `use_bundle=False` (or
  simply not implement the bundle branch at all) in their first version. Revisit only if profiling
  ever shows the fresh-struct-per-call cost matters for this path specifically.

## Risks / lookouts

1. **The one real open question is Step 0's gate** (does `wp.Tape` correctly reverse-differentiate a
   kernel whose forward pass already contains a derivative-level term, specifically through *these*
   kernels' specific custom-adjoint dependency chain) — addressed by empirical `gradcheck`, not
   assumed from the feasibility precedent alone, per this repo's own established discipline.
2. **Self-referencing double-counting.** `StateAwareWarpFunction.backward`'s `id(wa)`-keyed dedup
   only works correctly if the *same* source tensor reliably produces the *same* `wp.array` object
   when it appears at two `flat_tensors` positions in one call (query and reference roles). The new
   `build_fn`s must route repeated tensors through the *same* `getCachedWarpArray`-backed conversion
   `StateAwareWarpFunction.forward` already performs (i.e. just list the tensor twice in
   `flat_tensors`, don't independently `castTorchToWarp` it a second time inside `build_fn`) —
   `StateAwareWarpFunction.forward`'s own loop already handles this correctly by construction as long
   as the extraction step doesn't bypass it. Step 0's self-referencing test case exists specifically
   to catch a mistake here.
3. **Don't reach for `use_bundle=True`.** `arg_extract.py`'s own comment is explicit that struct
   reuse across grad-requiring calls is unsafe without a bespoke zero-on-acquire contract this
   plan's `build_fn`s won't have set up — a future optimization pass could get this wrong by copying
   the bundle pattern without also copying its safety net.
4. **This is new surface for the "reentrancy" class of bug** `lessons_learned.md` documents
   (never assign `.grad` directly, always `tape.backward(grads=...)`; always `.clone()` gradients
   read off a `wp.array`; always `tape.zero()` before returning) — all already handled *inside*
   `StateAwareWarpFunction`, so as long as this plan's new code calls `StateAwareWarpFunction.apply`
   rather than hand-rolling a fresh `torch.autograd.Function`, none of these rules need to be
   re-derived. Worth stating as a design constraint: **do not write a new `torch.autograd.Function`
   subclass for this** — reuse `StateAwareWarpFunction` exactly as-is, only supply new `build_fn`s.

## Verification (run after every step, not just at the end)

```bash
python scripts/gradcheck_density_native.py                 # existing, must stay green throughout
pytest tests/operations/test_gradcheck_scripts.py           # existing 17 scripts, must stay green
pytest tests/operations/test_forward_mode_tier2_*.py        # existing 254-ish JVP value tests, must stay green (no-grad path unchanged)
python scripts/operation_matrix.py --device cpu --ci --verbose   # baseline: OK=258, HIGH=0, ERR=0, NAN=0
```

Plus, once Step 4's new scripts exist, add them to the standing gate the same way.

## Critical files

- `src/warpSPHCore/autograd/stateAwareWarpFunction.py` — the bridge this plan reuses unmodified
- `src/warpSPHCore/autograd/arg_extract.py` — the pattern to mirror (not modify) for the new
  pair-kernel `build_fn`s
- `src/warpSPHCore/autograd/launcher.py` — `launch_kernel`, reused unmodified
- `src/warpSPHCore/coreOperations/_jvpCommon.py` — `launchPairKernelJVP`'s internals change (step 1)
- `src/warpSPHCore/coreOperations/wp_kernelGradientJVP.py` — `launchPairKernelGradientJVP`'s
  internals change (step 2)
- `src/warpSPHCore/coreOperations/wp_laplacianJVP.py` — `_launchPairKernelLaplacianJVP`'s internals
  change (step 3)
- `scripts/diagnostic_tier2_jvp_reverse_mode.py` — the existing diagnostic this plan's Step 0 spike
  should graduate into a real `gradcheck`-based confirmation
- new `scripts/gradcheck_tier2_jvp_*.py` (step 4), registered in
  `tests/operations/test_gradcheck_scripts.py`
- `docs/lessons_learned.md` — the reentrancy/double-counting/direct-`.grad`-assignment rules this
  plan's new code must not violate (all already enforced by reusing `StateAwareWarpFunction`
  unmodified, per Lookout 4 above)
