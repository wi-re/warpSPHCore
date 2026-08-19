# Tier-2 JVP wiring for the remaining five core SPH operators

## Context

`warpier_forward_mode_plan.md` Phase 4 originally called for extending `warpOperationJVP` to
all six core operators (Density/Interpolate, Gradient/Divergence/Curl/Laplacian), but was scoped
down to Density only because that's what the implicit-shifting comparison (Phase 4's actual goal)
needed. The math for every operator was still fully derived and validated in `warpier_adjoint.md`
Tiers 2.1-2.5, and prototyped in dense-all-pairs form in `scripts/spike_forward_mode_tier2_*.py` —
none of that was wasted, it just never got promoted to production. The user flagged this gap after
Phase 4 was marked done and asked to close it: wire Tier-2 JVP support for the remaining five
operators into `warpOperationJVP`, matching Density's existing production pattern. Scope choice
(user-selected): **JVP only** — no Hessian-vector products beyond Density's existing one, and no
CRK/renormalization correction paths (Tiers 2.4/2.5 stay out of scope, cleanly rejected).

Two rounds of research (three parallel Explore agents, then a Plan agent) already extracted the
exact formulas from the spike scripts, the exact structure of each operator's existing production
kernel, and the exact test-file conventions to mirror. This document is that synthesis, reviewed
and finalized against the source files before execution. Written to the repo (not just the local
plan-mode scratch file) because this is expected to span multiple sessions.

## Status: all scope done, including the optional stretch item (plan finalized 2026-08-19,
completed 2026-08-19 -- see the bottom "Status" section for the final summary)

**Step 0 done (2026-08-19).** `_buildParticleSoA`/`_buildDomainState`/`_buildKernelState` extracted
from `wp_densityJVP.py` into `coreOperations/_jvpCommon.py` (`buildParticleSoA`/`buildDomainState`/
`buildKernelState`, public names since more than one module now imports them), with an added
optional `densities` argument. `wp_densityJVP.py` and `wp_densityHVP.py` both updated to import from
there instead of defining/duplicating; `tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py`
(which reached into `wp_densityJVP.py`'s former private names directly) updated the same way. Pure
extract-function — `test_forward_mode_tier2_density.py`'s 9 cases and
`test_forward_mode_tier2_density_hvp_self_pair.py` both pass unmodified; full suite still
139 passed / 1 skipped.

**Differentiability diagnostic done (2026-08-19), confirms the suspicion in Lookout 2:**
`scripts/diagnostic_tier2_jvp_reverse_mode.py` builds `positions`/`tangentQueryPositions` with
`requires_grad=True`, calls `computeSPHDensityPositionJVP`, and checks connectivity with
`torch.autograd.grad(..., allow_unused=True)` (the unambiguous check — plain `.backward()` followed
by inspecting `.grad is None` gave a misleading intermediate result during this check: an
all-zero-but-non-None tensor for a leaf that undergoes `.contiguous()` but is never otherwise
consumed, which looks like "connected with zero gradient" but is not; `allow_unused=True` sidesteps
that ambiguity by returning `None` per-tensor for anything the graph never actually reaches). Result:
`positions` and `tangentQueryPositions` are **not** reachable (`None`) — confirmed not
reverse-mode-differentiable through the bare-`wp.launch` path, exactly as reasoned. Only
`tangentReferenceMasses` (routed through plain-torch indexing, `tangentReferenceMasses[adjacency.j.long()]`,
never touching the warp kernel) is reachable, and its gradient is nonzero. This is an inherited
limitation of Phase 4 step 1's own design choice (bare `wp.launch` bypassing `launchOperator`'s
`wp.Tape`-backed autograd wrapper), not a regression introduced by this plan — see Lookout 2 for the
practical workarounds to flag if this becomes load-bearing for someone later.

**Step 1 done (2026-08-19).** `operations.py`'s `warpOperationJVP` gained `queryValues=None,
referenceValues=None`, consulted only inside the Tier-2 branch. `isDensityPositionSupportCase` is
now the Density-only sub-branch of a per-operator dispatch on `_TIER2_OPERATIONS = (Density,
Interpolate, Gradient, Divergence, Curl, Laplacian)`; Density's own code is byte-for-byte the same
as before (unmoved, unedited). The five value-having operators get one shared validation block (all
seven scope-boundary checks from this doc's "Scope boundaries" section) before consulting a new
`_TIER2_VALUE_DISPATCH` dict, populated incrementally by steps 2/4-7; empty for now, so all five
still raise `NotImplementedError` naming `warpier_tier2_operators_plan.md`. Gate:
`test_forward_mode_tier2_density.py`'s 9 cases (incl. `test_otherOperators_tier2_still_raise`,
still pointed at Gradient) and `test_forward_mode_tier1.py`'s 8 cases all pass unmodified; full
suite still 139 passed / 1 skipped.

**Step 2 done (2026-08-19), pulling forward the shared pair-kernel launcher part of Step 3.**
Before building Interpolate, the per-pair `(W_ij, dW_ij)` `@wp.kernel` (previously private to
`wp_densityJVP.py` as `_computeSPHDensityJVP_PairKernel`) moved to `_jvpCommon.py` as
`_sphKernelJVP_PairKernel` + a `launchPairKernelJVP` wrapper (SoA-building is unaffected, still via
`buildParticleSoA`/etc.); `wp_densityJVP.py` now calls `launchPairKernelJVP` instead of launching its
own copy -- byte-identical kernel body, pure extract-function. (Step 3's own remaining scope --
`sphKernelGradientJVP`/`_ij` in `kernels/kernelJVP.py`, the `∇W_ij` JVP Gradient/Divergence/Curl/
Laplacian share -- is unrelated to this and still pending.)

New `coreOperations/wp_interpolateJVP.py`: `computeSPHInterpolatePositionJVP`, formula
`dInterpolate_i = sum_j fj*(dVj*W_ij + Vj*dW_ij)`, `Vj = mass_j/density_j`,
`dVj = dmass_j/density_j - mass_j*ddensity_j/density_j^2`, `fj` (`referenceValues`) frozen. Takes
`tangentReferenceDensities` (new tangent, not used by Density) since `Vj` depends on
`referenceParticles.densities`; has no `tangentQueryDensities`/`queryValues`/`gradientMode`/
`laplacianMode` at all (Interpolate's formula has no query-side field or density term) -- this
surfaced a gap in Step 1's dispatch call, fixed by making `operations.py`'s `dispatchKwargs`
construction conditional per-operator rather than one fixed kwarg set for all five (see Step 1's
code comment). `computeSPHInterpolatePositionJVP` also explicitly rejects a provided `queryValues`
(`ValueError`, not silent ignore) since production `warpOperation(Interpolate)` never reads a
query-side field either. Registered in `coreOperations/__init__.py` and wired into
`operations.py`'s `_TIER2_VALUE_DISPATCH`.

New `tests/operations/test_forward_mode_tier2_interpolate.py` (12 cases, cloning the Density test's
structure): 1D `SupportScheme`-parametrized + 2D Jacobian-reference matches (density obtained by
calling production `warpOperation(Density)` once and then treated as an independent frozen input
with its own tangent, exactly like `tangentReferenceMasses` -- not literally re-derived from
positions inside the Jacobian, matching the JVP's own contract), plus rejection tests for
non-`AdjacencyList` adjacency, Tier-1+Tier-2 combination, missing `referenceValues`, a provided
`queryValues`, `tangentQueryMasses`, and `queryVolumes`. All pass. `test_otherOperators_tier2_still_raise`
left pointed at Gradient (unchanged -- still correctly pending). Full suite: 151 passed / 1 skipped
(was 139; +12 for this file). `operation_matrix.py --device cpu --ci --verbose` baseline unchanged:
OK=258, HIGH=0, ERR=0, NAN=0.

**Step 3 done (2026-08-19).** `sphKernelGradientJVP_ij`/`sphKernelGradientJVP` added to
`kernels/kernelJVP.py`, directly below `sphKernelJVP_ij`/`sphKernelJVP` -- ported byte-for-byte
from `scripts/spike_forward_mode_tier2_gradient.py`'s already-validated `_kernelGradientJVP`
(`rel_err ~1e-9` in float64 there, re-confirmed by re-running the spike unchanged: still all-PASS),
only swapping the spike's generic `vector(dtype=scalar_t, length=Any)` for this file's fixed
`vector(dtype=scalar_t, length=dim_t)` convention. New `coreOperations/wp_kernelGradientJVP.py`:
one shared `@wp.kernel` (`_sphKernelGradientJVP_PairKernel`), one thread per adjacency pair,
`launchPairKernelGradientJVP` producing `(G_ij, dG_ij)` flat `(numPairs, dim)` tensors.

**Pitfall hit and fixed:** the output kernel arguments were first declared
`wp.array(dtype=vec_t)` using `type_config.vec_t` -- but `vec_t = vector(length=dim_t, dtype=scalar_t)`
resolves to a length-`Any` (ungrounded) vector type whenever `warpSPHCore_DIM` isn't pinned via env
var (the default in this repo/test suite), which Warp rejects at kernel-launch time
(`TypeError: ... cannot be generic, got array(ndim=1, dtype=vec0f)`). Fixed by declaring `outG`/
`outDG` as `wp.array(dtype=Any)` instead (matching every other dimension-generic production kernel,
e.g. `wp_gradient.py`'s `outputValues`) and allocating concrete `(numPairs, dim)` torch tensors cast
via `castTorchToWarpAsBuiltins` (which resolves the concrete vector width from the tensor's own
shape, the same way `buildParticleSoA` already does for `positions`) rather than from a fixed
constant. `launchPairKernelGradientJVP` therefore takes explicit `dim`/`device`/`dtype` args.

New `_jvpCommon.gradientWeights(massJ, densityI, densityJ, dMassJ, dDensityI, dDensityJ, scheme)`:
the `GradientScheme`-dispatched `(A, B, dA, dB)` table from `warpier_adjoint.md` Tier 2.2, operating
on flat `[numPairs]` tensors (the spike's dense-`(n,n)` version, re-expressed).

**Steps 4-6 done (2026-08-19).** New `coreOperations/wp_gradientJVP.py`/`wp_divergenceJVP.py`/
`wp_curlJVP.py` (`computeSPHGradientPositionJVP`/`computeSPHDivergencePositionJVP`/
`computeSPHCurlPositionJVP`), each pure-torch coefficient assembly on top of Step 3's shared pair
kernel and `gradientWeights` -- no new `@wp.kernel` in any of the three. Curl returns shape
`[numParticles, 1]` matching `wp_curl.py`'s own `[1]`-forced 2D output convention; rejects
`domain.dim != 2` (centralized in `operations.py`, not the function itself). Divergence rejects
`divergenceDotMode=True`/`consistentDivergence=True` (centralized).

**Step 7 done (2026-08-19).** New `coreOperations/wp_laplacianJVP.py`
(`computeSPHLaplacianBrookshawPositionJVP`): `q_ij=(fj-fi)*B_ij` (`B` from `gradientWeights`, same
coefficient as Gradient's `B` term -- not re-derived), the regularized-distance chain
(`D_ij=r_ij+eps*h_ij`, `n_ij=x_ij/D_ij`, `L_ij=-2*q_ij*dot(G_ij,n_ij)/D_ij`) ported from the spike.
Explicitly rejects `laplacianMode != Brookshaw` inside the function itself (not just centrally) --
see "Real bug found and fixed" below for why this guard turned out to be load-bearing. Step 8
(Naive scheme) was **not attempted** -- left as the optional/stretch scope the plan always called it,
now enforced as an explicit `NotImplementedError` rather than silently mis-computing.

**Two real bugs found and fixed during verification (both in `operations.py`'s dispatch wiring, not
the ported kernel math itself -- the math was re-confirmed correct by re-running the spike script
unchanged before any of this new code existed):**

1. **`gradientMode` was never threaded through to Laplacian's dispatch call**, only `laplacianMode`
   -- so `computeSPHLaplacianBrookshawPositionJVP` silently used its own default
   (`GradientScheme.Symmetric`) regardless of what `operationProperties.gradientMode` actually was.
   This is exactly the kind of "silently pick the wrong branch" bug `warpier_adjoint.md` Tier 2.2's
   own "Process notes" warned about for this class of dispatch code (a near-identical
   int-vs-enum comparison bug was caught the same way while writing the spike). Caught by the
   Jacobian-reference test suite: only `GradientScheme.Symmetric` cases passed (15/20 Laplacian
   cases failed, `rel_err` large, not a rounding-scale miss) -- every other scheme silently got
   Symmetric's math instead of its own. Fixed by adding `WarpOperation.Laplacian` to the
   `gradientMode`-forwarding condition in `operations.py`'s dispatch-kwargs construction.
2. **The centralized scope-boundary check allows `LaplacianScheme.Naive` through to dispatch**
   (per this doc's own "Scope boundaries" section, written before Step 7 was implemented, on the
   assumption a future Step 8 might land it) **but Step 8 was never built.** Without a guard,
   `computeSPHLaplacianBrookshawPositionJVP` would have silently computed Brookshaw's answer for a
   caller who asked for `Naive` -- caught before it could ship, not by a failing test (there wasn't
   one yet), by re-reading the scope-boundary logic against what was actually implemented while
   writing this status entry. Fixed by adding an explicit `laplacianMode is not Brookshaw` rejection
   inside `computeSPHLaplacianBrookshawPositionJVP` itself.

**Test files added:** `tests/operations/test_forward_mode_tier2_{interpolate,gradient,divergence,
curl,laplacian_brookshaw}.py` (12/36/20/17/25 cases respectively -- Gradient and Laplacian sweep all
four `GradientScheme`s across a representative `SupportScheme` subset in both 1D and 2D, mirroring
the spike's own "exhaustive on the cheapest operator, representative subset elsewhere" pattern;
Divergence/Curl use a smaller `SupportScheme` subset per `warpier_adjoint.md`'s own validation
scope). `test_forward_mode_tier2_density.py::test_otherOperators_tier2_still_raise` repointed at
`WarpOperation.Covariance` (the plan's own suggested target once all five value-having operators
land -- Covariance has no Tier-2 formula and never will).

**Step 8 done (2026-08-19), the optional/stretch item, attempted after all.** After the core plan
(Steps 0-7) landed, the user asked to keep going, so Laplacian's Naive scheme was picked up too --
`warpier_adjoint.md` Tier 2.3 had already fully derived and validated the math
(`scripts/spike_forward_mode_tier2_laplacian_naive.py`, done 2026-08-18), so this was porting, not
deriving. New `sphKernelLaplacianJVP_ij`/`sphKernelLaplacianJVP` in `kernels/kernelJVP.py`, ported
byte-for-byte from the spike's `_kernelLaplacianJVP`, built on `kernels/laplacian.py`'s already-
production `sphKernelLaplacianGradient_`/`sphKernelLaplacianDkDh_`. **Mirrors `sphKernelLaplacian`'s
own two-branch `SupportScheme` dispatch (SuperSymmetric explicit, everything else -- including
KernelMeanSymmetric -- falling through to the max-fallback), not the three-branch KernelMeanSymmetric/
SuperSymmetric-together dispatch every other Tier-2 kernel-derivative function in this plan uses** --
`warpier_adjoint.md` Tier 2.3's own structural finding, re-confirmed by a dedicated test
(`test_laplacianNaivePositionJVP_kernelMeanSymmetric_differs_from_superSymmetric`, asserts the two
schemes' assembled JVPs genuinely *differ* here, unlike everywhere else in this plan).

`wp_laplacianJVP.py` restructured: `computeSPHLaplacianBrookshawPositionJVP`/
`computeSPHLaplacianNaivePositionJVP` are now scheme-specific (neither takes `laplacianMode`
anymore), with a new `computeSPHLaplacianPositionJVP` dispatcher between them by `laplacianMode` --
this is the one `operations.py`'s `_TIER2_VALUE_DISPATCH[WarpOperation.Laplacian]` actually
registers. Naive's own per-pair `(L_ij, dL_ij)` kernel launcher lives locally in `wp_laplacianJVP.py`
(not promoted to a shared module) since Naive is its only consumer, structurally identical to
`_jvpCommon.launchPairKernelJVP` otherwise.

New `tests/operations/test_forward_mode_tier2_laplacian_naive.py` (21 cases): `GradientScheme` x a
`SupportScheme` subset in 1D/2D (matching the spike's own coverage), the KernelMeanSymmetric-vs-
SuperSymmetric divergence check, and a missing-`queryValues` rejection.
`test_forward_mode_tier2_laplacian_brookshaw.py`'s old "Naive still raises" case removed from its
now-renamed `test_laplacianPositionJVP_rejects_unimplemented_modes` (Dot/Default only).

## Status: all scope done, including the optional stretch item (2026-08-19)

Density, Interpolate, Gradient, Divergence, Curl, and **both** Laplacian schemes (Brookshaw, Naive)
are all wired into `warpOperationJVP`'s Tier-2 branch and covered by Jacobian-reference tests. Full
verification:

- `pytest tests/`: **254 passed, 1 skipped** (was 136/1 baseline before this plan).
- `python scripts/operation_matrix.py --device cpu --ci --verbose`: **OK=258, HIGH=0, ERR=0, NAN=0**
  -- unchanged from every prior tier's baseline (no production kernel math was touched, only new
  files added and `wp_densityJVP.py`/`wp_densityHVP.py`'s imports refactored).
- `pytest tests/operations/test_gradcheck_scripts.py` (all 17 gradcheck/spike subprocess scripts,
  covering every operator's reverse-mode AD path): **all PASS** -- reverse-mode unaffected by this
  plan's forward-mode-only additions.
- `scripts/spike_forward_mode_tier2_gradient.py` and `scripts/spike_forward_mode_tier2_laplacian_naive.py`
  both re-run standalone: still all-PASS (re-confirms the math this plan ported was correct before
  porting; the two bugs found were both in this plan's own new dispatch-wiring code, not inherited
  from either spike).

**Not done, by design:** HVP for the six newly-landed operators -- out of scope by user's original
choice (JVP only), see Lookout 1 below, unchanged. Laplacian's Dot/Default schemes -- never derived
by `warpier_adjoint.md` at all (Tier 2.2's own scope note: they need per-spatial-component block
indexing `computeLaplacianDot2`/`computeDotLaplacian` don't share with Brookshaw/Naive, "a genuinely
separate... JVP-assembly exercise"), correctly rejected. The differentiability-through-reverse-mode
gap (Lookout 2) is unchanged by this work -- it was already true of Density's existing pair-indexed-
kernel pattern, and every new operator in this plan reuses that exact same bare-`wp.launch` pattern,
so the gap is now six operators wider, not newly introduced.

`warpier_forward_mode_plan.md` updated with a new dated status entry recording this plan's
completion, per this doc's own "Verification" section.

## Approach

Mirror `coreOperations/wp_densityJVP.py`'s existing pattern for every new operator: a `@wp.kernel`
launched **one thread per real adjacency-list pair** (not the spike's dense all-pairs shortcut),
producing per-pair values into flat arrays, then a `torch.index_add_` scatter-reduce in Python to
assemble the per-query-particle result — bypassing `OperatorSpec`/`launchOperator` (which only
supports per-query-particle thread counts), the same way `wp_implicitShifting.computeShiftingPairTerms`
already does in the sibling `warpSPH` repo.

**Key finding that changes the file layout from "one file per operator":** Gradient, Divergence,
Curl, and Laplacian(Brookshaw) all consume the *exact same* pairwise `∇W_ij` JVP (confirmed
byte-identical in the spike, `warpier_adjoint.md` Tier 2.2 finding 2) — only the downstream
coefficient/combination step differs, and that step is pure torch, no warp. So there is exactly
**one** new shared per-pair warp kernel for those four operators, not four independent ones.

### Step 0 — refactor shared boilerplate out of `wp_densityJVP.py`, plus a differentiability diagnostic

`_buildParticleSoA`/`_buildDomainState`/`_buildKernelState` (currently private to
`wp_densityJVP.py`) move to a new `coreOperations/_jvpCommon.py`, extended with an optional
`densities` argument (Density's own dummy-zero path stays default). `wp_densityJVP.py` is updated
to import from it — pure extract-function, no logic change. Verify with
`pytest tests/operations/test_forward_mode_tier2_density.py` before touching anything else, so
every later step builds on an already-proven-identical shared module instead of re-proving it six
times.

**Also part of this step, before building anything new (see "Lookouts" item 2 below for the full
reasoning): a small standalone diagnostic checking whether `computeSPHDensityPositionJVP`'s output
carries a gradient back to its own inputs under standard torch autograd** (`positions.requires_grad_()`,
call the JVP, `.backward()` the summed output, check `positions.grad`). This determines, once, up
front, whether the pair-indexed-kernel pattern every new operator in this plan reuses is reverse-mode
differentiable through torch at all — relevant to anyone composing these JVP calls into a larger
differentiable solve (e.g. an implicit timestepper) and wanting to backprop through it. Record the
answer in this file's status section either way; if it's "not differentiable" (suspected, not yet
confirmed), that's inherited from Phase 4 step 1's own design choice (bare `wp.launch`, bypassing
`launchOperator`'s autograd-tape wrapper, because that ABI doesn't support pair-indexed threading),
not a new problem this plan introduces or needs to fix.

### Step 1 — extend `warpOperationJVP`'s signature and generalize its dispatch

`operations.py`'s `warpOperationJVP` currently has no way to pass **frozen** field values
(`fi`/`fj`) through for the Tier-2 branch — it only has `tangentQueryValues`/`tangentReferenceValues`
(Tier-1's tangent-of-values), and Density never needed plain values since it has no value input at
all. Add `queryValues=None, referenceValues=None` to the signature, consulted only inside the
Tier-2 (`providedTier2`) branch; raise `ValueError` if they're passed on a pure Tier-1 call (a
caller who passes them there almost certainly meant Tier-2 and forgot a tangent argument).

Replace the single `isDensityPositionSupportCase` special-case with a small per-operator dispatch
table (`_TIER2_OPERATIONS = (Density, Interpolate, Gradient, Divergence, Curl, Laplacian)`).
**Density's existing branch is left as literally the same code, unmoved and unedited** — this is
the one piece of this plan with zero tolerance for behavior drift, so it's not routed through any
new shared helper. The new value-having-operator branch adds one shared validation block (see
"scope boundaries" below) before dispatching to each operator's `computeSPH<Op>PositionJVP`.

Gate: `test_forward_mode_tier2_density.py`'s existing 9 cases (including both `NotImplementedError`
tests) pass unmodified.

### Step 2 — Interpolate (Tier 2.1's other operator, cheapest, no new kernel math)

`coreOperations/wp_interpolateJVP.py`: `computeSPHInterpolatePositionJVP`. Reuses the *existing*
`sphKernelJVP` (`kernels/kernelJVP.py`) — same `(W_ij, dW_ij)` per-pair computation `wp_densityJVP.py`
already launches, factored into a tiny shared pair-kernel launcher both files call (in
`_jvpCommon.py` or a small adjacent module) so `wp_densityJVP.py` is refactored to reuse it too.
Formula: `dInterpolate_i = sum_j fj*(dVj*W_ij + Vj*dW_ij)`, `Vj = mass_j/density_j`,
`dVj = dmass_j/density_j - mass_j*ddensity_j/density_j^2`. Proves the new `queryValues`/
`referenceValues` plumbing end to end on the simplest possible case before building anything on
top of it.

### Step 3 — shared `∇W_ij` JVP building block

New `@wp.func sphKernelGradientJVP_ij`/`sphKernelGradientJVP` in `kernels/kernelJVP.py`, directly
below `sphKernelJVP_ij`/`sphKernelJVP` — byte-for-byte the spike's already-validated
`_kernelGradientJVP` (KernelMeanSymmetric/SuperSymmetric two-term-average branch via
`sphGradient_`/`sphKernelHessian_`/`sphGradientDkDh_`; everything else via
`computePairwiseSupport`/`computePairwiseSupportJVP`'s single-`h` branch). New
`coreOperations/wp_kernelGradientJVP.py`: one shared `@wp.kernel`, one thread per adjacency pair,
producing `(G_ij, dG_ij)` flat `(numPairs, dim)` tensors — the single new warp kernel launch shared
by the next four operators.

### Steps 4-6 — Gradient, Divergence, Curl (thin, parallel-buildable)

Each a new `coreOperations/wp_<op>JVP.py`, pure-torch coefficient assembly on top of step 3's
shared pair kernel — no new `@wp.kernel` in any of these three files. Shared
`_gradientWeights(mass_j, density_i, density_j, dmass_j, ddensity_i, ddensity_j, scheme)` (the
`GradientScheme`-dispatched `A`/`B`/`dA`/`dB` table from `warpier_adjoint.md` Tier 2.2) lives once
in `_jvpCommon.py`, reused by all three plus Laplacian:

```
Naive: A=0, B=Vj  |  Difference: A=-Vj, B=Vj  |  Summation: A=Vj, B=Vj
Symmetric: A=mass_j/density_i, B=mass_j*density_i/density_j^2
```

`coeff_ij = fi*A_ij + fj*B_ij` (fi/fj frozen), combined as: Gradient `sum_j coeff_ij * G_ij`;
Divergence `sum_j dot(coeff_ij, G_ij)` (dotMode=False only); Curl (2D only)
`sum_j [G_ij.x*coeff_ij.y - G_ij.y*coeff_ij.x]`, matching `wp_curl.py`'s `[1]`-output-shape
convention.

### Step 7 — Laplacian (Brookshaw scheme)

`coreOperations/wp_laplacianJVP.py`: `computeSPHLaplacianBrookshawPositionJVP`. Reuses step 3's
shared pair kernel and `_gradientWeights`, adds the regularized-distance chain on top (pure torch,
ordinary calculus): `q_ij=(fj-fi)*B_ij`, `D_ij=r_ij+eps*h_ij` (`eps=1e-8`), `n_ij=x_ij/D_ij`,
`L_ij=-2*q_ij*dot(G_ij,n_ij)/D_ij`, with `dr_ij`, `dD_ij`, `dn_ij` chain-ruled through ordinarily.

### Step 8 (stretch, do last, optional) — Laplacian (Naive scheme)

Only if steps 0-7 land cleanly with time to spare — not required to call "Laplacian" done for this
plan's six-operator scope, since Brookshaw is what `wp_laplacian.py` itself treats as the
consistent estimator. If attempted: new `sphKernelLaplacianJVP_ij`/`sphKernelLaplacianJVP` in
`kernels/kernelJVP.py`, built from the *already-production* `sphKernelLaplacian_`/
`sphKernelLaplacianGradient_`/`sphKernelLaplacianDkDh_` (`kernels/laplacian.py`, landed under Tier
2.3, confirmed present). Same `_gradientWeights`-derived `q_ij`, different `L_ij`/`dL_ij` source.
Note the *already-documented* dispatch asymmetry: `sphKernelLaplacian`'s `SupportScheme` dispatch
gives `KernelMeanSymmetric` the *max-fallback* branch, not a two-term average like Brookshaw's —
carry this into the test as an explicit "these two schemes genuinely differ here" regression check
(the spike already does this), not a bug to "fix."

## Scope boundaries (raise `NotImplementedError`/`ValueError`, enforced centrally in step 1's dispatch)

Every new operator branch checks these before dispatching, so individual `wp_<op>JVP.py` files stay
focused on the math (mirroring `wp_densityJVP.py`, which has no internal validation of its own):

- `crkState`, `renormalizationState`, or `gradHState` provided → raise (Tiers 2.4/2.5 and grad-h
  coupling are all out of scope for this pass).
- `queryVolumes`/`referenceVolumes` provided → raise (the derived formulas always use
  `mass_j/density_j` directly, never a volume override).
- `tangentQueryValues`/`tangentReferenceValues` provided alongside Tier-2 tangents → raise
  (`fi`/`fj` must be frozen; combined Tier-1+Tier-2 was never derived).
- `tangentQueryMasses` provided → raise (no formula has an `m_i` term — matches Density's existing
  check).
- Divergence: `divergenceDotMode=True` or `consistentDivergence=True` → raise (neither is in the
  derived math; guessing a semantics for either risks a plausible-looking wrong answer).
- Curl: `domain.dim != 2` → raise (1D and 3D both undecided by the spike; don't special-case 1D's
  trivial-zero case without deriving it).
- Laplacian: `laplacianMode not in (Brookshaw, Naive)` or `positiveDivergence=True` → raise
  (Dot/Default explicitly deferred by Tier 2.2's own "Result" section; `positiveDotProduct`'s extra
  term isn't in the derived formula).

## Tests

One new file per operator in `tests/operations/`, cloning `test_forward_mode_tier2_density.py`'s
structure exactly: jittered (±15%) non-uniform supports (needed to disambiguate `SupportScheme`
dispatch, per that file's own documented rationale), `torch.autograd.functional.jacobian`-based
reference contracted with the tangent (never `torch.autograd.functional.jvp` — silently zero
through the warp/`wp.Tape` bridge, per every prior tier's documented gotcha), `rtol=1e-3,
atol=1e-5`, `SupportScheme`/`GradientScheme` parametrization mirroring each spike's own case
matrix. Each file also tests its own scope-boundary rejections (CRK/renorm/gradH/volumes, plus the
operator-specific ones from above).

`test_forward_mode_tier2_density.py::test_otherOperators_tier2_still_raise` currently hardcodes
Gradient as "still raises" — update it incrementally as each operator lands (repoint at whichever
operator is still pending), rather than leaving it silently vacuous or deleting it outright; once
all five land it can be retired or repointed at Covariance (which stays out of scope throughout).

## Verification (run after every step, not just at the end)

```bash
pytest tests/operations/test_forward_mode_tier2_<newly-landed-op>.py -v
pytest tests/operations/test_forward_mode_tier2_density.py -v   # must stay green throughout
pytest tests/operations/test_forward_mode_tier1.py -v           # unaffected by the signature change
pytest tests/                                                    # baseline: 136 passed / 1 skipped
python scripts/operation_matrix.py --device cpu                  # baseline: OK=258, HIGH=0, ERR=0, NAN=0
python scripts/gradcheck_<op>_native.py                          # for every operator touched, reverse-mode unaffected
```

Use this repo's `gradcheck`/`operation-matrix` skills for the last two rather than ad hoc
invocation. After all steps land, update `warpier_forward_mode_plan.md` with a new dated status
entry recording what shipped (mirroring every prior phase's own documentation discipline) — this
is new work beyond Phase 4's original scope, so it gets its own entry rather than silently
reopening Phase 4.

## Critical files

- `src/warpSPHCore/operations.py` — `warpOperationJVP`'s signature + dispatch (step 1)
- `src/warpSPHCore/coreOperations/wp_densityJVP.py` — existing pattern to mirror; refactored in step 0
- `src/warpSPHCore/coreOperations/_jvpCommon.py` — new shared helpers (step 0), `_gradientWeights` (steps 4-7)
- `src/warpSPHCore/kernels/kernelJVP.py` — `sphKernelGradientJVP`/`_ij` (step 3), `sphKernelLaplacianJVP`/`_ij` (step 8, done)
- `src/warpSPHCore/coreOperations/wp_interpolateJVP.py`, `wp_kernelGradientJVP.py`, `wp_gradientJVP.py`, `wp_divergenceJVP.py`, `wp_curlJVP.py`, `wp_laplacianJVP.py` — new, steps 2-8
- `tests/operations/test_forward_mode_tier2_{interpolate,gradient,divergence,curl,laplacian_brookshaw,laplacian_naive}.py` — new
- `warpier_forward_mode_plan.md` — new status entry (done)
- `scripts/diagnostic_tier2_jvp_reverse_mode.py` — new, step 0's differentiability diagnostic (done)

## Lookouts — explicitly deferred, don't lose track of these

**1. HVP for the other six operators.** This plan is JVP-only by user choice (Density already has
`warpOperationHVP`; nothing else will after this plan). Not widely needed today — the only current
consumer is the implicit-shifting comparison, which only ever touches Density — but if a future
Newton-style solve needs `Hess(Gradient/Divergence/Curl/Laplacian/Interpolate) @ v`, the pattern to
reuse is already established and documented: generic torch-level composition (`torch.func.jvp`
applied twice, nested `torch.autograd.forward_ad`) does **not** work through a warp-kernel-backed
function in this codebase (confirmed for Density, `wp_densityHVP.py`'s own docstring: immediate
`RuntimeError` from one path, a silently-dropped tangent from the other) — the fix each time is a
small hand-written "second-order helper" that differentiates the JVP's own formula once more by
hand (which for Density needed zero new kernel math, just reusing `sphKernelHessian`). Whoever
picks this up should expect the same shape of work per operator, not assume it composes for free.

**2. Differentiability of the JVP bridge itself, through standard torch autograd — needs an
empirical check, not an assumption, before this plan is considered fully verified.** The scenario
that matters: someone builds an implicit timestepper's Newton solve out of `warpOperationJVP`/
`warpOperationHVP` calls (this plan's whole point), then wants to differentiate *through that
solve* — either forward-mode again (composing a further JVP, which HVP already shows is not free,
per lookout 1) or reverse-mode (an ordinary `.backward()` call reaching back through the solve to
whatever produced its inputs, e.g. for training/calibration). Reverse-mode differentiability
through the solve requires every JVP call inside it to itself carry a valid autograd graph back to
its own inputs — and there's a concrete reason to suspect it currently doesn't:
`wp_densityJVP.py`'s pair-indexed kernel launch is a **bare `wp.launch`**, deliberately bypassing
`OperatorSpec`/`launchOperator` (Phase 4 step 1's own documented reason: that ABI only supports
per-query-particle thread counts, not pair-indexed threading) — and `launchOperator`'s reverse-mode
support comes from wrapping kernel launches in a `torch.autograd.Function` that records a
`wp.Tape` (`StateAwareWarpFunction`, referenced in `wp_densityHVP.py`'s own docstring for a related
but distinct reason). A bare `wp.launch` outside that wrapper does not register any autograd node,
so `W_t`/`dW_t` (and therefore `dDensity`) likely carry **no** gradient back to
`queryParticles.positions`/`tangentQueryPositions`/etc. through ordinary PyTorch autograd today —
only the plain-torch mass-tangent indexing (`massJ = referenceParticles.masses[adjacency.j.long()]`
etc.) would show up as differentiable, everything routed through the warp kernel would not.
**This has not been empirically confirmed yet, only reasoned from reading the code** — the very
first thing to do under this plan (fold into Step 0, before building anything new) is a small
diagnostic test: build `positions`/`tangentQueryPositions` with `requires_grad=True`, call the
*existing* `computeSPHDensityPositionJVP`, sum its output, call `.backward()`, and check whether
`positions.grad`/`tangentQueryPositions.grad` are populated or `None`. Establish the answer once,
up front, on the one operator that already exists, rather than each new operator's test rediscovering
it independently. If (as suspected) the answer is "not differentiable this way": that's an inherited
limitation of Phase 4 step 1's own design choice (pair-indexed threading bypassing `launchOperator`),
not a regression this plan introduces, and not something this plan needs to fix (fixing it would mean
wrapping every new `wp_<op>JVP.py` function in its own `torch.autograd.Function`+`wp.Tape`, a real
design task of its own) — but it must be written down prominently (this plan's status section, and
eventually `warpier_forward_mode_plan.md`) rather than left for a future confused user building a
differentiable implicit timestepper to rediscover the hard way. A user who needs that today would
need `torch.func` functional transforms (`torch.func.grad`/`vjp` composed around the *whole* Newton
solve as a black box, not autograd through its internals) or a hand-written custom
`torch.autograd.Function` around the solve — flag both as the practical workarounds if the
diagnostic confirms the gap.

## Source research (for whoever picks this up)

This plan was synthesized from three parallel Explore-agent research passes (spike-script
formulas, production coreOperations kernel structure, test/gradcheck conventions) and one Plan
agent design pass, all run against this repo's state as of 2026-08-19. If resuming this work in a
new session, re-reading `warpier_adjoint.md` Tiers 2.1-2.2 and `coreOperations/wp_densityJVP.py` +
`kernels/kernelJVP.py` is enough to reconstruct the same context from scratch if needed — nothing
in this plan depends on information that isn't already in the repo.
