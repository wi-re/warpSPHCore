# CSR-style backend port for the Tier-2 JVP kernels

## Status (2026-08-20): Steps 1-6 done, Step 7 not started

All seven operators (Density, Interpolate, Gradient, Divergence, Curl,
Laplacian-Brookshaw, Laplacian-Naive) are ported to the CSR shape and
cut over — the old COO/pair-indexed implementations, `_jvpCommon.launchPairKernelJVP`/
`_sphKernelJVP_PairKernel`, `wp_kernelGradientJVP.py` (deleted entirely),
and `wp_laplacianJVP.py`'s local Naive pair kernel are gone.
`computeSPH<Op>PositionJVP`'s names/signatures/return shapes are unchanged,
per Step 5's own requirement — `warpOperationJVP` and every existing
`test_forward_mode_tier2_*.py` file needed no changes. Full test suite green
(261 passed, 1 skipped) and `operation_matrix.py --device cpu --ci` baseline
unchanged (OK=258, HIGH=0, ERR=0, NAN=0) both before and after cutover.

**Grid (`CompactHashMap`) traversal landed too**, not just the CSR reshape —
flagged in the plan's own "explicitly out of scope" section below as a
natural pickup, and the user confirmed it was fine to add. Each
`computeSPH<Op>PositionJVP` now accepts an `AdjacencyList` *or*
`CompactHashMap`. `warpOperationJVP` itself still centrally rejects non-
`AdjacencyList` adjacency for Tier-2 (that gate in `operations.py` was left
untouched — grid support is available at the direct-import level, not yet
exposed through the public JVP entry point, since wiring that through wasn't
asked for). A `test_<op>PositionJVP_grid_traversal_matches_adjacency_traversal`
test was added to each operator's existing test file (comparing CSR-via-grid
against CSR-via-adjacency-list on the same particle set) to keep that
capability covered going forward.

**Equivalence proof, not preserved as standing tests**: each step's own
old-vs-new numerical equivalence check (Step 1's own gate, repeated per
operator) was run as throwaway test files during development and passed
across every `SupportScheme`/`GradientScheme` combination to `rtol=1e-5,
atol=1e-6` — tighter than the existing Jacobian-reference tests' own
`rtol=1e-3, atol=1e-5`, since old-vs-new is two independently-derived numeric
paths computing (up to launch-shape) the same floating point operations, not
a Jacobian-vs-analytic identity. Those equivalence test files were deleted
after the cutover (comparing an implementation against itself is moot once
there's only one) — see git history around 2026-08-20 if that evidence is
ever needed again; the grid-traversal tests that came out of the same files
were kept (see above), since that comparison stays meaningful post-cutover.

**Step 7 (benchmark memory/runtime on a large-particle-count case) was
explicitly deferred** — not started. Left for whoever picks this up next;
Step 7's own text below still describes what it should measure.

**Interaction with `warpier_tier2_jvp_reverse_mode_plan.md`, now actionable**:
the "Key interaction" section below predicted that once this CSR port landed,
that plan's own "three bespoke pair-kernel extraction closures" work would
shrink to "extend `extractStateInfo` with two more slots (tangent particle
state)" — that's now true, since every Tier-2 JVP kernel is per-query-
particle-threaded like every primal operator. Not acted on here; noting it
stays accurate for whoever picks up either plan next.

## Context

Every Tier-2 JVP kernel shipped by `warpier_tier2_operators_plan.md` (Density, Interpolate,
Gradient, Divergence, Curl, Laplacian-Brookshaw, Laplacian-Naive) uses the same launch shape: one
warp thread **per adjacency pair** (`edgeI[e]`, `edgeJ[e]`, `e` ranging over `numPairs`, a **COO**
layout — arbitrary, unsorted `(i, j)` pairs), each thread computing a per-pair scalar/vector
(`W_ij`/`dW_ij`, `G_ij`/`dG_ij`, `L_ij`/`dL_ij`), written into a flat `[numPairs]`-shaped output
array. The reduction down to a per-query-particle result then happens in **Python/torch**:
`torch.index_add_(0, adjacency.i.long(), pairContribution)`.

Every *primal* production operator (`wp_density.py`, `wp_interpolate.py`, `wp_gradient.py`,
`wp_divergence.py`, `wp_curl.py`, `wp_laplacian.py`) does the opposite: one warp thread **per query
particle** `i`, looping internally over that particle's own neighbors via the adjacency's flat,
grouped neighbor-list array (`beginIndex`/`numIndices`/`offsetArray` — a **CSR**-style layout, each
particle's neighbors contiguous), accumulating a local sum entirely in that thread's own registers,
and writing `outputValues[i]` exactly once. `warpier_core.md` calls this the "canonical structured
kernel ABI" — `computeSPHXXX_Func_i` (per-neighbor body) → `computeSPHXXX_Func_Adjacency` (dispatches
across `useAdjacency`/grid traversal) → `computeSPHXXX_Kernel` (the actual `@wp.kernel`), the same
three-function shape in every one of the six files above.

**The user's concern, raised after reviewing the shipped Tier-2 JVP code:** the COO/pair-indexed
approach the JVP kernels use has two real drawbacks relative to the CSR pattern every other operator
in this codebase already uses:

1. **Memory.** Materializing `O(numPairs)` (or `O(numPairs × dim)`) intermediate arrays — the per-pair
   `(W,dW)`/`(G,dG)`/`(L,dL)` outputs themselves, plus every subsequent torch-level elementwise op
   (`coeff`, `dcoeff`, `pairContribution`, etc.) — is a real memory cost that CSR-style accumulation
   avoids entirely (everything but the final `[numQuery]`-shaped result lives transiently in
   per-thread registers during the neighbor loop, never materialized as a standalone array).
2. **Atomics.** Reducing per-pair contributions into per-query-particle sums via `torch.index_add_`
   requires atomic adds under the hood on GPU (multiple pair-threads can target the same output
   index, unordered) — CSR-style needs none, since each query-particle thread owns exclusive write
   access to its own single output slot.

**Framing, in the user's own words: this is "technically only a porting" step — a backend change
that shouldn't make a numerical difference, to be done *after* the operators are correctness-verified
and wrapped up**, not urgent or blocking. This plan is written now so it doesn't get lost, per the
same "write it down, don't lose track" discipline every other Tier-2 lookout in this document family
has followed.

## Why this was COO/pair-indexed in the first place

Phase 4 step 1's own documented reason (inherited by every Tier-2 operator since): `OperatorSpec`/
`launchOperator` — the machinery that would otherwise let a kernel dispatch through the
per-query-particle "canonical structured kernel ABI" for free — only supports per-query-particle
thread counts, and (more importantly for this port) `extractStateInfo`'s `build_fn` has no concept
of a **tangent** particle state struct alongside the primal one. Bypassing that machinery with a
bare pair-indexed `wp.launch` was the fast path to a *correct* JVP, not a considered memory/atomics
tradeoff — this plan is the port back onto the CSR shape once correctness was established, per the
user's own sequencing.

## Key interaction worth flagging: this likely reshapes `warpier_tier2_jvp_reverse_mode_plan.md`

That plan (written earlier this session, not started) designs three bespoke `build_fn`/extraction
closures specifically *because* the current pair-indexed kernels don't fit `extractStateInfo`'s
per-query-particle-threaded convention. **Once this CSR port lands, the JVP kernels *would* fit that
convention** — same per-query-particle threading, same adjacency/grid traversal every primal kernel
already uses via `getIndexRange`/`checkDirectionality_i`/`_j`. The only remaining gap would be
`extractStateInfo`'s missing tangent-particle-state struct slots — a small, additive extension (two
more struct slots: `queryTangentState`/`referenceTangentState`) rather than a wholesale bespoke
bridge. **Sequencing recommendation, not a decision made here:** doing this CSR port before or
alongside `warpier_tier2_jvp_reverse_mode_plan.md` would mean that plan's own `build_fn` work shrinks
from "three bespoke pair-kernel extraction closures" to "extend `extractStateInfo` itself with two
more slots" — avoiding building a bespoke bridge for kernels that are about to be replaced anyway.
Left for whoever picks up either plan to decide the order; noting the dependency so it isn't
discovered mid-implementation.

## Approach

For each of the seven Tier-2 JVP kernels, write a new `computeSPHXXXJVP_Func_i` /
`computeSPHXXXJVP_Func_Adjacency` / `computeSPHXXXJVP_Kernel` triple, following the exact
"canonical structured kernel ABI" shape its *primal* counterpart already uses (e.g.
`wp_gradient.py`'s `computeSPHGradientTensor_Func_i`/`_Func_Adjacency`/`_Kernel`), with two changes:

1. **A tangent counterpart threaded alongside every primal argument**: `iTangentPtcl` alongside
   `iPtcl`, `referenceTangentState` alongside `referenceState` — same shape, same struct types
   (`particleDataSoA_1/2/3`), just holding tangent values instead of primal ones.
2. **The per-neighbor body computes both the primal contribution and its JVP, accumulating both** —
   reusing the already-validated per-pair `@wp.func`s (`sphKernelJVP`, `sphKernelGradientJVP`,
   `sphKernelLaplacianJVP`, `kernels/kernelJVP.py`) for the `(W,dW)`/`(G,dG)`/`(L,dL)` piece
   unchanged, verbatim, called from inside the neighbor loop instead of a standalone per-pair kernel.
   The operator-specific coefficient/combination math — currently pure torch in each
   `wp_<op>JVP.py` (`_gradientWeights`, `coeff_ij = fi*A_ij + fj*B_ij`, Laplacian's `n_ij`/`D_ij`
   chain) — moves **into** the kernel body, mirroring the *primal* kernel's own per-scheme branching
   exactly (e.g. `wp_gradient.py`'s `computeSPHGradientTensor_Func_i` doesn't factor into an
   `A_ij`/`B_ij` decomposition at all — it directly computes `fj*apparentVolume` for `Naive`,
   `mass_j*density_i*(fi/density_i² + fj/density_j²)` for `Symmetric`, etc., per `if/elif` branch on
   `kernelProperties.gradientMode`, then `out += outerTensorProduct(coeff, kernelGradient, ...)`).
   The JVP version needs the differentiated form of each branch (`dcoeff`, product-ruled against
   `kernelGradient`/`dKernelGradient`), one extra term per branch, not a new structure.
3. **Output: a single `outputValues[i]` write per query particle** — no per-pair intermediate array,
   no `torch.index_add_`, no atomics.

This is a materially larger port than either of this session's other two Tier-2 follow-up plans: it
touches the actual per-operator math assembly (currently Python/torch, would become warp kernel
code), not just dispatch wiring or pure summation. Each operator needs its **own** kernel now
(unlike the current shared-pair-kernel-launcher approach, where one `(G,dG)` launcher served four
operators) — the shared piece that survives is the per-pair building block (`sphKernelJVP`/
`sphKernelGradientJVP`/`sphKernelLaplacianJVP`), called from inside seven distinct per-query kernels
instead of from three distinct per-pair ones.

## Steps

(Steps 1-6 done, see Status above; Step 7 not started.)

1. **Prove the pattern on one operator first: Density** (simplest — no coefficient assembly beyond
   `dmj*W_ij + mj*dW_ij`, no `GradientScheme` branching). New `computeSPHDensityJVP_Func_i`/
   `_Func_Adjacency`/`_Kernel`. Validate numerically **equivalent** (not just plausible) to the
   current COO implementation's own output, on `test_forward_mode_tier2_density.py`'s existing
   cases — both the old and new implementation must agree to float32 round-off on the same inputs.
   This is the step that actually proves "backend swap, no numerical difference," per the user's own
   framing — don't skip straight to deleting the old implementation.
2. **Interpolate** — reuses Density's proven `(W,dW)`-per-neighbor math plus its own `Vj`/`dVj`
   coefficient (no scheme branching either). Same equivalence-to-old-implementation gate.
3. **The shared-`(G,dG)` four: Gradient, Divergence, Curl, Laplacian-Brookshaw** — each gets its own
   kernel (per-operator coefficient/combination differs and must be inlined), but all four reuse
   `sphKernelGradientJVP` verbatim for the per-pair piece. Port and equivalence-check one at a time,
   Gradient first (matches its primal kernel's branching structure most directly, quoted above).
4. **Laplacian-Naive** — reuses `sphKernelLaplacianJVP` verbatim; own kernel for the `q_ij`/combination
   step.
5. **Cutover, only once all seven are proven equivalent**: delete the old pair-indexed kernels/
   launchers (`_jvpCommon.launchPairKernelJVP`, `_jvpCommon._sphKernelJVP_PairKernel`,
   `wp_kernelGradientJVP.py`'s pair kernel, `wp_laplacianJVP.py`'s local Naive pair kernel) and the
   `torch.index_add_`-based Python assembly in each `wp_<op>JVP.py`; wire the new CSR kernels in
   their place. `computeSPH<Op>PositionJVP`'s function names, signatures, and return shapes stay
   **unchanged** — this is purely an internal-implementation swap, so no caller anywhere in this
   codebase (`warpOperationJVP`, every `test_forward_mode_tier2_*.py` file) needs to change.
6. **Re-run the full existing Tier-2 test suite unmodified** as the regression gate — since call
   signatures don't change, every existing test should pass without edits, which is itself the proof
   the swap was numerically transparent.
7. **Benchmark memory and runtime** on a large-particle-count case (the actual payoff this plan
   exists for) and report the before/after numbers — don't just infer the win from the design, measure
   it, matching this repo's general "measure, don't assume" discipline (see e.g. the
   `operation-matrix`/`gradcheck` skills' own emphasis on this).

## Explicitly out of scope

- **No math changes.** Every per-pair building block this port reuses (`sphKernelJVP`,
  `sphKernelGradientJVP`, `sphKernelLaplacianJVP`) is already validated and stays byte-for-byte
  unchanged; this plan is pure backend restructuring.
- **Sequencing relative to `warpier_tier2_jvp_reverse_mode_plan.md` and
  `warpier_tier2_combined_jvp_plan.md`** is flagged above as an open question, not resolved here.
- **HVP** (`wp_densityHVP.py`) is untouched — same bare-pair-indexed-launch shape, same class of
  potential future port, not attempted here.
- **Grid (non-adjacency) traversal.** The current pair-indexed kernels only support
  `AdjacencyList`-based traversal (`isinstance(adjacency, AdjacencyList)`, `NotImplementedError`
  otherwise) — the primal kernels' `_Func_Adjacency` pattern natively supports both `useAdjacency`
  and grid/`CompactHashMap` traversal, so this port *could* pick up grid support "for free" as a side
  effect of reusing that pattern. Worth flagging as a natural extension once the port is underway,
  not a required part of it — don't let scope creep into `warpier_tier2_operators_plan.md`'s own
  explicitly-scoped-out grid-traversal gap without a deliberate decision to do so.

## Verification (per step, and at the end)

```bash
pytest tests/operations/test_forward_mode_tier2_*.py   # must stay green, unmodified, throughout
pytest tests/                                            # full suite
python scripts/operation_matrix.py --device cpu --ci --verbose   # baseline: OK=258, HIGH=0, ERR=0, NAN=0
pytest tests/operations/test_gradcheck_scripts.py         # unaffected unless/until combined with the reverse-mode plan
```
Plus a new standalone equivalence check per operator (step 1-4's own gate): old COO implementation's
output vs. new CSR implementation's output, same inputs, same tolerance the existing Jacobian-
reference tests already use.

## Critical files

- `src/warpSPHCore/coreOperations/wp_density.py`, `wp_interpolate.py`, `wp_gradient.py`,
  `wp_divergence.py`, `wp_curl.py`, `wp_laplacian.py` — the primal kernels whose
  `Func_i`/`Func_Adjacency`/`Kernel` shape this port mirrors (read, don't modify)
- `src/warpSPHCore/coreOperations/wp_{density,interpolate,gradient,divergence,curl,laplacian}JVP.py`
  — new per-operator CSR kernels land here, replacing each file's current COO-based implementation
- `src/warpSPHCore/kernels/kernelJVP.py` — `sphKernelJVP`/`sphKernelGradientJVP`/
  `sphKernelLaplacianJVP`, reused verbatim, unmodified
- `src/warpSPHCore/coreOperations/_jvpCommon.py`, `wp_kernelGradientJVP.py` — the COO pair-kernel
  launchers this port ultimately deletes (step 5)
- `warpier_tier2_jvp_reverse_mode_plan.md` — read first if picking this plan up, for the sequencing
  interaction noted above
