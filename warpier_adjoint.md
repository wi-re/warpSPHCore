# Tier-2 Forward-Mode: JVP Derivation Plan

Companion to `warpier_core.md` Phase 6 and `warpier_fields.md` §3.6, which established the
Tier 1 / Tier 2 split and where this plan starts:

* **Tier 1** (tangents w.r.t. field values) is done and standing (`scripts/spike_forward_mode_tier1.py`,
  gated in `tests/operations/test_gradcheck_scripts.py`). Every SPH operator is exactly linear in the
  field values, so a Tier-1 tangent is the existing kernel re-launched on the tangent array. No new
  kernel math.
* **Tier 2** (tangents w.r.t. positions/supports/masses/densities) is where the kernel itself is
  genuinely nonlinear (a function of `|x_i - x_j|/h`), so it needs hand-derived JVPs -- "adjoint
  kernel functions" in the terminology this plan uses. `warpier_fields.md` flagged Tier 2 as "the
  expensive tier, costed separately" and stopped there. This document is that costing, broken into
  gated stages.

This plan does not touch `Field`/`ExecutionMode`/the torch-forward-mode bridge (`warpier_core.md`
Phase 6's Steps 1-4, still not started). It is scoped to one layer below that: **do the per-operator
JVP formulas exist and are they correct**, independent of how they eventually get wired into a
`Field.tangent` slot or a `jvp` staticmethod. That wiring is a separate, later step and should reuse
whatever bridge Phase 6 eventually builds -- this plan's deliverable is formulas plus proof they're
right, in wp.Tape/PyTorch scripts (`scripts/*_tier2_*.py`), the same shape as Tier 1's spike and this
session's `scripts/kernel_sanity_native.py`.

---

## Status as of 2026-08-18

**Tier 2.5 (CRK correction) is done -- the last tier in this plan.** All of Tier
2.0-2.5 are now complete; see "Suggested order and why" below for the final
scorecard. `scripts/spike_forward_mode_tier2_crk.py` assembles the JVP in four
gated stages -- apparentVolume (`crk_volume.py`, Gather mode, Tier 2.1's
single-h JVP dispatch), the CRK moments `m_0/m_1/m_2/dm_0dgamma/dm_1dgamma/
dm_2dgamma` (`crk_moments.py`, Scatter mode, ordinary product rule over Tier
2.1's kernel-value JVP and Tier 2.2's `kernelGradient` JVP), the CRK factors
`A/B/gradA/gradB` (`crk_terms.py`'s `computeCRKTermsWarp`), and
`correctGradientCRK`'s product-rule expansion (`crk/kernel.py`) combined with
Tier 2.2's field-value coefficient into a CRK-corrected Gradient operator.
Matches the production reverse-mode Jacobian to float64 round-off (`rel_err`
`1e-15`-`1e-16`) at every stage, 1D line of 7 and 2D 3x3 grid, all four
`GradientScheme`s in the final stage. Gate confirmed: new script green,
`gradcheck_crk_native.py`/`gradcheck_crk_correction_native.py` both still
PASS, `operation_matrix.py --device cpu` unaffected (`OK=258, HIGH=0, ERR=0,
NAN=0` -- identical baseline), `pytest tests/` unaffected (119 passed/1
skipped -- this tier added one script, touched no production code). The one
genuine departure from every earlier tier's methodology: `computeCRKTermsWarp`
is pure PyTorch with no Warp call anywhere in it, so (unlike every
Warp-kernel-backed operator) `torch.autograd.functional.jvp`'s double-backward
trick is exact on it, not silently zero -- confirmed empirically before
relying on it -- so that stage's JVP is obtained directly from torch's own
autodiff rather than hand-derived, sidestepping `gradATerm4`'s four-index
tensor contraction by hand. See the "Tier 2.5 -- Result" subsection for the
full derivation and this decision's justification.

**Tier 2.4 (Renormalization correction) is done.** `scripts/spike_forward_mode_tier2_renorm.py`
assembles the covariance matrix's JVP from Tier 2.2's already-validated `kernelGradient` JVP
building block (`useCRK=False`, no `useGradientRenormalization` -- the simple case
`computeRenormalizationMatrices` itself uses when computing `C`) plus ordinary product-rule
calculus for the `apparentVolume`/`fij` factors that turn a sum of kernel gradients into a
covariance matrix, then the standard `d(C^-1) = -C^-1 (dC) C^-1` matrix-inverse-derivative
identity for the pseudo-inverse -- no new kernel math, exactly as the plan entry predicted.
Matches the production reverse-mode Jacobian to float64 round-off across every `SupportScheme`
in 1D (line of 7) and a representative subset in 2D (3x3 grid), plus an explicit check that the
low-neighbor-count identity fallback's tangent is exactly zero (both dim=1 and dim=2). Gate
confirmed: new script green (including a forward-value parity check between the assembled and
production covariance matrix, which also passed to float64 round-off), `gradcheck_renorm_native.py`
and `gradcheck_pinv_native.py` both still PASS, `operation_matrix.py --device cpu` unaffected
(`OK=258, HIGH=0, ERR=0, NAN=0` -- identical baseline), `pytest tests/` unaffected (119 passed/1
skipped -- this tier added one script, touched no production code). The plan's flagged risk
(`pinv2x2_warpBackend`'s eigenvalue-relative rank cutoff being a genuine JVP discontinuity) was
deliberately not probed -- test geometries (a regular line, a regular grid) are comfortably
well-conditioned and nowhere near that boundary, consistent with how earlier tiers treated the
periodic-wrap and `SupportScheme`-tie boundaries. NEXT: Tier 2.5 (CRK), the last and hardest tier.

**Tier 2.3 (Laplacian's Naive scheme) is done.** This tier's own entry (below) had
recommended deferring it and asking whether `LaplacianScheme.Naive` matters in
practice before spending the derivation effort, since Brookshaw (Tier 2.2) is the
scheme `wp_laplacian.py`'s own comments treat as the consistent estimator and Naive is
not on any performance-relevant path. Asked; the answer was to derive it anyway, for
methodological completeness of the adjoint SPH scheme -- Naive calls
`sphKernelLaplacian` directly (the genuine analytic second-derivative-of-r estimator),
so stating its Tier-2 JVP completes the adjoint of the SPH Laplacian in general, not
just the one scheme every current consumer happens to use; Naive is nonetheless a
real, wired-in scheme (not hypothetical), so this is a genuine adjoint, not a
derivation of dead code. `scripts/spike_forward_mode_tier2_laplacian_naive.py`
assembles it from two new kernel-level building blocks (`sphKernelLaplacianGradient_`,
`sphKernelLaplacianDkDh_`, `kernels/laplacian.py`) validated against `wp.Tape` in
`kernel_sanity_native.py`'s new Section K, then chain-ruled through
`sphKernelLaplacian`'s own `SupportScheme` dispatch and Naive's `q_ij` coefficient
(shown to be the same `B_ij` Tier 2.2 already found for Brookshaw). Matches the
production reverse-mode Jacobian to float64 round-off across every `GradientScheme` x
a representative `SupportScheme` subset, in 1D and 2D. Gate confirmed: Section K green
inside `kernel_sanity_native.py`, the new spike script green,
`gradcheck_laplacian_native.py` (which already exercises `LaplacianScheme.Naive`)
still PASSES, `operation_matrix.py --device cpu` unaffected (`OK=258, HIGH=0, ERR=0,
NAN=0` -- identical baseline), `pytest tests/` unaffected (119 passed/1 skipped). See
the "Tier 2.3 -- Result" subsection for the derivation and the one genuine structural
finding it surfaced (`sphKernelLaplacian`'s `SupportScheme` dispatch treats
`KernelMeanSymmetric` differently from every other Tier-2.x building block).

**Tier 2.2 (Gradient, Divergence, Curl, Laplacian's Brookshaw scheme) is done.**
`scripts/spike_forward_mode_tier2_gradient.py` assembles the JVP of `sphKernelGradient_ij`
itself (a new dispatch function built from Tier 2.0's `sphGradient_`/`sphKernelHessian_`/
`sphGradientDkDh_`, structurally the SupportScheme-branch twin of Tier 2.1's `dW_ij`
dispatch), then chain-rules it through each operator's field-value coefficient (shown to be
the *same* coefficient across all four operators) and, for Laplacian, its `n_ij`/`r_ij`
regularized-distance algebra. Matches the production reverse-mode Jacobian to float64
round-off across every GradientScheme x a representative subset of SupportScheme, in 1D and
2D, for Gradient/Divergence/Curl/Laplacian(Brookshaw). Gate confirmed (all four existing
`gradcheck_{gradient,divergence,curl,laplacian}_native.py` scripts, `operation_matrix.py
--device cpu` at `OK=258, HIGH=0, ERR=0, NAN=0` -- identical to Tier 2.1's baseline, and
`pytest tests/` at 119 passed/1 skipped, all unaffected). Laplacian's Dot/Default schemes
are explicitly deferred (see the "Tier 2.2 -- Result" subsection). See that subsection for
the derivation and two findings it surfaced.

**Tier 2.1 (Density, Interpolate) is done.** `scripts/spike_forward_mode_tier2_density.py` assembles
the two operators' position/support/mass[/density] JVPs from Tier 2.0's validated building blocks
and matches the production reverse-mode Jacobian to float64 round-off, across every `SupportScheme`
`sphKernel_ij` actually implements. Gate confirmed (`gradcheck_density_native.py`,
`gradcheck_interpolate_native.py`, `operation_matrix.py --device cpu`, `pytest tests/` all
unaffected). See the "Tier 2.1 -- Result" subsection below for the derivation and two pre-existing
`SupportScheme` code facts it surfaced.

**Tier 2.0 (kernel-level building blocks) is done.** `scripts/kernel_sanity_native.py` validated,
against `wp.Tape`, every derivative of the raw pairwise kernel that Tier 2 needs to assemble from:

| Function | Gives you | File | Status |
|---|---|---|---|
| `sphKernel_` | `W(x,h)` | `kernels/kernel.py` | validated (Section A/B/C) |
| `sphGradient_` | `dW/dx` (vector) | `kernels/gradient.py` | validated (Section D) |
| `sphKernelHessian_` | `d(dW/dx)/dx` (matrix) | `kernels/hessian.py` | validated (Section G) |
| `sphKernelDkDh_` | `dW/dh` (scalar) | `kernels/gradH.py` | validated (Section F) |
| `sphGradientDkDh_` | `d(dW/dx)/dh` (vector) | `kernels/gradH.py` | validated (Section J, new this session) |
| `sphKernelLaplacian_` | `d²W/dr² + (dim-1)/r · dW/dr` (scalar, Brookshaw-style) | `kernels/laplacian.py` | validated (Section G/H) |
| `d(∇²W)/dx`, `d(∇²W)/dh` | -- | -- | **does not exist** (Tier 2.3, see below) |

These five validated functions are the complete first-derivative-of-position/support toolkit for a
*single pairwise kernel evaluation*. Every JVP below is built by chain-ruling an operator's neighbor
sum through these, plus ordinary vector calculus for `x_ij = x_i - x_j` (or its periodic
minimum-image variant) and `h_ij = computePairwiseSupport(h_i, h_j, mode)`.

---

## Validation methodology (carries over from Tier 1, restated because it is easy to get wrong)

Two levels, and they need two different references:

1. **Kernel-level** (does the new adjoint function match a direct derivative of an existing,
   already-validated kernel function): a raw `wp.Tape` backward, exactly `kernel_sanity_native.py`
   Sections C/F/G/J's pattern. No torch involved. Use this for any new `@wp.func` this plan adds
   (Tier 2.3's `d(∇²W)/dx` and `d(∇²W)/dh`).
2. **Operator-level** (does the assembled JVP match the operator's true directional derivative):
   **do not use `torch.autograd.functional.jvp`** -- `warpier_core.md`'s Phase 6 audit found it
   returns a silently zero tangent here (its double-backward trick needs a differentiable backward,
   and reading gradients off a `wp.Tape` is not one). Use a reverse-mode Jacobian
   (`torch.autograd.functional.jacobian`, which only needs the first-order backward the gradcheck
   suite already proves correct) contracted with the tangent -- exact on the small cases these
   scripts use, and it is what `spike_forward_mode_tier1.py` already validated Tier 1 against. Every
   Tier-2.x operator-level script below should follow that same shape.

Both levels check `f(0) == 0` (linear, not affine) alongside the JVP identity itself, per Tier 1's
spike -- an operator that happened to have a position-independent affine offset would pass a
careless JVP check and fail this one.

---

## Tier 2.1 -- Density and Interpolate

**Scope.** `Density_i = Σ_j m_j W(x_ij, h_ij)`; `Interpolate_i = Σ_j V_j f_j W(x_ij, h_ij)`
(Tier-1-differentiable in `f_j`, frozen here). No CRK, no renormalization, no grad-h correction --
the least structurally complex operators in the codebase (`coreOperations/wp_density.py` calls
`sphKernel` directly, not `computeKernelGradientCRK`).

**Building blocks needed.** `sphKernel_` (have), `sphGradient_` (have, for the `x_i` tangent
contribution `dW/dx_i · dx_i`), `sphKernelDkDh_` (have, for `dW/dh_i` and `dW/dh_j` -- note these
enter asymmetrically once `SupportScheme` is anything other than `MeanSymmetric`, see Risks below).
**No new kernel math.**

**Deliverable.** `scripts/spike_forward_mode_tier2_density.py`: assemble
`tangent_Density_i = Σ_j [ dm_j · W_ij + m_j · (∇_x W_ij · (dx_i - dx_j) + dW/dh_ij-contributions) ]`,
validated against the operator-level reference above. This is the cheapest possible proof that
"Tier 2 assembly from validated kernel derivatives" actually works end to end -- do this first even
though it is also the least interesting operator, for the same reason Tier 1 started with the
smallest case.

**Gate.** New script green, plus full `pytest tests/` / `operation_matrix.py` unaffected (this tier
adds a script, touches no production code).

---

### Tier 2.1 -- Result (done, 2026-08-18)

**Deliverable shipped.** `scripts/spike_forward_mode_tier2_density.py`, green on first full run
after fixing tensor-index bugs during writing (see "Process notes" below). Gate confirmed:
`gradcheck_density_native.py` and `gradcheck_interpolate_native.py` both still PASS,
`operation_matrix.py --device cpu` (default 2D/float32 sweep) is unaffected (`OK=258, HIGH=0,
ERR=0, NAN=0`), and `pytest tests/` is unaffected (this tier added a script, touched no production
code, per the Gate above).

**Math derivation, written down precisely (the scope note above was a sketch; this is what the
code actually required).** Both operators reduce to a sum over neighbors of `(coefficient)_j *
W_ij`:

```
Density_i      = Σ_j  m_j                * W_ij
Interpolate_i  = Σ_j  f_j * V_j           * W_ij ,   V_j = m_j / ρ_j   (f_j frozen: Tier 1)
```

so the product rule gives the tangent directly once `dW_ij` is known:

```
dDensity_i     = Σ_j [ dm_j       * W_ij  +  m_j     * dW_ij ]
dInterpolate_i = Σ_j [ f_j * dV_j * W_ij  +  f_j*V_j * dW_ij ],   dV_j = dm_j/ρ_j - m_j·dρ_j/ρ_j²
```

`dW_ij` needed case analysis on `SupportScheme`, because `sphKernel_ij` (`kernels/kernel.py`)
itself branches on it before ever touching a kernel-derivative building block. Reading the actual
code (not just the plan's one-line sketch) turned up two branches, not one:

* **Single-`h` branch** (`Gather`, `Scatter`, `MeanSymmetric`, and the `PartialSymmetric`/default
  fallthrough -- see below): `W_ij = sphKernel_(x_ij, h_ij, k)`, so
  `dW_ij = ∇_x[W](x_ij,h_ij)·dx_ij + dW/dh(x_ij,h_ij)·dh_ij`, with `dx_ij = dx_i - dx_j` and
  `dh_ij` the JVP of `computePairwiseSupport` (util/support.py) itself -- new, but ordinary
  calculus, not kernel math:
  `Gather: dh_ij=dh_i`; `Scatter: dh_ij=dh_j`; `MeanSymmetric: dh_ij=(dh_i+dh_j)/2`;
  else (max): `dh_ij = dh_i if h_i>=h_j else dh_j` -- a genuine subgradient, discontinuous at
  `h_i==h_j`, same class of forward-branch-boundary hazard Tier 2.4's plan flags for `pinv`'s rank
  cutoff. Test data keeps supports away from exact ties.
* **Two-term-average branch** (`KernelMeanSymmetric`, `SuperSymmetric`):
  `W_ij = 0.5·(W(x_ij,h_i)+W(x_ij,h_j))`, so
  `dW_ij = 0.5·[(∇_x[W](x_ij,h_i)+∇_x[W](x_ij,h_j))·dx_ij + dW/dh(x_ij,h_i)·dh_i + dW/dh(x_ij,h_j)·dh_j]`.

**Two pre-existing production-code facts this derivation surfaced** (documented in the script's
own docstring too; neither is a bug to fix under this plan):

1. **`SuperSymmetric` is provably identical to `KernelMeanSymmetric` at the value level.** The enum
   docstring (`enumTypes.py`) describes `SuperSymmetric` as `0.5·(W(x_ij,h_i) - W(x_ji,h_j))`
   (mirroring the *gradient* formula, where `x_ji` vs. `x_ij` genuinely matters because `∇W` is odd
   in `x`). But `W` is isotropic -- depends only on `|x|` -- so `W(x_ji,h_j) ≡ W(x_ij,h_j)`, and the
   docstring's `-W(x_ji,h_j)` collapses to `+W(x_ij,h_j)` once evaluated. `sphKernel_ij`'s code,
   which uses an identical `+` branch for both schemes, is therefore correct, not a copy-paste bug
   -- the two schemes are mathematically forced to coincide at the value level, and will
   legitimately diverge only at the gradient level (Tier 2.2), where the odd/even distinction
   actually bites. The script checks this explicitly (`assert`s both the assembled and the
   reference JVP agree bit-for-bit between the two schemes) rather than assuming it.
2. **`PartialSymmetric` is unimplemented at the kernel-value/-gradient level.** Its enum comment
   promises `f_i·W(h_i) + f_j·W(h_j)` (a field-value-weighted scheme, PESPH-style), and it is the
   *only* other place it's referenced in the whole `src/` tree: the neighbor-search radius
   (`radiusSearch/compactHash/{grid,wp_collectNeighbors,wp_countNeighbors}.py`). `sphKernel_ij`
   and `computePairwiseSupport` have no branch for it at all -- it silently falls through to the
   `else` (`max(h_i,h_j)`) case, i.e. behaves like an unweighted "largest support wins" scheme, not
   the documented one. The JVP assembled here matches what the code actually does (tested
   explicitly, labeled `PartialSymmetric` in the script's output), not the aspirational docstring.
   Worth flagging to whoever owns `SupportScheme` next, but out of scope to fix here.

**Validation, two independent code paths (both exact analytic derivatives, no finite differences
anywhere):**

* *Assembled side* -- a hand-written per-pair Warp kernel (`_pair_jvp_1d`/`_pair_jvp_2d`) built
  only from already-validated `kernel_sanity_native.py` functions (`sphKernel_`, `sphGradient_`,
  `sphKernelDkDh_`) plus the new `_pairwiseSupportTangent` building block above, summed over a
  **dense all-pairs loop** rather than the real neighbor list. This is safe, not a shortcut: every
  building block is exactly zero for `q=|x|/h > 1` (`kernel_sanity_native.py` Section I), so a pair
  outside the true support radius contributes nothing to either `W_ij` or `dW_ij` regardless of
  whether it would have appeared in the production adjacency list -- letting the test avoid
  touching internal `AdjacencyList`/`CompactHashMap` structures entirely.
* *Reference side* -- `torch.autograd.functional.jacobian` on the actual production
  `warpOperation(Density/Interpolate)` call (the same reverse-mode path every
  `gradcheck_*_native.py` script already validates), contracted with the tangent -- Tier 1's
  reference pattern (`spike_forward_mode_tier1.py`), just differentiating w.r.t.
  positions/supports/masses[/densities] instead of field values.

Both sides agreed to `rel_err ~ 1e-16` (float64 round-off) across Gather/Scatter/MeanSymmetric/
KernelMeanSymmetric/SuperSymmetric/PartialSymmetric in 1D, a 2D grid subset, and Interpolate with
frozen field values -- no tuning or tolerance-loosening needed.

**Process notes (how this actually got built, for the next tier).** Test cases were deliberately
given *non-uniform* supports (`h_i` perturbed ±15% across particles) -- the first draft used
`_gradcheck_common.py`'s uniform-`h` `line_case`/`grid_case_2d` unmodified, which made
Gather/Scatter/MeanSymmetric/KernelMeanSymmetric numerically indistinguishable (`h_i=h_j`
everywhere collapses every branch above to the same number) and would have silently passed even
with the branch dispatch wired wrong. Perturbing supports is now the standing pattern for any
Tier-2.x script that touches `SupportScheme`. The dense-all-pairs-instead-of-real-adjacency
simplification (above) was a deliberate choice made *before* writing code, not a fallback found
after struggling with `AdjacencyList` internals -- worth deciding up front for 2.2 as well, since
it generalizes to any operator whose building blocks are all compactly supported (true of
everything through Tier 2.4; CRK's moment sums in Tier 2.5 may not have this property and should
not assume it without checking).

---

## Tier 2.2 -- Gradient, Divergence, Curl, and Laplacian's Brookshaw/Dot/Default schemes

**Scope.** All four of these route through `computeKernelGradientCRK` (`crk/kernel.py`), which with
`useCRK=False` reduces to `sphKernelGradient_ij` -- i.e. **structurally the same `∇W_ij` building
block for all four operators**, only the outer contraction with the field/direction differs
(`kernel.py`'s `q_ij`-weighted sum for Gradient/Divergence/Curl; `wp_laplacian.py`'s
`-2·q_ij·(∇W·n_ij)/r_ij` for Laplacian's Brookshaw/Dot/Default `laplacianMode`s -- confirmed by
reading `coreOperations/wp_laplacian.py`: only `LaplacianScheme.Naive` calls `sphKernelLaplacian`
directly; Brookshaw/Dot/Default all go through the same `kernelGradient` as the other three
operators). Four operators, one shared kernel-derivative dependency.

**Building blocks needed.** `sphGradient_` (have), `sphKernelHessian_` (have, for `d(∇W)/dx`),
`sphGradientDkDh_` (have, for `d(∇W)/dh`) -- **the complete Tier-2.0 toolkit already covers this
tier**, plus ordinary calculus for `n_ij = x_ij / r_ij` and `1/r_ij`'s own position tangent (ordinary
vector calculus, not kernel-specific -- `d(x_ij/r_ij)/dx_i = (I - n_ij⊗n_ij)/r_ij`, a standard
identity, worth a direct wp.Tape check against `n_ij`/`r_ij` computed the way `wp_laplacian.py`
already does, since that file's `eps`-regularized `r_ij + eps·h_ij` denominator is exactly the same
class of bug this session found twice in `laplacian.py`/`hessian.py` -- check it does not leak the
same way before reusing it here). **No new kernel math**, assembly + one regularization-hygiene
check inherited from the Section G/H bug fix.

**Deliverable.** `scripts/spike_forward_mode_tier2_gradient.py` covering Gradient's four
`GradientScheme`s and every `SupportScheme`; then Divergence/Curl/Laplacian(Brookshaw/Dot/Default)
as thin follow-ups once Gradient's is green, since they share the dependency.

**Gate.** Same as 2.1, plus re-run `gradcheck_gradient_native.py`/`gradcheck_divergence_native.py`/
`gradcheck_curl_native.py`/`gradcheck_laplacian_native.py` to confirm nothing about how this tier
probes the kernel accidentally exercises a reverse-mode regression.

---

### Tier 2.2 -- Result (done, 2026-08-18)

**Deliverable shipped.** `scripts/spike_forward_mode_tier2_gradient.py`, green after fixing two
bugs found during writing (see "Process notes" below). Gate confirmed:
`gradcheck_gradient_native.py`/`gradcheck_divergence_native.py`/`gradcheck_curl_native.py`/
`gradcheck_laplacian_native.py` all still PASS, `operation_matrix.py --device cpu` is unaffected
(`OK=258, HIGH=0, ERR=0, NAN=0` -- identical to Tier 2.1's baseline), and `pytest tests/` is
unaffected (119 passed, 1 skipped -- this tier added a script, touched no production code).

**Math derivation.** `computeKernelGradientCRK` with `useCRK=False` reduces to
`kernelGradient = sphKernelGradient_ij(x_ij, hi, hj, ...)` (`kernels/gradient.py`) for all four
operators -- confirmed by reading `coreOperations/wp_{gradient,divergence,curl,laplacian}.py`, all
four call it identically. So there is exactly ONE new kernel-level building block this tier needs:
the JVP of `sphKernelGradient_ij` itself, `d(kernelGradient)/d{x,h}`. Everything downstream is
ordinary vector calculus, done directly in torch on the dense `(n,n)` pair grid rather than as new
`@wp.func`s -- a scope simplification the plan's entry didn't spell out but that held up cleanly.

*1. `kernelGradient`'s JVP* (`_kernelGradientJVP` in the script) mirrors `sphKernelGradient_ij`'s
own three-way `SupportScheme` dispatch, assembled from `sphGradient_`, `sphKernelHessian_`
(`d(gradW)/dx`, already validated as Tier 2.0's Section G/H deliverable and confirmed to be exactly
the Jacobian `sphGradient_` needs here -- Tier 2.0's table had flagged it "unused in production";
this tier is its first real consumer), and `sphGradientDkDh_` (`d(gradW)/dh`, Section J):

```
KernelMeanSymmetric/SuperSymmetric (provably identical -- see finding 1 below):
  G_ij  = 0.5*(sphGradient_(x,hi) + sphGradient_(x,hj))
  dG_ij = 0.5*[ sphKernelHessian_(x,hi)@dx + sphKernelHessian_(x,hj)@dx
                + sphGradientDkDh_(x,hi)*dhi + sphGradientDkDh_(x,hj)*dhj ]
else (Gather/Scatter/MeanSymmetric/max-fallback):
  h_ij = computePairwiseSupport(hi,hj,mode), dh_ij = Tier 2.1's _pairwiseSupportTangent(...)
  G_ij  = sphGradient_(x,h_ij)
  dG_ij = sphKernelHessian_(x,h_ij) @ dx + sphGradientDkDh_(x,h_ij)*dh_ij
```

*2. Field-value coefficient* (ordinary calculus, reused verbatim across all four operators --
see finding 2): every `GradientScheme` reduces to `coeff_ij = fi*A_ij + fj*B_ij` (fi, fj frozen --
Tier 1 territory), with `Vj = mass_j/density_j`:

```
Naive:      A=0,                B=Vj
Difference: A=-Vj,               B=Vj
Summation:  A=Vj,                B=Vj
Symmetric:  A=mass_j/density_i,  B=mass_j*density_i/density_j^2
```

(A, B differentiated by ordinary product/quotient rule through `mass_j`/`density_i`/`density_j`;
fi/fj are frozen so contribute no term of their own.) Gradient combines `coeff_ij` with `G_ij` by
scalar multiplication; Divergence via `dot(coeff_ij, G_ij)` (`dotMode=False`); Curl via the 2D
scalar cross `G_ij.x*coeff_ij.y - G_ij.y*coeff_ij.x` (`curlProduct`'s exact 2D formula,
`math/wp_cross.py`) -- all three bilinear in `(coeff_ij, G_ij)`, so `d(coeff*G) = dcoeff*G +
coeff*dG` by the ordinary product rule, no per-operator re-derivation needed.

*3. Laplacian(Brookshaw)'s regularized-distance chain* (`D_ij = r_ij + eps*h_ij`, `n_ij =
x_ij/D_ij`, `L_ij = -2*q_ij*dot(G_ij,n_ij)/D_ij`, `eps=1e-8` matching `wp_laplacian.py`'s literal
constant): `dr_ij = dot(x_ij,dx_ij)/r_ij` off the diagonal; `dh_ij` reuses Tier 2.1's
`_pairwiseSupportTangent` (`computePairwiseSupport`'s own dispatch is unchanged by this tier);
`dD_ij = dr_ij + eps*dh_ij`; `dn_ij = (dx_ij - n_ij*dD_ij)/D_ij` (the standard `d(x/D)/dx`
identity); `dL_ij` follows by the ordinary product/quotient rule through `dot(G,n)/D`.

**Two findings this derivation surfaced (documented in the script's docstring too; neither is a
bug to fix under this plan):**

1. **`SuperSymmetric` is provably identical to `KernelMeanSymmetric` at the *gradient* level too,
   not just the value level Tier 2.1 found.** `sphKernelGradient_ij`'s `SuperSymmetric` branch is
   literally `(sphGradient_(x,hi) - sphGradient_(-x,hj))/2`. `sphGradient_` is odd in its position
   argument (direction = `normalize(x)`, magnitude depends only on `|x|`), so
   `sphGradient_(-x,hj) = -sphGradient_(x,hj)`, collapsing the branch to
   `(sphGradient_(x,hi)+sphGradient_(x,hj))/2` -- bit-for-bit `KernelMeanSymmetric`. The plan's own
   entry had predicted the opposite ("will legitimately diverge only at the gradient level, where
   the odd/even distinction actually bites") -- reading the actual code before deriving showed the
   oddness is exactly what makes the two double-negatives cancel, not what makes them differ. The
   same argument carries one derivative further: `sphKernelHessian_` (the Jacobian of an odd
   function) is *even* in `x`, and `sphGradientDkDh_` is *odd* (both confirmed from their closed
   forms, not assumed), so the identity holds at the JVP level too -- checked explicitly (bit-for-
   bit `assert`, not just claimed) in the script's own output.
2. **The exact same field-value coefficient (`B_ij` above) serves as both Gradient's `B` term and
   Laplacian's `q_ij` weight, for every `GradientScheme`.** `wp_laplacian.py`'s `q_ij` for
   Naive/Difference/Summation is `(fj-fi)*apparentVolume` (`apparentVolume == Vj == B_Naive`) and
   for Symmetric is `(fj-fi)*mass_j*density_i/density_j^2` (`== B_Symmetric` exactly). Not
   rederived independently -- the script computes `B`/`dB` once (in `_gradient_weights`) and reuses
   it for both Gradient/Divergence/Curl's coefficient and Laplacian's `q_ij`/`dq_ij`.

**Scope note: Laplacian's Dot/Default schemes are NOT covered by this deliverable, despite the
plan entry's title.** `computeLaplacianDot2`/`computeDotLaplacian` (`math/wp_laplaciandot.py`) do
per-spatial-component block indexing into the field array (`q_ij[block*dim+k]`) that Brookshaw's
plain `dot(kernelGradient, n_ij)` doesn't need -- a genuinely separate (if likely mechanical)
JVP-assembly exercise, not just a formula swap the way Divergence/Curl were thin follow-ups on
Gradient. Deferred rather than attempted under this tier's time budget; Brookshaw is what
`wp_laplacian.py`'s own comments treat as the consistent estimator and is the scheme Tier 2.3
already assumes Tier 2.2 covers, so this does not block the suggested order below. Worth its own
small follow-up (`Tier 2.2b`, informally) before Tier 2.5 needs a Laplacian JVP under CRK.

**Validation, same two independent code paths as Tier 2.1 (both exact analytic derivatives, no
finite differences):** a hand-written dense all-pairs per-pair kernel built only from already-
validated functions, vs. `torch.autograd.functional.jacobian` on the actual production
`warpOperation` call contracted with the tangent. Agreed to `rel_err ~1e-15` (float64 round-off)
across all four `GradientScheme`s and a representative `SupportScheme` subset (Gather/Scatter/
MeanSymmetric/KernelMeanSymmetric/SuperSymmetric/PartialSymmetric for Gradient in 1D, a smaller
subset for Divergence/Curl/Laplacian and for Gradient's own 2D case -- mirroring Tier 2.1's
"exhaustive on the cheapest operator, representative subset elsewhere" pattern) -- no tuning or
tolerance-loosening needed once the two bugs below were fixed.

**Process notes -- two real bugs, both caught by the operator-level comparison exactly as the
methodology promises (neither would have been caught by a value-only check):**

1. **`SupportScheme` int-vs-enum comparison silently picking the wrong branch.** Every
   `assembled_*_jvp` function receives `mode` as the raw `SupportScheme.value` int (needed for the
   warp kernel launch), but Laplacian's new `_h_ij_and_tangent` torch helper compared it against
   `SupportScheme.Gather` etc. directly -- an int-vs-enum comparison that is always `False` in
   Python, so it silently fell through to the `else` (max) branch for every mode. This produced a
   *small* (`~1e-9` relative, not `O(1)`) error rather than an obviously-wrong one, because `h_ij`
   only enters Laplacian's formula through the tiny `eps=1e-8` regularization term -- a reminder
   that a silently-wrong branch dispatch does not always announce itself with a large error, and
   `rel_err ~1e-9` (nine orders above float64 round-off, but still "small-looking") is exactly the
   kind of near-miss worth treating as a real bug, not a tolerance to loosen. Fixed by coercing
   `mode = SupportScheme(mode)` at the top of the helper.
2. **A shape mismatch that produced a spuriously large "error" rather than no error at all** --
   the opposite failure mode from #1, useful to record for the same reason. Curl's production
   output shape is `(n,1)` (`wp_curl.py` forces `outputShape=[1]` for a 2D vector-field input,
   where Gradient/Divergence's outputs are shape `(n,dim)`/`(n,)`), but the assembled side returned
   shape `(n,)`. Subtracting them in `check()` silently broadcast to an `(n,n)` matrix instead of
   raising, comparing unrelated elements against each other and reporting `rel_err ~1.4-1.8` for
   every case -- large enough to look like a real formula bug (and initially mistaken for one; the
   actual JVP math checked out cleanly on separate inspection) but not large enough (`>1e9`-style)
   to obviously be a broadcast artifact either, since the underlying per-particle values are `O(1)`.
   Fixed by having `check()` flatten both sides and assert matching element counts before
   comparing -- now a genuine shape mismatch raises immediately instead of comparing the wrong
   elements. Worth carrying into any future Tier-2.x script that touches an operator whose output
   rank isn't the obvious `(n,)`/`(n,dim)` (Curl's `[1]`-forcing is the only one so far, but
   Laplacian-Vector or a future CRK-corrected shape could hit the same class of thing).

---

## Tier 2.3 -- Laplacian's `Naive` scheme (genuinely new kernel math)

**Scope.** `LaplacianScheme.Naive` calls `sphKernelLaplacian` directly -- the actual analytic
second-derivative-of-r kernel, not the gradient-based estimator Tier 2.2 covers. Its Tier-2 JVP needs
`d(∇²W)/dx` (vector) and `d(∇²W)/dh` (scalar), neither of which exists yet.

**Building blocks needed -- new.** Following the exact pattern that produced `sphGradientDkDh_`
cleanly (differentiate the closed form, verify against `wp.Tape`, don't trust the derivation alone):

* `d(∇²W)/dx`: `sphKernelLaplacian_`'s value is `s·k2 + t·k1` where `k1, k2` are radial derivatives
  (`dW/dr`, `d²W/dr²`) and `s, t` are algebraic functions of `x`/`r`. Differentiating this w.r.t. `x`
  needs `dk1/dx = sphGradient_` (have), `dk2/dx` (does not exist -- this is `d(sphKernelHessian_'s
  radial part)`, effectively **one more application of the same technique**, needing `eval_d3kdq3`,
  already validated in Section C), plus the `ds/dx`, `dt/dx` algebraic terms. This is the most
  involved single derivation in this plan -- budget for it separately, and expect (based on this
  session's track record) that the first written formula will not match `wp.Tape` on the first try.
* `d(∇²W)/dh`: same closed form, differentiate w.r.t. `h` instead -- structurally similar to how
  `sphGradientDkDh_` was derived from `sphGradient_`, one derivative order up.

**Note on priority.** `wp_laplacian.py`'s own long comment states `LaplacianScheme.Naive` -- unlike
Difference -- inherits an "uncancelled O(1/h²) residual" for non-constant-vs-constant field
consistency at the *forward* level already; whether that make it a low-priority scheme to Tier-2-ify
first is worth a direct question before investing here, rather than assumed. **Recommend deferring
this tier and asking whether `LaplacianScheme.Naive` is used anywhere production-relevant before
spending the (nontrivial) derivation effort** -- Brookshaw is what `coreOperations/wp_laplacian.py`
treats as the consistent estimator, and Tier 2.2 already covers it.

---

### Tier 2.3 -- Result (done, 2026-08-18)

**Asked, and answered.** The entry above recommended asking before deriving this --
asked, and the answer was to do it regardless of production relevance: Naive's
"uncancelled O(1/h²) residual" makes it a poor *forward*-accuracy choice, but that is
an orthogonal question from whether its adjoint is worth stating. It is real, wired-in
kernel math (`wp_laplacian.py`'s Naive branch, exercised by
`gradcheck_laplacian_native.py`), and completing its Tier-2 JVP is part of writing
down the adjoint of the SPH Laplacian in general -- the methodological foundation this
whole plan is building, not just the fast path every current consumer happens to take.

**Deliverable shipped.** Two new kernel-level building blocks in
`kernels/laplacian.py` (`sphKernelLaplacianGradient_` = `d(sphKernelLaplacian_)/dx`,
`sphKernelLaplacianDkDh_` = `d(sphKernelLaplacian_)/dh`), a new Section K in
`kernel_sanity_native.py` validating both against `wp.Tape`, and
`scripts/spike_forward_mode_tier2_laplacian_naive.py` assembling and validating the
operator-level JVP. All green on first full run (see "Process notes" below for the one
subtlety caught along the way -- not a bug, a deliberate design choice verified before
being relied on). Gate confirmed: `kernel_sanity_native.py` Section K PASS across every
`SPH_KERNELS` family x dim 1/2/3 x 4 directions, the new spike script PASS across every
`GradientScheme` x a representative `SupportScheme` subset in 1D/2D,
`gradcheck_laplacian_native.py` unaffected (still PASSES, including its existing
`LaplacianScheme.Naive` cases), `operation_matrix.py --device cpu` unaffected
(`OK=258, HIGH=0, ERR=0, NAN=0`), `pytest tests/` unaffected (119 passed/1 skipped --
this tier added two `@wp.func`s and two scripts, touched no other production code).

**Math derivation.** `sphKernelLaplacian_(x,h) = s*k2 + t*k1`, where
`k1 = dW/dr = eval_dkdq(q)*C_d/h^(dim+1)` (`== sphGradient_`'s magnitude),
`k2 = d2W/dr2 = eval_d2kdq2(q)*C_d/h^(dim+2)` (`== sphKernelHessian_`'s radial factor),
`s = dot(x,x)/r_eps^2`, `t = -dot(x,x)/r_eps^3 + dim/r_eps`, `r_eps = r + eps*h`,
`eps = get_epsilon(r)` -- a dtype-only constant (`1e-15` at float64), *not* itself a
function of `r` or `h`, confirmed by reading `math/wp_eps.py` before assuming it, so
`d(eps)/dx = d(eps)/dh = 0` throughout and `r_eps`'s only dependence on either is
through `r` itself. Differentiating by the product rule across `s`, `t`, `k1`, `k2`
needs one new scalar building block, `k3 = d3W/dr3 = eval_d3kdq3(q)*C_d/h^(dim+3)` --
exactly the "one more application of the same technique" the plan entry above
anticipated, using `eval_d3kdq3` (already validated, Section C) for the first time in
this codebase. The key identity, confirmed from the closed forms before touching
`wp.Tape` (mirroring how `sphGradientDkDh_`'s docstring derives its own mixed
partial): differentiating the `k1,k2` ladder shifts it up one rung --
`dk1/dx = k2*direction`, `dk2/dx = k3*direction`, `dk1/dh = -q*k2 - (dim+1)*k1/h`
(`== sphGradientDkDh_`'s own scalar magnitude, reused as a cross-check),
`dk2/dh = -q*k3 - (dim+2)*k2/h` (new, one rung up). Assembled:

```
d(Laplacian)/dx = k2*ds/dx + k1*dt/dx + (s*k3 + t*k2)*direction
  ds/dx = 2*r*eps*h/r_eps^3 * direction                       -- O(eps), vanishes as r_eps->r
  dt/dx = direction*[(3*dot(x,x) - 2*r*r_eps)/r_eps^4 - dim/r_eps^2]

d(Laplacian)/dh = k2*ds/dh + k1*dt/dh - q*(s*k3+t*k2) - (s*(dim+2)*k2 + t*(dim+1)*k1)/h
  ds/dh = -2*eps*dot(x,x)/r_eps^3
  dt/dh = 3*eps*dot(x,x)/r_eps^4 - dim*eps/r_eps^2
```

Cross-checked, before ever running `wp.Tape`, against the `eps->0` textbook closed form
`Laplacian = k2 + (dim-1)/r*k1` differentiated directly by ordinary calculus (`d/dr[...]
= k3 - (dim-1)/r^2*k1 + (dim-1)/r*k2`, `d/dh[...] `expands the same way) -- both match
the bracketed terms above term-for-term once `t -> (dim-1)/r`, `s -> 1`. `r=0`
(self-pair) needs no special-casing in either function: `direction` comes from
`vectorNormalize_warp`, which is exactly zero (not NaN) for a zero-length input, so
every term carrying an explicit `direction` factor vanishes there, and `dot(x,x)=0`
kills the rest before `r_eps=eps*h` in the denominator can blow anything up. Both
functions do replicate `sphKernelLaplacian_`'s own `q<eps` cutoff (the value is
identically zero on that open region, so its derivative is exactly zero there too, not
merely untested) but make no attempt to differentiate across the discrete jump at
`q==eps` itself -- a measure-zero point no validation sample lands on, the same
convention every other Section in `kernel_sanity_native.py` already follows.

**Operator-level assembly.** `wp_laplacian.py`'s Naive branch is
`laplacian_contribution = q_ij * sphKernelLaplacian(...)`. `sphKernelLaplacian`
(the `xi/xj/hi/hj` wrapper) has its own two-branch `SupportScheme` dispatch:

```
SuperSymmetric (explicit):
  L_ij  = 0.5*(sphKernelLaplacian_(xij,hi) + sphKernelLaplacian_(xij,hj))
  dL_ij = 0.5*[ dot(LG(xij,hi),dxij) + LDkDh(xij,hi)*dhi
                + dot(LG(xij,hj),dxij) + LDkDh(xij,hj)*dhj ]
else (Gather/Scatter/MeanSymmetric/KernelMeanSymmetric/max-fallback):
  h_ij = computePairwiseSupport(hi,hj,mode), dh_ij = Tier 2.1's _pairwiseSupportTangent(...)
  L_ij  = sphKernelLaplacian_(xij, h_ij)
  dL_ij = dot(LG(xij,h_ij),dxij) + LDkDh(xij,h_ij)*dh_ij
```

`q_ij` is exactly Tier 2.2's `B_ij` again (`wp_laplacian.py`'s `q_ij` depends only on
`gradientMode`, never on `laplacianMode`) -- not re-derived, reused verbatim, the same
finding as Tier 2.2 just re-confirmed under a different `laplacianMode`. So
`L = Sum_j q_ij*L_ij`, `dL = Sum_j (dq_ij*L_ij + q_ij*dL_ij)`.

**One genuine structural finding, not shared with Tier 2.2's Brookshaw scheme.**
`sphKernelLaplacian`'s `SupportScheme` dispatch has only *two* branches:
`SuperSymmetric` explicit, everything else (including `KernelMeanSymmetric`) falling
through to `computePairwiseSupport`'s own dispatch, which has no explicit
`KernelMeanSymmetric` branch either and so silently lands on the max-fallback `h_ij` --
a single evaluation, not a two-term average. This is *different* from `sphKernel_ij`
(Tier 2.1) and `sphKernelGradient_ij` (Tier 2.2), both of which give
`KernelMeanSymmetric` its own explicit two-term-average branch, structurally identical
to `SuperSymmetric`'s -- which is exactly why Tier 2.1/2.2 found the two schemes
provably identical at the value and gradient level. For the Naive Laplacian they are
*not* identical: `KernelMeanSymmetric` gets the single max-h evaluation,
`SuperSymmetric` gets the genuine two-term average. Checked explicitly in the script
(the mirror image of Tier 2.2's "assert identical" check: this one asserts the two
schemes' assembled JVPs genuinely *differ*, confirmed `PASS`) rather than assumed from
reading the dispatch code alone. Not a bug -- `sphKernelLaplacian` simply never had a
`KernelMeanSymmetric` branch added when `sphKernel_ij`/`sphKernelGradient_ij` got
theirs -- but worth flagging to whoever owns `SupportScheme` dispatch consistency
next, the same spirit as Tier 2.1's `PartialSymmetric` finding.

**Process note.** No bugs this time (unlike Tier 2.1's tensor-index slips or Tier 2.2's
int-vs-enum/shape-mismatch pair) -- the `eps->0` cross-check against the textbook
closed form was done on paper before writing any code and matched term-for-term on
the first attempt, and both `wp.Tape` (Section K) and the operator-level script
matched on their first full run too. Worth recording as a data point for the general
methodology rather than a specific catch: deriving the algebraic cross-check *before*
touching `wp.Tape`, the discipline every section of `kernel_sanity_native.py` already
follows, continued to pay off even on the most involved single derivation in this
plan (the plan's own Tier 2.3 entry had flagged it as such) -- it is cheap relative to
debugging a `wp.Tape` mismatch after the fact, and here it meant the `wp.Tape` runs
were confirmations rather than debugging sessions.

---

## Tier 2.4 -- Renormalization correction

**Scope.** `renorm.py` builds a moment matrix `C` (a `Σ V_j · ∇W_ij ⊗ x_ij`-shaped sum, confirmed by
reading the file's structure though not transcribed in full here) and inverts it via `pinv_warp`
(dispatches to `pinv1x1`/`pinv2x2_warpBackend`, already independently gradchecked in
`gradcheck_pinv_native.py`) to get the renormalization matrix `L`, with a low-neighbor-count identity
fallback.

**Building blocks needed.** Tier 2.2's `d(∇W)/dx`, `d(∇W)/dh` give `dC/dx`, `dC/dh` (a sum over the
same neighbor loop, just accumulating a matrix instead of a vector). The new piece is **not kernel
math but matrix calculus**: for the invertible branch, `d(C⁻¹) = -C⁻¹ (dC) C⁻¹` is a standard identity
(no new derivation, just careful contraction order in warp, matching the same first/second-axis care
`correctGradientCRK`'s docstring comment already had to get right once for `gradBi`). The low-neighbor
identity-fallback branch has an exactly-zero tangent by construction (constant output) -- worth an
explicit test the way `gradcheck_pinv_native.py`'s Test 3 checked the fallback's *reverse*-mode
gradient is exactly zero, not just small.

**Deliverable.** `scripts/spike_forward_mode_tier2_renorm.py`, gated the same way, plus confirmation
against `gradcheck_renorm_native.py` and `gradcheck_pinv_native.py` that nothing regresses.

**Risk.** `pinv2x2_warpBackend` branches on an eigenvalue-based rank/conditioning cutoff
(`gradcheck_pinv_native.py`'s docstring, `warpier_core.md`'s CRK notes) -- a position tangent that
pushes a case across that cutoff mid-JVP is a discontinuity the JVP formula cannot represent (same
class of issue as `warpier_fields.md` §3.3's non-contiguous-tensor caching hazard: a boundary the
*forward* pass already treats specially). Flag failing cases near that boundary rather than silently
producing a wrong tangent there.

---

### Tier 2.4 -- Result (done, 2026-08-18)

**Deliverable shipped.** `scripts/spike_forward_mode_tier2_renorm.py`, green on first full run
(no bugs this time, unlike Tier 2.1's tensor-index slips or Tier 2.2's int-vs-enum/shape-mismatch
pair -- reusing Tier 2.2's `_kernelGradientJVP` verbatim, per its own already-validated dispatch,
left much less new surface area for a mistake to hide in than either of those tiers had). Gate
confirmed: `gradcheck_renorm_native.py`/`gradcheck_pinv_native.py` both still PASS,
`operation_matrix.py --device cpu` unaffected (`OK=258, HIGH=0, ERR=0, NAN=0` -- identical to
Tier 2.3's baseline), `pytest tests/` unaffected (119 passed/1 skipped -- this tier added a
script, touched no production code).

**Math derivation.** `wp_covariance.py`'s per-neighbor contribution is
`out += wp.outer(fij * apparentVolume, kernelGradient)`, where
`fij = -computeDistanceVec(xi, xj) = -(x_i - x_j)`. Writing `y_ij = -x_ij = x_j - x_i` and
`Vj = apparentVolume = mass_j/density_j` (the `useVolume=False` case this script covers -- no
`queryVolumes`/`referenceVolumes`):

```
C_i  = Sum_j Vj * outer(y_ij, G_ij)
dC_i = Sum_j [ dVj * outer(y_ij, G_ij) + Vj * outer(dy_ij, G_ij) + Vj * outer(y_ij, dG_ij) ]
```

`G_ij, dG_ij` are exactly Tier 2.2's `kernelGradient`/its JVP -- confirmed by reading
`computeKernelGradientCRK`, which reduces to plain `sphKernelGradient_ij` when `useCRK=False`,
and by `computeRenormalizationMatrices_`, which is only ever called here with
`crkState=None, renormalizationState=None` (the base case; the more general "renormalize using an
existing CRK/renorm state" path a caller *could* wire up by passing those through is out of scope
for this tier, same as Tier 2.5 already claims CRK). `Vj, dVj` are literally Tier 2.2's
`GradientScheme.Naive` `B_ij, dB_ij` reused under a new name, not re-derived.

The low-neighbor-count identity fallback (`renorm.py`'s `lowNbrMask = num_nbrs < dim+2`) forces
`C` to a literal constant (`I`) on the masked branch, so `dC = 0` there by construction --
`num_nbrs` itself is consumed directly from production's own
`covarianceReturnNumNeighbors=True` output rather than re-derived, the same "non-differentiable
discrete branch decision, read off production rather than rebuilt" treatment every earlier tier
gave `SupportScheme` dispatch and `h_i>=h_j` ties. Checked explicitly with an isolated single
particle (`num_nbrs=1 < dim+2` for both dim=1 and dim=2): both the assembled and reference JVPs
are exactly/near-exactly zero, and `L` is exactly the identity, mirroring
`gradcheck_pinv_native.py`'s Test 3 for the analogous reverse-mode case.

For the pseudo-inverse itself, `d(C^-1) = -C^-1 (dC) C^-1` is the standard matrix-inverse
identity -- true whenever `L = C^-1` exactly (both eigenvalues on the "kept" side of
`pinv2x2_warp`'s `rcond`-relative cutoff, or `pinv1x1`'s `m[0,0] > 1e-10` branch). `L` itself is
taken directly from production's own `computeRenormalizationMatrices` output (already
independently gradchecked by `gradcheck_pinv_native.py`) rather than re-derived -- consistent
with how `num_nbrs` and every earlier tier's frozen densities were handled. Implemented as a
plain batched `-L @ dC @ L` in torch (`(n,dim,dim)` tensors); no warp-level contraction-order
subtlety actually bit here, since both `C` and `L` are symmetric by construction (a sum of
`x_ij (x) x_ij`-shaped terms, per `pinv2x2_warp`'s own comment) so left/right multiplication
order doesn't matter for these test cases -- the plan entry's caution about "the same
first/second-axis care `correctGradientCRK` needed for `gradBi`" turned out not to be a live
issue for the symmetric case this tier covers, though it may resurface once Tier 2.5's CRK
moment matrices (not guaranteed symmetric) reuse this identity.

**Risk from the plan entry, deliberately not tested.** `pinv2x2_warpBackend`'s eigenvalue-relative
rank cutoff (`rcond=1e-6`) is a genuine JVP discontinuity, the same class of hazard as a
`SupportScheme` tie or the periodic-wrap boundary. This script's test geometries (a regular 1D
line, a regular 2D grid, both with only mildly perturbed supports) are comfortably well-conditioned
and never approach that cutoff -- consistent with the "flag failing cases near that boundary
rather than silently producing a wrong tangent" guidance: there is nothing to flag if the
boundary is never approached, and manufacturing a near-singular covariance matrix on purpose to
probe it was judged out of scope for this tier (the plan asked for the identity to be validated,
not for the discontinuity itself to be characterized).

**Validation, same two independent code paths as every earlier tier (both exact analytic
derivatives, no finite differences):** a hand-written dense all-pairs assembly built only from
Tier 2.2's already-validated `kernelGradient` JVP, vs. `torch.autograd.functional.jacobian` on
the actual production `computeRenormalizationMatrices` call contracted with the tangent. Also
includes a forward-value parity check (assembled `C` vs. production `C`, unmasked) as a cheap
extra sanity gate, the same spirit as Tier 1's `f(0)==0` check, since a sign error in `fij`'s
`-x_ij` convention would have shown up there before ever reaching the JVP comparison. Agreed to
`rel_err ~1e-14` to `1e-15` (float64 round-off) across Gather/Scatter/MeanSymmetric/
KernelMeanSymmetric/SuperSymmetric/PartialSymmetric in 1D and Gather/MeanSymmetric/
KernelMeanSymmetric in 2D -- no tuning or tolerance-loosening needed.

---

## Tier 2.5 -- CRK correction (hardest tier)

**Scope.** `crk/kernel.py`'s `correctGradientCRK`/`computeKernelCRK` consume `A_i, B_i, gradA_i,
gradB_i` -- the CRK correction terms, themselves built from moment sums (`m0`, `m1`, `m2`) over the
same neighbor loop in `crk_terms.py`/`crk_moments.py`, and (per `crk_moments.py`, per
`docs`/memory: `project_crk_dualpath_and_latent_bugs`) going through a pseudo-inverse-adjacent
construction of their own. Propagating a position/support tangent through Gradient/Divergence/Curl
*with CRK enabled* needs `dA_i/dx`, `dB_i/dx`, `d(gradA_i)/dx`, `d(gradB_i)/dx` (and the `/dh`
equivalents) -- each is its own multi-stage chain rule through the moment sums, built on Tier 2.2's
`dW/dx`, `d(∇W)/dx` and Tier 2.4's matrix-inverse-derivative identity.

**Building blocks needed.** Everything from Tiers 2.2 and 2.4, plus working through
`crk_moments.py`'s specific moment-matrix construction (not characterized in detail in this plan --
that characterization is this tier's first concrete step, not something to derive blind). Expect this
tier to surface the same class of dynamic-loop/postloop-nonlinear-op issues that
`project_crk_dualpath_and_latent_bugs` already found once in CRK's *reverse*-mode path (a NaN-grad
bug from postprocessing a loop-accumulated value inside the same `@wp.func`) -- CRK has a track
record of exactly this failure mode, so budget verification time accordingly, and consider adapting
`scripts/debug_crk_backward.py`'s per-quantity NaN-tracing approach for the forward-mode case too.

**Deliverable.** `scripts/spike_forward_mode_tier2_crk.py`, gated against
`gradcheck_crk_native.py`/`gradcheck_crk_correction_native.py`.

**Recommend doing this tier last**, after 2.1/2.2/2.4 are solid -- it is a strict superset of their
building blocks plus CRK's own moment-sum chain rule, and CRK is the module this codebase's own
history flags as the most adjoint-fragile.

---

### Tier 2.5 -- Result (done, 2026-08-18)

**Deliverable shipped.** `scripts/spike_forward_mode_tier2_crk.py`, green after one shape bug
found during writing (see "Process notes" below). Gate confirmed:
`gradcheck_crk_native.py`/`gradcheck_crk_correction_native.py` both still PASS,
`operation_matrix.py --device cpu` unaffected (`OK=258, HIGH=0, ERR=0, NAN=0` -- identical to
Tier 2.4's baseline), `pytest tests/` unaffected (119 passed/1 skipped -- this tier added a
script, touched no production code).

**Math derivation, in the four stages the plan entry's "first concrete step" note called for
(characterizing `crk_moments.py`'s construction before deriving blind).** CRK's own pipeline
(`crk/crk_wrapper.py`'s `computeCRKFactors`) chains four functions, each gated independently
against its own production counterpart's reverse-mode Jacobian before being composed into the
next:

1. **`crk_volume.py`'s apparent volume, `V_i = 1/(Sum_j W_ij)`, always called with
   `supportMode=SupportScheme.Gather`** (hardcoded in `crk_wrapper.py`, not a caller-supplied
   mode) -- i.e. exactly Tier 2.1's single-`h` branch with `h_ij=h_i` always (Gather), plus the
   reciprocal: `dV_i = -dwsum_i/wsum_i^2`, `dwsum_i = Sum_j[gradW_ij.dx_ij + dW/dh(x_ij,h_i)*dh_i]`.
   No new kernel math -- Tier 2.0's `sphKernel_`/`sphGradient_`/`sphKernelDkDh_`, unchanged.

2. **`crk_moments.py`'s six accumulators, always called with `supportMode=SupportScheme.Scatter`**
   (also hardcoded, and *different* from stage 1's Gather -- easy to miss reading the plan alone,
   only visible by reading `crk_wrapper.py`'s two `OperationProperties` literally). `V_j` here is
   stage 1's `apparentVolume[j]` -- a *different* quantity from Tier 2.2/2.4's `Vj=mass_j/density_j`
   despite sharing the letter (CRK's `V` never touches mass or density at all; production code
   reuses the symbol for two structurally-similar but numerically-unrelated things). Given stage
   1's `V,dV`, the six accumulators' tangents follow by ordinary product rule through Tier 2.1's
   kernel-value JVP (`w_ij,dw_ij`, Scatter branch) and Tier 2.2's `kernelGradient` JVP
   (`gradw_ij,dgradw_ij`, Scatter branch) -- no new kernel math, "just" bookkeeping through six
   accumulators instead of one. `dm_2dgamma`'s flattened index convention
   (`gamma*dim*dim+alpha*dim+beta`, confirmed by reading `crk_moments.py`'s literal indexing
   expression, not assumed) had to be matched exactly in the assembled tensor's axis order for the
   result to feed correctly into stage 3.

3. **`crk_terms.py`'s `computeCRKTermsWarp` (moments -> A/B/gradA/gradB) uses
   `torch.autograd.functional.jvp` directly on the production function, not a hand-derived
   formula -- the plan entry's own "expect the same NaN-grad-class issues CRK's reverse-mode path
   already hit" caution turned out not to apply here for a specific, checkable reason: this
   function has ZERO Warp calls in it** (pure `torch.einsum`/`torch.linalg.pinv`/`torch.where`,
   confirmed by reading the whole file). The validation methodology section's ban on
   `torch.autograd.functional.jvp` is specifically about Warp-kernel-backed operators, where
   reading a gradient off a `wp.Tape` isn't itself differentiable, so the double-backward trick
   silently zeroes out; it says nothing about a plain-PyTorch sub-function that already supports
   ordinary `torch.autograd.gradcheck` (as `gradcheck_crk_native.py` already proves it does).
   Verified empirically before relying on it (not just argued): `torch.autograd.functional.jvp`
   on `computeCRKTermsWarp` alone, fed synthetic well-conditioned moment tensors, matched central
   finite differences to `~1e-10` (float64, `eps=1e-6`) on every one of A/B/gradA/gradB -- and
   the full pipeline (stage 1+2 assembled into stage 3) then matched the *production* reverse-mode
   Jacobian to `1e-15`-`1e-16`. This sidesteps hand-deriving `gradATerm4`'s
   `'nil,nklm,nmj,nj,ni->nk'`-style four-index tensor contraction (and its structural twins in
   `gradA`/`gradB`) by hand -- a large amount of error-prone matrix calculus for zero additional
   correctness margin over what torch's own autodiff already proves exactly, and the same spirit
   as Tier 2.4 consuming production's own `L`/`num_nbrs` rather than re-deriving them: reuse an
   already-correct piece rather than re-prove it when reuse is honestly available. `num_nbrs`/
   `supports` (the function's two non-differentiable/unused-in-body inputs) are taken directly
   from production's own output, same pattern as every earlier tier's discrete-branch-decision
   treatment. Test geometries are the same well-conditioned line/grid cases every earlier tier
   used, so `computeCRKTermsWarp`'s `is_singular`/`num_nbrs<2` masking branch is never engaged --
   consistent with Tier 2.4's choice not to manufacture a near-singular case on purpose.

4. **`crk/kernel.py`'s `correctGradientCRK`, assembled by direct product rule on its literal
   four-term formula**, using stage 3's `A,B,gradA,gradB` (and their tangents) as PER-QUERY-i
   values broadcast over the neighbor loop -- confirmed from `util/stateUtil.py`'s `getCRK_i`
   (`correctionData.queryA[i]` etc., indexed at `i` only, never `j`) before assuming it, exactly
   like `fi` in every earlier tier's field-value coefficient -- plus `W_ij,gradW_ij` (and their
   tangents) from Tier 2.1/2.2's dispatch evaluated at whatever `SupportScheme` the *consuming*
   operator uses (`SupportScheme.Gather` in this script's test, matching
   `gradcheck_crk_correction_native.py`). `term4`'s `matmul(wp.transpose(gradBi),x_ij)`
   contraction (the exact spot `gradcheck_crk_correction_native.py`'s own docstring records a
   real bug having lived, see that script's header) needed the same first/second-axis care in its
   JVP: `d(gradBi^T@x_ij) = dgradBi^T@x_ij + gradBi^T@dx_ij`, both terms contracting `gradBi`'s
   first (component) axis against the `x_ij`/`dx_ij` vector, matched here via
   `einsum('icl,ijc->ijl', ...)`. The corrected `kernelGradient`+tangent then combine with Tier
   2.2's `_gradient_weights` (mass/density-based `coeff_ij`, *not* CRK's `A_i/B_i` despite the
   naming echo) exactly as Tier 2.2/2.4 already do -- no new derivation for that last step, reused
   verbatim.

**Validation, two independent code paths at each of the four stages (all exact analytic
derivatives, no finite differences except the one-time Stage-3 bring-up sanity check against
central finite differences described above):** a hand-assembled dense all-pairs computation at
each Warp-backed stage (built only from already-validated Tier 2.0-2.2 building blocks) vs.
`torch.autograd.functional.jacobian` on the corresponding production function, contracted with
the tangent -- Tier 2.1's reference pattern, applied stage-by-stage rather than only at the very
end, so a bug in an early stage is caught there rather than surfacing as an unexplained mismatch
three stages later. Agreed to `rel_err ~1e-15`-`1e-16` at every stage, 1D line of 7 and 2D 3x3
grid (Stage 4: all four `GradientScheme`s, both dims) -- no tuning or tolerance-loosening needed.

**Process notes.** One bug, a scope/indexing slip rather than a math error: the first version of
Stage 4's 2D test case sized the frozen field-value tensors (`fv_q`,`fv_r`) using the raw
`n_per_side` parameter (3) instead of the actual particle count after `grid_case_2d(3)` expands
it to a 3x3=9-particle grid -- a shape mismatch caught immediately (`RuntimeError`, not a silent
wrong answer) when broadcasting `coeff_ij` against the field-value tensor. Fixed by reading
`pos0.shape[0]` after building the case, the same pattern every earlier tier's own `run_*_case`
helpers already use. No adjoint-level bugs this tier, despite CRK's documented track record
(`gradcheck_crk_native.py`'s/`gradcheck_crk_correction_native.py`'s own docstrings) of exactly
this failure class in the *reverse*-mode direction -- plausibly because this tier's stage-by-stage
gating caught anything of that shape immediately at the stage it would have appeared in, rather
than needing a from-scratch isolated repro the way the original reverse-mode bugs did.

**Scope note.** `crk_density.py`'s CRK-corrected consistency density (the third return value of
`computeCRKFactors`, used only as a diagnostic/consistency check, never consumed by
`correctGradientCRK` or any operator's actual force computation) is NOT covered by this
deliverable -- `gradcheck_crk_native.py` checks it, but no `correctGradientCRK`-style downstream
consumer ever reads it, so its JVP is not on the path this plan exists to validate (the path from
positions/supports to a CRK-corrected *operator* output). A mechanical follow-up if ever needed:
same four-stage shape, with `computeCRKDensity_Func_i`'s `mDensity/vol1` ratio needing the same
"return the raw accumulated sum, divide one level up" dynamic-loop treatment `crk_density.py`'s
own docstring already documents for its reverse-mode path.

**This closes out `warpier_adjoint.md`'s Tier 2 JVP plan.** All of Tier 2.0 (kernel-level
building blocks) through Tier 2.5 (CRK) are now done -- see "Suggested order and why" below for
the final scorecard, and "Explicitly out of scope for this plan" for what deliberately remains
outside this plan's boundary (grad-h/Omega cross-coupling, the `Field`/`ExecutionMode`/torch-jvp
bridge wiring, and periodic wrap, the last of which was independently verified already correct).

---

## Explicitly out of scope for this plan

* **Grad-h / Omega adaptive-smoothing-length cross-coupling.** `sphKernelDkDh_` is already validated
  and unused in production (`kernels/__init__.py` exports it, nothing calls it -- same status noted
  for `sphKernelHessian_`/`sphGradientDkDh_`). If/when `useGradHTerms`-style Omega corrections start
  consuming a Tier-2 tangent, that coupling crosses into `warpSPH`'s `adaptiveSupport` module
  territory (per `project_field_abstraction_plan` memory) and should get its own plan once there is
  a concrete consumer, not be speculatively derived now.
* **The `Field`/`ExecutionMode`/torch-`jvp`-bridge wiring** (`warpier_core.md` Phase 6 Steps 1-4).
  This plan produces validated formulas; wiring them into the actual forward-mode entry point is a
  separate, later step that should reuse whatever bridge design Phase 6 settles on.
* **Periodic minimum-image wrapping's tangent discontinuity.** `computeDistanceVec`'s minimum-image
  distance is piecewise-identity with a jump exactly at the periodic boundary. Irrelevant almost
  everywhere (measure zero to land exactly on a wrap boundary) but worth a one-line guard/assertion
  in each Tier-2.x script's domain construction (non-periodic, or particles kept well inside the
  domain) rather than rediscovering it as a spurious failure later.

  This is a genuine, unavoidable non-differentiability at `r=L/2` exactly (the wrap function's
  *value*, not just its derivative, jumps there -- no adjoint trick recovers a meaningful gradient
  at that point). It is provably unreachable by any pair that actually contributes to a kernel sum
  whenever `h < L/2` per periodic axis, which is already the standard periodic-SPH requirement
  (otherwise a particle self-interacts through its own periodic image, or double-counts a neighbor
  through two images) -- compactly-supported kernels are exactly zero once `q=r/h>1`, and `h<L/2`
  keeps that zero-region a strict superset of the boundary's neighborhood. No runtime guard was
  added for this (`h<L/2` would need pulling `supports`/domain bounds off the device every call --
  not worth it for a configuration nobody should hit anyway); it's a documented assumption instead.

  **Verified 2026-08-18**, independent of any Tier's math: `scripts/periodic_invariance_check.py`
  checks that translating a particle by an exact integer multiple of the periodic axis length is
  physically invisible -- `f(shifted) == f(original)` forward, and the full reverse-mode Jacobian
  w.r.t. positions/supports/masses[/densities] also matches exactly, for Density/Interpolate/
  Gradient/Divergence/Curl/Laplacian in 1D and 2D, including pairs that genuinely interact through
  a periodic image (not just a degenerate non-wrapping case). This exercises both places the wrap
  is implemented (`buildCompactHashMap`'s pre-hash wrap and `computeDistanceVec`'s per-pair wrap)
  and confirms they agree with each other and with the existing reverse-mode AD path, everywhere
  except the known, unreachable-under-`h<L/2` `r=L/2` point. This closes the loop on the assumption
  every Tier-2.x script's "just avoid periodic domains" workaround depends on: periodicity itself,
  where this plan doesn't touch it, was already correct.

---

## Suggested order and why

1. **Tier 2.1** (Density/Interpolate) -- DONE (2026-08-18). Proved the assembly pattern works end
   to end, zero new kernel math, cheapest possible validation. See the Tier 2.1 "Result" subsection
   above.
2. **Tier 2.2** (Gradient/Divergence/Curl/Laplacian-Brookshaw) -- DONE (2026-08-18). The highest-
   value tier: four production operators' base (non-CRK, non-renorm) paths, still zero new kernel
   math (one new dispatch function assembled from Tier 2.0's building blocks), all built on what
   Tier 2.0 already validated. Laplacian's Dot/Default schemes explicitly deferred -- see the Tier
   2.2 "Result" subsection's scope note. NEXT: Tier 2.4.
3. **Tier 2.4** (Renormalization) -- DONE (2026-08-18). The matrix-calculus (not
   kernel-derivative) identity `d(C^-1) = -C^-1(dC)C^-1`, bounded and well-precedented by
   `gradcheck_pinv_native.py`, assembled cleanly on the first attempt from Tier 2.2's
   `kernelGradient` JVP. See the Tier 2.4 "Result" subsection above. NEXT: Tier 2.5.
4. **Tier 2.5** (CRK) -- DONE (2026-08-18), last and hardest, as planned. Four gated stages
   (apparentVolume, moments, `computeCRKTermsWarp`'s A/B/gradA/gradB, `correctGradientCRK`'s
   product-rule assembly), the middle one obtained via `torch.autograd.functional.jvp` directly
   rather than hand-derived -- a deliberate, empirically-justified departure since that function
   has no Warp call in it, so the "silently zero" hazard the plan's own validation methodology
   warns about does not apply. See the Tier 2.5 "Result" subsection above.
5. **Tier 2.3** (Laplacian `Naive`) -- DONE (2026-08-18), out of the original numeric order.
   The entry below had recommended asking whether it matters in practice before deriving
   `d(∇²W)/dx`/`d(∇²W)/dh`, since Brookshaw (Tier 2.2) is the scheme the codebase's own comments
   treat as the consistent one; asked, and done anyway for the methodological completeness of the
   adjoint SPH derivation, independent of Tiers 2.4/2.5 (built only on Tier 2.0's `eval_d3kdq3`, not
   on anything from 2.4/2.5), so it did not need to wait its numeric turn. See the Tier 2.3 "Result"
   subsection above.

**All five sub-tiers (2.1-2.5) plus Tier 2.0's building blocks are now done -- this plan's scope is
complete**, modulo the three items "Explicitly out of scope for this plan" (below) deliberately
leaves for later or elsewhere: grad-h/Omega cross-coupling, the `Field`/`ExecutionMode`/torch-jvp
bridge wiring, and periodic minimum-image wrapping (independently verified already correct, not
part of this plan's derivation work).

Each tier's gate, per `warpier_fields.md`'s established discipline: the new spike script green,
`pytest tests/` and `scripts/operation_matrix.py` unaffected (these tiers add scripts, not production
code, until someone decides to wire a tier into the actual forward-mode bridge), and the relevant
existing `gradcheck_*.py` scripts re-run to confirm the reverse-mode path this tier's math is checked
against hasn't itself regressed.
