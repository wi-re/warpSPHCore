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

---

## Suggested order and why

1. **Tier 2.1** (Density/Interpolate) -- proves the assembly pattern works at all, zero new kernel
   math, cheapest possible validation.
2. **Tier 2.2** (Gradient/Divergence/Curl/Laplacian-non-Naive) -- the highest-value tier: four
   production operators' base (non-CRK, non-renorm) paths, still zero new kernel math, all built on
   what Tier 2.0 already validated.
3. **Tier 2.4** (Renormalization) -- first tier needing genuinely new (matrix-calculus, not
   kernel-derivative) work, but bounded and well-precedented by `gradcheck_pinv_native.py`.
4. **Tier 2.5** (CRK) -- last, hardest, highest historical bug rate; do only once 2.2/2.4 are solid
   since it depends on both.
5. **Tier 2.3** (Laplacian `Naive`) -- do only if confirmed to matter; ask before deriving `d(∇²W)/dx`
   and `d(∇²W)/dh`, since Brookshaw (covered by Tier 2.2) is the scheme the codebase's own comments
   treat as the consistent one.

Each tier's gate, per `warpier_fields.md`'s established discipline: the new spike script green,
`pytest tests/` and `scripts/operation_matrix.py` unaffected (these tiers add scripts, not production
code, until someone decides to wire a tier into the actual forward-mode bridge), and the relevant
existing `gradcheck_*.py` scripts re-run to confirm the reverse-mode path this tier's math is checked
against hasn't itself regressed.
