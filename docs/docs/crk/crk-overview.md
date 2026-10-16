# CRK corrections — overview

CRK ("corrected reproducing kernel", à la CRKSPH) lifts the raw SPH
operators to **linear-reproducing consistency**: per query particle $i$,
a scalar $A_i$ and a vector $B_i$ (and their position-derivatives
$\nabla A_i$, $\nabla B_i$) are solved from kernel-weighted moments so
that the corrected kernel and its gradient reproduce constants and
linear fields exactly on incomplete neighbor sets when the augmented moment
constraints are independent and numerically resolved. Degenerate stencils use
the constant identity fallback described below.

Module: `src/warpSPHCore/crk/`. Public API:

```python
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
```

## The corrected kernel and gradient

With $x_{ij} = x_i - x_j$ (the kernel's own convention) and the
uncorrected kernel value/gradient $W_{ij}$, $\nabla W_{ij}$:

$$
\widehat W_{ij}
= A_i\,\big(1 + B_i\cdot x_{ij}\big)\, W_{ij}
\qquad\texttt{(computeKernelCRK)}
$$

$$
\nabla\widehat W_{ij}
= A_i W_{ij}\, B_i
+ A_i\big(1 + B_i\cdot x_{ij}\big)\, \nabla W_{ij}
+ \big(1 + B_i\cdot x_{ij}\big)\, W_{ij}\, \nabla A_i
+ A_i W_{ij}\, \big(\nabla B_i^{\top} x_{ij}\big)
\qquad\texttt{(correctGradientCRK)}
$$

The last term contracts $\nabla B_i$'s **component** axis against
$x_{ij}$, leaving the differentiation axis free (the code is explicit
about this: `matmul(wp.transpose(gradBi), x_ij)`).

The JVP counterparts `computeKernelCRKJVP` /
`correctGradientCRKJVP` apply the ordinary product rule to the same
terms and are the shared building block behind the
[Gradient](../operations/gradient) / [Divergence](../operations/divergence) /
[Curl](../operations/curl) / [Laplacian](../operations/laplacian)
geometry JVPs.

## `computeCRKFactors` — the pipeline

```python
apparent_area, crk_density, crk_state = computeCRKFactors(
    queryParticles, domain, kernel,
    operationMode=OperationDirection.AllToAll,
    adjacency=None,                 # AdjacencyList | CompactHashMap | None
    referenceState=None,
    referenceVolumeMode=None,       # independent reference-to-reference rule
)
```

With a distinct `referenceState`, the returned apparent volume is defined
on the query population against the reference population. Moment/density
weights instead use apparent volumes estimated on the reference population
against itself. These arrays follow different particle IDs and can have
different lengths. The extra reference estimate builds its own grid, since
a query-to-reference neighbor list cannot describe reference-to-reference
interactions. With one population, the original estimate is reused.

`referenceVolumeMode=None` preserves that inherited interaction mode and
reuse policy. An explicit mode computes a separate reference-to-reference
estimate, including for an aliased population. Query volumes and moment
interactions still follow `operationMode`. A cross-category rule can reject
all reference-to-reference query rows, leaving zero weights and identity
fallbacks. When the whole reference cloud is the intended geometric sample,
`referenceVolumeMode=OperationDirection.TrueAllToToAll` gives it an independent
unfiltered volume estimate. The caller chooses the reference cloud and its
volume rule; ghosts included by that rule participate in volume estimation.
Downstream operators must use the same reference weights and Scatter support
to inherit the construction's reproduction identities.
To obtain those weights through the public API, take the first return from
`computeCRKFactors(referenceState, domain, kernel, operationMode=referenceVolumeMode)`
and pass it as `referenceVolumes` to the consuming operator. The first return
from the query/reference construction belongs to the query population.

Empty query populations preserve all output shapes. With no contributing
neighbors, apparent volume and diagnostic density are zero, and correction
terms use the existing constant identity fallback (`A=1`, `B=0`, zero spatial
derivatives). Their geometry tangents and adjoints are finite zeros while that
branch stays selected. These conventions do not supply a reproducing stencil.
Empty reference populations require explicit domain bounds for grid construction.

The correction stages and diagnostic use dual-path adjacency/grid kernels:

1. **Apparent volume** — $V_i = 1/\sum_j W_{ij}$, computed with the
   **Gather** scheme and the *uncorrected* kernel
   ([CRK volume](crk-volume)).
2. **Moments** — $m_0, m_1, m_2$ and their $\gamma$-derivatives,
   computed with the **Scatter** scheme, the *uncorrected* kernel, and
   the apparent volume as the per-particle weight ([CRK
   moments](crk-moments)).
3. **Solve for the correction terms** — `computeCRKTermsWarp` (pure
   torch, no Warp kernel):

   $$
   M_i = m_{2,i},\qquad
   A_i = \frac{1}{m_{0,i} - (M_i^{-1}m_{1,i})^{\top} m_{1,i}},\qquad
   B_i = -M_i^{-1} m_{1,i},
   $$

   with $M_i^{-1} = \mathrm{pinv}(M_i)$ and the gradient terms
   $\nabla A_i$, $\nabla B_i$ by the product rule through the
   $dm_{\ell}/d\gamma$ sums (the exact einsums are in
   `crk/crk_terms.py`).

   **Limiter:** particles with fewer than 2 neighbors or a singular
   moment matrix ($|m_{2,\mathrm{det}}| < 10^{-14}$) fall back to the
   *identity* correction $A_i = 1$, $B_i = 0$, $\nabla A_i = 0$,
   $\nabla B_i = 0$ — the uncorrected kernel. A warning is printed with
   the count.

   The limiter also rejects a nonfinite or unresolved Schur complement
   `s = m0 - m1ᵀ pinv(m2) m1`: `s <= 32*eps*(abs(m0)+abs(projection))`,
   where `projection = m1ᵀ pinv(m2) m1` and `eps` is the tensor dtype's
   machine epsilon. This detects singular augmented moment constraints even
   when `m2` is invertible, such as reference points on an affine plane away
   from the query. Rejected rows use finite constant algebra before division
   and have zero tangents/adjoints. The existing neighbor/determinant cutoffs
   remain in force. The guard is a numerical fallback, not a conditioning
   certificate for every accepted stencil; crossings are nonsmooth.

4. **Diagnostic density** — the CRK-corrected consistency density
   [CRK density](crk-density).

Returned `crk_state` is a `CRKState(A, B, gradA, gradB)` — pass it as
`crkState=` to `warpOperation` and every consuming operator
([Gradient](../operations/gradient), [Divergence](../operations/divergence),
[Curl](../operations/curl), [Laplacian](../operations/laplacian)
(Brookshaw/Dot/Default schemes), [Interpolate](../operations/interpolate))
uses the corrected kernel/gradient. Interpolate reads only $A_i, B_i$;
the gradient-family operators read all four tensors, and
`warpOperation` raises for them if `gradA`/`gradB` are missing.

## `computeCRKFactorsJVP` — the pipeline's JVP

```python
apparent_area, d_apparent_area, crk_state, crk_tangent_state = \
    computeCRKFactorsJVP(
        queryParticles, domain, kernel,
        queryTangentState,                     # ParticleTangentState
        referenceTangentState=None,
        operationMode=..., adjacency=None, referenceState=None,
        referenceVolumeMode=None,
    )
```

Chains the three stages' JVPs: the apparent-volume tangent
(`computeCRKVolumeGeometryJVP`) feeds the moment tangents
(`computeCRKMomentsGeometryJVP`), which feed the stage-3 solve. Stage 3
is differentiated with `torch.autograd.functional.jvp(...,
create_graph=True)` — valid there (and only there in this pipeline)
because `computeCRKTermsWarp` contains no Warp call, so double backward
through it is exact. `create_graph=True` is required so
$dA, dB, d\mathrm{grad}A, d\mathrm{grad}B$ stay differentiable back to
positions/supports for whole-pipeline gradcheck. The diagnostic
`crk_density` is not computed (no consumer in the JVP path).

For distinct populations, reference-volume tangents also use the
reference-to-reference estimate. Supply the reference geometry tangent
explicitly: an omitted reference tangent means zero even when primal
particles are shared between roles. All three traversal inputs, including
omitted adjacency, are supported. `tests/operations/test_crk_reference_roles.py`
qualifies unequal populations, reference permutation invariance, finite
differences and reverse-mode dot products with AllToAll and kind 0 particles.
With an explicit `referenceVolumeMode`, reference-to-reference volume
tangents use the reference tangent in both roles, including when the primal
population aliases the query. Supply both role tangents for one physical
moving population.

Both constructors propagate frozen particle kinds and `operationMode` through
volume and moment evaluation. The JVP differentiates the selected forward
pipeline, including its constant low-count/singular fallback. That fallback
uses finite inputs before inverse/reciprocal algebra so excluded rows have
finite zero derivatives. `tests/operations/test_kind_directions.py` qualifies
mixed-kind derivative parity for all interaction modes.
`tests/operations/test_correction_numerics.py` additionally qualifies the
independent unfiltered reference-volume option for all 15 directions with
mixed reference kinds: moment reproduction identities, downstream linear
interpolation/Gradient, finite differences and reverse dot products. This
uses well-resolved 2D stencils, Scatter support and Naive Gradient; it does
not certify every volume/boundary policy or poorly conditioned stencil.

This is the usual way to obtain the `CRKTangentState` that the
operators' geometry JVPs accept as `crkTangentState` (omitting it holds
the CRK correction frozen while the geometry moves).

## Design notes

- **Uncorrected moments.** The moments are computed with the plain
  kernel — CRK correction is never applied while computing the
  correction (the correction is defined by the uncorrected kernel's
  moments; applying it would be circular).
- **Scheme split on purpose.** Volume uses Gather ($h_i$), moments use
  Scatter ($h_j$); both are fixed inside `computeCRKFactors`, not
  caller options.
- **Traversal-agnostic neighbor count.** The moments kernel returns the
  per-particle neighbor count as a genuine kernel output (like
  Covariance does), because `AdjacencyList.numNeighbors` does not exist
  under grid traversal and the limiter needs it either way.
- **Warp adjoint workaround.** The volume and density kernels compute
  the reciprocal/ratio *one level up*, outside the function containing
  the dynamic neighbor loop — Warp's automatic adjoint for
  loop-accumulate-then-divide in one `@wp.func` produces NaN gradients
  (see `crk_volume.py`'s comment and `scripts/diagnostics/debug_crk_backward.py`).

## Tests

- `tests/operations/test_operations_crk_analytic.py` — the correction
  terms against analytic references (uniform lattice moments).
- `tests/operations/test_forward_mode_geometry_jvp_gradient.py`
  (and the divergence/curl/laplacian siblings) — the operators with
  `crkTangentState`, gradchecked end to end.
- The stage-3 `torch.func.jvp` pattern is validated by
  `scripts/spikes/spike_forward_mode_tier2_crk.py` (gated by
  `tests/operations/test_gradcheck_scripts.py`).

## See also

[CRK volume](crk-volume) · [CRK moments](crk-moments) ·
[CRK density](crk-density) · [Kernels](../kernels) ·
[Autodiff machinery](../autograd)
