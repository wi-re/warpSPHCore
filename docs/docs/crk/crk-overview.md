# CRK corrections — overview

CRK ("corrected reproducing kernel", à la CRKSPH) lifts the raw SPH
operators to **linear-reproducing consistency**: per query particle $i$,
a scalar $A_i$ and a vector $B_i$ (and their position-derivatives
$\nabla A_i$, $\nabla B_i$) are solved from kernel-weighted moments so
that the corrected kernel and its gradient reproduce constants and
linear fields exactly, even on incomplete neighbor sets (free surfaces,
boundaries, low counts).

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
)
```

Three stages, each a dual-path (adjacency + grid) kernel:

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
  (see `crk_volume.py`'s comment and `scripts/debug_crk_backward.py`).

## Tests

- `tests/operations/test_operations_crk_analytic.py` — the correction
  terms against analytic references (uniform lattice moments).
- `tests/operations/test_forward_mode_geometry_jvp_gradient.py`
  (and the divergence/curl/laplacian siblings) — the operators with
  `crkTangentState`, gradchecked end to end.
- The stage-3 `torch.func.jvp` pattern is validated by
  `scripts/spike_forward_mode_tier2_crk.py` (gated by
  `tests/operations/test_gradcheck_scripts.py`).

## See also

[CRK volume](crk-volume) · [CRK moments](crk-moments) ·
[CRK density](crk-density) · [Kernels](../kernels) ·
[Autodiff machinery](../autograd)
