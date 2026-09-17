# CRK stage 2 — moments

Kernel: `crk/crk_moments.py`
(`computeCRKMoments_Kernel`); JVP `crk/crk_moments_jvp.py`
(`computeCRKMomentsGeometryJVP`).

## Equations

For query particle $i$, with the apparent volume
$V_j$ ([stage 1](crk-volume)) as weight, the **uncorrected** kernel
value $W_{ij}$ and gradient $\nabla W_{ij}$ (Scatter support scheme),
and $x_{ij} = x_i - x_j$:

$$
m_{0,i} = \sum_j V_j W_{ij},
\qquad
m_{1,i} = \sum_j x_{ij}\, V_j W_{ij},
\qquad
m_{2,i} = \sum_j x_{ij}x_{ij}^{\top}\, V_j W_{ij},
$$

and their derivatives with respect to the query position (the
"$\gamma$-derivatives", needed for $\nabla A_i$, $\nabla B_i$):

$$
\frac{\partial m_0}{\partial \gamma}
  = \sum_j V_j\, \nabla W_{ij},
$$

$$
\frac{\partial m_1}{\partial \gamma}
  = \sum_j V_j\left(x_{ij}\,\nabla W_{ij}^{\top} + W_{ij}\, I\right),
$$

$$
\frac{\partial m_2}{\partial \gamma}
  = \sum_j V_j\Big(
      x_{\alpha} x_{\beta}\, (\nabla W)_{\gamma}
    + W_{ij}\big(x_{\alpha}\,\delta_{\beta\gamma}
                + \delta_{\alpha\gamma}\, x_{\beta}\big)
    \Big)
$$

(the kernel stores the third-order tensor flattened to
`dim**3` components, `[gamma * dim*dim + alpha*dim + beta]`, because
Warp's rank-3 handling is awkward — the backend reshapes it back to
`[N, D, D, D]`).

The moments are computed with the **Scatter** scheme and **no
corrections of any kind** — `crkState`/`gradHState`/
`renormalizationState` are never applied while computing the
correction (see [CRK overview](crk-overview#design-notes)).

The **per-particle neighbor count** is a genuine kernel output (same
pattern as [Covariance](../operations/covariance)), not read off
`adjacency.numNeighbors` — that field does not exist under grid
traversal, and the [correction-term
limiter](crk-overview#computecrkfactors--the-pipeline) needs the count
under both traversal modes.

## Public API

Internal backend `_computeCRKMoments_stateBackend(queryParticles,
operationProperties, domain, queryVolumes, referenceVolumes,
adjacency=None, referenceParticles=None)`. The public path is through
[`computeCRKFactors`](crk-overview#computecrkfactors--the-pipeline),
which passes the apparent volume as both `queryVolumes` and
`referenceVolumes` and fixes the Scatter scheme.

**Outputs:** `(m_0 [N], m_1 [N,D], m_2 [N,D,D], dm_0dgamma [N,D],
dm_1dgamma [N,D,D], dm_2dgamma [N,D,D,D], numNeighbors [N] int32)`.

## JVP

`computeCRKMomentsGeometryJVP(queryParticles, domain, kernel,
adjacency, referenceVolumes, tangentReferenceVolumes,
queryTangentState, referenceParticles=None,
referenceTangentState=None)` — the product rule through every moment:
position/support tangents flow through $x_{ij}$, $W_{ij}$,
$\nabla W_{ij}$ (via `sphKernelJVP` / `sphKernelGradientJVP`), and the
volume weight flows through `tangentReferenceVolumes` (stage 1's
output). It is stage 2 of `computeCRKFactorsJVP`.

## Tests

- `tests/operations/test_operations_crk_analytic.py` — moments against
  analytic references.
- `tests/operations/test_forward_mode_geometry_jvp_gradient.py`
  (siblings) — through the full CRK-tangent pipeline.
- `scripts/spikes/spike_forward_mode_tier2_crk.py` — the stage assembly,
  float64, gated by `test_gradcheck_scripts.py`.

## See also

[CRK overview](crk-overview) · [CRK volume](crk-volume) ·
[CRK density](crk-density)
