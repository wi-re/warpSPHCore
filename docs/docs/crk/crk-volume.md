# CRK stage 1 — apparent volume

Kernel: `crk/crk_volume.py` (`computeCRKVolume_Kernel`); JVP
`crk/crk_volume_jvp.py` (`computeCRKVolumeGeometryJVP`).

## Equation

The per-particle **apparent volume** (called `apparentArea` in the
code, being an area in 2D):

$$
V_i \;=\; \frac{1}{\displaystyle\sum_{j \in \mathcal{N}(i)} W_{ij}}
$$

— the reciprocal of the plain kernel sum. It estimates the volume of
fluid associated with particle $i$ from the kernel's own normalization
($\sum_j W_{ij} \approx 1/V_i$), and is the weight the CRK moments and
density use in place of $m_j/\rho_j$.

Computed with the **Gather** support scheme and the **uncorrected**
kernel (no CRK, no renormalization, no apparent volume — the kernel's
`correctionData` argument is present for ABI consistency but unused).
The lattice-normalization factor is applied through the kernel layer as
for any operator.

The Warp kernel accumulates the raw sum
$\sum_j W_{ij}$ per particle; the reciprocal is applied **one level
up**, in `computeCRKVolume_Kernel`, outside the function containing the
dynamic neighbor loop. That placement is load-bearing: Warp's
automatic reverse-mode adjoint for "accumulate into a local in a
runtime-length loop, then divide" in a single `@wp.func` produces NaN
gradients (reproduced in `scripts/diagnostics/debug_crk_backward.py`).

## Public API

Internal backend `_computeCRKVolume_stateBackend(queryParticles,
operationProperties, domain, adjacency=None, referenceParticles=None)`.
The public path is through
[`computeCRKFactors`](crk-overview#computecrkfactors--the-pipeline)
(which fixes the Gather scheme itself).

**Output:** `torch.Tensor` `[N]`.

## JVP

`computeCRKVolumeGeometryJVP(queryParticles, domain, kernel, adjacency,
apparentArea, queryTangentState, referenceTangentState=None)` — the
chain rule through the reciprocal:

$$
dV_i = -\frac{1}{\left(\sum_j W_{ij}\right)^2}
      \sum_j dW_{ij},
$$

with $dW_{ij}$ from `sphKernelJVP` (position/support tangents). It is
stage 1 of `computeCRKFactorsJVP` — its output feeds the moment
JVP's `tangentReferenceVolumes`.

## Tests

- `tests/operations/test_operations_crk_analytic.py` — the volume
  against analytic lattice references.
- `tests/operations/test_forward_mode_geometry_jvp_gradient.py`
  (siblings) — exercises it indirectly through the full CRK-tangent
  pipeline.

## See also

[CRK overview](crk-overview) · [CRK moments](crk-moments) ·
[CRK density](crk-density)
