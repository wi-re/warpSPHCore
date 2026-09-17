# CRK stage 4 — corrected density (diagnostic)

Kernel: `crk/crk_density.py`
(`computeCRKDensity_Kernel`).

## Equation

A CRK-consistency density, returned by
[`computeCRKFactors`](crk-overview) as a diagnostic:

$$
\rho^{\mathrm{CRK}}_i
= \frac{\displaystyle\sum_j m_j\, V_j\, \widehat W_{ij}}
       {\displaystyle\sum_j V_j^2\, \widehat W_{ij}},
$$

with the **CRK-corrected** kernel
$\widehat W_{ij} = A_i(1 + B_i\cdot x_{ij})\, W_{ij}$ (Scatter support
scheme, $V_j$ the apparent volume from
[stage 1](crk-volume)). The numerator is a mass-weighted sum and the
denominator a volume-normalizing sum, so on a consistent neighbor set
the ratio reads the nominal density with the linear-reproduction error
removed — useful to *see* what the correction is doing, not a
replacement for the [Density](../operations/density) operator in a
simulation.

Implementation notes:

- The kernel accumulates the two sums and returns them; the ratio is
  applied one level up, in `computeCRKDensity_Kernel`, outside the
  dynamic-loop function — the same Warp-adjoint NaN workaround as the
  [volume kernel](crk-volume).
- The CRK kernel properties are built fresh with the Scatter scheme and
  **deliberately do not carry the lattice-normalization flag**: the
  lattice-sum defect the calibration exists to remove is not present in
  this construction, and applying $1/L$ on top would double-correct
  (see `crk_density.py`'s comment; `LATTICE_DENSITY_PLAN.md` §3.3).

## Public API

Internal backend `_computeCRKDensity_stateBackend(queryParticles,
operationProperties, domain, crkState, queryVolumes, referenceVolumes,
adjacency=None, referenceParticles=None)`. The public path is through
[`computeCRKFactors`](crk-overview#computecrkfactors--the-pipeline)
only — it is not wired into `warpOperation`.

**Output:** `torch.Tensor` `[N]`.

## JVP

None. The density diagnostic has no consumer in any JVP path;
`computeCRKFactorsJVP` does not compute it (see [CRK
overview](crk-overview#computecrkfactorsjvp--the-pipelines-jvp)).

## Tests

- `tests/operations/test_operations_crk_analytic.py` — the corrected
  density against analytic references.

## See also

[CRK overview](crk-overview) · [CRK moments](crk-moments) ·
[Density](../operations/density)
