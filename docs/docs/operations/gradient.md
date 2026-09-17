# Gradient

SPH gradient of a field, in four coefficient schemes. Kernel:
`coreOperations/wp_gradient.py` (`computeSPHGradientTensor_Kernel`);
JVP `coreOperations/wp_gradientJVP.py`.

## Equation

$$
\nabla f_i \;=\; \sum_{j \in \mathcal{N}(i)} \mathrm{coeff}_{ij}\; \nabla W_{ij},
$$

with $\mathrm{coeff}_{ij}$ selected by `gradientMode`
(`GradientScheme`):

| Scheme | $\mathrm{coeff}_{ij}$ | Notes |
|---|---|---|
| `Naive` (1) | $f_j\, V_j$ | the standard "scatter" form |
| `Symmetric` (2) | $m_j\,\rho_i\!\left(\dfrac{f_i}{\rho_i^2} + \dfrac{f_j}{\rho_j^2}\right)$ | density-symmetric; no $V_j$ term at all |
| `Difference` (3) | $(f_j - f_i)\, V_j$ | the consistent two-particle difference form; reproduces linear fields exactly on a complete neighbor set |
| `Summation` (4) | $(f_j + f_i)\, V_j$ | |

where

$$
V_j = \begin{cases}
m_j / \rho_j & \text{default} \\
\mathrm{referenceVolumes}[j] & \text{if an apparent volume is supplied}
\end{cases}
$$

and, if grad-h terms are supplied (`GradHState`), both field values are
rescaled by the local smoothing-length ratio:
$f_i \leftarrow f_i/\omega_i$, $f_j \leftarrow f_j/\omega_j$.

$\nabla W_{ij}$ is the kernel gradient `sphKernelGradient` under the
chosen [support scheme](../kernels#pairwise-evaluation-and-the-support-schemes), optionally passed
through the **CRK correction** (`computeKernelGradientCRK`, see
[CRK overview](../crk/crk-overview)) and then **gradient
renormalization** (left-multiply by $L_i$, see
[Renormalization](../renorm)). The composition order is fixed everywhere:
CRK first, then renormalization, applied to the single resulting
$\nabla W_{ij}$.

**Output:** shape `queryValues.shape[1:] + (dim,)` — a scalar field
gives `[N, dim]`, a vector field `[N, dim, dim]`.

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    queryValues=f, referenceValues=f,      # required, shapes [N] or [N, ...]
    adjacency=None,
    queryVolumes=None, referenceVolumes=None,  # apparent-volume substitution
    crkState=None,                        # CRKState
    gradHState=None,                      # GradHState | (qOmega, rOmega) | omega
    renormalizationState=None,            # RenormalizationState | [N,D,D] tensor
)
```

with `operation == WarpOperation.Gradient`.

| Option | Default | Effect |
|---|---|---|
| `gradientMode` | `Naive` | coefficient scheme (table above) |
| `supportMode` | `Gather` | pairwise support of $\nabla W_{ij}$ |
| `operationMode` | `AllToAll` | directional filtering on `kinds` |
| `queryVolumes`/`referenceVolumes` | `None` | substitute $V_j$ for $m_j/\rho_j$ (only the reference side is ever read) |
| `crkState` | `None` | CRK-corrected kernel gradient; for Gradient the full state including `gradA`/`gradB` is used |
| `gradHState` | `None` | rescale $f_i$, $f_j$ by $1/\omega$ |
| `renormalizationState` | `None` | left-multiply $\nabla W_{ij}$ by $L_i$ |
| `calibrateNormalization`, `n_h` | `False`/`None` | lattice-normalization factor on the kernel |

The dispatcher requires **both** `queryValues` and `referenceValues`
for Gradient; `referenceValues` defaults to `queryValues` inside
`warpOperation`.

## Corrections, in order

1. `f_j \leftarrow f_j/\omega_j`, `f_i \leftarrow f_i/\omega_i` if
   `gradHState` is passed.
2. $\nabla W_{ij}$: plain kernel gradient → CRK-corrected if
   `crkState` → $L_i \nabla W_{ij}$ if `renormalizationState`.
3. `V_j` substituted by `referenceVolumes[j]` if supplied.

## JVP / HVP

**Value JVP** — the operator is exactly linear and homogeneous in
`queryValues`/`referenceValues`, so the value-tangent JVP is the same
operator relaunched on the tangent arrays:
`warpOperationJVP(..., tangentQueryValues=..., tangentReferenceValues=...)`.
Gated by `tests/operations/test_forward_mode_value_jvp.py`.

**Geometry JVP** — `warpOperationJVP(..., queryTangentState=...,
referenceTangentState=...)` computes, per neighbor, the product rule
($G_{ij} = \nabla W_{ij}$ after CRK/renorm, with its own tangent $dG_{ij}$
from `computeKernelGradientCRKJVP`):

$$
d(\nabla f)_i \;=\; \sum_j
\Big[
  \mathrm{dcoeff}_{ij}\cdot G_{ij}
+ \mathrm{coeff}_{ij}\cdot dG_{ij}
\Big]
\;+\; \underbrace{\sum_j (\, df_i A_{ij} + df_j B_{ij}\,)\cdot G_{ij}\,}_{\text{value-tangent term, fused into the same loop}},
$$

with $\mathrm{coeff} = f_i A + f_j B$ (values frozen at primal) and the
per-pair weights from `_jvpCommon.gradientWeightsJVP`:

| Scheme | $A_{ij}$ | $B_{ij}$ |
|---|---|---|
| `Naive` | $0$ | $V_j$ |
| `Difference` | $-V_j$ | $V_j$ |
| `Summation` | $V_j$ | $V_j$ |
| `Symmetric` | $m_j/\rho_i$ | $m_j\rho_i/\rho_j^2$ |

with tangents

$$
dV_j = \frac{dm_j}{\rho_j} - \frac{m_j\, d\rho_j}{\rho_j^2}
\quad\text{(or the supplied apparent-volume tangent)},
$$

$$
dA = \frac{dm_j}{\rho_i} - \frac{m_j\, d\rho_i}{\rho_i^2},
\qquad
dB = \frac{dm_j\,\rho_i + m_j\, d\rho_i}{\rho_j^2}
      - \frac{2\,m_j\,\rho_i\, d\rho_j}{\rho_j^3}
\quad\text{(Symmetric only)}.
$$

Notes on scope:

- **Query-side mass tangents are rejected** — no formula has an $m_i$
  term. Query-side density tangents are used (Symmetric scheme reads
  $\rho_i$).
- **CRK tangent**: pass `crkTangentState` (a `CRKTangentState`) to
  differentiate the correction itself; omitting it holds CRK frozen.
  The usual way to obtain a real tangent is
  `computeCRKFactorsJVP` on the same geometry tangents.
- **Renormalization tangent**: pass
  `renormalizationTangentState`; the kernel applies
  $dG = dL_i\, G + L_i\, dG$ (product rule, after CRK). Obtain one via
  `computeRenormalizationMatricesJVP`.
- **CRK and renormalization tangents may be supplied simultaneously** —
  they compose in the same fixed order as the primal kernel.
- `gradHState` is **not supported** for the geometry JVP (no consumer
  exists).
- The value-tangent term is fused into the geometry-JVP neighbor loop
  (`_FUSED_VALUE_JVP_OPERATIONS` in `operations.py`) so a combined
  value+geometry JVP costs one pass, not two.

Gated by `tests/operations/test_forward_mode_geometry_jvp_gradient.py`
(gradcheck against production, all correction combinations).

No HVP exists for Gradient.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness for
  scalar, vector and matrix fields.
- `tests/operations/test_operations_consistency.py` — e.g. the trace of
  the Difference-scheme gradient of a vector field matches the
  Difference-scheme divergence (interior mask).
- `tests/operations/test_forward_mode_value_jvp.py` — value-JVP identity
  (relaunch on tangents).
- `tests/operations/test_forward_mode_geometry_jvp_gradient.py` —
  geometry JVP + CRK/renorm tangents vs reverse-mode Jacobian.
- `tests/operations/test_grid_modes.py` — adjacency vs grid traversal
  parity.

## See also

[Divergence](divergence) · [Curl](curl) · [Laplacian](laplacian) ·
[CRK overview](../crk/crk-overview) · [Renormalization](../renorm)
