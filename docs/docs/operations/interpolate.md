# Interpolate

Kernel-weighted (re)construction of a field at the query positions.
Kernel: `coreOperations/wp_interpolate.py`
(`computeSPHInterpolation_Kernel`); JVP
`coreOperations/wp_interpolateJVP.py`.

## Equation

$$
\tilde f_i \;=\; \sum_{j \in \mathcal{N}(i)} f_j\, V_j\, \widehat W_{ij},
\qquad
V_j = \begin{cases}
m_j / \rho_j & \text{default} \\
\mathrm{referenceVolumes}[j] & \text{if an apparent volume is supplied}
\end{cases}
$$

$\widehat W_{ij}$ is the kernel value `sphKernel` under the chosen
[support scheme](../kernels#pairwise-evaluation-and-the-support-schemes), **CRK-corrected** when a
`CRKState` is supplied:

$$
\widehat W_{ij} = A_i\,\big(1 + B_i\cdot x_{ij}\big)\, W_{ij}
\qquad(\text{`computeKernelCRK`}),
$$

see [CRK overview](../crk/crk-overview). Interpolate is the only
value-having operator that **never reads `queryValues`** — the field is
read from the reference side only — and it never reads the CRK
gradient terms (`gradA`/`gradB`; if you pass a `CRKState` without them
the backend substitutes correctly-shaped dummies, since the kernel only
uses $A_i, B_i$).

**Output:** `[N_query, ...]`, preserving the trailing shape of
`referenceValues`. The query and reference counts can differ. Warp kernels
support rank-1/2 field types, so fields of rank $>2$ are flattened to
one vector dimension before the launch and reshaped back afterwards.

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    referenceValues=f,               # required
    queryVolumes=None, referenceVolumes=None,
    crkState=None,
    adjacency=None,
    referenceParticles=None,
)
```

with `operation == WarpOperation.Interpolate`. `warpOperation` raises
if `referenceValues` is missing.

| Option | Default | Effect |
|---|---|---|
| `supportMode` | `Gather` | pairwise support of $W_{ij}$ |
| `operationMode` | `AllToAll` | directional filtering on `kinds` |
| `referenceVolumes` | `None` | substitute $V_j$ |
| `crkState` | `None` | CRK-corrected kernel value ($A_i, B_i$ only) |
| `calibrateNormalization`, `n_h` | `False`/`None` | lattice-normalization factor on $W$ |

`gradientMode`, `laplacianMode`, `positiveDivergence` and the
renormalization/grad-h states are not consumed by Interpolate.

## JVP / HVP

**Value JVP** — relaunch on `tangentReferenceValues` (there is no
query-side value input, so `warpOperationJVP` only ever threads a
reference tangent for Interpolate).

**Geometry JVP** — the volume-weighted product rule:

$$
d\tilde f_i \;=\; \sum_j
\Big[
  f_j\big(dV_j\, W_{ij} + V_j\, dW_{ij}\big)
\Big]
\;+\; \underbrace{\sum_j df_j\, V_j\, W_{ij}}_{\text{value-tangent term, fused}},
$$

with $f_j$ frozen at primal for the geometry part and

$$
dV_j = \frac{dm_j}{\rho_j}
      - \frac{m_j\, d\rho_j}{\rho_j^2}
\quad\text{(or the supplied apparent-volume tangent)},
$$

$dW_{ij}$ from `sphKernelJVP`. Interpolate has **no query-side density term**. The Warp explicit geometry
JVP has no CRK correction path and rejects query-mass tangents; omitted
tangents default to zero. Native JVPs additionally support query-side CRK
$A_i,B_i$ products and classic forward duals; unused particle/correction
roles have zero products.

Gated by `tests/operations/test_forward_mode_geometry_jvp_interpolate.py`.

No HVP exists for Interpolate.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness for
  scalar, vector and matrix fields.
- `tests/operations/test_operations_consistency.py` — e.g.
  interpolating a linear field reproduces it at interior points.
- `tests/operations/test_forward_mode_value_jvp.py`,
  `tests/operations/test_forward_mode_geometry_jvp_interpolate.py`,
  `tests/operations/test_grid_modes.py`.

## See also

[Density](density) · [CRK overview](../crk/crk-overview) ·
[CRK volume](../crk/crk-volume)

## Native derivative contract

The native Torch-facing path supports all 15 kernels, six support schemes,
all operation directions, FP32/FP64 and 1D–3D. It preserves arbitrary nonempty
trailing field shapes by launching bounded scalar/vector/matrix blocks.
Mass/density volume weights or explicit reference volumes and query CRK
$A_i,B_i$ participate in backward/VJP and JVP. Unread query fields, grad-h,
renormalization and CRK gradient/reference coefficients do not enter the
interpolation equation. Supplied query volumes are the reference-volume
fallback when reference volumes are omitted.

Reverse uses deterministic query CSR/reference CSC ownership with no floating
atomics or per-edge Jacobians. Supplied CSR, native auto-search and rebuilding
supplied Warp grids retain original particle order. Noncontiguous floating
inputs, shared query/reference tensors, empty populations, sparse/duplicate
edges and fixed-CSR CUDA capture/replay are covered. Normalization calibration
is a constant output factor.

Products are first-order, matching Warp's detached taped reverse. Functional
transforms, higher products and row-chunk AD are inherited unsupported paths.
At singular self pairs/cutoffs, native preserves the reference nonfinites and
its separate reverse/analytic-JVP conventions, including max-support ties.
The framework-free C++ ABI supports its declared scalar/vector/matrix fields
and raw uncorrected interpolation adjoint for all 15 kernels; that ABI exposes
no JVP or correction bundles. Deterministic ownership describes the Torch
bridge, not the standalone scatter/accumulation API.

Maintained qualification: `tests/native/torch/test_interpolate_contract.py`,
`tests/parity/test_interpolate_contract.py` and the standalone interpolation
product checks. Accepted frontend PESPH steps qualify primal simulation use;
independent three-step feedback products are toy AD evidence.
