# Density

The SPH density sum. Kernel:
`coreOperations/wp_density.py` (`computeSPHDensityTensor_Kernel`); JVP
`coreOperations/wp_densityJVP.py`; HVP `coreOperations/wp_densityHVP.py`.

## Equation

For query particle $i$ with neighbor set $\mathcal{N}(i)$:

$$
\rho_i \;=\; \sum_{j \in \mathcal{N}(i)} m_j\, W_{ij},
\qquad
W_{ij} \;=\; C_d\, h_{ij}^{-d}\, k\!\left(\frac{r_{ij}}{h_{ij}}\right),
$$

where

- $r_{ij} = \lVert x_i - x_j\rVert$ is the (minimum-image,
  periodic-aware) distance from `computeDistanceVec`,
- $h_{ij} = \mathrm{computePairwiseSupport}(h_i, h_j, \text{supportMode})$
  resolves the pairwise support per the [support scheme](../kernels#pairwise-evaluation-and-the-support-schemes)
  (Gather: $h_i$; Scatter: $h_j$; MeanSymmetric: $(h_i+h_j)/2$; the
  three symmetric schemes evaluate the kernel at both $h_i$ and $h_j$),
- $k$ and $C_d$ are the compact-form kernel and its normalizing constant
  (see [Kernels](../kernels)),
- and the optional lattice-normalization factor multiplies $W$
  (see [Lattice calibration](../lattice-calibration)).

Density is the **only operator with no field values and no correction
inputs** — it reads `queryParticles.masses` directly and applies no CRK,
renormalization, apparent-volume, or grad-h term.

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    adjacency=None,            # AdjacencyList | CompactHashMap | None
    referenceParticles=None,   # defaults to queryParticles
)
```

with `operationProperties.operation == WarpOperation.Density`. The
dispatcher (`operations.py`) routes Density directly to
`_computeSPHDensity_stateBackend` before any value-argument validation,
since it takes no `queryValues`/`referenceValues`.

| Option | Type | Default | Effect |
|---|---|---|---|
| `kernel` | `KernelFunctions` | — | which compact-form kernel $k$ (any of the 12) |
| `supportMode` | `SupportScheme` | `Gather` | pairwise support resolution (see [Kernels](../kernels#pairwise-evaluation-and-the-support-schemes)) |
| `operationMode` | `OperationDirection` | `AllToAll` | directional filtering on `ParticleState.kinds` (fluid/boundary/ghost) |
| `calibrateNormalization`, `n_h` | `bool`, `float` | `False`, `None` | lattice-normalization correction; requires a positive `n_h` (raises in `OperationProperties.__post_init__` otherwise) |
| `adjacency` | `AdjacencyList` \| `CompactHashMap` \| `None` | `None` | neighbor list, grid traversal, or build-a-grid-on-the-fly |
| `referenceParticles` | `ParticleState` | `queryParticles` | a separate reference particle set (positions/supports/masses) |

**Output:** a `torch.Tensor` of shape `[N]`.

## JVP / HVP

Density has no value input, so **no value JVP exists** — passing a value
tangent to `warpOperationJVP` for Density raises
`NotImplementedError`.

**Geometry JVP** — `warpOperationJVP(..., queryTangentState=...,
referenceTangentState=...)` with a `ParticleTangentState` (positions,
supports, masses) computes the exact total derivative:

$$
d\rho_i \;=\; \sum_{j}\left(
   d m_j\, W_{ij}
 + m_j\, dW_{ij}
\right),
$$

where $dW_{ij}$ is the kernel JVP `sphKernelJVP`
(`kernels/kernelJVP.py`): chain rule through
$x_{ij} = x_i - x_j$ and $h_{ij} = \mathrm{computePairwiseSupport}(h_i,
h_j, \text{mode})$, built from the already-validated
`sphKernel_` / `sphGradient_` / `sphKernelDkDh_` building blocks.
Only the **reference-side mass** tangent appears: $\rho_i$ never reads
its own mass, so a query-side mass tangent has no formula
(`warpOperationJVP` raises if you try). Omitted tangent fields are
treated as zero.

**Hessian-vector product** — `warpOperationHVP(...,
tangentQueryPositions=v)` computes $\mathrm{Hess}(\rho_i)\, v$, the only
second-order entry point in the library. It differentiates the position
tangent of the JVP once more in the same direction; the self-pair
($r_{ij}=0$) term is exact rather than a limit: at the peak the second
derivative is the kernel's own curvature there, finite and physically
meaningful (the identity is pinned by
`test_forward_mode_tier2_density_hvp_self_pair.py`). It requires a
torch-facing `AdjacencyList` (`.i`/`.j` pairs); grid traversal is not
implemented for the HVP.

## Tests

- `tests/operations/test_operations_core.py` — density is positive and
  finite on the lattice case; basic operator smoke tests.
- `tests/operations/test_forward_mode_geometry_jvp_density.py` — the
  geometry JVP against the reverse-mode Jacobian (the Tier-2 gate for
  Density).
- `tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py` —
  the HVP's self-pair Hessian identity.
- `tests/operations/test_grid_modes.py` — neighbor-list vs
  `CompactHashMap` traversal give identical results.
- `tests/operations/test_operations_consistency.py` — cross-operator
  identities (e.g. against interpolated linear fields).

## See also

[Gradient](gradient) · [Interpolate](interpolate) ·
[Kernels](../kernels) · [Autodiff machinery](../autograd)
