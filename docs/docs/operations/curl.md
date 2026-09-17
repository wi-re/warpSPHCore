# Curl

SPH curl of a vector field. Kernel:
`coreOperations/wp_curl.py` (`computeSPHCurlTensor_Kernel`); JVP
`coreOperations/wp_curlJVP.py`.

## Equation

$$
(\nabla\times\mathbf{f})_i \;=\; \sum_{j \in \mathcal{N}(i)}
   \mathrm{curlProduct}(\mathrm{coeff}_{ij},\, \nabla W_{ij}),
$$

with $\mathrm{coeff}_{ij}$ from the same
[Gradient](gradient#equation) coefficient table (Naive / Symmetric /
Difference / Summation, apparent volume $V_j$, grad-h rescaling) and
the same CRK-then-renormalized kernel gradient $G_{ij}$.
`curlProduct` (`math/wp_cross.py`) applies the per-dimension curl
operator to the coefficient:

- **2D** — scalar output, the $z$-component of the 3D curl:
  $$(\nabla\times\mathbf{f})_i = \sum_j
     \left(G_{ij,x}\,\mathrm{coeff}_{ij,y}
          - G_{ij,y}\,\mathrm{coeff}_{ij,x}\right),$$
- **3D** — vector output, the standard right-hand-rule cross product
  $(\mathbf{c}\times\nabla W_{ij})$ per field block. (An earlier
  implementation negated the 3D result; the negation was confirmed to
  be a bug and removed — see `math/wp_cross.py`'s own note and
  `warpier_core.md`),
- **1D** — the curl is identically zero; the output is a scalar.

**Output shape:** 2D: `[N, 1]` (a vector field collapses to the
scalar $z$-component); 3D: `inputShape` (a vector field stays a vector);
1D: `[N]`.

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    queryValues=f, referenceValues=f,   # required, vector fields
    queryVolumes=None, referenceVolumes=None,
    crkState=None, gradHState=None, renormalizationState=None,
    adjacency=None,
)
```

with `operation == WarpOperation.Curl`.

| Option | Default | Effect |
|---|---|---|
| `gradientMode` | `Naive` | coefficient scheme (same table as Gradient) |
| `supportMode` / `operationMode` / corrections / calibration | — | as in [Gradient](gradient#public-api) |

There is no `consistentDivergence` or dot-mode analogue for Curl.

## JVP / HVP

**Value JVP** — relaunch-on-tangents identity, as usual
(`tests/operations/test_forward_mode_value_jvp.py`).

**Geometry JVP — 2D only.** The product rule expanded through the
2D cross:

$$
d(\nabla\times\mathbf{f})_i = \sum_j
\Big[
  G_x\, d\mathrm{coeff}_y - G_y\, d\mathrm{coeff}_x
+ dG_x\, \mathrm{coeff}_y - dG_y\, \mathrm{coeff}_x
\Big]
+ \underbrace{\sum_j
  \left(G_x\, v\mathrm{coeff}_y - G_y\, v\mathrm{coeff}_x\right)}_{\text{value-tangent term, fused}},
$$

with the same $A/B$ weights and $G/dG$ chain as
[Gradient](gradient#jvp--hvp). `warpOperationJVP` rejects
`domain.dim != 2` for Curl (1D and 3D are both undecided by the
underlying derivation spike); the same CRK/renorm-tangent,
grad-h-unsupported, and query-mass-rejected rules as Gradient apply.
The kernel produces a scalar per query particle internally; the
torch-level entry point unsqueezes to `[nQuery, 1]`.

Gated by `tests/operations/test_forward_mode_geometry_jvp_curl.py`.

No HVP exists for Curl.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness.
- `tests/operations/test_forward_mode_value_jvp.py` — value-JVP
  identity.
- `tests/operations/test_forward_mode_geometry_jvp_curl.py` — 2D
  geometry JVP vs reverse-mode Jacobian, including CRK/renorm
  correction tangents.
- `tests/operations/test_grid_modes.py` — adjacency vs grid traversal
  parity.

## See also

[Gradient](gradient) · [Divergence](divergence) · [Laplacian](laplacian)
