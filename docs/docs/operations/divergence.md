# Divergence

SPH divergence of a field. Kernel:
`coreOperations/wp_divergence.py` (`computeSPHDivergenceTensor_Kernel`);
JVP `coreOperations/wp_divergenceJVP.py`.

## Equation

$$
\nabla\!\cdot\! \mathbf{f}_i \;=\; \sum_{j \in \mathcal{N}(i)}
   \mathrm{divergenceProduct}(\mathrm{coeff}_{ij},\, \nabla W_{ij}),
$$

where $\mathrm{coeff}_{ij}$ is the **same coefficient table as
[Gradient](gradient)** (Naive / Symmetric / Difference / Summation,
with the same apparent-volume $V_j$ and grad-h rescaling), and
$\nabla W_{ij}$ is the same CRK-then-renormalized kernel gradient.
Only the combination with the gradient differs:

- **standard mode** (default): for a field of shape
  `[N, ..., d]` the last axis is the spatial one and the contribution
  is the component sum
  $\sum_{\alpha=1}^{d} \mathrm{coeff}_{ij}[\alpha]\;
   \partial_\alpha W_{ij}$, contracted over the spatial axis —
  a vector field `[N, d]` yields a scalar output `[N]`;
- **dot mode** (`divergenceDotMode=True`): the field is a matrix
  `[N, d, d]` and the contraction is over the *last* index only,
  $\sum_\alpha \mathrm{coeff}_{ij}[\beta, \alpha]\; \partial_\alpha
  W_{ij}$ for each remaining index $\beta$ — a tensor field yields
  `[N, d]`.

**Output shape:** `inputShape[:-1]` in standard mode,
`inputShape[1:]` in dot mode (see `_computeSPHDivergence_stateBackend`).

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    queryValues=f, referenceValues=f,      # required
    consistentDivergence=False,
    queryVolumes=None, referenceVolumes=None,
    crkState=None, gradHState=None, renormalizationState=None,
    adjacency=None,
)
```

with `operation == WarpOperation.Divergence`.

| Option | Default | Effect |
|---|---|---|
| `gradientMode` | `Naive` | coefficient scheme (same table as Gradient) |
| `divergenceDotMode` | `False` | matrix-field contraction over the last index (output keeps one axis) |
| `consistentDivergence` | `False` | scales the apparent volume by the density ratio: $V_j \leftarrow V_j\,\rho_j/\rho_i$ (or $V_j \leftarrow m_j/\rho_i$ without supplied volumes) — the "consistent" form of $\nabla\!\cdot\!(\rho^{-1}\nabla f)$-type bookkeeping |
| `supportMode` / `operationMode` / corrections / calibration | — | as in [Gradient](gradient#public-api) |

## JVP / HVP

**Value JVP** — same relaunch-on-tangents identity as every
value-having operator (`tests/operations/test_forward_mode_value_jvp.py`).

**Geometry JVP** — the product rule with vector-valued coefficients:

$$
d(\nabla\!\cdot\!\mathbf{f})_i \;=\; \sum_j
\Big[
  \mathrm{dcoeff}_{ij}\cdot G_{ij}
+ \mathrm{coeff}_{ij}\cdot dG_{ij}
\Big]
\;+\; \sum_j (df_i A_{ij} + df_j B_{ij})\cdot G_{ij}\;\text{(fused)},
$$

with the same $A/B/dA/dB$ weights as
[Gradient](gradient#jvp--hvp) (`_jvpCommon.gradientWeightsJVP`) and
$G/dG$ from `computeKernelGradientCRKJVP` followed by the
renormalization product rule $dG = dL_i G + L_i\, dG$.

Scope restrictions (enforced centrally in `operations.py`):

- `divergenceDotMode` and `consistentDivergence` are **not** in the
  derived math — the geometry JVP rejects both.
- CRK / renormalization tangents: same rules as Gradient (including
  simultaneous use); `gradHState` unsupported; query-mass tangents
  rejected.

Gated by `tests/operations/test_forward_mode_geometry_jvp_divergence.py`.

No HVP exists for Divergence.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness.
- `tests/operations/test_operations_consistency.py` — trace of the
  Difference-scheme Gradient equals the Difference-scheme Divergence
  (interior mask), the key cross-operator identity.
- `tests/operations/test_forward_mode_value_jvp.py`,
  `tests/operations/test_forward_mode_geometry_jvp_divergence.py`,
  `tests/operations/test_grid_modes.py`.

## See also

[Gradient](gradient) · [Curl](curl) · [Laplacian](laplacian)
