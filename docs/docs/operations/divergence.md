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
- **dot mode** (`divergenceDotMode=True`): for a field `[N, d, ...]`,
  the first field axis is spatial. The contribution contracts
  $\sum_\alpha \mathrm{coeff}_{ij}[\alpha, \beta]\; \partial_\alpha
  W_{ij}$ for each remaining component $\beta$. A matrix yields `[N, d]`.

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
| `divergenceDotMode` | `False` | contraction over the first field axis (remaining axes are preserved) |
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


## Native derivative contract

Native divergence supports primal, first-order backward/VJP and JVP for all
15 kernels, six support enums, directions and four coefficient schemes in
FP32/FP64 and 1D–3D. Both contraction modes preserve arbitrary nonempty packed
field axes, with the contracted axis required to have spatial extent D.
Scalar fields are supported in 1D. Compiled scalar/vector/matrix blocks bound
native per-row storage.

Products include fields, positions/supports, bounds, mass/density or supplied
volumes, query CRK A/B/gradA/gradB, both grad-h omega roles and renormalization.
The consistency option uses `mass_j / density_i` directly when no volume is
supplied, so an unread zero reference density cannot introduce a spurious
singularity. With supplied volumes it uses `volume_j * density_j / density_i`.
The symmetric scheme retains its ordinary density weighting. Query-volume
fallback and unused roles preserve their reference semantics.

Torch reverse uses deterministic query CSR/reference CSC ownership without
floating atomics. Noncontiguous tensors, aliases, sparse/duplicate/empty CSR,
native search, grid rebuilding, calibration, topology lifetime, current streams
and fixed-CSR capture/replay are covered. Standalone C++ retains its uncorrected
scalar/vector/matrix adjoint ABI and scatter policy; it has no JVP/correction API.

Geometry products preserve Warp's custom regularized gradient Hessian at
self/tiny pairs, singular nonfinites and separate reverse/JVP max-support ties.
The reference reverse remains detached and first-order; functional/batched
transforms and row-chunk AD remain unsupported. Native classic duals and its
explicit JVP additionally handle packed contractions, consistency, and omega or
bound products as permitted by their APIs. The explicit API has no omega-tangent
argument. Warp's explicit geometry JVP remains limited to ordinary vector
contraction without consistency/grad-h; numerical comparisons respect that scope.

Maintained checks are `tests/native/torch/test_divergence_contract.py`,
`tests/parity/test_divergence_contract.py`, standalone all-kernel products and
the paired frontend's accepted Monaghan steps. Accepted steps qualify primal
simulation use; three-step feedback qualifies toy AD.
