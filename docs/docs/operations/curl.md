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
  $(\nabla W_{ij}\times\mathbf{c})$ per field block. (An earlier
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
  \left(G_x\, (df_i A_{ij} + df_j B_{ij})_y - G_y\, (df_i A_{ij} + df_j B_{ij})_x\right)}_{\text{value-tangent term, fused}},
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


## Native derivative contract

Native curl supports primal, first-order backward/VJP and JVP for all 15
kernels, six support enums, operation directions and four coefficient schemes
in FP32/FP64 and 1D–3D. Products include geometry/bounds, fields, reference
mass/density or supplied volumes, query CRK A/B/gradA/gradB, query/reference
omega and renormalization. Unread tensor roles retain zero products, and
query-volume fallback is preserved for unequal populations.

Packed fields retain the existing flattened reference layout. In 3D, the
operator crosses three groups of `floor(field_elements / 3)` components and
leaves any remainder zero; scalar/short fields therefore produce zero outputs.
In 2D, the public output drops the final field axis (with scalar output stored
as `[N,1]`), and the reference loop fills `floor(output_elements / 2) + 1`
components from two adjacent groups. Remaining output components stay zero.
This historical packed-field convention is preserved, rather than reinterpreted.
Layouts whose reference loop would read beyond available fields are rejected
before launch. Bounded compiled blocks cover all defined nonempty layouts.

The 1D primal and JVP are zero, while taped reverse retains the reference's
singular geometry nonfinites (including viscosity self-pairs). Geometry products
use Warp's custom regularized gradient Hessian and separate reverse/JVP
max-support ties. Raw paths evaluate canonical vector-gradient components and reverse the field
coefficient before its scalar mass/density quotient. Cross adjoints use explicitly
rounded device products and CPU scalar-product reductions; otherwise tiny-pair
Hessians and singular weights amplify cancellation residuals. JVPs preserve the
reference coefficient/geometry/value contribution ordering.

Torch reverse uses deterministic query CSR/reference CSC ownership without
floating atomics. Noncontiguous inputs, aliases, sparse/duplicate/empty CSR,
sole omega dependencies, native search/grid rebuilding, calibration, topology
lifetime and current-stream fixed-CSR capture/replay are checked. Standalone C++
retains its uncorrected scalar/vector/matrix adjoint ABI and scatter policy;
it has no correction/JVP API.

The reference reverse is detached and first-order, with functional/batched
transforms and row-chunk AD unsupported. Warp's explicit geometry curl JVP is
2D-vector only and excludes grad-h. Native classic duals and explicit JVP also
cover the defined 1D/3D and packed layouts; classic duals carry omega and bound
tangents. The explicit API has no omega-tangent argument. Three-dimensional
boundary extensions are checked against composed supported scalar Warp-gradient
products, without claiming a public Warp 3D-curl geometry JVP.

Maintained checks: `tests/native/torch/test_curl_contract.py`,
`tests/parity/test_curl_contract.py`, standalone all-kernel products and the
paired frontend's accepted Monaghan/Balsara Gresho steps. Accepted simulation
steps qualify primal only; independent three-step feedback qualifies toy AD.
