# Laplacian

SPH Laplacian of a field, in four estimator schemes. Kernel:
`coreOperations/wp_laplacian.py` (`computeSPHLaplacianTensor_Kernel`);
JVP `coreOperations/wp_laplacianJVP.py` (one function per scheme plus a
dispatcher).

## Equation

All four schemes share one per-pair quantity, `q_ij`, which depends on
`gradientMode` only (never on `laplacianMode`):

$$
q_{ij} =
\begin{cases}
(f_j - f_i)\, V_j & \text{Naive, Difference, Summation} \\[4pt]
(f_j - f_i)\, \dfrac{m_j\,\rho_i}{\rho_j^2} & \text{Symmetric}
\end{cases}
$$

(with the usual apparent-volume / grad-h substitutions). The schemes
then combine $q_{ij}$ differently:

| Scheme | Estimator |
|---|---|
| `Naive` (1) | $\nabla^2 f_i = \sum_j q_{ij}\, \nabla^2 W_{ij}$ — the analytic second-derivative-of-$r$ kernel Laplacian `sphKernelLaplacian`: $\nabla^2 W = s\,k_2 + t\,k_1$ with $k_1 = C_d h^{-(d+1)} k'(q)$, $k_2 = C_d h^{-(d+2)} k''(q)$, $s = r^2/r_{\varepsilon}^2$, $t = -r^2/r_{\varepsilon}^3 + d/r_{\varepsilon}$, $r_{\varepsilon} = r + \varepsilon h$ (and $0$ for $q < \varepsilon$ or $q > 1$). Never reads the kernel gradient, so no correction can affect it. |
| `Brookshaw` (2) | $\nabla^2 f_i = \sum_j -2\, q_{ij}\, \dfrac{\nabla W_{ij}\cdot \hat{n}_{ij}}{D_{ij}}$, $\quad D_{ij} = r_{ij} + 10^{-8} h_{ij},\;\; \hat n_{ij} = x_{ij}/D_{ij}$ |
| `Dot` (3) | DJ Price SPH/MHD eq. 96 (`computeLaplacianDot2`): the field is processed in blocks of `dim` components; each block is projected against $\hat n_{ij}$ and combined with the same $P_{ij} = (\nabla W_{ij}\cdot \hat n_{ij})/D_{ij}$ Brookshaw uses. Requires the flattened field size to be a multiple of `dim` — scalar fields in $>1$D are rejected with a `ValueError`. |
| `Default` (4) | `computeDotLaplacian`: $\nabla^2 f_i[\beta] = \sum_j -2\, q_{ij}[\beta]\; \hat n_{ij}^{(2)}\cdot \nabla W_{ij}$ with $\hat n_{ij}^{(2)} = \hat n_{ij}/D_{ij}^{(2)}$, $D_{ij}^{(2)} = r_{ij} + 10^{-12} h_{ij}$ — Brookshaw's exact $-2qP$ shape one quotient-rule level deeper (a second, tighter regularized distance). |

**Consistency constraint** (documented in `wp_laplacian.py`): for the
$\sum_j K_{ij} q_{ij}$ family to be a consistent estimator, $q_{ij}$
must vanish for spatially constant $f$ — otherwise the sum picks up an
uncancelled $O(1/h^2)$ residual from the kernel's own second-moment
scaling that grows without bound as resolution increases.
$(f_j - f_i)V_j$ satisfies this by construction; reusing the
Gradient/Divergence/Curl coefficient tables unmodified would inherit
the divergent residual, which is why the Laplacian's `q_ij` is its own
table (the Symmetric row above is the self-term-subtracted,
density-weighted variant of the same family).

**Output:** same shape as the input field (scalar $\to$ scalar, vector
$\to$ vector).

## Public API

```python
warpOperation(
    queryParticles, operationProperties, domain,
    queryValues=f, referenceValues=f,
    queryVolumes=None, referenceVolumes=None,
    crkState=None, gradHState=None, renormalizationState=None,
    adjacency=None,
)
```

with `operation == WarpOperation.Laplacian`.

| Option | Default | Effect |
|---|---|---|
| `laplacianMode` | `Brookshaw` | estimator scheme (table above) |
| `gradientMode` | `Naive` | selects only `q_ij` |
| `positiveDivergence` | `False` | adds the `positiveDotProduct` extra term to every scheme's contribution |
| `supportMode` / `operationMode` / corrections / calibration | — | as in [Gradient](gradient#public-api); CRK/renormalization act on the same kernel gradient Brookshaw/Dot/Default consume |

## JVP / HVP

**Value JVP** — relaunch-on-tangents, as usual.

**Geometry JVP** — `warpOperationJVP` dispatches
`computeSPHLaplacianGeometryJVP`, which routes by `laplacianMode` to one
of four hand-derived kernels sharing a
`_laplacianGeometryChainJVP` building block (the
$(G, dG, n_{ij}, dn_{ij}, D_{ij}, dD_{ij}, r_{ij}, dr_{ij}, h_{ij},
dh_{ij})$ regularized-distance chain on top of
`sphKernelGradientJVP`):

- **Brookshaw**: $L = \sum_j -2 q_{ij} P_{ij}$,
  $dL = \sum_j -2\,(dq_{ij} P_{ij} + q_{ij}\, dP_{ij})$ — ordinary
  product/quotient rule; generalized to scalar *and* vector fields
  (2026-08-25, after a vector-field diffusion consumer hit the
  `scalar_t`-only boundary).
- **Naive**: $L = \sum_j q_{ij} L^{W}_{ij}$ with the analytic kernel
  Laplacian, $dL = \sum_j (dq_{ij} L^{W}_{ij} + q_{ij}\, dL^{W}_{ij})$
  using `sphKernelLaplacianJVP` (built from the
  $k_1\to k_2\to k_3$ derivative ladder, cross-checked against the
  $\varepsilon\to0$ textbook form).
- **Dot**: same shared chain plus the per-`dim`-block projection
  $\mathrm{proj}_b = q_{ij}[b]\cdot \hat n_{ij}$ and its tangent; the
  projection reduction lives in its own `@wp.func` returning
  `(proj, dproj)` — Warp's automatic reverse-mode adjoint is wrong
  when a loop-accumulated local is consumed by a later nonlinear op in
  the same function body (found and fixed 2026-08-20, pinned by
  gradcheck).
- **Default**: one quotient-rule level deeper
  ($dn_{ij}^{(2)}/dD_{ij}^{(2)}$), no block structure.

Scope restrictions:

- **CRK / renormalization tangents**: supported for Brookshaw, Dot and
  Default (Dot/Default landed in the 2026-08-21 follow-up); **Naive is
  permanently excluded** — its estimator never reads the corrected
  kernel gradient, so there is no formula for either correction to
  affect. Omitted tangent states hold the correction frozen.
- `positiveDivergence` is rejected (its extra term is in no derived
  formula).
- `gradHState` unsupported; query-mass tangents rejected; the
  value-tangent term is fused into the same loop.

Gated per scheme by
`tests/operations/test_forward_mode_geometry_jvp_laplacian_naive.py`,
`..._brookshaw.py`, `..._dot.py`, `..._default.py` (gradcheck +
finite-difference references; the Brookshaw/Naive files also carry the
2026-08-25 vector-field generalization tests).

No HVP exists for the Laplacian.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness
  (including the Dot-scheme scalar-field `ValueError`).
- `tests/operations/test_operations_consistency.py` — scheme
  comparison on linear/quadratic fields.
- `tests/operations/test_forward_mode_value_jvp.py` — value JVP.
- `tests/operations/test_forward_mode_geometry_jvp_laplacian_*.py` —
  the four geometry JVPs, with and without CRK/renorm tangents.
- `tests/operations/test_grid_modes.py` — adjacency vs grid parity.

## See also

[Gradient](gradient) · [Kernels](../kernels#the-naive-kernel-laplacian) ·
[CRK overview](../crk/crk-overview)
