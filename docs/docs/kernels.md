# Kernel library

All kernels are evaluated in **compact form**:

$$
W(\mathbf{r}, h) \;=\; C_d\, h^{-d}\, k(q),
\qquad q = \frac{r}{h} \in [0, 1],
$$

with every kernel supported on $q \le 1$ (i.e. a support radius of
$1\cdot h$, regardless of the classical convention some splines use).
The smoothstep-style `cpow_warp(x, p) = \max(x, 0)^p` convention
implements the knot terms. Derivatives are stored as the compact
ladder $k, k', k'', k'''$ and re-scaled with powers of $h$ at the
evaluation level, so a kernel "family" is one module in
`kernels/kernelFunctions/` exposing `*_k`, `*_dkdq`, `*_d2kdq2`,
`*_d3kdq3`, `*_C_d`, `*_kernelScale`, `*_packingRatio`.

## The kernels

Select with `KernelFunctions.<Name>` in `OperationProperties`.

| Kernel | Compact form $k(q)$ (per dim where it differs) | $C_d$ (1D / 2D / 3D) | Smoothness |
|---|---|---|---|
| `Poly6` | $(1 - q^2)^3$ | $35/32$ / $4/\pi$ / $315/(64\pi)$ | $C^2$ |
| `Spiky` | $(1-q)^3$ | $2$ / $10/\pi$ / $15/\pi$ | $C^1$ |
| `CubicSpline` | $(1-q)^3 - 4(0.5-q)^3$ | $8/3$ / $80/(7\pi)$ / $16/\pi$ | $C^2$ |
| `QuarticSpline` | $(1-q)^4 - 5(0.6-q)^4 + 10(0.2-q)^4$ | $3125/768$ / $46875/(2398\pi)$ / $15625/(512\pi)$ | $C^3$ |
| `QuinticSpline` | $(1-q)^5 - 6(\tfrac{2}{3}-q)^5 + 15(\tfrac{1}{3}-q)^5$ | $243/40$ / $15309/(478\pi)$ / $2187/(40\pi)$ | $C^4$ |
| `B7` | $56(q-0.25)^7 - 28(q-0.5)^7 + 8(q-0.75)^7 - (q-1)^7$ | $4096/315$ / $589824/(7435\pi)$ / $16384/(105\pi)$ | $C^6$ |
| `Wendland2` | 1D: $(1-q)^3(1+3q)$; d≥2: $(1-q)^4(1+4q)$ | $5/4$ / $7/\pi$ / $21/(2\pi)$ | $C^2$, positive definite |
| `Wendland4` | 1D: $(1-q)^5(1+5q+8q^2)$; d≥2: $(1-q)^6(1+6q+\tfrac{35}{3}q^2)$ | $3/2$ / $9/\pi$ / $495/(32\pi)$ | $C^4$, positive definite |
| `Wendland6` | 1D: $(1-q)^7(1+7q+19q^2+21q^3)$; d≥2: $(1-q)^8(1+8q+25q^2+32q^3)$ | $55/32$ / $78/(7\pi)$ / $1365/(64\pi)$ | $C^6$, positive definite |
| `ViscosityKernel` | $-\tfrac12 q^3 + q^2 + \tfrac{1}{2q} - 1$ (XSPH viscosity) | $15/(2\pi)$ / $10/(9\pi)$ / $15/(2\pi)$ | — |
| `AdhesionKernel` | $(-4q^2 + 6q - 2)^{1/4}$ for $q > 0.5$, else $0$ | $0.007$ (all dims) | — |
| `CohesionKernel` | $2(1-q)^3 q^3 - 1/64$ for $q < 0.5$; $(1-q)^3 q^3$ else | $32/\pi$ (all dims) | — |

The Wendland family is the default in the notebooks and tests
(`KernelFunctions.Wendland2`); it is the family the
[lattice-normalization calibration](renorm#lattice-normalization-calibration)
has a closed-form analysis for (positive definite $\Rightarrow$
$L(n_h) > 1$ strictly, no root in $h$).

### Packing and support properties

Each kernel module also provides (Dehnen & Aly, *Improving convergence
in SPH simulations*) the properties `properties.py` exposes:

- `sphKernelScale(kernel, dim)` — the $h$-to-spacing scale factor
  (the support in particle-spacings for a well-packed lattice);
- `sphKernelN_H(kernel, dim)` — the expected neighbor count
  $N_H = \mathrm{fac}_d\, \mathrm{pack}^{d}\,
  \mathrm{scale}^{d}$ with $\mathrm{fac}_1 = 2$,
  $\mathrm{fac}_2 = \pi$, $\mathrm{fac}_3 = \tfrac{4}{3}\pi$;
- `sphKernel_xi(kernel, dim)` — packing ratio × scale, the support in
  spacing units.

These are *properties* of the kernel, not part of the evaluation: the
lattice-normalization factor is deliberately **not** applied to them.

## Pairwise evaluation and the support schemes

The pairwise functions take both particles' supports $(h_i, h_j)$ and
resolve the effective support with `computePairwiseSupport`
(`util/support.py`):

| `SupportScheme` | value | $W_{ij}$ | $\nabla W_{ij}$ |
|---|---|---|---|
| `Gather` | 11 | $W(x_{ij}, h_i)$ | $\nabla W(x_{ij}, h_i)$ |
| `Scatter` | 12 | $W(x_{ij}, h_j)$ | $\nabla W(x_{ij}, h_j)$ |
| `MeanSymmetric` | 13 | $W(x_{ij}, (h_i+h_j)/2)$ | $\nabla W(x_{ij}, (h_i+h_j)/2)$ |
| `KernelMeanSymmetric` | 14 | $\tfrac12\big(W(x_{ij},h_i) + W(x_{ij},h_j)\big)$ | $\tfrac12\big(\nabla W(x_{ij},h_i) + \nabla W(x_{ij},h_j)\big)$ |
| `SuperSymmetric` | 15 | $\tfrac12\big(W(x_{ij},h_i) + W(x_{ij},h_j)\big)$ | $\tfrac12\big(\nabla W(x_{ij},h_i) - \nabla W(x_{ji},h_j)\big)$ — the CRK-SPH formulation |
| `PartialSymmetric` | 16 | (PESPH) $f_i$-weighted $h_i$ + $f_j$-weighted $h_j$ kernel combination | same |

$KernelMeanSymmetric$ and $SuperSymmetric$ are provably identical for
the kernel *value* and gradient *at $q$*, but genuinely differ for the
[Naive kernel Laplacian](#the-naive-kernel-laplacian) (which is why
`sphKernelLaplacian` keeps a two-branch dispatch and the Laplacian JVP
mirrors it).

`x_{ij} = x_i - x_j` is always the periodic minimum-image distance
(`computeDistanceVec` with the domain's periodicity); tangents use the
plain difference $dx_i - dx_j$ (periodic-wrap discontinuity is out of
scope for the JVPs, per `warpier_adjoint.md` Tier 2.1).

### The building blocks

| Function | Computes | Notes |
|---|---|---|
| `sphKernel(x_i, x_j, h_i, h_j, kernelProperties, domain)` | $W_{ij}$ | + lattice factor |
| `sphKernelGradient(...)` | $\nabla_i W_{ij}$ | + lattice factor |
| `sphKernelHessian(...)` | $\nabla^2 W_{ij}$ | $\varepsilon$-regularized; the $q<\varepsilon$ branch gives the finite peak curvature |
| `sphKernelDkDh(...)` | $\partial W_{ij}/\partial h$ | $= -C_d h^{-(d+2)}\big(d\,h\,k + r\,k'\big)$ |
| `sphGradientDkDh(...)` | $\partial\nabla W_{ij}/\partial h$ | $= -C_d h^{-(d+2)}\big(q\,k'' + (d+1)k'\big)\,\hat x$ |
| `sphKernelLaplacian(...)` | the analytic kernel Laplacian (see below) | |

Every function above is multiplied by the lattice-normalization factor
`resolveNormalization(kernelState)` — `1.0` unless the
[calibration](renorm#lattice-normalization-calibration) is enabled, in
which case $1/L$. The scaling is applied at the public boundary of each
function (not at the 13 internal `eval_C_d` sites) because everything
is linear in $C_d$, so one scale cannot be forgotten for a derivative.

### The naive kernel Laplacian

`sphKernelLaplacian_` evaluates

$$
\nabla^2 W = s\,k_2 + t\,k_1,
\quad
k_1 = C_d h^{-(d+1)} k'(q),\;\; k_2 = C_d h^{-(d+2)} k''(q),
$$

$$
s = \frac{r^2}{r_\varepsilon^2},\qquad
t = -\frac{r^2}{r_\varepsilon^3} + \frac{d}{r_\varepsilon},
\qquad r_\varepsilon = r + \varepsilon h,
$$

returning $0$ for $q < \varepsilon$ or $q > 1$ — the estimator the
[Laplacian](operations/laplacian) `Naive` scheme consumes. Its
position-derivative `sphKernelLaplacianGradient_` is derived from the
$k_1 \to k_2 \to k_3$ ladder (cross-checked against the
$\varepsilon\to0$ closed form $k_3 - \frac{d-1}{r^2}k_1 +
\frac{d-1}{r}k_2$) and needs no self-pair branch: every term carries a
factor of the zero-at-origin direction vector.

### Custom adjoint of the kernel gradient

`adj_sphGradient_*` replaces Warp's automatic reverse-mode adjoint of
`sphGradient_` with a closed form built on `sphKernelHessian_` and
`sphGradientDkDh_`. The automatic composition is correct for $r > 0$
but wrong at the exact self-pair ($r=0$): the $\varepsilon$-regularized
Jacobian of `vectorNormalize_warp` blows up like $O(1/\varepsilon)$ and
gets multiplied by $k'(q=0) = 0$ — "huge × 0.0" silently collapses to
zero instead of the true finite limit (a removable
$0 \times \infty$ singularity). The closed form has an explicit
near-origin branch with the right $r\to0$ limit and is validated
independently at $r=0$ and $r>0$ (see `wp_densityHVP.py`'s docstring
and `kernel_sanity_native.py` sections 6/8).

## Kernel JVPs

`kernels/kernelJVP.py` provides the forward-mode building blocks the
operator JVPs are built from — each mirrors its primal function
branch-for-branch on `SupportScheme`:

| Function | Returns |
|---|---|
| `sphKernelJVP(x_i, x_j, h_i, h_j, dx_i, dx_j, dh_i, dh_j, ...)` | $(W_{ij}, dW_{ij})$ |
| `sphKernelGradientJVP(...)` | $(\nabla W_{ij}, d\nabla W_{ij})$ — chain rule through $H$ (the Hessian) and $\partial\nabla W/\partial h$ |
| `sphKernelLaplacianJVP(...)` | $(\nabla^2 W_{ij}, d\nabla^2 W_{ij})$ — built from `sphKernelLaplacianGradient_`/`sphKernelLaplacianDkDh_`; keeps the primal's two-branch dispatch (SuperSymmetric explicit, everything else via `computePairwiseSupport`'s max-fallback) |

$dW$ and friends are chain-rule through $x_{ij} = x_i - x_j$ and
$h_{ij} = \mathrm{computePairwiseSupport}(h_i, h_j, \mathrm{mode})$
(`computePairwiseSupportJVP` is an ordinary subgradient on the
$\max$ branch, exact away from the $h_i = h_j$ kink). All three apply
the lattice factor to both primal and tangent.

## Tests

- `tests/operations/test_operations_core.py` — kernel-driven operator
  smoke tests.
- `scripts/kernel_sanity_native.py` (gated by
  `tests/operations/test_gradcheck_scripts.py`) — the derivative
  ladder, the custom adjoint at $r=0$/$r>0$, and the Laplacian
  derivatives against `wp.Tape`.
- `scripts/gradcheck_*.py` — operator-level gradcheck at float64,
  which exercises every kernel × scheme combination.

## See also

[Support schemes in the operator pages](operations/gradient) ·
[CRK corrected kernel](crk/crk-overview#the-corrected-kernel-and-gradient) ·
[Renormalization & lattice calibration](renorm)
