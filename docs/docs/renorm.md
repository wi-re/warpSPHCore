# Renormalization

The [De Courcy](https://doi.org/10.1016/j.jcp.2024.113192)
gradient-renormalization matrix $\mathbf{L}_i$ that the
[Gradient](operations/gradient) operator (and the
[Divergence](operations/divergence)/[Curl](operations/curl)/
[Laplacian](operations/laplacian) operators in their
CRK-corrected configurations) apply to the *uncorrected* gradient
result:

$$
\mathbf{G}_i \;\longleftarrow\; \mathbf{L}_i \, \mathbf{G}_i,
\qquad
\mathbf{L}_i = \mathrm{pinv}\!\left(\mathbf{C}_i\right),
$$

where $\mathbf{C}_i$ is the [Covariance](operations/covariance) of the
kernel gradient field at $i$. $\mathbf{C}_i$ measures the linear
reproduction error of the neighbor set; inverting it (pseudo-inverse,
since free-surface particles give a rank-deficient matrix) makes the
corrected operator reproduce linear fields exactly. The order of
application is fixed: CRK correction first, then
$\mathbf{L}_i\mathbf{G}_i$ (see the
[Gradient](operations/gradient) page).

Module: `src/warpSPHCore/renorm.py`.

## `computeRenormalizationMatrices`

```python
computeRenormalizationMatrices(
    queryParticles, operationProperties, domain,
    queryVolumes=None, referenceVolumes=None,
    adjacency=None,                 # AdjacencyList | CompactHashMap | None
    referenceParticles=None,
    crkState=None, gradHState=None, renormalizationState=None,
    returnEigVals=True,
)  -> (C, eigVals, RenormalizationState)      # returnEigVals=True
   -> RenormalizationState                    # returnEigVals=False
```

The pipeline (the `_`-suffixed internal function):

### 1. Covariance — on a *copy* of the properties

The operation is forced to `WarpOperation.Covariance` via
`dataclasses.replace(operationProperties, ...)` — never written back
into the caller's object. The old code did
`operationProperties.operation = WarpOperation.Covariance` in place; a
caller that reused its properties object afterwards would silently get
a covariance launch where it asked for a gradient, and the `(N, D, D)`
result is plausible enough to go unnoticed. The bug was found by a
Tier-1 spike that reused one properties object across the renorm call
and the gradient call that consumed its output; it is pinned by
`tests/operations/test_renorm_no_caller_mutation.py`.

The covariance call passes `covarianceReturnNumNeighbors=True` so the
low-neighbor fallback (below) works under **all three traversal
inputs** — explicit `AdjacencyList`, explicit `CompactHashMap`, or
`adjacency=None`: the count is a per-particle output of the
[Covariance kernel](operations/covariance), computed identically under
either traversal mode, rather than read from
`adjacency.numNeighbors`, which only exists for a list.

### 2. Low-neighbor-count fallback — identity, branch-free

A particle with too few neighbors to trust its covariance (free-surface
fingers, isolated particles) gets the **identity** instead of its own
untrustworthy matrix, so the pseudo-inverse does not amplify noise:

$$
\mathbf{C}_i \;\longleftarrow\;
\begin{cases} \mathbf{I} & n_i < d + 2 \\ \mathbf{C}_i & \text{otherwise} \end{cases}
\qquad
\text{implemented as } \mathrm{torch.where}(n_i < d+2,\; \mathbf{I},\; \mathbf{C}_i)
$$

This is deliberately a single elementwise `torch.where`, not
`if torch.any(mask): C[mask] = eye` — that guard was **two host
readbacks per step** in a real run (the `torch.any` in the Python
`if`, plus the mask count read by the boolean-masked assignment, which
is itself a synchronizing op): the fast path paid a stall to *maybe*
avoid a stall. `where` has neither, needs no clone, and gives the same
values and the same gradients (a low-neighbor row is replaced by a
constant either way, so nothing flows back through it). See
`docs/regression/real_workload_bottleneck_audit.md` (Step H sync
census) and the pinning test
`tests/operations/test_no_host_sync.py::test_renormalization_does_not_read_back`.

The 2D pseudo-inverse re-checks the same cutoff internally (see below);
applying it here as well covers 3D.

### 3. Pseudo-inverse — `pinv_warp`, dispatched on dimension

`pinv/wrapper.py::pinv_warp(C, numNbrs)`:

| dim | backend |
|---|---|
| 1 | `pinv1x1` (trivial reciprocal) |
| 2 | `pinv2x2_warpBackend` — a Warp kernel with the **closed-form symmetric eigendecomposition** (the general 2×2-SVD formula this used to use is unstable for near-isotropic covariances). Eigenvalue cutoff: an eigenvalue is inverted only if it is finite in absolute value *and* above `rcond = 1e-6` times the dominant one, so anisotropic/thin neighborhoods don't produce huge inverted eigenvalues. Re-checks `num_nbrs < 4` (i.e. `< dim + 2`) → identity. |
| 3 | `torch.linalg.pinv(C, rtol=1e-6)` (the 3×3 Warp kernel exists, `pinv/wp_pinv3x3.py`, but is not wired in yet); `eigVals` from `torch.linalg.eigvals`, sorted by descending magnitude. |

**Outputs:** `C` `[N,D,D]` (the covariance, *after* the identity
fallback — the matrix the pseudo-inverse actually inverted), `eigVals`
`[N, D]`, and `RenormalizationState(renormalizationMatrices=L)`
`[N,D,D]`.

`computeRenormalizationMatrices` accepts `crkState`/`gradHState`/
`renormalizationState` and forwards them to its internal covariance
call — the renormalization matrix is then a correction *on top of* the
other corrections, which is the configuration the gradient JVP
scripts exercise.

## JVP

`computeRenormalizationMatricesJVP(queryParticles, operationProperties,
domain, queryTangentState, queryVolumes=None, referenceVolumes=None,
tangentReferenceVolumes=None, adjacency=None, referenceParticles=None,
referenceTangentState=None, returnEigVals=True)` →
`(C, eigVals, RenormalizationState, RenormalizationTangentState)` (or
the two states only if `returnEigVals=False`).

Given only *geometry* tangents (no pre-existing state), it returns
$d\mathbf{L}_i$ by the same two steps the primal applies to
$\mathbf{C}_i$:

1. **`dC`** from [`computeCovarianceGeometryJVP`](operations/covariance#jvp--hvp)
   (the raw, unmasked covariance tangent).
2. **The low-neighbor mask's exact-zero tangent**: the fallback
   replaces low-neighbor rows by the *constant* identity, so their
   tangent is exactly zero —
   `dC = torch.where(lowNbrMask, 0, dC_raw)`.
3. **The matrix-inverse derivative**:

$$
d\mathbf{L}_i = -\,\mathbf{L}_i\, d\mathbf{C}_i\, \mathbf{L}_i
$$

with $\mathbf{L}_i$ consumed from this function's own primal call
(same "consume an already-validated value" pattern the CRK JVP uses).
The neighbor count is recomputed via a second internal covariance call
(the primal function does not return it).

`crkState`/`gradHState` are **not** accepted here — the covariance JVP
kernel is deliberately CRK/grad-h-free, matching the "renorm alone
first" scoping (CRK+renorm tangents *simultaneously* through this path
is a documented fast follow-up). Validated to float64 round-off by
`scripts/spike_forward_mode_tier2_renorm.py`.

## Lattice normalization calibration

An orthogonal feature that shares this page because it is a
*normalization* of the kernel: `OperationProperties` carries
`calibrateNormalization: bool = False` and `n_h: Optional[float] =
None`.

### Why: the quadrature offset

The SPH density

$$
\rho_i = \sum_j m_j\, W(|x_{ij}|, h)
$$

is a **midpoint quadrature** of $\int W\, dV = 1$ sampled on the
particle lattice. The kernel normalizes its *integral*, not its
lattice *sum* — so a defect-free cubic lattice of spacing $s$ carrying
the nominal mass $m = \rho_0\, s^d$ does not read $\rho_0$; it reads
$\rho_0 \cdot L$, with

$$
L(n_h) \;=\; \frac{C_d}{n_h^{d}}\sum_{\mathbf{n} \in \mathbb{Z}^{d}}
k\!\left(\frac{|\mathbf{n}|}{n_h}\right),
\qquad n_h = \frac{h}{s}.
$$

The $s^d$ from the mass and the $h^{-d}$ from the kernel cancel
exactly, which is why $L$ depends on the *ratio* $n_h$ alone and never
on the resolution: the offset does not vanish as $s \to 0$, only as
$n_h \to \infty$. It is a resolution-independent systematic error,
separate from the O(h²) quadrature error.

Enabling `calibrateNormalization` scales **every kernel value and
derivative** (all ops, all schemes) by $1/L$, so the ideal lattice
reads $\rho_0$ exactly and every derivative is corrected by the same
factor (everything is linear in $C_d$). See the
[Kernel library](kernels) for where the scaling is applied and what it
is *not* applied to.

### The math, three ways — `util/latticeDensity.py`

`latticeDensity(kernel, n_h, dim, method='shells')` returns $L$ (cached
on `(kernel, n_h, dim, method)`); `latticeDensityFactor(...)` returns
$1/L$. Three evaluation methods:

**`shells` (default — exact, dependency-free).** $|\mathbf{n}|^2 = j$
is an integer, so the $d$-fold integer loop collapses to a single sum
over shells weighted by the multiplicity $r_d(j) = \#\{\mathbf{n} \in
\mathbb{Z}^d : |\mathbf{n}|^2 = j\}$:

$$
L(n_h) = \frac{C_d}{n_h^d}\sum_{j=0}^{\lfloor n_h^2\rfloor}
r_d(j)\; k\!\left(\frac{\sqrt{j}}{n_h}\right)
$$

In 2D the multiplicity is closed form — Jacobi's two-square theorem,
$r_2(j) = 4\sum_{e\mid j,\, e\ \text{odd}} (-1)^{(e-1)/2}$
(`jacobiR2`, asserted equal to the direct `shellCounts` reference).
Exact for any positive *real* $n_h$, at the cost of $O(n_h^2)$ kernel
evaluations instead of $O(n_h^d)$.

**`fourier` (exact identity — the one that explains the behaviour).**
Poisson summation turns the real-space lattice sum into a
reciprocal-space one: the offset *is* the kernel's own Fourier
transform sampled on the reciprocal lattice,

$$
L(n_h) = \sum_{\mathbf{m} \in \mathbb{Z}^d}
\widehat W(2\pi n_h\, |\mathbf{m}|)
\;=\; 1 + \sum_{j \ge 1} r_d(j)\,
\widehat W\!\left(2\pi n_h\sqrt{j}\right),
$$

with $\widehat W$ the $d$-dimensional radial transform of the
normalized kernel ($\widehat W(0) = 1$). This is pure aliasing, and it
makes two things immediate:

- $L - 1$ decays at the rate of the kernel's spectral tail;
- for a **positive-definite** kernel every term is positive (Bochner),
  so $L > 1$ *strictly*.

**`closed` (no summation at all — Wendland family only).** Only the
odd powers of $k$'s Taylor expansion at $q=0$ produce an algebraic tail
in $\widehat W$, and each shell sum $\sum_j r_d(j) j^{-\sigma}$ is an
Epstein zeta constant, so

$$
L(n_h) \sim 1 + \sum_{\text{m odd}}
\frac{A_m\, Z_d\!\left(\frac{d+m}{2}\right)}{(2\pi n_h)^{d+m}},
\qquad
A_m = c_m\, C_d\, (2\pi)^{d/2}\, 2^{(m+d)/2}\,
\frac{\Gamma\!\left(\frac{d+m}{2}\right)}{\Gamma(-m/2)},
$$

with $Z_1(s) = 2\zeta(2s)$, $Z_2(s) = 4\zeta(s)\beta(s)$ (Dirichlet
beta), and $A_m \in \mathbb{Z}$ for the Wendland kernels
(`WENDLAND_TAIL_COEFFICIENTS`). The residual is the *oscillatory* part
of $\widehat W$, one half-power below the term kept; measured over
$n_h \in [3, 6]$, the closed form recovers **94.6–99.8 %** of the
offset in 2D and **72.8–99.5 %** in 3D (1D much worse — Wendland2 1D
has a single odd Taylor coefficient and sits at a flat 66.7 %). Use
`shells` when you want the number and `closed` when you want the
scaling law.

`latticeDensityClosed` **raises `KeyError` for the B-splines** rather
than returning a number dominated by the oscillatory knot term it
omits: a spline's interior knots generate an oscillatory term that
decays *slower* than the algebraic tail, so a spline's offset changes
sign with $n_h$ (CubicSpline 2D reads 1.00344 at $n_h = 3$ but 0.99996
at $n_h = 4$) and has no smooth closed form. Use `shells` there.

### Wendland: no root in $h$

`latticeDensityIsStrictlyAbove1(kernel)` — the Wendland2/4/6 functions
are positive definite by construction, so $\widehat W > 0$ everywhere
and a perfect lattice reads *over* $\rho_0$ at **any** support radius:
no $h$ solves $L = 1$. A Newton/bisection search on $h$ cannot
converge — **the mass (or an explicit kernel correction like this one)
is the only lever**. The B-splines are not positive definite in
2D/3D, $L$ crosses 1, and roots in $h$ do exist.

### How it is wired

- **Opt-in, loud failure.** `OperationProperties` is a frozen
  dataclass whose `__post_init__` raises
  `calibrateNormalization=True requires a positive n_h` — the earliest
  point the invalid combination can be caught, covering every backend
  at once. Deliberately loud: silently treating a missing $n_h$ as
  "off" makes "calibration requested, never computed"
  indistinguishable from "calibration off".
- **Resolution.** `extractStateInfo` (in
  `autograd/arg_extract.py`) computes
  `latticeDensityFactor(kernel, n_h, dim)` (an `lru_cache` hit after
  the first call — negligible against everything else that function
  does per call) into `normalizationCoefficient`, and **raises** if
  the factor is not finite and $> 0$: anything failing there is a bad
  `(kernel, n_h, dim)` combination that would otherwise reach a kernel
  as a NaN. The flag and the coefficient are carried into the Warp
  `kernelState` struct; the boolean flag guards against Warp's
  zero-initialization of the scalar (a bare `0.0` coefficient would be
  indistinguishable from "off" without it).
- **Application point.** Every public kernel function
  (`sphKernel`, `sphKernelGradient`, …) multiplies by the factor at its
  boundary — see the [Kernel library](kernels). It is **not** applied
  to the packing properties `sphKernelScale`/`sphKernel_xi`/
  `sphKernelN_H` (those are properties of the kernel, not evaluations),
  and the [CRK density diagnostic](crk/crk-density) deliberately does
  not carry the flag (applying $1/L$ on top would double-correct).
- **Scope.** $n_h$ is the *nominal* support-to-spacing ratio the case
  was configured with, not a measured one, and the correction is
  meaningful only for a uniform-resolution lattice — with adaptive
  support $n_h$ is per particle and a single scalar cannot represent
  it.

The design document is `LATTICE_DENSITY_PLAN.md` in the sibling
warpSPH repo; the CLI over this module is
`scripts/lattice_density_offset.py` (there), and the consumer that
matters is `cases/weaklyCompressible.calibrateRestDensityMasses`
(warpSPH), which needs $L$ to separate the ideal-lattice quadrature
offset from the sampler's block-fit error — two independent problems
that both show up as "the initial state does not read $\rho_0$".
Performance records for the calibration work live in
`docs/regression/`.

## Tests

- `tests/operations/test_no_host_sync.py` — the branch-free low-neighbor
  fallback (readback counter on `renorm.py`) and
  `test_low_neighbour_fallback_still_replaces_those_rows` (identity,
  not zero, for `n_i < dim + 2`).
- `tests/operations/test_renorm_no_caller_mutation.py` — the
  properties-copy behavior.
- `scripts/gradcheck_renorm_native.py`,
  `scripts/gradcheck_renorm_uniform_grid_native.py` — forward-value
  parity + `torch.autograd.gradcheck` across all three traversal
  inputs (gated by `tests/operations/test_gradcheck_scripts.py`).
- `scripts/spike_forward_mode_tier2_renorm.py` — the JVP, float64.
- The lattice math itself (`util/latticeDensity.py`) is exercised in
  the warpSPH repo, not by this test suite.

## See also

[Covariance](operations/covariance) · [Gradient — where $\mathbf{L}$
is applied](operations/gradient) · [Kernel library — the scaling
point](kernels)
