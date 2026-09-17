# Lattice normalization calibration

A correction to the SPH kernel's own normalization — unrelated to
[gradient renormalization](renorm) beyond both being, in a loose sense,
"normalizations" around the kernel. `OperationProperties` carries
`calibrateNormalization: bool = False` and `n_h: Optional[float] =
None`.

## Why: the quadrature offset

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

## The math, three ways — `util/latticeDensity.py`

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

## Wendland: no root in $h$

`latticeDensityIsStrictlyAbove1(kernel)` — the Wendland2/4/6 functions
are positive definite by construction, so $\widehat W > 0$ everywhere
and a perfect lattice reads *over* $\rho_0$ at **any** support radius:
no $h$ solves $L = 1$. A Newton/bisection search on $h$ cannot
converge — **the mass (or an explicit kernel correction like this one)
is the only lever**. The B-splines are not positive definite in
2D/3D, $L$ crosses 1, and roots in $h$ do exist.

## How it is wired

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

- The lattice math itself (`util/latticeDensity.py`) is exercised in
  the warpSPH repo, not by this test suite.

## See also

[Renormalization](renorm) · [Kernel library — where the scaling is
applied](kernels#pairwise-evaluation-and-the-support-schemes) ·
[Autodiff machinery — where it's resolved](autograd#the-36-slot-flat-layout)
