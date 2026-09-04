"""The summation density of a *perfect* lattice, in closed form.

An SPH density `rho_i = sum_j m_j W(|x_ij|, h)` is a midpoint quadrature of
`int W dV = 1` sampled on the particle lattice. The kernel normalises its
*integral*, not its lattice *sum*, so a defect-free cubic lattice of spacing
`s` carrying the nominal mass `m = rho0 s^d` does not read `rho0`; it reads
`rho0 * L`, with

    L(n_h) = C_d / n_h^d * sum_{n in Z^d} k(|n| / n_h),        n_h = h / s.

The `s^d` from the mass and the `h^-d` from the kernel cancel exactly, which is
why `L` depends on the *ratio* `n_h` alone and never on the resolution: it does
not vanish as `s -> 0`, only as `n_h -> infinity`.

Three ways to evaluate it, all implemented here:

`shells` (default, exact, dependency-free)
    `|n|^2 = j` is an integer, so the `d`-fold integer loop collapses to a
    single sum over shells weighted by `r_d(j) = #{n in Z^d : |n|^2 = j}`:

        L(n_h) = C_d / n_h^d * sum_{j=0}^{floor(n_h^2)} r_d(j) k(sqrt(j) / n_h)

    In 2D the multiplicity itself is closed form -- Jacobi's two-square theorem
    `r_2(j) = 4 * sum_{e | j, e odd} (-1)^((e-1)/2)`, see `jacobiR2`. Exact for
    any positive *real* `n_h`, and it needs `O(n_h^2)` kernel evaluations
    instead of `O(n_h^d)`.

`fourier` (exact identity; the one that explains the behaviour)
    Poisson summation turns the real-space lattice sum into a reciprocal-space
    one: the offset *is* the kernel's own Fourier transform sampled on the
    reciprocal lattice,

        L(n_h) = sum_{m in Z^d} what(2 pi n_h |m|) = 1 + sum_{j>=1} r_d(j) what(2 pi n_h sqrt(j))

    with `what` the `d`-dimensional radial transform of the normalised kernel
    (`what(0) = 1`). This is pure aliasing, and it makes two things immediate:
    `L - 1` decays at the rate of the kernel's spectral tail, and for a
    positive-definite kernel every term is positive, so `L > 1` *strictly* --
    see `latticeDensityIsStrictlyAbove1`.

`closed` (no summation at all; Wendland family only)
    Only the odd powers of `k`'s Taylor expansion at `q = 0` produce an
    algebraic tail in `what`, and each shell sum `sum_j r_d(j) j^-sigma` is an
    Epstein zeta constant, so

        L(n_h) ~ 1 + sum_{m odd} A_m Z_d((d+m)/2) / (2 pi n_h)^(d+m),
        A_m = c_m C_d (2 pi)^(d/2) 2^(m+d/2) Gamma((d+m)/2) / Gamma(-m/2),

    with `Z_1(s) = 2 zeta(2s)`, `Z_2(s) = 4 zeta(s) beta(s)`. The `A_m` are
    integers (`WENDLAND_TAIL_COEFFICIENTS`). The residual is the *oscillatory*
    part of `what`, one half-power below the term kept, and how big a share of
    the offset that is depends strongly on the dimension. Measured over
    `n_h` in [3, 6]: **2D recovers 94.6-99.8 %** of the offset, 3D 72.8-99.5 %,
    and 1D only 65.9-92.9 % -- Wendland2 in 1D has a single odd Taylor
    coefficient and sits at a flat 66.7 %. Use `shells` when you want the
    number and this when you want the scaling law.

    Not available for the B-splines, and not by omission: their interior knots
    generate an oscillatory term that decays *slower* than the algebraic tail,
    so a spline's offset changes sign with `n_h` (CubicSpline 2D reads 1.00344
    at `n_h = 3` but 0.99996 at `n_h = 4`) and has no smooth closed form. Use
    `shells` there.

`scripts/lattice_density_offset.py` in warpSPH is the CLI over this, and
`cases/weaklyCompressible.calibrateRestDensityMasses` is the consumer that
matters: it needs `L` to separate the ideal-lattice quadrature offset from the
sampler's block-fit error, which are two independent problems that both show up
as "the initial state does not read rho0".
"""
from __future__ import annotations

import functools
import math
from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = [
    'shellCounts', 'jacobiR2',
    'latticeDensity', 'latticeDensityFactor',
    'latticeDensityShells', 'latticeDensityFourier', 'latticeDensityClosed',
    'kernelFourierTransform', 'epsteinZeta',
    'latticeDensityIsStrictlyAbove1', 'WENDLAND_TAIL_COEFFICIENTS',
]


def _kernelInt(kernel) -> int:
    """`KernelFunctions` member, its name, or a raw int -> the warp enum value."""
    if isinstance(kernel, (int, np.integer)):
        return int(kernel)
    from ..enumTypes import KernelFunctions
    if isinstance(kernel, str):
        return int(KernelFunctions[kernel].value)
    return int(kernel.value)


def _evaluators():
    """Lazy import: `kernels` imports `util.support`, so `util` must not pull
    `kernels` in at module scope.

    `scalar_t` is read through the module rather than bound at import so this
    follows a `configure(precision=...)` that ran after import. It has to be
    applied: a warp `@wp.func` called from Python only accepts a raw Python
    float when `scalar_t` is float32 -- under a float64 configuration
    `eval_k(0.25, 2, 0)` raises "no overload found".
    """
    from .. import type_config
    from ..kernels.eval_kernel import eval_k, eval_C_d
    scalar = type_config.scalar_t
    return (lambda q, dim, kern: float(eval_k(scalar(q), dim, kern)),
            lambda dim, kern: float(eval_C_d(dim, kern)))


# --------------------------------------------------------------- lattice shells
@functools.lru_cache(maxsize=32)
def shellCounts(dim: int, jmax: int) -> np.ndarray:
    """`r_d(j) = #{n in Z^dim : |n|^2 == j}` for `j = 0..jmax`, by direct count.

    `jacobiR2` is the closed form of the `dim == 2` case; this one covers every
    dimension and is the reference the two are asserted equal against.
    """
    N = int(math.floor(math.sqrt(jmax)))
    G = np.meshgrid(*([np.arange(-N, N + 1)] * dim), indexing='ij')
    n2 = np.square(np.stack(G, 0).astype(np.int64)).sum(0).ravel()
    counts = np.bincount(n2[n2 <= jmax], minlength=jmax + 1)
    counts.flags.writeable = False
    return counts


@functools.lru_cache(maxsize=32)
def jacobiR2(jmax: int) -> np.ndarray:
    """`r_2(j)` from Jacobi's two-square theorem, `4 * (d_1(j) - d_3(j))` over
    the odd divisors -- a closed form for the multiplicity of the 2D integer
    double sum, with no `meshgrid` and no `O(jmax)` memory in `dim`."""
    out = np.zeros(jmax + 1, dtype=np.int64)
    out[0] = 1
    for e in range(1, jmax + 1, 2):
        out[e::e] += 4 * (1 if e % 4 == 1 else -1)
    out.flags.writeable = False
    return out


# ------------------------------------------------ 1. exact, real-space shell sum
def latticeDensityShells(kernel, n_h: float, dim: int) -> float:
    """`L(n_h)`, exact for the infinite lattice. Any positive real `n_h`.

    Exact and not an approximation: every lattice point with `W != 0` has
    `|n|^2 <= n_h^2`, so the truncation at `floor(n_h^2)` misses nothing.
    """
    eval_k, eval_C_d = _evaluators()
    kern = _kernelInt(kernel)
    counts = shellCounts(dim, int(math.floor(n_h * n_h)))
    j = np.nonzero(counts)[0]
    k = np.array([eval_k(math.sqrt(jj) / n_h, dim, kern) for jj in j])
    return float(eval_C_d(dim, kern) / n_h ** dim * (counts[j] * k).sum())


# ------------------------------------ 2. exact, reciprocal space (Poisson sum)
def _besselJ0(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import j0
        return j0(x)
    except ImportError:                     # torch's is only ~1e-11 absolute
        import torch
        return torch.special.bessel_j0(torch.as_tensor(x, dtype=torch.float64)).numpy()


def kernelFourierTransform(kernel, dim: int, s, *,
                           knots: Sequence[float] = (0, .2, 1/3, .5, .6, 2/3, 1),
                           nGauss: int = 64) -> np.ndarray:
    """`what(s)` -- the `dim`-dimensional radial Fourier transform of the
    *normalised* kernel `W(r) = C_d h^-d k(r/h)`, at `|xi| h = s`. `what(0) = 1`.

    Elementary in 1D and 3D (a polynomial against cos / sin); only 2D needs a
    Bessel function::

        dim 1:  2 C_1        int_0^1 k(q) cos(s q) dq
        dim 2:  2 pi C_2     int_0^1 k(q) q J_0(s q) dq
        dim 3:  4 pi C_3 / s int_0^1 k(q) q sin(s q) dq

    `knots` are panel boundaries for the Gauss-Legendre rule; the defaults are
    the union of every B-spline breakpoint in the repo, so the quadrature never
    straddles a discontinuity in a derivative.
    """
    eval_k, eval_C_d = _evaluators()
    kern = _kernelInt(kernel)
    x, wg = np.polynomial.legendre.leggauss(nGauss)
    a, b = np.asarray(knots[:-1])[:, None], np.asarray(knots[1:])[:, None]
    q = ((b - a) / 2 * x + (a + b) / 2).ravel()
    f = ((b - a) / 2 * wg).ravel() * np.array([eval_k(qi, dim, kern) for qi in q])
    C_d = eval_C_d(dim, kern)
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))[:, None]
    if dim == 1:
        return 2.0 * C_d * (f * np.cos(s * q)).sum(-1)
    if dim == 2:
        return 2.0 * np.pi * C_d * ((f * q) * _besselJ0(s * q)).sum(-1)
    if dim == 3:
        return 4.0 * np.pi * C_d / s[:, 0] * ((f * q) * np.sin(s * q)).sum(-1)
    raise ValueError(f'dim must be 1, 2 or 3, got {dim}')


def latticeDensityFourier(kernel, n_h: float, dim: int, *,
                          sMax: float = 150.0, **kw) -> float:
    """`L(n_h) = sum_j r_d(j) what(2 pi n_h sqrt(j))` -- Poisson summation.

    The identity is exact; the series is cut at `|s| <= sMax` because `what` is
    evaluated by quadrature in float64 and cancels catastrophically once it
    falls below ~1e-12 (the integrand stays O(1) while the answer decays like
    `s^-(d+2k+1)`). That cut is far past convergence -- agreement with
    `latticeDensityShells` is 1e-6..1e-11 -- so prefer `shells` when you want
    the number and this when you want the structure.
    """
    counts = shellCounts(dim, max(9, int((sMax / (2 * np.pi * n_h)) ** 2)))
    j = np.nonzero(counts[1:])[0] + 1
    what = kernelFourierTransform(kernel, dim, 2 * np.pi * n_h * np.sqrt(j), **kw)
    return 1.0 + float((counts[j] * what).sum())


def latticeDensityIsStrictlyAbove1(kernel) -> bool:
    """Is `L(n_h) > 1` for every `n_h` -- i.e. is choosing `h` a dead end?

    The Wendland functions are positive definite by construction, so
    `what(s) > 0` for all `s` (Bochner) and every term of the Poisson sum above
    is strictly positive. A perfect lattice therefore reads *over* `rho0` at any
    support radius, and no `h` solves `L(h) = 1`: a Newton/bisection search on
    `h` cannot converge, and the mass (or an explicit kernel correction) is the
    only lever. The B-splines are not positive definite in 2D/3D -- `what`
    changes sign, `L` crosses 1, and roots in `h` do exist.
    """
    from ..enumTypes import KernelFunctions
    return _kernelInt(kernel) in (KernelFunctions.Wendland2.value,
                                  KernelFunctions.Wendland4.value,
                                  KernelFunctions.Wendland6.value)


# ------------------------------------------------------ 3. closed form, no sums
#: `A_m = c_m C_d Lambda_d(m)` per kernel and dimension, keyed by the odd Taylor
#: power `m` of `k` at `q = 0`. Derived symbolically; they come out integral.
WENDLAND_TAIL_COEFFICIENTS: Dict[str, Dict[int, Dict[int, int]]] = {
    'Wendland2': {1: {3: 120},
                  2: {3: 2520, 5: -12600},
                  3: {3: 20160, 5: -120960}},
    'Wendland4': {1: {5: 20160, 7: -120960},
                  2: {5: 604800, 7: -12700800},
                  3: {5: 6652800, 7: -159667200}},
    'Wendland6': {1: {7: 6652800, 9: -159667200},
                  2: {7: 259459200, 9: -14010796800, 11: 77059382400},
                  3: {7: 3632428800, 9: -217945728000, 11: 1307674368000}},
}


def _hurwitzZeta(s: float, a: float, N: int = 24, M: int = 10) -> float:
    """`zeta(s, a)` by Euler-Maclaurin. Local so this module needs no scipy.

    `N` direct terms then the integral plus the first `M` Bernoulli corrections;
    at the arguments used here (`s >= 2`) that is good to ~1e-14.
    """
    #: B_2k / (2k)! ... folded into the standard Euler-Maclaurin coefficients.
    B = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510, 43867/798, -174611/330]
    total = sum((a + n) ** -s for n in range(N))
    z = a + N
    total += z ** (1 - s) / (s - 1) + 0.5 * z ** -s
    term, fac = z ** -(s + 1), s
    for k in range(min(M, len(B))):
        total += B[k] / math.factorial(2 * k + 2) * fac * term
        fac *= (s + 2 * k + 1) * (s + 2 * k + 2)
        term /= z * z
    return total


@functools.lru_cache(maxsize=256)
def epsteinZeta(dim: int, sigma: float) -> float:
    """`Z_d(sigma) = sum_{n in Z^d \\ 0} |n|^(-2 sigma)`.

    `Z_1 = 2 zeta(2 sigma)`; `Z_2 = 4 zeta(sigma) beta(sigma)` (Dirichlet beta),
    both from `_hurwitzZeta`; `Z_3` has no such factorisation and is summed
    directly with a shell-count tail.
    """
    if dim == 1:
        return 2.0 * _hurwitzZeta(2 * sigma, 1.0)
    if dim == 2:
        beta = (_hurwitzZeta(sigma, 0.25) - _hurwitzZeta(sigma, 0.75)) / 4.0 ** sigma
        return 4.0 * _hurwitzZeta(sigma, 1.0) * beta
    counts = shellCounts(3, 6400)
    j = np.nonzero(counts[1:])[0] + 1
    return float((counts[j] * j.astype(float) ** -sigma).sum())


def latticeDensityClosed(kernel, n_h: float, dim: int) -> float:
    """`L(n_h)` with no summation at all -- Wendland kernels only.

    Raises `KeyError` for the B-splines rather than returning a number that
    would be dominated by the oscillatory knot term it omits (see the module
    docstring).
    """
    from ..enumTypes import KernelFunctions
    name = KernelFunctions(_kernelInt(kernel)).name
    coefficients = WENDLAND_TAIL_COEFFICIENTS[name][dim]
    return 1.0 + float(np.sum([A * epsteinZeta(dim, (dim + m) / 2)
                               / (2 * np.pi * n_h) ** (dim + m)
                               for m, A in coefficients.items()]))


# ------------------------------------------------------------------- dispatcher
_METHODS = {
    'shells': latticeDensityShells,
    'fourier': latticeDensityFourier,
    'closed': latticeDensityClosed,
}


@functools.lru_cache(maxsize=1024)
def _latticeDensityCached(kern: int, n_h: float, dim: int, method: str) -> float:
    return _METHODS[method](kern, n_h, dim)


def latticeDensity(kernel, n_h: float, dim: int, method: str = 'shells') -> float:
    """`L` -- what a perfect lattice at `h / s = n_h` measures, in units of `rho0`.

    Cached on `(kernel, n_h, dim, method)`: it is a pure function of those four
    and the caller (mass calibration, a diagnostic banner) asks for the same
    handful of values every step.
    """
    if method not in _METHODS:
        raise ValueError(f'method must be one of {sorted(_METHODS)}, got {method!r}')
    return _latticeDensityCached(_kernelInt(kernel), float(n_h), int(dim), method)


def latticeDensityFactor(kernel, n_h: float, dim: int, method: str = 'shells') -> float:
    """`1 / L` -- what the nominal mass `rho0 s^d` has to be scaled by so a
    defect-free lattice measures exactly `rho0`."""
    return 1.0 / latticeDensity(kernel, n_h, dim, method)
