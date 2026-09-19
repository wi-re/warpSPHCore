"""The D&A (2012) eq. 18/19 density self-term correction.

    rho_corr,i = rho_hat_i - eps_i m_i W(0, h_i)        (eq. 18)
    eps_i      = eps_100 (N_H,i / 100)^(-alpha)         (eq. 19)

The self-contribution `m_i W(0, h_i)` biases the summation estimate
high for the kernels that over-estimate (the Wendland family); eq. 18
subtracts a fraction `eps` of it, with `eps` falling as a power law in
the neighbour number. The correction is a per-particle remap of the
FINISHED raw estimate (eps depends on N_H, which depends on rho_hat),
applied after the density operator and before the EOS -- the warpSPH
wiring is `modules/density/density.py::computeDensities`, gated by
`SimulationConfig.densityCorrection` (a sibling of
`calibrateNormalization`, which is a DIFFERENT correction: a constant
1/L lattice-quadrature rescale applied at the kernel level; see
LATTICE_DENSITY_PLAN.md and the replication's renorm_eps_design_note.md
section 7 -- do not enable both without refitting eps).

Consistency (why "correct after the fact" is legitimate): replacing
rho_hat by rho_corr in the Lagrangian leaves the equations of motion
otherwise identical, because `h^d rho_corr = h^d rho_hat - eps m C_d
f(0)` contributes a constant (in h) to `h^d rho_hat`, so the
conservation properties are unaffected.

Constants. Only the entries where the power law actually fits the
densest-lattice bias are shipped (refit: the replication's
`scripts/eps_constants_multidim.py`, `results/eps_constants_multidim.
json`; 3D window 40-400 as in the paper, 2D is our extension):

  * Wendland C2/C4/C6 in 2D and 3D. The 3D values reproduce the paper's
    (0.0294, 0.977) / (0.01342, 1.579) / (0.0116, 2.236) within x1.04
    and are tighter: corrected band over the closed 40-400 window
    (dense sweep, measured with the runtime N_H convention below) is
    0.27-1.0 % in 3D and 3e-5-2.2e-4 in 2D, vs ~5 % for the paper's
    constants in 3D. The worst case is the N_H ~ 40 edge of the window
    (alpha * bias^2, bias up to ~5 % there); mid-window the corrected
    estimate is within ~1e-4 (3D) / 1e-5 (2D) of 1.
  * NOT shipped, with the measured reason (all in the refit JSON):
    - 1D (all kernels): the raw 1D estimate is already within 1e-6 of
      exact over 40 <= N_H <= 400 (the 1D continuum limit is reached
      immediately), and the implied eps is not a power law -- it
      crosses zero, so no fit exists.
    - B-splines (2D and 3D): they under-estimate, and their lattice
      bias oscillates with N_H (the spline offset changes sign, see
      latticeDensity), so a single (eps_100, alpha) mis-corrects: the
      fitted power law has log residuals of 0.78-1.29 decades and the
      "corrected" curve is WORSE than the raw one for 8 of the 10
      spline entries.

`W(0, h) = C_d f(0) / h^d` is evaluated from the SHIPPED eval
functions at the code support radius (the paper's H -- evaluating at
h/kernelScale would be wrong by kernelScale^d; see the design note).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ..enumTypes import KernelFunctions
from ..type_config import scalar_t
from ..kernels.eval_kernel import eval_C_d, eval_k


def _kernelInt(kernel) -> int:
    """`KernelFunctions` member, its name, or a raw int -> the enum value
    (same normalisation as latticeDensity)."""
    if isinstance(kernel, int):
        return kernel
    if isinstance(kernel, str):
        return int(KernelFunctions[kernel].value)
    return int(kernel.value)


#: Volume of the unit ball: N_H = V_d h^d rho / m.
_V = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}

#: Shipped eq.-19 constants: (dim, kernel) -> (eps_100, alpha, (nh_min,
#: nh_max)). Refit on the densest lattice (1D uniform / 2D hexagonal /
#: 3D FCC) at rho = 1 from the shipped kernels; values transcribed from
#: results/eps_constants_multidim.json (window 40-400 for every entry).
_EPS_CONSTANTS: Dict[Tuple[int, int], Tuple[float, float, Tuple[float, float]]] = {
    # (eps_100, alpha, validity window)
    (2, KernelFunctions.Wendland2.value):
        (0.0030233012061203278, 1.5016297296462702, (40.0, 400.0)),
    (2, KernelFunctions.Wendland4.value):
        (0.00035064837551512925, 2.4815365353197136, (40.0, 400.0)),
    (2, KernelFunctions.Wendland6.value):
        (8.005994569170169e-05, 3.4708508575608708, (40.0, 400.0)),
    (3, KernelFunctions.Wendland2.value):
        (0.029488222560940334, 0.9989656162766523, (40.0, 400.0)),
    (3, KernelFunctions.Wendland4.value):
        (0.013611521340766316, 1.6332544351850704, (40.0, 400.0)),
    (3, KernelFunctions.Wendland6.value):
        (0.011308235376902351, 2.218472363304362, (40.0, 400.0)),
}


def densityCorrectionConstants(kernel, dim: int
                               ) -> Tuple[float, float, Tuple[float, float]]:
    """The shipped (eps_100, alpha, validity window) for (dim, kernel).

    Raises KeyError (mirroring the replication script) for kernels or
    dimensions without fitted constants -- see the module docstring for
    why each missing entry is missing."""
    key = (int(dim), _kernelInt(kernel))
    try:
        return _EPS_CONSTANTS[key]
    except KeyError:
        try:
            name = KernelFunctions(_kernelInt(kernel)).name
        except ValueError:
            name = str(kernel)
        have = sorted(f"{d}D {KernelFunctions(k).name}"
                      for (d, k), _ in _EPS_CONSTANTS.items())
        raise KeyError(
            f"no eq.-19 self-term correction constants for {name} in "
            f"{dim}D; shipped: {have} (refit: scripts/eps_constants_"
            f"multidim.py, see renorm_eps_design_note.md)"
        ) from None


_CENTRAL_CACHE: Dict[Tuple[int, int], float] = {}


def _centralValue(dim: int, kernelInt: int) -> float:
    """`C_d f(0)` of the SHIPPED kernel, from the shipped eval functions
    (host-callable @wp.func, as the latticeDensity tests use them).
    Cached per (dim, kernel)."""
    key = (int(dim), kernelInt)
    if key not in _CENTRAL_CACHE:
        _CENTRAL_CACHE[key] = (float(eval_C_d(int(dim), kernelInt))
                               * float(eval_k(scalar_t(0.0), int(dim), kernelInt)))
    return _CENTRAL_CACHE[key]


def selfTermW0(kernel, dim: int, supports: torch.Tensor) -> torch.Tensor:
    """The per-particle self-term kernel value `m W(0, h)` WITHOUT the
    mass: `W(0, h) = C_d f(0) / h^d` at the code support radius (the
    paper's H). Parameterisation-invariant by construction (the shipped
    functions are the same ones the density operator sums)."""
    dim = int(dim)
    return _centralValue(dim, _kernelInt(kernel)) * supports.pow(-dim)


def applyDensityCorrection(densities: torch.Tensor, masses: torch.Tensor,
                           supports: torch.Tensor, kernel, dim: int,
                           eps100: Optional[float] = None,
                           alpha: Optional[float] = None) -> torch.Tensor:
    """Eq. 18 on a finished raw estimate: `rho - eps m W(0, h)`.

    `N_H = V_d h^d rho / m` is recovered per particle from the RAW
    estimate -- the only density available at the hook (the true rho is
    unknown at runtime; using the corrected one instead would shift eps
    by O(alpha eps), second order). The runtime N_H is therefore
    biased by the estimate's own bias, which shifts eps by
    O(alpha * bias) -- negligible mid-window (bias ~ 1e-3) but the
    dominant term of the window-edge band (alpha * bias^2, bias up to
    ~5 % at the N_H ~ 40 edge; see the module docstring's bands).
    Pure torch (three elementwise ops next to the density operator's
    neighbour sums); the only warp call is the one-time `C_d f(0)`
    cache fill. `eps100` / `alpha` override the shipped table (refit
    experiments); the power law is applied as fitted -- the table's
    validity window is where it is graded, outside it the correction
    extrapolates.
    """
    dim = int(dim)
    if dim not in _V:
        raise ValueError(f"dim must be 1, 2 or 3 (got {dim})")
    c_eps100, c_alpha, _ = densityCorrectionConstants(kernel, dim)
    eps100 = c_eps100 if eps100 is None else float(eps100)
    alpha = c_alpha if alpha is None else float(alpha)
    n_h = _V[dim] * supports.pow(dim) * densities / masses
    eps = eps100 * (n_h / 100.0).pow(-alpha)
    return densities - eps * masses * selfTermW0(kernel, dim, supports)
