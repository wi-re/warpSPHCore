"""`warpSPHCore.util.densityCorrection` -- the D&A (2012) eq. 18/19
self-term correction.

Reference values: the densest-lattice refit in the replication's
`scripts/eps_constants_multidim.py` (`results/eps_constants_multidim.
json`; 3D path = fig03 check #4's computation) and, for the 3D Wendland
kernels, the paper's own eq.-19 constants (agreement x1.00-1.03).
"""
import math

import numpy as np
import pytest
import torch

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.kernels.eval_kernel import eval_k, eval_C_d, eval_kernelScale
from warpSPHCore.sampling.lattice import sampleDensestLattice
from warpSPHCore.type_config import scalar_t
from warpSPHCore.util import (applyDensityCorrection,
                              densityCorrectionConstants, selfTermW0)

WENDLANDS = [KernelFunctions.Wendland2, KernelFunctions.Wendland4,
             KernelFunctions.Wendland6]

#: The shipped table (refit, window 40-400): (dim, kernel) -> (eps_100,
#: alpha). A transcription guard, not a re-fit.
SHIPPED = {
    (2, KernelFunctions.Wendland2): (0.0030233012061203278, 1.5016297296462702),
    (2, KernelFunctions.Wendland4): (0.00035064837551512925, 2.4815365353197136),
    (2, KernelFunctions.Wendland6): (8.005994569170169e-05, 3.4708508575608708),
    (3, KernelFunctions.Wendland2): (0.029488222560940334, 0.9989656162766523),
    (3, KernelFunctions.Wendland4): (0.013611521340766316, 1.6332544351850704),
    (3, KernelFunctions.Wendland6): (0.011308235376902351, 2.218472363304362),
}
#: Corrected-curve band max |corr - 1| over the closed window 40 <= N_H
#: <= 400: the refit's DENSE-sweep value, measured with the LIBRARY's
#: runtime convention (eps at N_H,est = V_d h^d rho_hat / m -- the exact
#: N_H is not available at runtime; at the window edges the two
#: conventions differ by O(alpha * bias)) plus headroom for the kernel
#: evaluation's build-precision noise (the refit is float64; the tests
#: run in the repo's working precision).
BAND = {
    (2, KernelFunctions.Wendland2): 3.0e-4,   # dense 2.15e-4
    (2, KernelFunctions.Wendland4): 7.5e-5,   # dense 5.27e-5
    (2, KernelFunctions.Wendland6): 4.5e-5,   # dense 3.02e-5
    (3, KernelFunctions.Wendland2): 6.0e-3,   # dense 4.30e-3
    (3, KernelFunctions.Wendland4): 3.5e-3,   # dense 2.69e-3
    (3, KernelFunctions.Wendland6): 1.4e-2,   # dense 9.97e-3
}


# --------------------------------------------------------------------------- #
# W(0, h): the self-term central value
# --------------------------------------------------------------------------- #

# No Gaussian here: its shape uses `wp.where`, which has no host-call
# (Python-mode) implementation in this Warp version, so `eval_k` cannot
# be called outside a launch. The utility is unaffected -- the constants
# table raises before `selfTermW0` is ever reached for a non-Wendland
# kernel.
@pytest.mark.parametrize('kernel', WENDLANDS + [KernelFunctions.CubicSpline])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_selfTermW0IsTheShippedKernelCentre(kernel, dim):
    """`C_d f(0) / h^d` straight from the shipped eval functions --
    parameterisation-invariant by construction."""
    h = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    want = float(eval_C_d(dim, kernel.value)) * \
        float(eval_k(scalar_t(0.0), dim, kernel.value))
    got = selfTermW0(kernel, dim, h)
    assert float((got - want / h ** dim).abs().max()) < 1e-15


@pytest.mark.parametrize('kernel', WENDLANDS)
@pytest.mark.parametrize('dim', [2, 3])
def test_w0ConventionIsTheCodeSupport(kernel, dim):
    """The W0 gotcha: `W(0, h)` at the CODE support h (the paper's H),
    not at h/kernelScale -- the two differ by kernelScale^dim (x7.26 for
    Wendland2 3D)."""
    h = torch.ones(1, dtype=torch.float64)
    ks = float(eval_kernelScale(kernel.value, dim))
    ratio = float(selfTermW0(kernel, dim, h)
                  / selfTermW0(kernel, dim, h / ks))
    assert ratio == pytest.approx(ks ** (-dim), rel=1e-12)


# --------------------------------------------------------------------------- #
# The constants table
# --------------------------------------------------------------------------- #

def test_constantsTableRegression():
    for (dim, kernel), (eps100, alpha) in SHIPPED.items():
        c_eps100, c_alpha, window = densityCorrectionConstants(kernel, dim)
        assert c_eps100 == pytest.approx(eps100, rel=1e-12)
        assert c_alpha == pytest.approx(alpha, rel=1e-12)
        assert window == (40.0, 400.0)


def test_3dWendlandMatchesThePaper():
    """The 3D refit is the paper's fit on the same lattice: within x1.04
    of the published (eps_100, alpha) pairs (the design note's figure)."""
    paper = {
        KernelFunctions.Wendland2: (0.0294, 0.977),
        KernelFunctions.Wendland4: (0.01342, 1.579),
        KernelFunctions.Wendland6: (0.0116, 2.236),
    }
    for kernel, (p_eps, p_alpha) in paper.items():
        c_eps, c_alpha, _ = densityCorrectionConstants(kernel, 3)
        assert c_eps / p_eps <= 1.04 and p_eps / c_eps <= 1.04
        assert c_alpha / p_alpha <= 1.04 and p_alpha / c_alpha <= 1.04


@pytest.mark.parametrize('kernel', [KernelFunctions.Gaussian,
                                    KernelFunctions.HOCT4,
                                    KernelFunctions.CubicSpline,
                                    KernelFunctions.QuarticSpline,
                                    KernelFunctions.QuinticSpline,
                                    KernelFunctions.B7,
                                    KernelFunctions.B8])
@pytest.mark.parametrize('dim', [2, 3])
def test_noConstantsForNonShippedEntries(kernel, dim):
    """B-splines: their lattice bias is not a power law (the refit's log
    residuals are 0.78-1.29 decades and the 'corrected' curve is WORSE
    for 8 of the 10 entries) -- no constants ship. Gaussian/HOCT4: no
    paper fit and the Gaussian's small-N_H bias is self-term dominated.
    A KeyError, not a silent wrong answer."""
    with pytest.raises(KeyError):
        densityCorrectionConstants(kernel, dim)


@pytest.mark.parametrize('kernel', WENDLANDS)
def test_no1DConstants(kernel):
    """1D: the raw estimate is already within 1e-6 of exact over
    40 <= N_H <= 400 and the implied eps is not a power law (it crosses
    zero) -- nothing to correct with, nothing to fit."""
    with pytest.raises(KeyError, match='1D'):
        densityCorrectionConstants(kernel, 1)


# --------------------------------------------------------------------------- #
# The correction itself
# --------------------------------------------------------------------------- #

_V = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


def _densestLatticeDistances(dim):
    """Sorted min-image distances from a lattice point, on the densest
    lattice at rho = 1 (the refit's own setup)."""
    n = {1: 2048, 2: 4096, 3: 4000}[dim]
    lat = sampleDensestLattice(n, 1.0, dim)
    box = lat.box
    pos = (lat.positions - lat.positions[0]) % box
    d = pos - box * np.round(pos / box)
    r = np.sqrt((d ** 2).sum(-1))
    mass = float(np.prod(box)) / lat.count
    return np.sort(r), mass


def _rawLatticeDensity(r_sorted, mass, dim, kernel, H):
    """rho_hat = m sum_j W(|x_j|, H) at the origin (self term included),
    the shipped kernel evaluated pointwise (the test's own reference --
    it does not share code with the utility under test)."""
    rr = r_sorted[r_sorted < H]
    q = rr / H
    f = np.array([float(eval_k(scalar_t(x), dim, kernel.value)) for x in q])
    return float(eval_C_d(dim, kernel.value)) * mass * f.sum() / H ** dim


@pytest.mark.parametrize('kernel', WENDLANDS)
@pytest.mark.parametrize('dim', [2, 3])
def test_correctedLatticeWithinRefitBand(kernel, dim):
    """The shipped constants do what the refit says: over the window the
    corrected densest-lattice estimate sits within the refit band, and
    the RAW estimate is clearly outside it (the correction is not a
    no-op)."""
    r, mass = _densestLatticeDistances(dim)
    band = BAND[(dim, kernel)]
    for N_H in (40.0, 100.0, 200.0, 400.0):
        H = (N_H * mass / _V[dim]) ** (1.0 / dim)
        rho = _rawLatticeDensity(r, mass, dim, kernel, H)
        supports = torch.full((1,), H, dtype=torch.float64)
        masses = torch.full((1,), mass, dtype=torch.float64)
        densities = torch.full((1,), rho, dtype=torch.float64)
        corr = float(applyDensityCorrection(densities, masses, supports,
                                            kernel, dim)[0])
        assert abs(corr - 1.0) <= band, \
            f"{kernel.name} {dim}D N_H={N_H}: |corr-1| = {abs(corr-1.0):.3e} > {band}"
        if N_H == 40.0:
            # The bias is monotone in N_H (decreasing), so the window
            # edge is the worst case for the raw deviation. 2x, not 3x:
            # at the edge the band itself is inflated by the
            # runtime-N_H convention's alpha*bias^2, which is a
            # fraction of the raw bias (W6 3D: raw ~4.6e-2, band 1.4e-2).
            assert abs(rho - 1.0) >= 2.0 * band, \
                f"{kernel.name} {dim}D N_H=40: raw {abs(rho-1.0):.3e} is " \
                f"not clearly outside the {band} band"


def test_correctionMatchesTheManualFormula():
    """rho - eps m W(0, h) with N_H = V_d h^d rho / m from the raw
    estimate -- checked term by term, plus the eps100/alpha override."""
    dim = 3
    kernel = KernelFunctions.Wendland2
    dx = 0.1
    H = 0.35
    n = 8
    masses = torch.full((n,), dx ** dim, dtype=torch.float64)
    supports = torch.linspace(0.25, 0.5, n, dtype=torch.float64)
    densities = torch.linspace(0.8, 1.4, n, dtype=torch.float64)
    eps100, alpha, _ = densityCorrectionConstants(kernel, dim)
    w0 = float(eval_C_d(dim, kernel.value)) * \
        float(eval_k(scalar_t(0.0), dim, kernel.value))
    nh = _V[dim] * supports ** dim * densities / masses
    eps = eps100 * (nh / 100.0) ** (-alpha)
    w0vec = torch.tensor([w0 / h ** dim for h in supports.tolist()],
                         dtype=torch.float64)
    manual = densities - eps * masses * w0vec
    got = applyDensityCorrection(densities, masses, supports, kernel, dim)
    assert float((got - manual).abs().max()) < 1e-15
    manual2 = densities - (2.0 * eps) * masses * w0vec
    got2 = applyDensityCorrection(densities, masses, supports, kernel, dim,
                                  eps100=2.0 * eps100)
    assert float((got2 - manual2).abs().max()) < 1e-15


def test_zeroEpsIsTheIdentity():
    n = 4
    densities = torch.linspace(0.9, 1.1, n, dtype=torch.float64)
    masses = torch.full((n,), 1e-4, dtype=torch.float64)
    supports = torch.full((n,), 0.4, dtype=torch.float64)
    out = applyDensityCorrection(densities, masses, supports,
                                 KernelFunctions.Wendland2, 3, eps100=0.0)
    assert torch.equal(out, densities)


def test_badDimRaises():
    densities = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError):
        applyDensityCorrection(densities, densities, densities,
                               KernelFunctions.Wendland2, 4)
