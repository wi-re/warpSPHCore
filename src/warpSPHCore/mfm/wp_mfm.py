"""MFM / MFV on the CSR neighbourhood: warp kernels (Hopkins 2015).

The same algebra as the pure-torch reference (``geometry.py``, ``limiters.py``,
``riemann.py``, ``scheme.py`` -- which stays the differentiable oracle these
kernels are tested against), but with no materialised pair list: every kernel
is *particle-centred* (one thread per particle ``i``), walks its row of the
CSR neighbour list (or the compact-hash grid cells) and evaluates the pair
algebra on the fly, accumulating into thread-local variables. Memory is O(N);
nothing scales with the number of pairs.

Passes (each a kernel in the canonical structured ABI, so adjacency and grid
traversal both work; ``r > max(h_i, h_j)`` pairs have no face and are skipped):

1. ``mfmMoments``    -- ``omega_i = sum_j W_ij``, ``Ehat_i = sum_j W_ij d_ij d_ij^T``
2. (torch, O(N))    -- ``Ehat^-1``, ``N_cond``, volumes ``1/omega``
3. ``mfmGradients``  -- matrix least-squares gradients of ``(rho, v, P)``
4. ``mfmLimiter``    -- Barth-Jespersen-type slope factors (App. B1-B3)
5. ``mfmClosure*``   -- residual ``sum_j A_ij`` and the matvec of the graph
                        Laplacian ``sum_j kappa_ij (p_i - p_j)`` for the closure
                        correction (CG itself runs on the torch side, O(N))
6. ``mfmFlux``       -- per pair: effective face (with closure correction),
                        limited reconstruction, half-step prediction, HLLC /
                        MFM / MFV flux; ``dQ_i += -/+ |A| F``
7. ``mfmTimestep``   -- signal-velocity CFL per particle

**Antisymmetry.** Both threads of a pair evaluate the *same* canonical pair
``(a, b) = (min, max)(i, j)`` with the same operand order, so the pair flux is
bitwise identical on both sides and mass, momentum and energy cancel exactly
(not just to the solver's mirror symmetry) at the price of evaluating each
Riemann problem twice.

Forward evaluation only (no adjoint / JVP); use the torch reference for
derivatives. Single particle set. Ideal gas. ``kernelProperties.calibrate
Normalization`` is not applied (the torch path does not either).
"""

import warp as wp
from typing import Any, Optional, Union
from warp.types import vector, matrix
import torch

from ..profiling import record_function
from ..type_config import *
from ..autograd import *
from ..dataTypes import *
from ..radiusSearch.grid_util import getIndexRangeLane
from ..math import *
from ..kernels import *
from ..kernels.kernel import sphKernel_
from ..kernels.eval_kernel import eval_dkdq, eval_C_d
from ..util import *
from ..enumTypes import *

__all__ = ["mfmMomentsWarp", "mfmGradientsWarp", "mfmLimiterWarp", "mfmClosureResidualWarp",
           "mfmClosureMatvecWarp", "mfmClosureWeightsWarp", "mfmFluxWarp", "mfmTimestepWarp", "mfmCoarseStencilWarp"]

_TINY = 1.0e-30            # (1e-300 underflows to 0 in float32)


# ---------------------------------------------------------------------------------------
# pair algebra (shared by the kernels)
# ---------------------------------------------------------------------------------------

@wp.func
def faceVector(d: Any, Wi: scalar_t, Wj: scalar_t, Vi: scalar_t, Vj: scalar_t, EinvI: Any, EinvJ: Any):
    """``A_ij = V_i W_ij(h_i) Ehat_i^-1 d + V_j W_ij(h_j) Ehat_j^-1 d`` with ``d = x_j - x_i``."""
    return (Vi * Wi) * matmul(EinvI, d) + (Vj * Wj) * matmul(EinvJ, d)


@wp.func
def kernelDerivativeValue(r: scalar_t, h: scalar_t, kernel: wp.int32, dim: wp.int32):
    """``dW/dr (r, h) = C_d k'(q) / h^(dim + 1)``."""
    q = r / h
    if q > scalar_t(1.0):
        return scalar_t(0.0)
    return eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / wp.pow(h, scalar_t(dim + 1))


@wp.func
def guardedFace(d: Any, Vi: scalar_t, Vj: scalar_t, EinvI: Any, EinvJ: Any, hi: scalar_t, hj: scalar_t,
                badI: wp.int32, badJ: wp.int32, kernel: wp.int32, dim: wp.int32, centred: wp.int32, areaCap: wp.int32):
    """The face vector of the pair (``d = x_j - x_i``) with GIZMO's guards (see ``geometry.py``): centred volume
    weights for large volume jumps, SPH-style fallback face where either matrix is ill-conditioned or
    ``A . d < 0``, optional geometric area cap."""
    tiny = scalar_t(_TINY)
    r = wp.length(d)
    Wi = sphKernel_(d, hi, kernel)
    Wj = sphKernel_(d, hj, kernel)
    wti = Vi
    wtj = Vj
    if centred != 0:
        if wp.abs(Vi - Vj) / wp.min(Vi, Vj) / scalar_t(dim) > scalar_t(1.25):
            den = Vi * Wi + Vj * Wj
            if den > scalar_t(0.0):
                wc = Vi * Vj * (Wi + Wj) / den
                wti = wc
                wtj = wc
    A = faceVector(d, Wi, Wj, wti, wtj, EinvI, EinvJ)
    bad = badI != 0 or badJ != 0 or wp.dot(A, d) < scalar_t(0.0)
    if not wp.isfinite(wp.length(A)):
        bad = True
    if bad:
        dWi = kernelDerivativeValue(r, hi, kernel, dim)
        dWj = kernelDerivativeValue(r, hj, kernel, dim)
        A = (-(wti * Vi * dWi + wtj * Vj * dWj) / wp.max(r, tiny)) * d
    if areaCap != 0:
        pi = scalar_t(3.141592653589793)
        ai = scalar_t(2.0)
        aj = scalar_t(2.0)
        if dim == 2:
            ai = scalar_t(2.0) * pi * wp.sqrt(Vi)
            aj = scalar_t(2.0) * pi * wp.sqrt(Vj)
        if dim == 3:
            ai = scalar_t(4.0) * pi * wp.pow(Vi, scalar_t(2.0 / 3.0))
            aj = scalar_t(4.0) * pi * wp.pow(Vj, scalar_t(2.0 / 3.0))
        Amax = wp.min(ai, aj)
        mag = wp.length(A)
        if mag > Amax:
            A = A * (Amax / mag)
    return A


@wp.func
def alphaScalar(phi: scalar_t, ngbHi: scalar_t, ngbLo: scalar_t, midHi: scalar_t, midLo: scalar_t, beta: scalar_t):
    """Eq. B2 for one component: ``min(1, beta min(up_ratio, dn_ratio))``."""
    tiny = scalar_t(_TINY)
    big = scalar_t(1.0e30)
    up = midHi - phi
    dn = phi - midLo
    rUp = big
    rDn = big
    if up > tiny:
        rUp = (ngbHi - phi) / up
    if dn > tiny:
        rDn = (phi - ngbLo) / dn
    return wp.min(scalar_t(1.0), beta * wp.min(rUp, rDn))


@wp.func
def _sgn(x: scalar_t):
    if x > scalar_t(0.0):
        return scalar_t(1.0)
    if x < scalar_t(0.0):
        return scalar_t(-1.0)
    return scalar_t(0.0)


@wp.func
def pairLimit(phiI: scalar_t, phiJ: scalar_t, phi0: scalar_t, frac: scalar_t, psi1: scalar_t, psi2: scalar_t):
    """Eq. B4: the limited face value seen from ``i``."""
    if phiI == phiJ:
        return phiI
    tiny = scalar_t(_TINY)
    dphi = wp.abs(phiI - phiJ)
    d1 = psi1 * dphi
    d2 = psi2 * dphi
    bar = phiI + frac * (phiJ - phiI)
    pmin = wp.min(phiI, phiJ)
    pmax = wp.max(phiI, phiJ)
    lo = pmin - d1
    if _sgn(lo) != _sgn(pmin):
        lo = pmin / (scalar_t(1.0) + d1 / wp.max(wp.abs(pmin), tiny))
    hi = pmax + d1
    if _sgn(hi) != _sgn(pmax):
        hi = pmax / (scalar_t(1.0) + d1 / wp.max(wp.abs(pmax), tiny))
    if phiI < phiJ:
        return wp.max(lo, wp.min(bar + d2, phi0))
    return wp.min(hi, wp.max(bar - d2, phi0))


@wp.func
def _speedsRoe(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t, aL: scalar_t,
               rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, aR: scalar_t, gamma: scalar_t):
    R = wp.sqrt(rhoR / rhoL)
    EL = PL / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rhoL * (uL * uL + wp.dot(vtL, vtL))
    ER = PR / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rhoR * (uR * uR + wp.dot(vtR, vtR))
    HL = (EL + PL) / rhoL
    HR = (ER + PR) / rhoR
    ut = (uL + R * uR) / (scalar_t(1.0) + R)
    vtt = (vtL + R * vtR) / (scalar_t(1.0) + R)
    H = (HL + R * HR) / (scalar_t(1.0) + R)
    a2 = (gamma - scalar_t(1.0)) * (H - scalar_t(0.5) * (ut * ut + wp.dot(vtt, vtt)))
    at = wp.sqrt(wp.max(a2, scalar_t(0.0)))
    return wp.min(uL - aL, ut - at), wp.max(uR + aR, ut + at)


@wp.func
def _speedsPvrs(rhoL: scalar_t, uL: scalar_t, PL: scalar_t, aL: scalar_t,
                rhoR: scalar_t, uR: scalar_t, PR: scalar_t, aR: scalar_t, gamma: scalar_t):
    rbar = scalar_t(0.5) * (rhoL + rhoR)
    abar = scalar_t(0.5) * (aL + aR)
    pstar = wp.max(scalar_t(0.5) * (PL + PR) - scalar_t(0.5) * (uR - uL) * rbar * abar, scalar_t(0.0))
    qL = scalar_t(1.0)
    qR = scalar_t(1.0)
    if pstar > PL:
        qL = wp.sqrt(scalar_t(1.0) + (gamma + scalar_t(1.0)) / (scalar_t(2.0) * gamma) * (pstar / PL - scalar_t(1.0)))
    if pstar > PR:
        qR = wp.sqrt(scalar_t(1.0) + (gamma + scalar_t(1.0)) / (scalar_t(2.0) * gamma) * (pstar / PR - scalar_t(1.0)))
    return uL - aL * qL, uR + aR * qR


@wp.func
def _starFromSpeeds(SL: scalar_t, SR: scalar_t, rhoL: scalar_t, uL: scalar_t, PL: scalar_t,
                    rhoR: scalar_t, uR: scalar_t, PR: scalar_t):
    num = PR - PL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    Ss = num / den
    Ps = PL + rhoL * (SL - uL) * (Ss - uL)
    return Ss, Ps


@wp.func
def _okStar(Ss: scalar_t, Ps: scalar_t):
    return Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss)


@wp.func
def hllcStar(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t,
             rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, gamma: scalar_t):
    """``(S_L, S_R, S*, P*)`` with the Roe -> PVRS -> Rusanov fallback chain."""
    aL = wp.sqrt(gamma * PL / rhoL)
    aR = wp.sqrt(gamma * PR / rhoR)
    SL, SR = _speedsRoe(rhoL, uL, vtL, PL, aL, rhoR, uR, vtR, PR, aR, gamma)
    Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
    if not _okStar(Ss, Ps):
        SL, SR = _speedsPvrs(rhoL, uL, PL, aL, rhoR, uR, PR, aR, gamma)
        Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
        if not _okStar(Ss, Ps):
            s = wp.max(wp.abs(uL) + aL, wp.abs(uR) + aR)
            SL = -s
            SR = s
            Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
    return SL, SR, Ss, Ps


@wp.func
def _hllcSide(rho: scalar_t, u: scalar_t, vt: Any, P: scalar_t, S: scalar_t, Ss: scalar_t, gamma: scalar_t, star: wp.bool):
    """Mass / normal-momentum / tangential-momentum / energy flux of one side (``star``: its star region)."""
    E = P / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rho * (u * u + wp.dot(vt, vt))
    Fm = rho * u
    Fn = rho * u * u + P
    Ft = Fm * vt
    Fe = (E + P) * u
    if star:
        coef = rho * (S - u) / (S - Ss)
        Us_m = coef
        Us_n = coef * Ss
        Us_t = coef * vt
        Us_e = coef * (E / rho + (Ss - u) * (Ss + P / (rho * (S - u))))
        Fm = Fm + S * (Us_m - rho)
        Fn = Fn + S * (Us_n - rho * u)
        Ft = Ft + S * (Us_t - rho * vt)
        Fe = Fe + S * (Us_e - E)
    return Fm, Fn, Ft, Fe


@wp.func
def faceFluxPair(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t,
                 rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, gamma: scalar_t,
                 mode: wp.int32, normal: Any, vframe: Any):
    """Lab-frame flux per unit face area: ``(mass, momentum vector, energy)``."""
    SL, SR, Ss, Ps = hllcStar(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR, gamma)
    vn = wp.dot(vframe, normal)
    if mode == 0:  # MFM: the face moves with the contact -> no mass flux
        return scalar_t(0.0), Ps * normal, Ps * (Ss + vn), Ss, Ps
    # MFV: full HLLC in the frame of the quadrature point, then de-boost (Eq. A8)
    Fm = scalar_t(0.0)
    Fn = scalar_t(0.0)
    Ft = vtL * scalar_t(0.0)
    Fe = scalar_t(0.0)
    if SL >= scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoL, uL, vtL, PL, SL, Ss, gamma, False)
    elif Ss >= scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoL, uL, vtL, PL, SL, Ss, gamma, True)
    elif SR > scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoR, uR, vtR, PR, SR, Ss, gamma, True)
    else:
        Fm, Fn, Ft, Fe = _hllcSide(rhoR, uR, vtR, PR, SR, Ss, gamma, False)
    mom = Fn * normal + Ft
    momLab = mom + Fm * vframe
    eLab = Fe + wp.dot(vframe, mom) + scalar_t(0.5) * wp.dot(vframe, vframe) * Fm
    return Fm, momLab, eLab, Ss, Ps


# ---------------------------------------------------------------------------------------
# helpers shared by every kernel
# ---------------------------------------------------------------------------------------

def _dimOf(ctx, extras):
    return ctx.query.positions.shape[1]


def _vecDtype(ctx, extras):
    return vector(length=_dimOf(ctx, extras), dtype=scalar_t)


def _matDtype(ctx, extras):
    d = _dimOf(ctx, extras)
    return matrix(shape=(d, d), dtype=scalar_t)


# ---------------------------------------------------------------------------------------
# 1. moments: omega, Ehat, neighbour count
# ---------------------------------------------------------------------------------------

@wp.func
def mfmMoments_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    EZero: Any,
):
    omega = scalar_t(0.0)
    E = zero_like_warp(EZero)
    cnt = wp.int32(0)
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + n])
        if j == i:
            continue
        jPtcl = getParticleData(referenceState, j)
        d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        r = wp.length(d)
        if r > wp.max(iPtcl.support, jPtcl.support):
            continue
        W = sphKernel_(d, iPtcl.support, kernelProperties.kernelFunction)
        omega += W
        E += W * wp.outer(d, d)
        if W > scalar_t(0.0):
            cnt += 1
    return omega, E, cnt


@wp.kernel
def mfmMoments_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    output_omega: wp.array(dtype=scalar_t),  # type: ignore
    output_E: wp.array(dtype=Any),  # type: ignore
    output_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    omega = kernel_self(iPtcl, kernelProperties)
    E = zero_like_warp(output_E[i])
    cnt = wp.int32(0)
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        w, e, c = mfmMoments_Func_i(
            i, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex, E)
        omega += w
        E += e
        cnt += c
    output_omega[i] = omega
    output_E[i] = E
    output_count[i] = cnt + 1


@wp.func
def kernel_self(iPtcl: Any, kernelProperties: kernelState):
    """``W(0, h_i)``."""
    z = iPtcl.position * scalar_t(0.0)
    return sphKernel_(z, iPtcl.support, kernelProperties.kernelFunction)


_MOMENTS_SPEC = OperatorSpec(
    kernel=mfmMoments_Kernel,
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=_matDtype), OutputSpec(dtype=wp.int32)),
)


# ---------------------------------------------------------------------------------------
# 2. matrix gradients of (rho, v, P)
# ---------------------------------------------------------------------------------------

@wp.func
def mfmGradients_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    sRhoZero: Any, sVelZero: Any,
):
    sRho = zero_like_warp(sRhoZero)
    sVel = zero_like_warp(sVelZero)
    sP = zero_like_warp(sRhoZero)
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + n])
        if j == i:
            continue
        jPtcl = getParticleData(referenceState, j)
        d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        r = wp.length(d)
        if r > wp.max(iPtcl.support, jPtcl.support):
            continue
        W = sphKernel_(d, iPtcl.support, kernelProperties.kernelFunction)
        if W > scalar_t(0.0):
            sRho += W * (rho[j] - rho[i]) * d
            sVel += W * wp.outer(vel[j] - vel[i], d)
            sP += W * (pres[j] - pres[i]) * d
    return sRho, sVel, sP


@wp.kernel
def mfmGradients_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    Einv: wp.array(dtype=Any), deficient: wp.array(dtype=wp.int32),  # type: ignore

    output_gRho: wp.array(dtype=Any), output_gVel: wp.array(dtype=Any), output_gP: wp.array(dtype=Any),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    sRho = zero_like_warp(output_gRho[i])
    sVel = zero_like_warp(output_gVel[i])
    sP = zero_like_warp(output_gP[i])
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        a, b, c = mfmGradients_Func_i(
            i, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            rho, vel, pres, sRho, sVel)
        sRho += a
        sVel += b
        sP += c
    if deficient[i] != 0:
        output_gRho[i] = zero_like_warp(sRho)
        output_gVel[i] = zero_like_warp(sVel)
        output_gP[i] = zero_like_warp(sP)
    else:
        Ei = Einv[i]
        output_gRho[i] = matmul(Ei, sRho)
        output_gVel[i] = sVel * Ei            # S[a, k] Einv[k, b]; Einv is symmetric
        output_gP[i] = matmul(Ei, sP)


_GRADIENTS_SPEC = OperatorSpec(
    kernel=mfmGradients_Kernel,
    outputs=(OutputSpec(dtype=_vecDtype), OutputSpec(dtype=_matDtype), OutputSpec(dtype=_vecDtype)),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR), ExtraSpec("vel", ExtraKind.TENSOR), ExtraSpec("pres", ExtraKind.TENSOR),
            ExtraSpec("Einv", ExtraKind.TENSOR), ExtraSpec("deficient", ExtraKind.TENSOR)),
)


# ---------------------------------------------------------------------------------------
# 3. slope limiter factors
# ---------------------------------------------------------------------------------------

@wp.func
def mfmLimiter_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    nbHiRho: scalar_t, nbLoRho: scalar_t, mdHiRho: scalar_t, mdLoRho: scalar_t,
    nbHiP: scalar_t, nbLoP: scalar_t, mdHiP: scalar_t, mdLoP: scalar_t,
    nbHiV: Any, nbLoV: Any, mdHiV: Any, mdLoV: Any,
):
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + n])
        if j == i:
            continue
        jPtcl = getParticleData(referenceState, j)
        d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        r = wp.length(d)
        if r > wp.max(iPtcl.support, jPtcl.support):
            continue
        frac = iPtcl.support / (iPtcl.support + jPtcl.support)
        disp = frac * d
        nbHiRho = wp.max(nbHiRho, rho[j])
        nbLoRho = wp.min(nbLoRho, rho[j])
        nbHiP = wp.max(nbHiP, pres[j])
        nbLoP = wp.min(nbLoP, pres[j])
        mR = rho[i] + wp.dot(gRho[i], disp)
        mP = pres[i] + wp.dot(gP[i], disp)
        mdHiRho = wp.max(mdHiRho, mR)
        mdLoRho = wp.min(mdLoRho, mR)
        mdHiP = wp.max(mdHiP, mP)
        mdLoP = wp.min(mdLoP, mP)
        vj = vel[j]
        mV = vel[i] + matmul(gVel[i], disp)
        for c in range(nbHiV.length):
            nbHiV[c] = wp.max(nbHiV[c], vj[c])
            nbLoV[c] = wp.min(nbLoV[c], vj[c])
            mdHiV[c] = wp.max(mdHiV[c], mV[c])
            mdLoV[c] = wp.min(mdLoV[c], mV[c])
    return nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV


@wp.kernel
def mfmLimiter_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    beta: wp.array(dtype=scalar_t),

    output_aRho: wp.array(dtype=scalar_t), output_aVel: wp.array(dtype=Any), output_aP: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    vi = vel[i]
    nbHiRho = rho[i]
    nbLoRho = rho[i]
    mdHiRho = rho[i]
    mdLoRho = rho[i]
    nbHiP = pres[i]
    nbLoP = pres[i]
    mdHiP = pres[i]
    mdLoP = pres[i]
    nbHiV = vi
    nbLoV = vi
    mdHiV = vi
    mdLoV = vi
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV = mfmLimiter_Func_i(
            i, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            rho, vel, pres, gRho, gVel, gP,
            nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV)
    b = beta[i]
    output_aRho[i] = alphaScalar(rho[i], nbHiRho, nbLoRho, mdHiRho, mdLoRho, b)
    output_aP[i] = alphaScalar(pres[i], nbHiP, nbLoP, mdHiP, mdLoP, b)
    aV = zero_like_warp(vi)
    for c in range(vi.length):
        aV[c] = alphaScalar(vi[c], nbHiV[c], nbLoV[c], mdHiV[c], mdLoV[c], b)
    output_aVel[i] = aV


_LIMITER_SPEC = OperatorSpec(
    kernel=mfmLimiter_Kernel,
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR), ExtraSpec("vel", ExtraKind.TENSOR), ExtraSpec("pres", ExtraKind.TENSOR),
            ExtraSpec("gRho", ExtraKind.TENSOR), ExtraSpec("gVel", ExtraKind.TENSOR), ExtraSpec("gP", ExtraKind.TENSOR),
            ExtraSpec("beta", ExtraKind.TENSOR)),
)


# ---------------------------------------------------------------------------------------
# 4. closure: residual sum_j A_ij and the Laplacian matvec (CG runs on the torch side)
# ---------------------------------------------------------------------------------------

@wp.func
def pairFace(i: wp.int32, j: wp.int32, iPtcl: Any, jPtcl: Any, domainState: domainData, kernelProperties: kernelState,
             vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
             centred: wp.int32, areaCap: wp.int32):
    """Guarded ``A_ij`` from the point of view of ``i`` (``d = x_j - x_i``)."""
    d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
    return guardedFace(d, vol[i], vol[j], Einv[i], Einv[j], iPtcl.support, jPtcl.support, condBad[i], condBad[j],
                       kernelProperties.kernelFunction, domainState.dim, centred, areaCap)


@wp.kernel
def mfmClosureResidual_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
    centred: wp.int32, areaCap: wp.int32,

    output_a: wp.array(dtype=Any), output_a2: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    acc = zero_like_warp(output_a[i])
    acc2 = scalar_t(0.0)
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        offsetArray = adjacencyState.neighborList if useAdjacency else gridState.sortIndex
        for n in range(numIndices):
            j = wp.int32(offsetArray[beginIndex + n])
            if j == i:
                continue
            jPtcl = getParticleData(referenceState, j)
            if wp.length(computeDistanceVec(iPtcl.position, jPtcl.position, domainState)) > wp.max(iPtcl.support, jPtcl.support):
                continue
            Aij = pairFace(i, j, iPtcl, jPtcl, domainState, kernelProperties, vol, Einv, condBad, centred, areaCap)
            acc += Aij
            acc2 += wp.dot(Aij, Aij)
    output_a[i] = acc
    output_a2[i] = acc2


@wp.kernel
def mfmClosureWeights_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
    centred: wp.int32, areaCap: wp.int32, power: scalar_t,

    output_kappa: wp.array(dtype=scalar_t),  # type: ignore
):
    """``kappa = |A_ij|^power`` for every entry of the CSR neighbour list (zero for self and out-of-face pairs).
    One float per directed pair: the closure solve applies the graph Laplacian dozens of times and the face
    algebra is too expensive to redo each time. Adjacency traversal only."""
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    beginIndex = adjacencyState.neighborOffsets[i]
    numIndices = adjacencyState.numNeighbors[i]
    for n in range(numIndices):
        j = wp.int32(adjacencyState.neighborList[beginIndex + n])
        k = scalar_t(0.0)
        if j != i:
            jPtcl = getParticleData(referenceState, j)
            if wp.length(computeDistanceVec(iPtcl.position, jPtcl.position, domainState)) <= wp.max(iPtcl.support, jPtcl.support):
                A = pairFace(i, j, iPtcl, jPtcl, domainState, kernelProperties, vol, Einv, condBad, centred, areaCap)
                k = wp.pow(wp.length(A), power)
        output_kappa[beginIndex + n] = k


@wp.kernel
def mfmClosureMatvec_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    kappa: wp.array(dtype=scalar_t), p: wp.array(dtype=Any),  # type: ignore

    output_y: wp.array(dtype=Any), output_diag: wp.array(dtype=scalar_t),  # type: ignore
):
    """``y_i = sum_j kappa_ij (p_i - p_j)`` from the cached weights; also the diagonal ``sum_j kappa_ij``
    (the Jacobi preconditioner / multigrid smoother of the closure solve)."""
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    beginIndex = adjacencyState.neighborOffsets[i]
    numIndices = adjacencyState.numNeighbors[i]
    acc = zero_like_warp(output_y[i])
    diag = scalar_t(0.0)
    pi = p[i]
    for n in range(numIndices):
        k = kappa[beginIndex + n]
        if k > scalar_t(0.0):
            j = wp.int32(adjacencyState.neighborList[beginIndex + n])
            acc += k * (pi - p[j])
            diag += k
    output_y[i] = acc
    output_diag[i] = diag


_CLOSURE_RES_SPEC = OperatorSpec(
    kernel=mfmClosureResidual_Kernel, outputs=(OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("vol", ExtraKind.TENSOR), ExtraSpec("Einv", ExtraKind.TENSOR), ExtraSpec("condBad", ExtraKind.TENSOR),
            ExtraSpec("centred", ExtraKind.SCALAR), ExtraSpec("areaCap", ExtraKind.SCALAR)),
)
_CLOSURE_MV_SPEC = OperatorSpec(
    kernel=mfmClosureMatvec_Kernel, outputs=(OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("kappa", ExtraKind.TENSOR), ExtraSpec("p", ExtraKind.TENSOR)),
)
_CLOSURE_W_SPEC = OperatorSpec(
    kernel=mfmClosureWeights_Kernel,
    outputs=(OutputSpec(dtype=scalar_t, shape=lambda ctx, extras: ctx.adjacency.j.shape[0]),),
    extras=(ExtraSpec("vol", ExtraKind.TENSOR), ExtraSpec("Einv", ExtraKind.TENSOR), ExtraSpec("condBad", ExtraKind.TENSOR),
            ExtraSpec("centred", ExtraKind.SCALAR), ExtraSpec("areaCap", ExtraKind.SCALAR), ExtraSpec("power", ExtraKind.SCALAR)),
    numThreads=lambda ctx, extras: ctx.query.positions.shape[0],
)


# ---------------------------------------------------------------------------------------
# 5. the flux pass
# ---------------------------------------------------------------------------------------

@wp.func
def sideTimeDerivative(rho: scalar_t, vBoost: Any, pres: scalar_t, gRho: Any, gVel: Any, gP: Any, gamma: scalar_t):
    """Eq. A4 for one particle's (limited) gradients; ``vBoost`` the velocity in the face frame."""
    divV = scalar_t(0.0)
    for c in range(vBoost.length):
        divV += gVel[c, c]
    dRho = -(wp.dot(vBoost, gRho) + rho * divV)
    dV = -(matmul(gVel, vBoost) + gP / rho)
    dP = -(wp.dot(vBoost, gP) + gamma * pres * divV)
    return dRho, dV, dP


@wp.func
def reconstructSide(
    rhoS: scalar_t, vS: Any, pS: scalar_t, rhoO: scalar_t, vO: Any, pO: scalar_t,
    gRho: Any, gVel: Any, gP: Any, aRho: scalar_t, aVel: Any, aP: scalar_t,
    disp: Any, frac: scalar_t, vframe: Any, dt: scalar_t, gamma: scalar_t, order: wp.int32,
    psi1: scalar_t, psi2: scalar_t, vdt: wp.int32,
):
    """Face state of one side: limited linear reconstruction (lab frame), pairwise limiter,
    boost, half-step prediction. Returns the boosted ``(rho, v, P)``."""
    rhoF = rhoS
    vF = vS
    pF = pS
    if order >= 2:
        rhoF = pairLimit(rhoS, rhoO, rhoS + aRho * wp.dot(gRho, disp), frac, psi1, psi2)
        pF = pairLimit(pS, pO, pS + aP * wp.dot(gP, disp), frac, psi1, psi2)
        lin = matmul(gVel, disp)
        for c in range(vS.length):
            vF[c] = pairLimit(vS[c], vO[c], vS[c] + aVel[c] * lin[c], frac, psi1, psi2)
    vF = vF - vframe
    if order >= 2 and dt > scalar_t(0.0):
        vB = vS - vframe
        gVelL = gVel * scalar_t(0.0)
        for c in range(vS.length):
            for k in range(vS.length):
                gVelL[c, k] = aVel[c] * gVel[c, k]
        dR, dV, dPp = sideTimeDerivative(rhoS, vB, pS, aRho * gRho, gVelL, aP * gP, gamma)
        half = scalar_t(0.5) * dt
        rP = rhoF + half * dR
        pP = pF + half * dPp
        if rP > scalar_t(0.0) and pP > scalar_t(0.0):
            rhoF = rP
            pF = pP
            vF = vF + half * dV
    return rhoF, vF, pF


@wp.func
def mfmFlux_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    aRho: wp.array(dtype=scalar_t), aVel: wp.array(dtype=Any), aP: wp.array(dtype=scalar_t),  # type: ignore
    vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), lam: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
    gamma: scalar_t, dt: scalar_t, mode: wp.int32, order: wp.int32, psi1: scalar_t, psi2: scalar_t,
    power: scalar_t, timeCentred: wp.int32, centred: wp.int32, areaCap: wp.int32, retry: wp.int32, massLimit: scalar_t,
    dmIn: scalar_t, dpIn: Any, deIn: scalar_t,
):
    dm = dmIn
    dp = zero_like_warp(dpIn)
    de = deIn
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + n])
        if j == i:
            continue
        # canonical pair (a, b), a < b: both threads of the pair compute the identical flux
        a = wp.min(i, j)
        b = wp.max(i, j)
        aPtcl = getParticleData(referenceState, a)
        bPtcl = getParticleData(referenceState, b)
        d = -computeDistanceVec(aPtcl.position, bPtcl.position, domainState)        # x_b - x_a
        r = wp.length(d)
        if r > wp.max(aPtcl.support, bPtcl.support):
            continue
        A0 = guardedFace(d, vol[a], vol[b], Einv[a], Einv[b], aPtcl.support, bPtcl.support, condBad[a], condBad[b],
                         kernelProperties.kernelFunction, domainState.dim, centred, areaCap)
        kappa = wp.pow(wp.length(A0), power)
        A = A0 + kappa * (lam[b] - lam[a])
        Amag = wp.length(A)
        if Amag <= scalar_t(0.0):
            continue
        nrm = A / Amag

        frac = aPtcl.support / (aPtcl.support + bPtcl.support)
        va = vel[a]
        vb = vel[b]
        # face (quadrature point) velocity from the half-step particle velocities (Eq. 21)
        vha = va
        vhb = vb
        if order >= 2 and dt > scalar_t(0.0) and timeCentred != 0:
            gvl_a = gVel[a] * scalar_t(0.0)
            gvl_b = gVel[b] * scalar_t(0.0)
            for c in range(va.length):
                for k in range(va.length):
                    gvl_a[c, k] = aVel[a][c] * gVel[a][c, k]
                    gvl_b[c, k] = aVel[b][c] * gVel[b][c, k]
            _r1, dva, _p1 = sideTimeDerivative(rho[a], va, pres[a], aRho[a] * gRho[a], gvl_a, aP[a] * gP[a], gamma)
            _r2, dvb, _p2 = sideTimeDerivative(rho[b], vb, pres[b], aRho[b] * gRho[b], gvl_b, aP[b] * gP[b], gamma)
            vha = va + scalar_t(0.5) * dt * dva
            vhb = vb + scalar_t(0.5) * dt * dvb
        vframe = vha + frac * (vhb - vha)

        rL, vL, pL = reconstructSide(
            rho[a], va, pres[a], rho[b], vb, pres[b], gRho[a], gVel[a], gP[a], aRho[a], aVel[a], aP[a],
            frac * d, frac, vframe, dt, gamma, order, psi1, psi2, timeCentred)
        rR, vR, pR = reconstructSide(
            rho[b], vb, pres[b], rho[a], va, pres[a], gRho[b], gVel[b], gP[b], aRho[b], aVel[b], aP[b],
            -(scalar_t(1.0) - frac) * d, scalar_t(1.0) - frac, vframe, dt, gamma, order, psi1, psi2, timeCentred)
        rL = wp.max(rL, scalar_t(_TINY))
        rR = wp.max(rR, scalar_t(_TINY))
        pL = wp.max(pL, scalar_t(_TINY))
        pR = wp.max(pR, scalar_t(_TINY))
        uL = wp.dot(vL, nrm)
        uR = wp.dot(vR, nrm)
        vtL = vL - uL * nrm
        vtR = vR - uR * nrm
        Fm, Fp, Fe, Ss, Ps = faceFluxPair(rL, uL, vtL, pL, rR, uR, vtR, pR, gamma, mode, nrm, vframe)
        if retry != 0:
            # GIZMO's failure handling: invalid star pressure -> first-order states -> zero velocity jump
            s1 = wp.max(scalar_t(0.0), -wp.dot(vb - va, d) / wp.max(r, scalar_t(_TINY)))
            s2 = wp.max(scalar_t(0.0), wp.dot(va - vb, nrm))
            v2 = wp.max(s1, s2)
            v2 = v2 * v2
            limit = scalar_t(1.1) * wp.max(pres[a] + rho[a] * v2, pres[b] + rho[b] * v2)
            if mode == 1:
                limit = limit * scalar_t(2.0)
            ok = Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss) and Ps <= scalar_t(1.4) * limit and wp.isfinite(Fm) and wp.isfinite(Fe)
            if not ok:
                vL1 = va - vframe
                vR1 = vb - vframe
                uL1 = wp.dot(vL1, nrm)
                uR1 = wp.dot(vR1, nrm)
                r1L = wp.max(rho[a], scalar_t(_TINY))
                r1R = wp.max(rho[b], scalar_t(_TINY))
                p1L = wp.max(pres[a], scalar_t(_TINY))
                p1R = wp.max(pres[b], scalar_t(_TINY))
                Fm, Fp, Fe, Ss, Ps = faceFluxPair(r1L, uL1, vL1 - uL1 * nrm, p1L, r1R, uR1, vR1 - uR1 * nrm, p1R,
                                                  gamma, mode, nrm, vframe)
                ok2 = Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss) and wp.isfinite(Fm) and wp.isfinite(Fe)
                if not ok2:
                    zero = scalar_t(0.0)
                    Fm, Fp, Fe, Ss, Ps = faceFluxPair(r1L, zero, vL1 * zero, p1L, r1R, zero, vR1 * zero, p1R,
                                                      gamma, mode, nrm, vframe)
        if mode == 1 and massLimit > scalar_t(0.0) and dt > scalar_t(0.0):
            # GIZMO: a pair moves at most massLimit of the donor's mass per step (mass update only)
            dmass = Amag * Fm * dt
            cap = massLimit * rho[a] * vol[a]
            if dmass < scalar_t(0.0):
                cap = massLimit * rho[b] * vol[b]
            Fm = wp.max(wp.min(dmass, cap), -cap) / (Amag * dt)
        # dQ_a -= |A| F, dQ_b += |A| F
        sgn = scalar_t(1.0)
        if i == a:
            sgn = scalar_t(-1.0)
        dm += sgn * Amag * Fm
        dp += sgn * Amag * Fp
        de += sgn * Amag * Fe
    return dm, dp, de


@wp.kernel
def mfmFlux_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    aRho: wp.array(dtype=scalar_t), aVel: wp.array(dtype=Any), aP: wp.array(dtype=scalar_t),  # type: ignore
    vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), lam: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
    gamma: scalar_t, dt: scalar_t, mode: wp.int32, order: wp.int32, psi1: scalar_t, psi2: scalar_t,
    power: scalar_t, timeCentred: wp.int32, centred: wp.int32, areaCap: wp.int32, retry: wp.int32, massLimit: scalar_t,

    output_dm: wp.array(dtype=scalar_t), output_dp: wp.array(dtype=Any), output_de: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    dm = scalar_t(0.0)
    dp = zero_like_warp(output_dp[i])
    de = scalar_t(0.0)
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        dm, dp, de = mfmFlux_Func_i(
            i, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            rho, vel, pres, gRho, gVel, gP, aRho, aVel, aP, vol, Einv, lam, condBad,
            gamma, dt, mode, order, psi1, psi2, power, timeCentred, centred, areaCap, retry, massLimit, dm, dp, de)
    # rates dQ/dt = -sum_j |A| F with the sign convention folded into the accumulation above
    output_dm[i] = dm
    output_dp[i] = dp
    output_de[i] = de


_FLUX_SPEC = OperatorSpec(
    kernel=mfmFlux_Kernel,
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR), ExtraSpec("vel", ExtraKind.TENSOR), ExtraSpec("pres", ExtraKind.TENSOR),
            ExtraSpec("gRho", ExtraKind.TENSOR), ExtraSpec("gVel", ExtraKind.TENSOR), ExtraSpec("gP", ExtraKind.TENSOR),
            ExtraSpec("aRho", ExtraKind.TENSOR), ExtraSpec("aVel", ExtraKind.TENSOR), ExtraSpec("aP", ExtraKind.TENSOR),
            ExtraSpec("vol", ExtraKind.TENSOR), ExtraSpec("Einv", ExtraKind.TENSOR), ExtraSpec("lam", ExtraKind.TENSOR),
            ExtraSpec("condBad", ExtraKind.TENSOR),
            ExtraSpec("gamma", ExtraKind.SCALAR), ExtraSpec("dt", ExtraKind.SCALAR), ExtraSpec("mode", ExtraKind.SCALAR),
            ExtraSpec("order", ExtraKind.SCALAR), ExtraSpec("psi1", ExtraKind.SCALAR), ExtraSpec("psi2", ExtraKind.SCALAR),
            ExtraSpec("power", ExtraKind.SCALAR), ExtraSpec("timeCentred", ExtraKind.SCALAR),
            ExtraSpec("centred", ExtraKind.SCALAR), ExtraSpec("areaCap", ExtraKind.SCALAR), ExtraSpec("retry", ExtraKind.SCALAR),
            ExtraSpec("massLimit", ExtraKind.SCALAR)),
)



# ---------------------------------------------------------------------------------------
# 5b. per-particle coarse-stencil sums for the cell multigrid of the closure solve
# ---------------------------------------------------------------------------------------

@wp.func
def _wrapDiff(diff: wp.int32, n: wp.int32, periodic: wp.int32):
    if periodic != 0:
        if diff > 1:
            diff = diff - n
        if diff < -1:
            diff = diff + n
    return diff


@wp.kernel
def mfmCoarseStencil_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    kappa: wp.array(dtype=scalar_t), cellId: wp.array(dtype=wp.int32),
    n0: wp.int32, n1: wp.int32, n2: wp.int32, p0: wp.int32, p1: wp.int32, p2: wp.int32,

    output_S: wp.array(dtype=Any),  # type: ignore
):
    """``S[i, slot(delta)] = sum_j kappa_ij`` over the neighbours ``j`` whose cell is ``delta`` away from ``i``'s
    (``delta != 0``): the particle's share of the Galerkin coarse operator ``P^T L P`` of the cell aggregation."""
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    S = zero_like_warp(output_S[i])
    dim = domainState.dim
    ci = cellId[i]
    cz = ci % n2
    cy = (ci // n2) % n1
    cx = ci // (n2 * n1)
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        offsetArray = adjacencyState.neighborList if useAdjacency else gridState.sortIndex
        for n in range(numIndices):
            j = wp.int32(offsetArray[beginIndex + n])
            if j == i:
                continue
            jPtcl = getParticleData(referenceState, j)
            if wp.length(computeDistanceVec(iPtcl.position, jPtcl.position, domainState)) > wp.max(iPtcl.support, jPtcl.support):
                continue
            cj = cellId[j]
            jz = cj % n2
            jy = (cj // n2) % n1
            jx = cj // (n2 * n1)
            dx = _wrapDiff(jx - cx, n0, p0)
            dy = _wrapDiff(jy - cy, n1, p1)
            dz = _wrapDiff(jz - cz, n2, p2)
            if dx == 0 and dy == 0 and dz == 0:
                continue
            if dx < -1 or dx > 1 or dy < -1 or dy > 1 or dz < -1 or dz > 1:
                continue
            slot = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
            if dim == 2:
                slot = (dx + 1) * 3 + (dy + 1)
            if dim == 1:
                slot = dx + 1
            S[slot] += kappa[beginIndex + n]
    output_S[i] = S


def _stencilDtype(ctx, extras):
    return vector(length=3 ** _dimOf(ctx, extras), dtype=scalar_t)


_COARSE_STENCIL_SPEC = OperatorSpec(
    kernel=mfmCoarseStencil_Kernel, outputs=(OutputSpec(dtype=_stencilDtype),),
    extras=(ExtraSpec("kappa", ExtraKind.TENSOR), ExtraSpec("cellId", ExtraKind.TENSOR),
            ExtraSpec("n0", ExtraKind.SCALAR), ExtraSpec("n1", ExtraKind.SCALAR), ExtraSpec("n2", ExtraKind.SCALAR),
            ExtraSpec("p0", ExtraKind.SCALAR), ExtraSpec("p1", ExtraKind.SCALAR), ExtraSpec("p2", ExtraKind.SCALAR)),
)

# ---------------------------------------------------------------------------------------
# 6. signal-velocity time step
# ---------------------------------------------------------------------------------------

@wp.kernel
def mfmTimestep_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gamma: scalar_t, cfl: scalar_t,

    output_dt: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    iPtcl = getParticleData(queryState, i)
    ci = wp.sqrt(gamma * pres[i] / rho[i])
    vsig = scalar_t(2.0) * ci
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        offsetArray = adjacencyState.neighborList if useAdjacency else gridState.sortIndex
        for n in range(numIndices):
            j = wp.int32(offsetArray[beginIndex + n])
            if j == i:
                continue
            jPtcl = getParticleData(referenceState, j)
            d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
            r = wp.length(d)
            if r > wp.max(iPtcl.support, jPtcl.support):
                continue
            cj = wp.sqrt(gamma * pres[j] / rho[j])
            approach = wp.min(scalar_t(0.0), wp.dot(vel[j] - vel[i], d) / wp.max(r, scalar_t(_TINY)))
            vsig = wp.max(vsig, ci + cj - approach)
    output_dt[i] = scalar_t(2.0) * cfl * iPtcl.support / vsig


_TIMESTEP_SPEC = OperatorSpec(
    kernel=mfmTimestep_Kernel, outputs=(OutputSpec(dtype=scalar_t),),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR), ExtraSpec("vel", ExtraKind.TENSOR), ExtraSpec("pres", ExtraKind.TENSOR),
            ExtraSpec("gamma", ExtraKind.SCALAR), ExtraSpec("cfl", ExtraKind.SCALAR)),
)


# ---------------------------------------------------------------------------------------
# python entry points (thin: build the context, launch)
# ---------------------------------------------------------------------------------------

def _ctx(particles, props, domain, adjacency):
    return SPHContext(query=particles, properties=props, domain=domain, adjacency=adjacency,
                      corrections=Corrections(volumes=(None, None)))


def mfmMomentsWarp(particles, props, domain, adjacency=None):
    with record_function("warpSPH[MFMMoments]"):
        return launchOperator(_MOMENTS_SPEC, _ctx(particles, props, domain, adjacency))


def mfmGradientsWarp(particles, props, domain, adjacency, rho, vel, pres, Einv, deficient):
    with record_function("warpSPH[MFMGradients]"):
        return launchOperator(_GRADIENTS_SPEC, _ctx(particles, props, domain, adjacency), rho=rho, vel=vel, pres=pres,
                              Einv=Einv, deficient=deficient)


def mfmLimiterWarp(particles, props, domain, adjacency, rho, vel, pres, gRho, gVel, gP, beta):
    with record_function("warpSPH[MFMLimiter]"):
        return launchOperator(_LIMITER_SPEC, _ctx(particles, props, domain, adjacency), rho=rho, vel=vel, pres=pres,
                              gRho=gRho, gVel=gVel, gP=gP, beta=beta)


def mfmClosureResidualWarp(particles, props, domain, adjacency, vol, Einv, condBad, centred=1, areaCap=0):
    with record_function("warpSPH[MFMClosureResidual]"):
        return launchOperator(_CLOSURE_RES_SPEC, _ctx(particles, props, domain, adjacency), vol=vol, Einv=Einv,
                              condBad=condBad, centred=int(centred), areaCap=int(areaCap))


def mfmClosureWeightsWarp(particles, props, domain, adjacency, vol, Einv, condBad, power, centred=1, areaCap=0):
    with record_function("warpSPH[MFMClosureWeights]"):
        return launchOperator(_CLOSURE_W_SPEC, _ctx(particles, props, domain, adjacency), vol=vol, Einv=Einv,
                              condBad=condBad, centred=int(centred), areaCap=int(areaCap), power=scalar_t(power))


def mfmClosureMatvecWarp(particles, props, domain, adjacency, kappa, p):
    with record_function("warpSPH[MFMClosureMatvec]"):
        return launchOperator(_CLOSURE_MV_SPEC, _ctx(particles, props, domain, adjacency), kappa=kappa, p=p)


def mfmFluxWarp(particles, props, domain, adjacency, rho, vel, pres, gRho, gVel, gP, aRho, aVel, aP, vol, Einv, lam,
                condBad, gamma, dt, mode, order, psi1, psi2, power, timeCentred, centred=1, areaCap=0, retry=1,
                massLimit=0.1):
    with record_function("warpSPH[MFMFlux]"):
        return launchOperator(
            _FLUX_SPEC, _ctx(particles, props, domain, adjacency), rho=rho, vel=vel, pres=pres, gRho=gRho, gVel=gVel,
            gP=gP, aRho=aRho, aVel=aVel, aP=aP, vol=vol, Einv=Einv, lam=lam, condBad=condBad, gamma=scalar_t(gamma), dt=scalar_t(dt),
            mode=int(mode), order=int(order), psi1=scalar_t(psi1), psi2=scalar_t(psi2), power=scalar_t(power),
            timeCentred=int(timeCentred), centred=int(centred), areaCap=int(areaCap), retry=int(retry), massLimit=scalar_t(massLimit))


def mfmTimestepWarp(particles, props, domain, adjacency, rho, vel, pres, gamma, cfl):
    with record_function("warpSPH[MFMTimestep]"):
        return launchOperator(_TIMESTEP_SPEC, _ctx(particles, props, domain, adjacency), rho=rho, vel=vel, pres=pres,
                              gamma=scalar_t(gamma), cfl=scalar_t(cfl))


def mfmCoarseStencilWarp(particles, props, domain, adjacency, kappa, cellId, n, periodic):
    with record_function("warpSPH[MFMCoarseStencil]"):
        return launchOperator(
            _COARSE_STENCIL_SPEC, _ctx(particles, props, domain, adjacency), kappa=kappa, cellId=cellId,
            n0=int(n[0]), n1=int(n[1]), n2=int(n[2]), p0=int(periodic[0]), p1=int(periodic[1]), p2=int(periodic[2]))
