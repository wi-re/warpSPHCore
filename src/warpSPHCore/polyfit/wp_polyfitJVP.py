"""Geometry-tangent (forward-mode) JVP of the polynomial-fit kernels
(`wp_polyfit.py`), following the Tier-2 conventions of `coreOperations/
wp_*JVP.py`: canonical JVP ABI (`queryState, referenceState,
queryTangentState, referenceTangentState, domainState, useAdjacency,
adjacencyState, gridState, correctionData, correctionTangentData,
kernelProperties`, extra tensors, extra scalars, output), launched through
`launchGeometryJVP` so the JVP kernels are themselves reverse-mode
differentiable.

The primal entries are ``w_ij P_a(xi_ij) P_b(xi_ij)`` (moments) and
``w_ij P_a(xi_ij) (f_j - kappa f_i)`` (right-hand side), ``w_ij = V_j W_ij``,
``xi_ij = (x_j - x_i)/h_i``, ``V_j = m_j/rho_j``. Their tangents are the
product rule, with ``dW`` from the existing `sphKernelJVP`,
``dV_j = dm_j/rho_j - m_j drho_j/rho_j^2``,
``dxi = -(dx_i - dx_j)/h_i + x_ij dh_i/h_i^2`` (with ``x_ij = x_i - x_j``) and
``dP_a = sum_k e_k xi_k^(e_k - 1) dxi_k prod_{m != k} xi_m^e_m`` written as
straight-line code (no loop-carried multiplication, see lessons_learned.md).
The right-hand-side kernel also folds in the *field* tangent
(``w P (df_j - kappa df_i)``) so one launch gives the total ``db``.

The dense part of the JVP of ``c = M^-1 b`` is done by the caller in torch:
``dc = M^-1 (db - dM c)``.
"""

from typing import Any, Optional

import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import *
from ..kernels.kernelJVP import sphKernelJVP
from ..radiusSearch.grid_util import getIndexRangeLane
from ..util import (checkDirectionality_i, checkDirectionality_j, getParticleData,
                    getParticleCorrectionData_i)
from ..coreOperations._jvpCommon import launchGeometryJVP as _launchGeometryJVP
from .wp_polyfit import polyIntPow, polyInSector, _momentRowCol

__all__ = ["computePolyMomentsGeometryJVP", "computePolyRHSGeometryJVP"]


@wp.func
def polyIntPowD(x: scalar_t, n: wp.int32):
    """``d/dx x**n = n x**(n-1)``, exactly 0 for ``n == 0``."""
    if n == 0:
        return scalar_t(0.0)
    return scalar_t(n) * polyIntPow(x, n - 1)


@wp.func
def polyMonomialJVP(e: Any, xi: Any, dxi: Any, dim: wp.int32):
    """``(P, dP)`` of the monomial ``prod_k xi_k**e_k`` (dim <= 3), straight-line."""
    r = polyIntPow(xi[0], e[0])
    dr = polyIntPowD(xi[0], e[0]) * dxi[0]
    if dim > 1:
        r1 = polyIntPow(xi[1], e[1])
        d1 = polyIntPowD(xi[1], e[1]) * dxi[1]
        dr = dr * r1 + r * d1
        r = r * r1
    if dim > 2:
        r2 = polyIntPow(xi[2], e[2])
        d2 = polyIntPowD(xi[2], e[2]) * dxi[2]
        dr = dr * r2 + r * d2
        r = r * r2
    return r, dr


@wp.func
def polyFitWeightJVP(
    i: wp.int32, j: wp.int32,
    iPtcl: Any, jPtcl: Any, iTan: Any, jTan: Any,
    domainState: domainData, kernelProperties: kernelState,
    correctionData: Any, correctionTangentData: Any, dim: wp.int32,
    sector: wp.int32, numSectors: wp.int32, unitWeights: wp.int32,
):
    """``(w, dw, xi, dxi)`` -- mirrors `wp_polyfit.polyFitWeight`."""
    if correctionData.useVolume:
        vj = correctionData.referenceVolumes[j]
        dvj = correctionTangentData.referenceVolumes[j]
    else:
        vj = jPtcl.mass / jPtcl.density
        dvj = jTan.mass / jPtcl.density - jPtcl.mass * jTan.density / (jPtcl.density * jPtcl.density)
    x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)   # x_i - x_j
    dx_ij = iTan.position - jTan.position
    h = iPtcl.support
    dh = iTan.support
    xi = (-x_ij) / h
    dxi = (-dx_ij) / h + x_ij * (dh / (h * h))
    w = scalar_t(0.0)
    dw = scalar_t(0.0)
    if unitWeights != 0:
        if wp.length(x_ij) < h:
            w = scalar_t(1.0)
    else:
        W, dW = sphKernelJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTan.position, jTan.position, iTan.support, jTan.support,
            kernelProperties, domainState)
        w = vj * W
        dw = dvj * W + vj * dW
    if i != j:
        if not polyInSector(-x_ij, dim, sector, numSectors):
            w = scalar_t(0.0)
            dw = scalar_t(0.0)
    return w, dw, xi, dxi


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------

@wp.func
def computePolyMomentsJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTan: Any, referenceState: Any, referenceTangentState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    correctionData: Any, correctionTangentData: Any,

    ea: Any,  # type: ignore
    eb: Any,  # type: ignore
    sector: wp.int32, numSectors: wp.int32, unitWeights: wp.int32,
):
    s = scalar_t(0.0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        jTan = getParticleData(referenceTangentState, j)
        w, dw, xi, dxi = polyFitWeightJVP(
            i, j, iPtcl, jPtcl, iTan, jTan, domainState, kernelProperties,
            correctionData, correctionTangentData, dim, sector, numSectors, unitWeights)
        pa, dpa = polyMonomialJVP(ea, xi, dxi, dim)
        pb, dpb = polyMonomialJVP(eb, xi, dxi, dim)
        s += dw * pa * pb + w * (dpa * pb + pa * dpb)
    return s


@wp.kernel
def computePolyMomentsJVP_Kernel(
    queryState: Any, referenceState: Any, queryTangentState: Any, referenceTangentState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,
    kernelProperties: kernelState,
    # canonical JVP ABI prefix -- do not change

    exponents: wp.array(dtype=Any),  # type: ignore
    rowCol: wp.array(dtype=wp.vec2i),
    sector: wp.int32, numSectors: wp.int32, unitWeights: wp.int32,

    output_dM: wp.array2d(dtype=scalar_t),  # type: ignore
):
    tid = wp.tid()
    L = rowCol.shape[0]
    i = tid // L
    l = tid - i * L
    if i >= queryState.positions.shape[0]:
        return
    rc = rowCol[l]
    dim = domainState.dim
    iPtcl = getParticleData(queryState, i)
    iTan = getParticleData(queryTangentState, i)
    s = scalar_t(0.0)
    ok = True
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        ok = checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode)
    if ok:
        numOffsets = gridState.numOffsets if not useAdjacency else 1
        for o in range(numOffsets):
            beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
            if beginIndex < 0:
                continue
            s += computePolyMomentsJVP_Func_i(
                i, dim, iPtcl, iTan, referenceState, referenceTangentState,
                domainState, kernelProperties,
                beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
                correctionData, correctionTangentData,
                exponents[rc[0]], exponents[rc[1]], sector, numSectors, unitWeights)
    output_dM[i, l] = s


# --------------------------------------------------------------------------
# right-hand side
# --------------------------------------------------------------------------

@wp.func
def computePolyRHSJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTan: Any, referenceState: Any, referenceTangentState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    correctionData: Any, correctionTangentData: Any,

    ea: Any,  # type: ignore
    values: wp.array(dtype=Any),  # type: ignore
    tangentValues: wp.array(dtype=Any),  # type: ignore
    centerValue: Any, dCenterValue: Any,  # type: ignore
    comp: wp.int32,
    sector: wp.int32, numSectors: wp.int32, unitWeights: wp.int32,
):
    s = scalar_t(0.0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        jTan = getParticleData(referenceTangentState, j)
        w, dw, xi, dxi = polyFitWeightJVP(
            i, j, iPtcl, jPtcl, iTan, jTan, domainState, kernelProperties,
            correctionData, correctionTangentData, dim, sector, numSectors, unitWeights)
        pa, dpa = polyMonomialJVP(ea, xi, dxi, dim)
        fj = values[j]
        dfj = tangentValues[j]
        s += (dw * pa + w * dpa) * (fj[comp] - centerValue[comp]) \
            + w * pa * (dfj[comp] - dCenterValue[comp])
    return s


@wp.kernel
def computePolyRHSJVP_Kernel(
    queryState: Any, referenceState: Any, queryTangentState: Any, referenceTangentState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,
    kernelProperties: kernelState,
    # canonical JVP ABI prefix -- do not change

    exponents: wp.array(dtype=Any),  # type: ignore
    values: wp.array(dtype=Any),  # type: ignore
    tangentValues: wp.array(dtype=Any),  # type: ignore
    subtractCenter: wp.int32, numComponents: wp.int32,
    sector: wp.int32, numSectors: wp.int32, unitWeights: wp.int32,

    output_db: wp.array2d(dtype=scalar_t),  # type: ignore
):
    tid = wp.tid()
    width = exponents.shape[0] * numComponents
    i = tid // width
    l = tid - i * width
    if i >= queryState.positions.shape[0]:
        return
    a = l // numComponents
    comp = l - a * numComponents
    dim = domainState.dim
    iPtcl = getParticleData(queryState, i)
    iTan = getParticleData(queryTangentState, i)
    centerValue = values[i] * scalar_t(0.0)
    dCenterValue = tangentValues[i] * scalar_t(0.0)
    if subtractCenter != 0:
        centerValue = values[i]
        dCenterValue = tangentValues[i]
    s = scalar_t(0.0)
    ok = True
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        ok = checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode)
    if ok:
        numOffsets = gridState.numOffsets if not useAdjacency else 1
        for o in range(numOffsets):
            beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
            if beginIndex < 0:
                continue
            s += computePolyRHSJVP_Func_i(
                i, dim, iPtcl, iTan, referenceState, referenceTangentState,
                domainState, kernelProperties,
                beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
                correctionData, correctionTangentData,
                exponents[a], values, tangentValues, centerValue, dCenterValue, comp,
                sector, numSectors, unitWeights)
    output_db[i, l] = s


# --------------------------------------------------------------------------
# Python entry points
# --------------------------------------------------------------------------

def _fullTangent(t: Optional[ParticleTangentState], like: ParticleState) -> ParticleTangentState:
    pos = like.positions
    n, dtype, device = pos.shape[0], pos.dtype, pos.device
    z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)
    if t is None:
        return ParticleTangentState(positions=z(n, pos.shape[1]), supports=z(n), masses=z(n), densities=z(n))
    return ParticleTangentState(
        positions=t.positions if t.positions is not None else z(n, pos.shape[1]),
        supports=t.supports if t.supports is not None else z(n),
        masses=t.masses if t.masses is not None else z(n),
        densities=t.densities if t.densities is not None else z(n))


def computePolyMomentsGeometryJVP(
    queryParticles: ParticleState, domain: DomainDescription, kernel: KernelFunctions,
    adjacency, queryTangentState: ParticleTangentState, exponents: torch.Tensor,
    supportMode: SupportScheme = SupportScheme.Gather,
    sector: int = -1, numSectors: int = 8, unitWeights: bool = False,
) -> torch.Tensor:
    """Tangent of the packed moment matrices, ``(N, n(n+1)/2)`` (same packing
    as `_computePolyMoments_stateBackend`). Query and reference roles are the
    same particle set."""
    tan = _fullTangent(queryTangentState, queryParticles)
    rowCol = _momentRowCol(exponents)
    N = queryParticles.positions.shape[0]
    L = rowCol.shape[0]
    return _launchGeometryJVP(
        computePolyMomentsJVP_Kernel, domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        tan, tan,
        outputShape=(N, L), outputDtype=scalar_t,
        queryDensities=queryParticles.densities, referenceDensities=queryParticles.densities,
        extraTensors=(exponents, rowCol),
        extraScalars=(int(sector), int(numSectors), int(unitWeights)),
        numThreads=N * L,
    )


def computePolyRHSGeometryJVP(
    queryParticles: ParticleState, domain: DomainDescription, kernel: KernelFunctions,
    adjacency, queryTangentState: Optional[ParticleTangentState], exponents: torch.Tensor,
    values: torch.Tensor, tangentValues: Optional[torch.Tensor], subtractCenter: bool,
    supportMode: SupportScheme = SupportScheme.Gather,
    sector: int = -1, numSectors: int = 8, unitWeights: bool = False,
) -> torch.Tensor:
    """Total tangent ``db`` of the right-hand sides, ``(N, n * D)``: geometry
    tangent plus the field tangent ``tangentValues`` (zeros if None).
    ``values`` is the 2-D ``(N, D)`` primal field."""
    tan = _fullTangent(queryTangentState, queryParticles)
    if tangentValues is None:
        tangentValues = torch.zeros_like(values)
    N = queryParticles.positions.shape[0]
    n = exponents.shape[0]
    D = int(values.shape[1])
    return _launchGeometryJVP(
        computePolyRHSJVP_Kernel, domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        tan, tan,
        outputShape=(N, n * D), outputDtype=scalar_t,
        queryDensities=queryParticles.densities, referenceDensities=queryParticles.densities,
        extraTensors=(exponents, values.contiguous(), tangentValues.contiguous()),
        extraScalars=(int(subtractCenter), D, int(sector), int(numSectors), int(unitWeights)),
        numThreads=N * n * D,
    )
