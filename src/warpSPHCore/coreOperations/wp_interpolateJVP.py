"""Geometry-tangent JVP of the Interpolate
operator (`warpier_tier2_operators_plan.md` Step 2, `warpier_adjoint.md`
Tier 2.1): `dInterpolate_i = sum_j fj * (dVj * W_ij + Vj * dW_ij)`, with
`Vj = mass_j / density_j`, `dVj = dmass_j/density_j - mass_j*ddensity_j/density_j^2`,
`fj` (`referenceValues`) held **frozen** here -- this is the geometry-tangent
**partial** contribution only; `warpOperationJVP` sums it with the
value-tangent (value JVP) contribution when both are supplied
(`warpier_tier2_combined_jvp_plan.md`).

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`
Step 2): same canonical structured kernel ABI shape as `wp_interpolate.py`'s
`computeSPHInterpolation_Func_i`/`_Func_Adjacency`/`_Kernel`, reusing
Density's proven `sphKernelJVP`-per-neighbor building block plus this
operator's own `Vj`/`dVj` coefficient, no `GradientScheme`-style branching
either. Unlike the primal kernel, `correctionData.useVolume`/`.useCRK` are
never read here (the geometry JVP has no CRK/volume support, enforced centrally in
`operations.py`), so `Vj` is always `mass_j/density_j`, not the CRK-generic
primal formula. Also supports grid (`CompactHashMap`) traversal. Replaced
the original pair-indexed (COO) implementation once proven numerically
equivalent to float32 round-off; see git history around 2026-08-20 for that
implementation and its own equivalence tests if reference is ever needed.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp
from ..kernels.kernelJVP import sphKernelJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import castTorchToWarpAsBuiltins
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i, getParticleCorrectionTangentData_i
from ._jvpCommon import launchGeometryJVP as _launchGeometryJVP

__all__ = ['computeSPHInterpolateGeometryJVP']


@wp.func
def computeSPHInterpolateJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        jTangentPtcl = getParticleData(referenceTangentState, j)

        W, dW = sphKernelJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        Vj = jPtcl.mass / jPtcl.density
        dVj = jTangentPtcl.mass / jPtcl.density - jPtcl.mass * jTangentPtcl.density / (jPtcl.density * jPtcl.density)

        fv = referenceValues[j]
        out += fv * (dVj * W + Vj * dW)

    return out


@wp.func
def computeSPHInterpolateJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHInterpolateJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            iCorrectionTangentData, correctionTangentData,

            referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHInterpolateJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHInterpolateJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHInterpolateGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`dInterpolate_i`, shape `[numParticles, *referenceValues.shape[1:]]`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Interpolate's JVP -- `referenceValues` is held at its **primal**
    (non-tangent) value here. It is **not** the full derivative on its own;
    add the value-tangent (value JVP) contribution (`warpOperation` relaunched
    with the tangent value array) for that, or call `warpOperationJVP`
    directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).

    `referenceValues` (`fj`) is required and frozen (no tangent on it --
    that would be the value JVP). `queryValues` (`fi`) is not part of Interpolate's
    formula at all and must not be provided. `adjacency` is an `AdjacencyList`
    or `CompactHashMap`.
    """
    if referenceValues is None:
        raise ValueError("computeSPHInterpolateGeometryJVP: referenceValues (frozen fj) is required.")
    if queryValues is not None:
        raise ValueError(
            "computeSPHInterpolateGeometryJVP: Interpolate has no queryValues (fi) term "
            "-- only referenceValues (fj) is used, matching warpOperation(Interpolate)."
        )

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )

    outputDtype = castTorchToWarpAsBuiltins(referenceValues).dtype

    return _launchGeometryJVP(
        computeSPHInterpolateJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=outputDtype,
        referenceDensities=referenceParticles.densities,
        extraTensors=(referenceValues,),
    )
