"""Geometry-tangent JVP of the Divergence
operator (`warpier_tier2_operators_plan.md` Step 5, `warpier_adjoint.md`
Tier 2.2): `dDivergence_i = sum_j [ dot(dcoeff_ij, G_ij) + dot(coeff_ij,
dG_ij) ]`, `coeff_ij = fi*A_ij + fj*B_ij` (`fi`/`fj` = frozen vector-valued
`queryValues`/`referenceValues`), `dotMode=False` only (`divergenceDotMode`
is out of geometry-JVP scope, enforced centrally in `operations.py`).

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`
Step 3): same `_jvpCommon.gradientWeightsJVP` coefficient building block
Gradient's CSR port uses, combined via `wp.dot` instead of a bare scalar
multiply since `fi`/`fj` here are vector-valued. Also supports grid
(`CompactHashMap`) traversal. Replaced the original pair-indexed (COO)
implementation once proven numerically equivalent to float32 round-off; see
git history around 2026-08-20 for that implementation and its own
equivalence tests if reference is ever needed.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp
from ..crk import computeKernelGradientCRKJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i, getParticleCorrectionTangentData_i
from ._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)

__all__ = ['computeSPHDivergenceGeometryJVP']


# ---------------------------------------------------------------------------
# CSR (per-query-particle) launch shape.
# ---------------------------------------------------------------------------


@wp.func
def computeSPHDivergenceJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

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

        # CRK tangent extension (`warpier_tier2_correction_jvp_plan.md` phase (e)):
        # same computeKernelGradientCRKJVP swap Gradient's own JVP uses
        # (wp_gradientJVP.py) -- dispatches on correctionData.useCRK, so this is a
        # no-op (identical to the plain sphKernelGradientJVP call it replaces) when
        # CRK isn't in use.
        G, dG = computeKernelGradientCRKJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK,
            iCorrectionData.A, iCorrectionData.B, iCorrectionData.gradA, iCorrectionData.gradB,
            iCorrectionTangentData.A, iCorrectionTangentData.B, iCorrectionTangentData.gradA, iCorrectionTangentData.gradB,
        )

        A, B, dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        fj = referenceValues[j]
        coeff = fi * A + fj * B
        dcoeff = fi * dA + fj * dB

        out += wp.dot(dcoeff, G) + wp.dot(coeff, dG)

    return out


@wp.func
def computeSPHDivergenceJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: Any, referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHDivergenceJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            iCorrectionTangentData, correctionTangentData,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHDivergenceJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHDivergenceJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHDivergenceGeometryJVP(
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
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    crkTangentState: Optional[CRKTangentState] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dDivergence_i`, shape `[numParticles]`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Divergence's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).

    `queryValues`/`referenceValues` (`fi`/`fj`, `[numParticles, dim]` vector
    fields) are required and frozen here. `queryParticles.densities`/
    `referenceParticles.densities` must already hold real values, same
    requirement as `computeSPHGradientGeometryJVP`. `adjacency` is an
    `AdjacencyList` or `CompactHashMap`. `referenceVolumes`/
    `tangentReferenceVolumes` (`warpier_tier2_correction_jvp_plan.md` phase
    b) enable apparent-volume support and its tangent, same as Gradient
    (`consistentDivergence`/`divergenceDotMode` stay unsupported here,
    rejected centrally by `operations.py`, so the plain `apparentVolume`
    formula applies unconditionally). `crkState`/`crkTangentState` (phase
    (e)) enable CRK correction and its tangent, matching
    `warpOperation(..., crkState=...)` -- reuses phase (c)'s CRK JVP
    building block (`crk.computeKernelGradientCRKJVP`) verbatim, combined
    via Divergence's own `dot(dcoeff,G) + dot(coeff,dG)` formula instead of
    Gradient's `dcoeff*G + coeff*dG`. `crkTangentState` may be omitted
    (treated as an all-zero tangent), same as Gradient.
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHDivergenceGeometryJVP: queryValues and referenceValues "
            "(frozen fi/fj) are both required."
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
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
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

    return _launchGeometryJVP(
        computeSPHDivergenceJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=scalar_t,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        crkState=crkState,
        crkTangentState=crkTangentState,
        extraTensors=(queryValues, referenceValues),
    )
