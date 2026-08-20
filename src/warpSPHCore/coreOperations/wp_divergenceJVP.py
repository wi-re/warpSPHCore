"""Tier-2 position/support/mass/density-tangent JVP of the Divergence
operator (`warpier_tier2_operators_plan.md` Step 5, `warpier_adjoint.md`
Tier 2.2): `dDivergence_i = sum_j [ dot(dcoeff_ij, G_ij) + dot(coeff_ij,
dG_ij) ]`, `coeff_ij = fi*A_ij + fj*B_ij` (`fi`/`fj` = frozen vector-valued
`queryValues`/`referenceValues`), `dotMode=False` only (`divergenceDotMode`
is out of Tier-2 scope, enforced centrally in `operations.py`).

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
from ..kernels.kernelJVP import sphKernelGradientJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import allocateTorchWarp, castTorchToWarpAsBuiltins
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i
from ._jvpCommon import (
    buildParticleSoA as _buildParticleSoA,
    buildDomainState as _buildDomainState,
    buildKernelState as _buildKernelState,
    gradientWeightsJVP as _gradientWeightsJVP,
    buildAdjacencyOrGridState as _buildAdjacencyOrGridState,
    buildNullCorrectionData as _buildNullCorrectionData,
)

__all__ = ['computeSPHDivergencePositionJVP']


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

        G, dG = sphKernelGradientJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        A, B, dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
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
    correctionData: Any,
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
    correctionData: Any,

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
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHDivergencePositionJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
    tangentQuerySupports: Optional[torch.Tensor] = None,
    tangentReferenceSupports: Optional[torch.Tensor] = None,
    tangentReferenceMasses: Optional[torch.Tensor] = None,
    tangentQueryDensities: Optional[torch.Tensor] = None,
    tangentReferenceDensities: Optional[torch.Tensor] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dDivergence_i`, shape `[numParticles]`. `queryValues`/
    `referenceValues` (`fi`/`fj`, `[numParticles, dim]` vector fields) are
    required and frozen. `queryParticles.densities`/
    `referenceParticles.densities` must already hold real values, same
    requirement as `computeSPHGradientPositionJVP`. `adjacency` is an
    `AdjacencyList` or `CompactHashMap`.
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHDivergencePositionJVP: queryValues and referenceValues "
            "(frozen fi/fj) are both required."
        )

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    tangentReferencePositions = tangentReferencePositions if tangentReferencePositions is not None else zerosVec(nRef)
    tangentQuerySupports = tangentQuerySupports if tangentQuerySupports is not None else zerosScalar(nQuery)
    tangentReferenceSupports = tangentReferenceSupports if tangentReferenceSupports is not None else zerosScalar(nRef)
    tangentReferenceMasses = tangentReferenceMasses if tangentReferenceMasses is not None else zerosScalar(nRef)
    tangentQueryDensities = tangentQueryDensities if tangentQueryDensities is not None else zerosScalar(nQuery)
    tangentReferenceDensities = tangentReferenceDensities if tangentReferenceDensities is not None else zerosScalar(nRef)

    queryState = _buildParticleSoA(
        dim, queryParticles.positions, queryParticles.supports, queryParticles.masses, queryParticles.densities,
    )
    referenceState = _buildParticleSoA(
        dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        referenceParticles.densities,
    )
    queryTangentState = _buildParticleSoA(
        dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery), tangentQueryDensities,
    )
    referenceTangentState = _buildParticleSoA(
        dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses, tangentReferenceDensities,
    )
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode, gradientMode=gradientMode)
    correctionData = _buildNullCorrectionData(dim, device)

    useAdjacency, adjacencyState, gridState, _numOffsets = _buildAdjacencyOrGridState(adjacency, domain)

    queryValuesWarp = castTorchToWarpAsBuiltins(queryValues.contiguous())
    referenceValuesWarp = castTorchToWarpAsBuiltins(referenceValues.contiguous())
    warpDevice = queryState.positions.device
    dDivergence_t, dDivergence_w = allocateTorchWarp(nQuery, queryState.masses.dtype, warpDevice)

    wp.launch(
        computeSPHDivergenceJVP_Kernel,
        dim=nQuery,
        inputs=[
            queryState, referenceState,
            queryTangentState, referenceTangentState,
            domainState,
            useAdjacency, adjacencyState, gridState,
            correctionData,
            kernelProperties,
            queryValuesWarp, referenceValuesWarp,
            dDivergence_w,
        ],
        device=warpDevice,
    )
    return dDivergence_t
