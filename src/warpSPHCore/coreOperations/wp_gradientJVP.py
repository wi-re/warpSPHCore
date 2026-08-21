"""Geometry-tangent JVP of the Gradient operator
(`warpier_tier2_operators_plan.md` Step 4, `warpier_adjoint.md` Tier 2.2):
`dGradient_i = sum_j [ dcoeff_ij*G_ij + coeff_ij*dG_ij ]`, `coeff_ij =
fi*A_ij + fj*B_ij` (`fi`/`fj` = frozen `queryValues`/`referenceValues`),
`(G_ij, dG_ij)` from `kernels.kernelJVP.sphKernelGradientJVP`, `(A, B, dA,
dB)` from `_jvpCommon.gradientWeightsJVP`.

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`
Step 3, first of "the shared-(G,dG) four"): the `A`/`B`/`dA`/`dB`
coefficient/combination step lives in the kernel body via
`_jvpCommon.gradientWeightsJVP`, mirroring the primal `wp_gradient.py`
kernel's own per-`GradientScheme` branching. The geometry JVP.s scope is scalar
`fi`/`fj` only (no CRK/volume/grad-h, no vector- or higher-rank fields), so
unlike primal `wp_gradient.py`'s generic `outerTensorProduct`/
`flatOutputShape` machinery, the output here is always exactly a
`dim`-length vector -- `coeff*kernelGradient` directly. Also supports grid
(`CompactHashMap`) traversal. Replaced the original pair-indexed (COO,
pure-torch-coefficient-assembly-on-a-shared-pair-kernel) implementation once
proven numerically equivalent to float32 round-off; see git history around
2026-08-20 for that implementation and its own equivalence tests if
reference is ever needed.
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
from ..util import _get_warp_vector_dtype
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i, getParticleCorrectionTangentData_i
from ._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)

__all__ = ['computeSPHGradientGeometryJVP']


# ---------------------------------------------------------------------------
# CSR (per-query-particle) launch shape.
# ---------------------------------------------------------------------------


@wp.func
def computeSPHGradientJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    fi: scalar_t, referenceValues: wp.array(dtype = scalar_t), # type: ignore

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
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        fj = referenceValues[j]
        coeff = fi * A + fj * B
        dcoeff = fi * dA + fj * dB

        out += dcoeff * G + coeff * dG

    return out


@wp.func
def computeSPHGradientJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

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

        out += computeSPHGradientJVP_Func_i(
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
def computeSPHGradientJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHGradientJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHGradientGeometryJVP(
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
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dGradient_i`, shape `[numParticles, dim]`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Gradient's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).

    `queryValues`/`referenceValues` (`fi`/`fj`, scalar fields) are required
    and frozen (no tangent on them here). `queryParticles.densities`/
    `referenceParticles.densities` must already hold real (non-dummy) values
    -- `coeff_ij` depends on them directly, same requirement as
    `computeSPHInterpolateGeometryJVP`. `adjacency` is an `AdjacencyList` or
    `CompactHashMap`. `referenceVolumes`/`tangentReferenceVolumes`
    (`warpier_tier2_correction_jvp_plan.md` phase b) enable apparent-volume
    support and its tangent, matching `warpOperation(..., referenceVolumes=...)`
    -- `GradientScheme.Symmetric` ignores both (its coefficient has no
    apparent-volume term).
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHGradientGeometryJVP: queryValues and referenceValues "
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
        computeSPHGradientJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=_get_warp_vector_dtype(dim, dtype),
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        extraTensors=(queryValues, referenceValues),
    )
