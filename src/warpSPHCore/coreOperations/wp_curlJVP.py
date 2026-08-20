"""Geometry-tangent JVP of the Curl operator, 2D
only (`warpier_tier2_operators_plan.md` Step 6, `warpier_adjoint.md` Tier
2.2): `dCurl_i = sum_j [ G_ij.x*dcoeff_ij.y - G_ij.y*dcoeff_ij.x +
dG_ij.x*coeff_ij.y - dG_ij.y*coeff_ij.x ]` (the product-rule expansion of
`wp_curl.py`'s 2D scalar cross, `curlProduct`, `math/wp_cross.py`),
`coeff_ij = fi*A_ij + fj*B_ij` (`fi`/`fj` = frozen vector-valued
`queryValues`/`referenceValues`). `domain.dim != 2` is rejected centrally in
`operations.py` (1D/3D both undecided by the spike).

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`
Step 3), fixed at `dim=2`. Produces a `scalar_t` per query particle
internally; the torch-level entry point unsqueezes to `[nQuery, 1]`
afterward. Also supports grid (`CompactHashMap`) traversal. Replaced the
original pair-indexed (COO) implementation once proven numerically
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
from ..kernels.kernelJVP import sphKernelGradientJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i
from ._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)

__all__ = ['computeSPHCurlGeometryJVP']


# ---------------------------------------------------------------------------
# CSR (per-query-particle) launch shape.
# ---------------------------------------------------------------------------


@wp.func
def computeSPHCurlJVP_Func_i(
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

        out += G[0] * dcoeff[1] - G[1] * dcoeff[0] + dG[0] * coeff[1] - dG[1] * coeff[0]

    return out


@wp.func
def computeSPHCurlJVP_Func_Adjacency(
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

        out += computeSPHCurlJVP_Func_i(
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
def computeSPHCurlJVP_Kernel(
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

    outputValues[i] = computeSPHCurlJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHCurlGeometryJVP(
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
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dCurl_i`, shape `[numParticles, 1]` (matching production
    `warpOperation(Curl)`'s own `[1]`-forced output shape for a 2D
    vector-field input, `wp_curl.py`).

    This is the geometry/mass/density-tangent **partial** contribution to
    Curl's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).

    `queryValues`/`referenceValues` (`fi`/`fj`, `[numParticles, 2]` vector
    fields) are required and frozen here. `queryParticles.densities`/
    `referenceParticles.densities` must already hold real values, same
    requirement as `computeSPHGradientGeometryJVP`. `adjacency` is an
    `AdjacencyList` or `CompactHashMap`.
    """
    if domain.dim != 2:
        raise ValueError("computeSPHCurlGeometryJVP: only domain.dim == 2 is implemented.")
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHCurlGeometryJVP: queryValues and referenceValues "
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

    dCurl_t = _launchGeometryJVP(
        computeSPHCurlJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=scalar_t,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        extraTensors=(queryValues, referenceValues),
    )
    return dCurl_t.unsqueeze(-1)
