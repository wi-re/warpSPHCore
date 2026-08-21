"""Geometry-tangent JVP of the raw covariance matrix `wp_covariance.py` computes
(`warpier_tier2_correction_jvp_plan.md` phase (d), Step 1;
`scripts/spike_forward_mode_tier2_renorm.py`'s already-validated math):

    C_i  = Sum_j Vj * outer(y_ij, G_ij)
    dC_i = Sum_j [ dVj * outer(y_ij, G_ij) + Vj * outer(dy_ij, G_ij) + Vj * outer(y_ij, dG_ij) ]

`y_ij = -computeDistanceVec(xi, xj) = xj - xi` (`wp_covariance.py`'s own `fij`),
`Vj = apparentVolume` (`gradientWeightsJVP`'s `GradientScheme.Naive` branch,
`B_ij = Vj` -- literally the same `mass_j/density_j` formula, reused here rather
than re-derived, exactly like the spike's own `_apparent_volume_jvp` docstring
notes), `(G_ij, dG_ij)` from `kernels.kernelJVP.sphKernelGradientJVP`.

Deliberately **plain, uncorrected** kernel-gradient JVP -- no CRK, no
renormalization dispatch inside this kernel -- matching
`computeRenormalizationMatrices_`'s own internal covariance call, which is
"only ever called here with `crkState=None, renormalizationState=None`"
(confirmed by reading it; see the spike's module docstring point 1). CRK
applied simultaneously with renormalization is explicitly out of scope for
this plan (see the plan's phase (d) entry and "Explicitly out of scope"
section).

This file produces only the raw, UNMASKED `dC_i` (this operator's own JVP
counterpart to `wp_covariance.py`'s `C_i`) -- the low-neighbor-count identity
fallback's zero-tangent masking and the `-L(dC)L` pseudo-inverse-derivative
identity that turn this into a renormalization-matrix tangent `dL_i` are
`renorm.py`'s `computeRenormalizationMatricesJVP`'s job (mirroring how
`wp_covariance.py` itself only computes the covariance matrix, leaving the
fallback/pseudo-inverse step to `renorm.py`'s `computeRenormalizationMatrices_`).
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp, computeDistanceVec
from ..kernels.kernelJVP import sphKernelGradientJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import _get_warp_matrix_dtype, checkDirectionality_i, checkDirectionality_j, getParticleData
from ._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)

__all__ = ['computeCovarianceGeometryJVP']


@wp.func
def computeCovarianceJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    correctionData: Any, correctionTangentData: Any,

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

        # GradientScheme.Naive's B/dB is literally Vj/dVj (apparentVolume and its
        # tangent, `useVolume`-aware) -- reused rather than re-derived, see module
        # docstring. A/dA (Naive's unused first return) are discarded.
        _unusedA, Vj, _unusedDA, dVj = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            wp.static(GradientScheme.Naive.value),
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        G, dG = sphKernelGradientJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        fij = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        dfij = jTangentPtcl.position - iTangentPtcl.position

        out += dVj * wp.outer(fij, G) + Vj * wp.outer(dfij, G) + Vj * wp.outer(fij, dG)

    return out


@wp.func
def computeCovarianceJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)
    iTangentPtcl = getParticleData(queryTangentState, i)

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeCovarianceJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState, kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            correctionData, correctionTangentData,

            outputValue,
        )
    return out


@wp.kernel
def computeCovarianceJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured JVP kernel ABI, see
    # coreOperations/_jvpCommon.py's launchGeometryJVP docstring.

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeCovarianceJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,

        zero_like_warp(outputValues[i]),
    )


def computeCovarianceGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`dC_i`, the RAW (unmasked) covariance-matrix JVP, shape
    `[numParticles, dim, dim]` -- see module docstring. `queryParticles.densities`/
    `referenceParticles.densities` must already hold real values (`Vj`'s
    `useVolume=False` branch depends on `densityJ` directly, same requirement
    as every other value-having operator's geometry JVP). `referenceVolumes`/
    `tangentReferenceVolumes` (`warpier_tier2_correction_jvp_plan.md` phase b)
    enable apparent-volume support and its tangent for `Vj`, matching
    `wp_covariance.py`'s own `correctionData.useVolume` branch. No query-side
    mass tangent term exists (`wp_covariance.py` never reads `iPtcl.mass`).
    """
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
        computeCovarianceJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=_get_warp_matrix_dtype(dim, dim, dtype),
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
    )
