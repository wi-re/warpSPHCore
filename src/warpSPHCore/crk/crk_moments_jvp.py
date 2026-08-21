"""JVP counterpart to `crk_moments.py`'s raw kernel-weighted geometric
moments (`warpier_tier2_correction_jvp_plan.md` phase (c), Stage 2): given
Stage 1's apparent-volume value/tangent (`V_j`/`dV_j`, threaded through the
already-shipped `correctionData.referenceVolumes`/
`correctionTangentData.referenceVolumes` plumbing from phase (b) -- CRK's
`V_j` is a direct substitute for the same `useVolume` slot every value-having
operator's `gradientWeightsJVP` already reads, despite being numerically
unrelated to `mass_j/density_j`), produces the tangent of each of the six
moment accumulators (`m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma`) by
ordinary product rule. Always `SupportScheme.Scatter` for BOTH the kernel
value and its gradient (matching `_computeCRKMoments_stateBackend`'s own
hardcode, `crk_wrapper.py`'s `momentsProperties`).

Operator-agnostic (produces moment tangents regardless of which operator
eventually consumes A/B/gradA/gradB), so built once here for reuse by phase
(e)'s Divergence/Curl/Laplacian extension, same as `crk_volume_jvp.py`.

Ported from `scripts/spike_forward_mode_tier2_crk.py`'s
`assembled_crk_moments_jvp` (validated there to float64 round-off against
`_computeCRKMoments_stateBackend`'s own reverse-mode Jacobian). Mirrors
`computeCRKMoments_Func_i`/`_Func_Adjacency`/`_Kernel`'s dual-path
(adjacency/grid) structure and per-term formulas field-for-field -- no
loop-accumulator/nonlinear-post-loop-op concern here (unlike Stage 1): every
output is a plain linear accumulation returned directly, the same shape the
primal moments kernel itself already has.
"""

from typing import Any, Optional
import torch
import warp as wp
from warp.types import vector, matrix

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..radiusSearch.grid_util import getIndexRange
from ..math import zero_like_warp, computeDistanceVec, kroneckerDelta, warp_eye
from ..kernels.kernelJVP import sphKernelJVP_ij, sphKernelGradientJVP_ij
from ..util import (
    checkDirectionality_i, checkDirectionality_j, getParticleData,
    getParticleCorrectionData_i, getParticleCorrectionTangentData_i,
)
from ..util.stateUtil import getVolume_j, getVolumeTangent_j

__all__ = ['computeCRKMomentsGeometryJVP']


@wp.func
def computeCRKMomentsJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    output_dm_0: scalar_t, # type: ignore
    output_dm_1: vector(length=Any, dtype=scalar_t), # type: ignore
    output_dm_2: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_d_dm_0dgamma: vector(length=Any, dtype=scalar_t), # type: ignore
    output_d_dm_1dgamma: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_d_dm_2dgamma: vector(length=Any, dtype=scalar_t) # type: ignore (flattened, see crk_moments.py)
):
    dm_0 = zero_like_warp(output_dm_0)
    dm_1 = zero_like_warp(output_dm_1)
    dm_2 = zero_like_warp(output_dm_2)
    d_dm_0dgamma = zero_like_warp(output_d_dm_0dgamma)
    d_dm_1dgamma = zero_like_warp(output_d_dm_1dgamma)
    d_dm_2dgamma = zero_like_warp(output_d_dm_2dgamma)

    eye = warp_eye(iPtcl.position)

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

        _, V_j = getVolume_j(correctionData, j)
        _, dV_j = getVolumeTangent_j(correctionData, correctionTangentData, j)

        x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        dx_ij = iTangentPtcl.position - jTangentPtcl.position

        w_ij, dw_ij = sphKernelJVP_ij(
            x_ij, iPtcl.support, jPtcl.support, dx_ij, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )
        gradw_ij, dgradw_ij = sphKernelGradientJVP_ij(
            x_ij, iPtcl.support, jPtcl.support, dx_ij, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        VW = V_j * w_ij
        dVW = dV_j * w_ij + V_j * dw_ij

        dm_0 += dVW
        dm_1 += dx_ij * VW + x_ij * dVW
        dm_2 += (wp.outer(dx_ij, x_ij) + wp.outer(x_ij, dx_ij)) * VW + wp.outer(x_ij, x_ij) * dVW

        d_dm_0dgamma += dV_j * gradw_ij + V_j * dgradw_ij
        d_dm_1dgamma += (dV_j * (wp.outer(x_ij, gradw_ij) + w_ij * eye)
                         + V_j * (wp.outer(dx_ij, gradw_ij) + wp.outer(x_ij, dgradw_ij) + dw_ij * eye))

        for alpha in range(dim):
            for beta in range(dim):
                for gamma in range(dim):
                    gradTerm = x_ij[alpha] * x_ij[beta] * gradw_ij[gamma]
                    d_gradTerm = (dx_ij[alpha] * x_ij[beta] * gradw_ij[gamma]
                                  + x_ij[alpha] * dx_ij[beta] * gradw_ij[gamma]
                                  + x_ij[alpha] * x_ij[beta] * dgradw_ij[gamma])
                    deltaA = x_ij[alpha] * kroneckerDelta(beta, gamma)
                    deltaB = kroneckerDelta(alpha, gamma) * x_ij[beta]
                    d_deltaA = dx_ij[alpha] * kroneckerDelta(beta, gamma)
                    d_deltaB = kroneckerDelta(alpha, gamma) * dx_ij[beta]
                    kernelTerm = w_ij * (deltaA + deltaB)
                    d_kernelTerm = dw_ij * (deltaA + deltaB) + w_ij * (d_deltaA + d_deltaB)
                    d_dm_2dgamma[gamma * dim * dim + alpha * dim + beta] += dV_j * (gradTerm + kernelTerm) + V_j * (d_gradTerm + d_kernelTerm)

    return dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma


@wp.func
def computeCRKMomentsJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    output_dm_0: scalar_t, # type: ignore
    output_dm_1: vector(length=Any, dtype=scalar_t), # type: ignore
    output_dm_2: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_d_dm_0dgamma: vector(length=Any, dtype=scalar_t), # type: ignore
    output_d_dm_1dgamma: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_d_dm_2dgamma: vector(length=Any, dtype=scalar_t) # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return (
                zero_like_warp(output_dm_0), zero_like_warp(output_dm_1), zero_like_warp(output_dm_2),
                zero_like_warp(output_d_dm_0dgamma), zero_like_warp(output_d_dm_1dgamma), zero_like_warp(output_d_dm_2dgamma),
            )
    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

    dm_0 = zero_like_warp(output_dm_0)
    dm_1 = zero_like_warp(output_dm_1)
    dm_2 = zero_like_warp(output_dm_2)
    d_dm_0dgamma = zero_like_warp(output_d_dm_0dgamma)
    d_dm_1dgamma = zero_like_warp(output_d_dm_1dgamma)
    d_dm_2dgamma = zero_like_warp(output_d_dm_2dgamma)

    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        s_dm0, s_dm1, s_dm2, s_ddm0, s_ddm1, s_ddm2 = computeCRKMomentsJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,
            domainState, kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData, iCorrectionTangentData, correctionTangentData,

            output_dm_0, output_dm_1, output_dm_2, output_d_dm_0dgamma, output_d_dm_1dgamma, output_d_dm_2dgamma,
        )
        dm_0 += s_dm0
        dm_1 += s_dm1
        dm_2 += s_dm2
        d_dm_0dgamma += s_ddm0
        d_dm_1dgamma += s_ddm1
        d_dm_2dgamma += s_ddm2

    return dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma


@wp.kernel
def computeCRKMomentsJVP_Kernel(
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

    # The last parameters are always the output arrays and should not be changed
    output_dm_0 : wp.array(dtype = Any), # type: ignore
    output_dm_1 : wp.array(dtype = Any), # type: ignore
    output_dm_2 : wp.array(dtype = Any), # type: ignore
    output_d_dm_0dgamma : wp.array(dtype = Any), # type: ignore
    output_d_dm_1dgamma : wp.array(dtype = Any), # type: ignore
    output_d_dm_2dgamma : wp.array(dtype = Any), # type: ignore (flattened, see crk_moments.py)
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma = computeCRKMomentsJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,

        zero_like_warp(output_dm_0[i]), zero_like_warp(output_dm_1[i]), zero_like_warp(output_dm_2[i]),
        zero_like_warp(output_d_dm_0dgamma[i]), zero_like_warp(output_d_dm_1dgamma[i]), zero_like_warp(output_d_dm_2dgamma[i]),
    )

    output_dm_0[i] = dm_0
    output_dm_1[i] = dm_1
    output_dm_2[i] = dm_2
    output_d_dm_0dgamma[i] = d_dm_0dgamma
    output_d_dm_1dgamma[i] = d_dm_1dgamma
    output_d_dm_2dgamma[i] = d_dm_2dgamma


def computeCRKMomentsGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    adjacency: 'AdjacencyList | CompactHashMap',
    referenceVolumes: torch.Tensor,
    tangentReferenceVolumes: torch.Tensor,
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
):
    """`(dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma)` -- the
    JVP counterpart to `_computeCRKMoments_stateBackend`. Always
    `SupportScheme.Scatter`, matching that function's own hardcode.
    `referenceVolumes`/`tangentReferenceVolumes` are Stage 1's own primal `V`
    and JVP `dV` (`computeCRKVolumeGeometryJVP`) -- threaded through the same
    `correctionData.referenceVolumes`/`correctionTangentData.referenceVolumes`
    plumbing phase (b) already wired into `launchGeometryJVP`, since CRK's
    apparent volume occupies the exact same `useVolume` slot every
    value-having operator's `gradientWeightsJVP` reads (see module
    docstring). `d_dm_2dgamma` is returned pre-reshaped to `[N, dim, dim,
    dim]`, matching `_computeCRKMoments_stateBackend`'s own `.view(...)`.
    """
    from ..coreOperations._jvpCommon import launchGeometryJVP

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    dim = domain.dim
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    # No mass/density term anywhere in Stage 2's formula either (moments only
    # ever read positions/supports and the apparent volume) -- zero those
    # tangents unconditionally, same as computeCRKVolumeGeometryJVP.
    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=zerosScalar(nRef),
            densities=zerosScalar(nRef),
        )

    vecD = vector(length=dim, dtype=scalar_t)
    matDD = matrix(shape=(dim, dim), dtype=scalar_t)
    vecD3 = vector(length=dim ** 3, dtype=scalar_t)

    dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma = launchGeometryJVP(
        computeCRKMomentsJVP_Kernel,
        domain, kernel, SupportScheme.Scatter, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=[nQuery] * 6,
        outputDtype=[scalar_t, vecD, matDD, vecD, matDD, vecD3],
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
    )
    return dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma.view(-1, dim, dim, dim)
