import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *

from ..radiusSearch.grid_util import checkOffset
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *

# Unified CRK-moments kernel: same dual-path design as every migrated operator (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list ("adjacency") and compact-hash-grid traversal. This
# replaces the former neighbor-list-only kernel, which took a pre-flattened
# (neighborList, neighborOffset, numNeighs) triple and could never be driven by a
# CompactHashMap. The moments are the raw (uncorrected) kernel-weighted geometric
# moments used to solve for the CRK correction terms A/B/gradA/gradB (see
# crk_terms.py) -- CRK correction is never applied while computing them.
#
# The per-particle neighbor count is now a genuine kernel output (outputNumNeighbors,
# following the same pattern coreOperations/wp_covariance.py uses for its own neighbor
# count) rather than being read off adjacency.numNeighbors on the Python side -- that
# field only exists for AdjacencyList/AdjacencyListWarp, not for CompactHashMap, so
# crk_wrapper.py's low-neighbor-count fallback (crk_terms.py's computeCRKTermsWarp)
# needs a traversal-mode-agnostic source for it to work under grid traversal too.


@wp.func
def delta(a: wp.int32, b: wp.int32):
    return scalar_t(1.0) if a == b else scalar_t(0.0)


@wp.func
def computeCRKMoments_Func_i(
    # General shape parameters
    i: wp.int32, dim: wp.int32,

    # SPH properties for the query point (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA_1/2/3

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    # Neighbor range within offsetArray to iterate; offsetArray is either the adjacency
    # neighbor list or the grid's sorted particle index, depending on the caller.
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation mode for masking certain kinds of interactions, e.g. for directional operations
    ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    correctionData: Any, # correctionData_1/2/3 -- only used here for the (query==reference) apparent volumes

    output_m_0: scalar_t, # type: ignore
    output_m_1: vector(length=Any, dtype=scalar_t), # type: ignore
    output_m_2: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_0dgamma: vector(length=Any, dtype=scalar_t), # type: ignore
    output_dm_1dgamma: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_2dgamma: vector(length=Any, dtype=scalar_t) # type: ignore (flattened to avoid issues with warp's handling of rank-3 tensors)
):
    m_0 = zero_like_warp(output_m_0)
    m_1 = zero_like_warp(output_m_1)
    m_2 = zero_like_warp(output_m_2)
    dm_0dgamma = zero_like_warp(output_dm_0dgamma)
    dm_1dgamma = zero_like_warp(output_dm_1dgamma)
    dm_2dgamma = zero_like_warp(output_dm_2dgamma)
    numNeighbors = wp.int32(0)

    eye = warp_eye(xi)

    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        _, V_j = getVolume_j(correctionData, j)

        x_ij = computeDistanceVec(xi, xj, domainState)
        w_ij = sphKernel_ij(x_ij, hi, hj, kernelProperties, domainState)
        gradw_ij = sphKernelGradient_ij(x_ij, hi, hj, kernelProperties, domainState)

        m_0 += V_j * w_ij
        m_1 += x_ij * (V_j * w_ij)
        m_2 += wp.outer(x_ij, x_ij) * (V_j * w_ij)

        dm_0dgamma += V_j * gradw_ij
        dm_1dgamma += V_j * (wp.outer(x_ij, gradw_ij) + w_ij * eye)

        for alpha in range(dim):
            for beta in range(dim):
                for gamma in range(dim):
                    gradTerm = x_ij[alpha] * x_ij[beta] * gradw_ij[gamma]
                    deltaA = x_ij[alpha] * delta(beta, gamma)
                    deltaB = delta(alpha, gamma) * x_ij[beta]
                    kernelTerm = w_ij * (deltaA + deltaB)
                    dm_2dgamma[gamma * dim * dim + alpha * dim + beta] += V_j * (gradTerm + kernelTerm)

        numNeighbors += 1

    return m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors


@wp.func
def computeCRKMoments_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    kernelProperties: kernelState,

    output_m_0: scalar_t, # type: ignore
    output_m_1: vector(length=Any, dtype=scalar_t), # type: ignore
    output_m_2: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_0dgamma: vector(length=Any, dtype=scalar_t), # type: ignore
    output_dm_1dgamma: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_2dgamma: vector(length=Any, dtype=scalar_t) # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return (
                zero_like_warp(output_m_0), zero_like_warp(output_m_1), zero_like_warp(output_m_2),
                zero_like_warp(output_dm_0dgamma), zero_like_warp(output_dm_1dgamma), zero_like_warp(output_dm_2dgamma),
                wp.int32(0),
            )

    m_0 = zero_like_warp(output_m_0)
    m_1 = zero_like_warp(output_m_1)
    m_2 = zero_like_warp(output_m_2)
    dm_0dgamma = zero_like_warp(output_dm_0dgamma)
    dm_1dgamma = zero_like_warp(output_dm_1dgamma)
    dm_2dgamma = zero_like_warp(output_dm_2dgamma)
    numNeighbors = wp.int32(0)

    for o in range(numOffsets):
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue

        s_m0, s_m1, s_m2, s_dm0, s_dm1, s_dm2, s_n = computeCRKMoments_Func_i(
            i, dim,
            xi, hi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            correctionData,

            output_m_0, output_m_1, output_m_2, output_dm_0dgamma, output_dm_1dgamma, output_dm_2dgamma,
        )
        m_0 += s_m0
        m_1 += s_m1
        m_2 += s_m2
        dm_0dgamma += s_dm0
        dm_1dgamma += s_dm1
        dm_2dgamma += s_dm2
        numNeighbors += s_n

    return m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors


@wp.kernel
def computeCRKMoments_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    # The last parameters are always the output arrays and should not be changed
    output_m_0 : wp.array(dtype = Any), # type: ignore
    output_m_1 : wp.array(dtype = Any), # type: ignore
    output_m_2 : wp.array(dtype = Any), # type: ignore
    output_dm_0dgamma : wp.array(dtype = Any), # type: ignore
    output_dm_1dgamma : wp.array(dtype = Any), # type: ignore
    output_dm_2dgamma : wp.array(dtype = Any), # type: ignore (flattened to avoid issues with warp's handling of rank-3 tensors)
    output_numNeighbors : wp.array(dtype = wp.int32) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors = computeCRKMoments_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and should not be changed

        # zero_like_warp on the *array itself* only has overloads up to a 3-vector /
        # 3x3-matrix (see math/wp_zero.py) -- dm_2dgamma flattens to dim**3 components,
        # which exceeds that for dim>1, so index into the array first (matches the
        # pattern every migrated operator's kernel uses, see warpier_core.md's
        # "Landing Gradient" section).
        zero_like_warp(output_m_0[i]), zero_like_warp(output_m_1[i]), zero_like_warp(output_m_2[i]),
        zero_like_warp(output_dm_0dgamma[i]), zero_like_warp(output_dm_1dgamma[i]), zero_like_warp(output_dm_2dgamma[i]),
    )

    output_m_0[i] = m_0
    output_m_1[i] = m_1
    output_m_2[i] = m_2
    output_dm_0dgamma[i] = dm_0dgamma
    output_dm_1dgamma[i] = dm_1dgamma
    output_dm_2dgamma[i] = dm_2dgamma
    output_numNeighbors[i] = numNeighbors


def _computeCRKMoments_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryVolumes: torch.Tensor, referenceVolumes: torch.Tensor,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # None or CompactHashMap -> grid traversal; AdjacencyList -> neighbor-list traversal.
    referenceParticles: Optional[ParticleState] = None,
):
    """Computes the raw (uncorrected) kernel-weighted geometric moments m_0/m_1/m_2 and
    their gamma-derivatives dm_0dgamma/dm_1dgamma/dm_2dgamma used by crk_terms.py's
    computeCRKTermsWarp to solve for the CRK correction terms A/B/gradA/gradB, plus the
    per-particle neighbor count (needed by the low-neighbor-count fallback there, and
    computed here rather than read off adjacency.numNeighbors since that field doesn't
    exist for grid/CompactHashMap traversal).
    """
    with record_function("warpSPH[CRKMoments]"):
        with record_function("warpSPH[CRKMoments] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]
            dim = queryPositions.shape[1]

            outputSizes = [outputSize] * 7
            outputDtypes = [
                scalar_t,
                vector(length=dim, dtype=scalar_t),
                matrix(shape=(dim, dim), dtype=scalar_t),
                vector(length=dim, dtype=scalar_t),
                matrix(shape=(dim, dim), dtype=scalar_t),
                vector(length=dim**3, dtype=scalar_t),
                wp.int32,
            ]

        with record_function("warpSPH[CRKMoments] - Kernel Execution"):
            m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeCRKMoments_Kernel,
                outputSizes=outputSizes,
                outputDtypes=outputDtypes,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    None, None, None, # crkState, gradHState, renormalizationState -- moments never apply corrections
                ),
                additionalArguments=(),
            )

    return m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma.view(-1, dim, dim, dim), numNeighbors
