import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..types import *
from ..autograd import *

from ..dataTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap
from ..radiusSearch.grid_util import checkOffset
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *

from .kernel import computeKernelCRK

# Unified CRK-density kernel: same dual-path design as every migrated operator (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list ("adjacency") and compact-hash-grid traversal.
# Computes a CRK-corrected density estimate (mDensity/vol1, a Shepard-style ratio) used
# by computeCRKFactors as a diagnostic/consistency companion to A/B/gradA/gradB -- not
# the general SPH Density operator (coreOperations/wp_density.py).


@wp.func
def computeCRKDensity_Func_i(
    i: wp.int32, dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    correctionData: Any, # correctionData_1/2/3, for the reference-side apparent volumes
):
    mDensity = scalar_t(0.0)
    vol1 = scalar_t(0.0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        _, Vj = getVolume_j(correctionData, j)

        # mode/useCRK are hardcoded here (not mode_uint/a useCRK flag) -- pre-existing
        # behavior carried over unchanged from the neighbor-list-only kernel this
        # replaces; not touched by this traversal-style migration.
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernel_int, wp.uint32(12), domainState.periodicity, domainState.domainMin, domainState.domainMax,
            True, Ai, Bi
        )

        mDensity += mj * Vj * w_ij
        vol1 += Vj * Vj * w_ij

    return mDensity, vol1


@wp.func
def computeCRKDensity_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, opInt: wp.int32,
):
    # Returns (mDensity, vol1, masked) rather than the final mDensity/vol1 ratio --
    # Warp's adjoint for a dynamic for-loop (numOffsets is a runtime value) that
    # accumulates into locals via += and then feeds them into a nonlinear post-loop op
    # (division), all inside the same @wp.func, produces NaN gradients here -- same
    # issue as computeCRKVolume_Func_Adjacency, see its docstring comment and
    # scripts/debug_crk_backward.py for the minimal repro. The ratio is applied one
    # level up, in computeCRKDensity_Kernel, outside the function that contains the loop.
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return scalar_t(0.0), scalar_t(0.0), True

    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    mDensity = scalar_t(0.0)
    vol1 = scalar_t(0.0)
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

        d_mDensity, d_vol1 = computeCRKDensity_Func_i(
            i, dim,
            xi, hi, Ai, Bi,
            referenceState, domainState,
            mode_uint, kernel_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            correctionData,
        )
        mDensity += d_mDensity
        vol1 += d_vol1

    return mDensity, vol1, False


@wp.kernel
def computeCRKDensity_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    mDensity, vol1, masked = computeCRKDensity_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, opInt,
    )
    # The ratio is applied here, outside computeCRKDensity_Func_Adjacency's dynamic
    # loop -- see that function's docstring comment for why.
    if masked:
        outputValues[i] = scalar_t(0.0)
    else:
        outputValues[i] = mDensity / vol1


def _computeCRKDensity_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    crkState: CRKState,
    queryVolumes: torch.Tensor, referenceVolumes: torch.Tensor,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
) -> torch.Tensor:
    """Computes the CRK-corrected consistency density estimate for every query
    particle, using the CRK terms (A/B) and apparent volumes already solved for by
    computeCRKMoments/computeCRKTermsWarp.
    """
    with record_function("warpSPH[CRKDensity]"):
        with record_function("warpSPH[CRKDensity] - Preprocessing"):
            outputSize = queryParticles.positions.shape[0]

        with record_function("warpSPH[CRKDensity] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeCRKDensity_Kernel,
                outputSizes=outputSize,
                outputDtypes=scalar_t,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState, None, None,
                ),
                additionalArguments=(),
            )

    return result
