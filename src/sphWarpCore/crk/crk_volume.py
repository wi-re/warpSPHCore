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

# Unified CRK-volume kernel: same dual-path design as every migrated operator (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list ("adjacency") and compact-hash-grid traversal.
# Computes the apparent volume estimate 1/sum_j(W_ij) used as the "volume" input to
# CRK moments/density -- no correction terms (CRK/grad-h/renorm) apply here, since this
# *is* one of the raw inputs those corrections are built from.


@wp.func
def computeCRKVolume_Func_i(
    i: wp.int32, dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore
):
    out = scalar_t(0.0)
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
        x_ij = computeDistanceVec(xi, xj, domainState)
        w_ij = sphKernel_ij(x_ij, hi, hj, kernelProperties, domainState)

        out += w_ij

    return out


@wp.func
def computeCRKVolume_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    kernelProperties: kernelState,
):
    # Returns (wsum, masked) rather than the final 1/wsum reciprocal -- Warp's adjoint
    # for a *dynamic* for-loop (numOffsets is a runtime value, not a compile-time
    # constant) that accumulates into a local via += and then feeds that local into a
    # nonlinear post-loop op (division), all inside the same @wp.func, produces NaN
    # gradients here (confirmed via a minimal standalone repro against just this
    # function -- see scripts/debug_crk_backward.py). Every other migrated operator's
    # _Func_Adjacency avoids this because it returns the loop-accumulated value
    # directly with no further transform. The reciprocal is applied one level up, in
    # computeCRKVolume_Kernel, outside the function that contains the loop -- verified
    # to fix the NaN gradient (same data, same case, clean backward once moved).
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return scalar_t(0.0), True

    wsum = scalar_t(0.0)
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

        wsum += computeCRKVolume_Func_i(
            i, dim,
            xi, hi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
        )

    return wsum, False


@wp.kernel
def computeCRKVolume_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, # unused (no corrections apply to the volume estimate itself) -- kept for ABI consistency, see coreOperations/wp_density.py for the same pattern

    kernelProperties: kernelState,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    wsum, masked = computeCRKVolume_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
    )
    # The reciprocal is applied here, outside computeCRKVolume_Func_Adjacency's dynamic
    # loop -- see that function's docstring comment for why.
    if masked:
        outputValues[i] = scalar_t(0.0)
    else:
        outputValues[i] = scalar_t(1.0) / wsum


def _computeCRKVolume_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
) -> torch.Tensor:
    """Computes the CRK apparent-volume estimate V_i = 1 / sum_j W_ij for every query
    particle. This is the raw volume estimate only, used as an input to computeCRKMoments
    and _computeCRKDensity_stateBackend (crk_moments.py / crk_density.py) -- it applies
    no corrections of its own.
    """
    with record_function("warpSPH[CRKVolume]"):
        with record_function("warpSPH[CRKVolume] - Preprocessing"):
            outputSize = queryParticles.positions.shape[0]

        with record_function("warpSPH[CRKVolume] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeCRKVolume_Kernel,
                outputSizes=outputSize,
                outputDtypes=scalar_t,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    None, None,
                    adjacency,
                    referenceParticles,
                    None, None, None,
                ),
                additionalArguments=(),
            )

    return result
