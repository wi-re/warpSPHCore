import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from ..profiling import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *
from ..radiusSearch.grid_util import getIndexRange, checkOffset
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *
from ..autograd import *

from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified covariance kernel: a single wp.func/wp.kernel pair drives both neighbor-list
# ("adjacency") and compact-hash-grid traversal, following the same recipe every other
# operator uses (see warpier_core.md's "Working Prototype -> Production" section) --
# this file's structured kernel ABI is in fact the one the rest of that recipe was
# modeled on, since covariance was the original demonstration of the structured-state
# kernel interface. computeCovarianceMatrix (below) is the thin Python wrapper; the
# low-neighbor-count fallback and pseudo-inverse that turn a covariance matrix into a
# renormalization matrix live in wp_renormalization.py, not here -- this file only
# computes the covariance matrix itself.
#
# For matrices we need to implement the logic manually using outer products, since Warp
# does not support rank-2 field types natively. The output is stored as a matrix-typed
# array (not a flattened vector, unlike Gradient's arbitrary-rank output) since the
# covariance matrix is always DxD for a D-dimensional domain, i.e. at most 3x3.


@wp.func
def computeCovariance_Func_i(
    i: wp.int32, dim: wp.int32,
    # SPH properties for the query point (indexed by i)
    iPtcl: Any, # WarpParticle_1/2/3, picked by dimensionality
    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA_1/2/3, picked by dimensionality

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    # Neighbor range within offsetArray to iterate; offsetArray is either the adjacency
    # neighbor list or the grid's sorted particle index, depending on the caller.
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Optional correction terms
    iCorrectionData: Any, # ParticleCorrectionData_1/2/3, picked by dimensionality
    correctionData: Any, # correctionData_1/2/3
    # End of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    numNeighbors = wp.int32(0)
    # # Loop over neighbors to compute the gradient contribution from each neighbor
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        apparentVolume = jPtcl.mass / jPtcl.density if not correctionData.useVolume else correctionData.referenceVolumes[j]

        fij = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)

        kernelGradient = computeKernelGradientCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK, iCorrectionData.A, iCorrectionData.B, iCorrectionData.gradA, iCorrectionData.gradB
        )

        if correctionData.useGradientRenormalization:
            kernelGradient = matmul(iCorrectionData.renormalizationMatrix, kernelGradient)

        out += wp.outer(fij * apparentVolume, kernelGradient)
        kernel = computeKernelCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK, iCorrectionData.A, iCorrectionData.B
        )
        if kernel > 0.0:
            numNeighbors += 1

    return out, numNeighbors


@wp.func
def computeCovariance_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    # SPH properties for the points and the corrections
    queryState: Any, referenceState: Any, correctionData: Any,
    # Domain properties 
    domainState: domainData,
    # Adjacency / grid traversal properties
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    # Kernel properties, e.g., kernel type, support scheme, gradient scheme, etc.
    kernelProperties: kernelState,
    # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.

    outputValue : Any, # type: ignore
    outputNeighbors : wp.int32 # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue), 0

    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    out = zero_like_warp(outputValue)
    numNeighbors = wp.int32(0)

    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        stepC, stepN= computeCovariance_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.


            outputValue,
        )
        out += stepC
        numNeighbors += stepN
    return out, numNeighbors

@wp.kernel
def computeCovariance_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any), # type: ignore
    outputNeighbors : wp.array(dtype = wp.int32) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    C, N = computeCovariance_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed

        # zero_like_warp on the *array itself* only has overloads up to a 3x3 matrix
        # (see wp_util.py) -- harmless here since a covariance matrix is always DxD for
        # D<=3, but indexing into the array first matches the pattern every other
        # operator's kernel uses (see warpier_core.md's "Landing Gradient" section) so
        # this stays correct if that ever changes.
        zero_like_warp(outputValues[i]),
        0,
    )
    outputValues[i] = C
    outputNeighbors[i] = N


def _computeSPHCovariance_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # None or CompactHashMap -> grid traversal; AdjacencyList -> neighbor-list traversal. Both go through the same kernel branch (useAdjacency); if None, extractStateInfo builds a CompactHashMap.
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
    returnNumNeighbors: bool = False,
) -> torch.Tensor:
    """Computes the SPH covariance matrix C_i = sum_j V_j (x_i - x_j) (x) gradW_ij for
    every query particle. This is the raw covariance tensor only -- turning it into a
    gradient-renormalization matrix (low-neighbor-count fallback + pseudo-inverse) is
    wp_renormalization.py's job, not this function's.
    """
    with record_function("warpSPH[Covariance]"):
        with record_function("warpSPH[Covariance] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSizes  = [queryPositions.shape[0], queryPositions.shape[0]]
            outputDtypes = [_get_warp_matrix_dtype(queryPositions.shape[1], queryPositions.shape[1], queryPositions.dtype), wp.int32]

        with record_function("warpSPH[Covariance] - Kernel Execution"):
            C = warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeCovariance_Kernel,
                outputSizes  = outputSizes,
                outputDtypes = outputDtypes,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState,
                    gradHState,
                    renormalizationState,
                ),
                additionalArguments=(
                ),
            )
    return C[0] if not returnNumNeighbors else (C[0], C[1])
