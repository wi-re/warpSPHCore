import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..utils.wp_autograd import *

from ..dataTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap
from ..radiusSearch.grid_util import checkOffset
from ..math import *
from ..kernels import *
from ..utils.wp_util import (
    getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j,
    zero_like_warp, _get_warp_matrix_dtype,
)
from ..utils import *

from ..enumTypes import *
from ..warp_state_util import warpWrapper2

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
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32,

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32,

    # Operation specific parameters
    gradientMode_int: wp.int32, # type: ignore

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

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
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        fij = -computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)

        kernelGradient = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)

        out += wp.outer(fij * apparentVolume, kernelGradient)
        kernel = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi
        )
        if kernel > 0.0:
            numNeighbors += 1

    return out, numNeighbors


@wp.func
def computeCovariance_Func_Adjacency(
    i : wp.int32, dim: wp.int32,

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, opInt: wp.int32,

    outputValue : Any, # type: ignore
    outputNeighbors : wp.int32 # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue), wp.int32(0)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = zero_like_warp(outputValue)
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

        stepC, stepN= computeCovariance_Func_i(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            correctionData,


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

    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
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
        mode_uint, kernel_int, gradientMode_int,  opInt,
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
