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
from torch.profiler import profile, record_function, ProfilerActivity

from ..enumTypes import *
from ..autograd import *

from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified Gradient kernel: a single wp.func/wp.kernel pair drives both neighbor-list
# ("adjacency") and compact-hash-grid traversal. The two only differ in how
# (beginIndex, numIndices, offsetArray) are produced -- see
# computeSPHGradientTensor_Func_Adjacency below -- so the per-neighbor physics is
# written exactly once. This replaced a former split between an adjacency-only flat-args
# kernel in this file and a separate grid-only kernel with duplicated physics in the
# now-deleted operations_grid/ package. See warpier_core.md's "Working Prototype" section
# for the design this follows
# (prototyped in the repo-root wp_grad.py / warp_gradient.ipynb).
#
# For matrices we need to implement the logic manually using outer products, since Warp
# does not support rank-2 field types natively. The output is stored as a flattened
# vector and reshaped back into its tensor shape on the Python side.


@wp.func
def computeSPHGradientTensor_Func_i(
    # General shape parameters
    i: wp.int32, dim: wp.int32,

    # SPH properties for the query point (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA_1/2/3, picked by dimensionality

    # Domain and kernel parameters
    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,

    # Neighbor range within offsetArray to iterate; offsetArray is either the adjacency
    # neighbor list or the grid's sorted particle index, depending on the caller.
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    # Optional correction terms
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t), # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    correctionData: Any, # correctionData_1/2/3

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        # Explicit if/else, not a ternary: both branches read referenceValues[j], and a
        # ternary assigned to a local where both branches index the *same* array silently
        # zeroes that array's adjoint (compiles fine, runs the correct branch, wrong
        # gradient) -- see docs/lessons_learned.md, the bug that broke Interpolate this way.
        if useGradHTerms:
            fj = referenceValues[j] / referenceOmegas[j]
        else:
            fj = referenceValues[j]

        kernelGradient = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)

        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            out += outerTensorProduct(fj * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += outerTensorProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)), kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            out += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            out += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)

    return out


@wp.func
def computeSPHGradientTensor_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValue: Any, referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
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

        out += computeSPHGradientTensor_Func_i(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi, correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            correctionData,

            numDims, flatInputShape, flatOutputShape,
            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHGradientTensor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHGradientTensor_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        # zero_like_warp on the *array itself* only has overloads for output
        # lengths 1-3 (see wp_util.py) and silently falls back to a broken
        # generic `type(array)()` for anything longer -- vector/matrix-valued
        # fields flatten to longer outputs than that. Index into the array
        # first so this generalizes to any flatOutputShape, matching the
        # pattern the previous (adjacency-only) kernel used.
        zero_like_warp(outputValues[i]),
    )


def _computeSPHGradient_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    gradientMode: GradientScheme,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None -- both traversal modes go through this one path
    queryValues: torch.Tensor, referenceValues: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
):
    with record_function("warpSPH[Gradient]"):
        with record_function("warpSPH[Gradient] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]

            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for d in inputShape:
                flatInputShape *= d

            outputShape = inputShape + (queryPositions.shape[1],) # add an extra dimension for the gradient
            flatOutputShape = 1
            for d in outputShape:
                flatOutputShape *= d
            numDims = len(inputShape)

            outputDtype = _get_warp_vector_dtype(flatOutputShape, queryValues.dtype)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Gradient,
                gradientMode=gradientMode,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Gradient] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHGradientTensor_Kernel,
                outputSizes=outputSize,
                outputDtypes=outputDtype,
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
                    wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                    queryValues.view(-1, flatInputShape), referenceValues.view(-1, flatInputShape),
                ),
            )

    return result.view(outputSize, *outputShape)
