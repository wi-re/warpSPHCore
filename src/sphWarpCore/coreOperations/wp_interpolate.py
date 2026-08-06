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
from ..autograd import *
from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified Interpolate kernel: same design as the unified Gradient kernel (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list and compact-hash-grid traversal, replacing the former
# split between this file (adjacency-only) and operations_grid/wp_interpolate_grid.py
# (grid-only, duplicated physics).


@wp.func
def computeSPHInterpolation_Func_i(
    i: wp.int32, dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    useVolume: wp.bool, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: wp.bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), # type: ignore

    referenceValues: wp.array(dtype = Any), # type: ignore

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

        fv = referenceValues[j]
        vj = mj / rhoj if not useVolume else referenceVolumes[j]

        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi
        )

        out += fv * vj * w_ij

    return out


@wp.func
def computeSPHInterpolation_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,

    referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)

    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    useVolume = correctionData.useVolume

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

        out += computeSPHInterpolation_Func_i(
            i, dim,
            xi, hi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useVolume, correctionData.referenceVolumes,
            useCRK, Ai, Bi,

            referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHInterpolation_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHInterpolation_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt,
        referenceValues,

        zero_like_warp(outputValues[i]),
    )


def _computeSPHInterpolant_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None
    referenceValues: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
):
    if crkState is not None and (crkState.gradA is None or crkState.gradB is None):
        # CRKState requires gradA/gradB (shapes [N,D]/[N,D,D]), but Interpolate never
        # reads correctionData.queryGradA/queryGradB -- only Ai/Bi. Fill correctly-shaped
        # dummies rather than leaving them unset -- extractStateInfo accesses
        # crkState.gradA/.gradB unconditionally whenever a CRKState is provided, and
        # reusing e.g. crkState.B as a stand-in would be a shape mismatch ([N,D] where
        # gradB needs [N,D,D]).
        dim = queryParticles.positions.shape[1]
        dummy_gradA = getCachedDummyTensor((1, dim), device=crkState.A.device, dtype=crkState.A.dtype)
        dummy_gradB = getCachedDummyTensor((1, dim, dim), device=crkState.A.device, dtype=crkState.A.dtype)
        crkState = CRKState(A=crkState.A, B=crkState.B, gradA=dummy_gradA, gradB=dummy_gradB)

    with record_function("warpSPH[Interpolation]"):
        with record_function("warpSPH[Interpolation] - Preprocessing"):
            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            # For higher-rank inputs (e.g. shape (n, p, m, d)) flatten the field dims to
            # a single vector dimension, interpolate, then restore the original shape.
            field_shape = referenceValues.shape[1:]
            needs_flatten = referenceValues.dim() > 3
            if needs_flatten:
                flat_len = referenceValues[0].numel()
                referenceValues = referenceValues.reshape(referenceValues.shape[0], flat_len).contiguous()

            outputSize = queryParticles.positions.shape[0]
            outputDtype = castTorchToWarpAsBuiltins(referenceValues).dtype

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Interpolate,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Interpolation] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHInterpolation_Kernel,
                outputSizes=outputSize,
                outputDtypes=outputDtype,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState,
                    None,
                    None,
                ),
                additionalArguments=(
                    referenceValues,
                ),
            )

        if needs_flatten:
            result = result.reshape(result.shape[0], *field_shape)

    return result
