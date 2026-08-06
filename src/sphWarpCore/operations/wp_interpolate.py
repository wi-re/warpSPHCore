import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, DomainDescription, PointCloud
from ..radiusSearch.wp_compactHash import CompactHashMap
from ..radiusSearch.grid_util import checkOffset
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j, zero_like_warp, castTorchToWarpAsBuiltins, getCachedDummyTensor

from ..enumTypes import *
from ..warp_state import (
    domainData, adjacencyData, gridData,
    getParticle, getCRK_i,
)
from ..warp_state_util import warpWrapper2
from ..state import ParticleState, OperationProperties, CRKState

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


def computeSPHInterpolant_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    operationMode: OperationDirection,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    scatteredQuantities: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None,
):
    """Public entry point kept for ``sphOperation_warp``: same flat-tensor signature as
    before, now a thin adapter over the unified state-based kernel above. Interpolate's
    output only depends on referenceValues (not queryValues -- there is no "difference
    from self" term the way Gradient has), matching the pre-migration kernel.
    """
    if scatteredQuantities is not None:
        raise NotImplementedError(
            "Pre-scattered quantities are no longer supported by the Interpolate operator: "
            "no caller in this repo relies on them (same rationale as Gradient's migration -- "
            "see warpier_core.md). Pass referenceValues instead."
        )
    if referenceValues is None:
        raise ValueError("referenceValues must be provided for the interpolation computation.")

    queryParticles = ParticleState(positions=queryPositions, supports=querySupports, masses=queryMasses, densities=queryDensities, kinds=queryKinds)
    referenceParticles = ParticleState(positions=referencePositions, supports=referenceSupports, masses=referenceMasses, densities=referenceDensities, kinds=referenceKinds)

    if useCRK:
        # CRKState requires gradA/gradB (shapes [N,D]/[N,D,D]), but Interpolate never
        # reads correctionData.queryGradA/queryGradB -- only Ai/Bi. Fill correctly-shaped
        # dummies rather than reusing crk_B (wrong shape for gradB) or leaving them unset.
        dim = queryPositions.shape[1]
        dummy_gradA = getCachedDummyTensor((1, dim), device=crk_A.device, dtype=crk_A.dtype)
        dummy_gradB = getCachedDummyTensor((1, dim, dim), device=crk_A.device, dtype=crk_A.dtype)
        crkState = CRKState(A=crk_A, B=crk_B, gradA=dummy_gradA, gradB=dummy_gradB)
    else:
        crkState = None

    return _computeSPHInterpolant_stateBackend(
        queryParticles, referenceParticles, domain, mode, kernel, operationMode,
        adjacency, referenceValues,
        queryVolumes=queryVolumes if useVolume else None,
        referenceVolumes=referenceVolumes if useVolume else None,
        crkState=crkState,
    )
