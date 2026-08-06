import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, DomainDescription, PointCloud
from ..radiusSearch.wp_compactHash import CompactHashMap
from ..radiusSearch.grid_util import checkOffset
from ..math import *
from ..kernels import *
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j, zero_like_warp, castTorchToWarpAsBuiltins

from ..enumTypes import *
from ..warp_state import (
    domainData, adjacencyData, gridData,
    getParticle,
)
from ..warp_state_util import warpWrapper2
from ..state import ParticleState, OperationProperties
from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified Density kernel: same design as the unified Gradient/Interpolate kernels (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list and compact-hash-grid traversal, replacing the former
# split between this file (adjacency-only) and operations_grid/wp_density_grid.py
# (grid-only, duplicated physics). Density is the simplest operator in the family: no
# queryValues/referenceValues, no correction paths (CRK/volume/grad-h/renorm) -- it just
# sums reference masses weighted by the kernel.


@wp.func
def computeSPHDensity_Func_i(
    i: wp.int32, dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

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

        out += mj * sphKernel(xi, xj, hi, hj, kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax)

    return out


@wp.func
def computeSPHDensity_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,

    outputValue: Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)

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

        out += computeSPHDensity_Func_i(
            i, dim,
            xi, hi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHDensity_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHDensity_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt,

        zero_like_warp(outputValues[i]),
    )


def _computeSPHDensity_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    operationMode: OperationDirection,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
):
    """Unified state-based Density backend -- see warpier_core.md's "Working Prototype ->
    Production" section. Handles adjacency-list, compact-hash-grid, and
    ``adjacency=None`` traversal alike -- see ``extractStateInfo`` in
    ``warp_state_util.py`` for the dispatch. Density has no queryValues/referenceValues
    and no correction paths (CRK/volume/grad-h/renorm), so its state footprint is just
    the two particle states.
    """
    with record_function("warpSPH[Density]"):
        with record_function("warpSPH[Density] - Preprocessing"):
            outputSize = queryParticles.positions.shape[0]
            outputDtype = castTorchToWarpAsBuiltins(queryParticles.masses).dtype

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Density,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Density] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHDensity_Kernel,
                outputSizes=outputSize,
                outputDtypes=outputDtype,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    None, None,
                    adjacency,
                    referenceParticles,
                    None,
                    None,
                    None,
                ),
                additionalArguments=(),
            )

    return result
