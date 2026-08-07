import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
import torch
from torch.profiler import record_function

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

# Unified Interpolate kernel: same design as the unified Gradient kernel (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list and compact-hash-grid traversal. This replaced a former
# split between an adjacency-only kernel in this file and a separate grid-only kernel
# with duplicated physics in the now-deleted operations_grid/ package.


@wp.func
def computeSPHInterpolation_Func_i(
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


    referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
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

        fv = referenceValues[j]

        vj = jPtcl.mass / jPtcl.density if not correctionData.useVolume else correctionData.referenceVolumes[j]

        w_ij = computeKernelCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK, iCorrectionData.A, iCorrectionData.B
        )

        out += fv * vj * w_ij

    return out


@wp.func
def computeSPHInterpolation_Func_Adjacency(
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

    referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHInterpolation_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.

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

    kernelProperties: kernelState,
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
        kernelProperties,
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
