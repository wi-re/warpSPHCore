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

# Unified Curl kernel: same design as the unified Gradient/Divergence kernels (see
# warpier_core.md's "Working Prototype -> Production" section). Curl shares Gradient's
# correction-path machinery (CRK, grad-h, volume, renormalization) and differs only in
# the neighbor-level contraction: `curlProduct` (Levi-Civita / cross-product contraction)
# in place of Gradient's `outerTensorProduct`.

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=3), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    # This used to `return -R`, per a comment claiming the negation was needed
    # to match the right-hand rule. Confirmed via operation_matrix.py --dim 3
    # that this made every 3D curl output the exact negative of the true
    # (right-hand-rule) curl -- the 2D overload below has no such negation
    # and is dimensionally consistent with the standard convention, so the
    # 3D `-R` was the actual bug, not a deliberate convention choice. See
    # warpier_core.md.
    R = type(output)(scalar_t(0.0))
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.
    for s in range(stride):
        # Flattened locations for T[0][s], T[1][s], T[2][s]
        k0 = wp.int32(s)
        k1 = wp.int32(s + stride)
        k2 = wp.int32(s + 2 * stride)
        # Levi-Civita / cross-product logic:
        R[0 * stride + s] = V[1] * T[k2] - V[2] * T[k1];
        R[1 * stride + s] = V[2] * T[k0] - V[0] * T[k2];
        R[2 * stride + s] = V[0] * T[k1] - V[1] * T[k0];
    return R

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=2), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(scalar_t(0.0))
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.
    for s in range(stride+1): # loop to stride+1: in 2D the output has one less dimension than the input
        k0 = wp.int32(s)
        k1 = wp.int32(s + stride+1)
        # 2D cross-product logic: collapses the first dimension of T and the dimension of V
        R[s] = V[0] * T[k1] - V[1] * T[k0];
    return R

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=1), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(scalar_t(0.0))
    # in 1D the curl is identically zero
    R[0] = scalar_t(0.0)
    return R


@wp.func
def computeSPHCurlTensor_Func_i(
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

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

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

        apparentVolume = jPtcl.mass / jPtcl.density if not correctionData.useVolume else correctionData.referenceVolumes[j]

        # Explicit if/else, not a ternary: both branches read referenceValues[j], and a
        # ternary assigned to a local where both branches index the *same* array silently
        # zeroes that array's adjoint (compiles fine, runs the correct branch, wrong
        # gradient) -- see docs/lessons_learned.md, the bug that broke Interpolate this way.
        if correctionData.useGradHTerms:
            fj = referenceValues[j] / correctionData.referenceOmegas[j]
        else:
            fj = referenceValues[j]

        kernelGradient = computeKernelGradientCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK, iCorrectionData.A, iCorrectionData.B, iCorrectionData.gradA, iCorrectionData.gradB
        )

        if correctionData.useGradientRenormalization:
            kernelGradient = matmul(iCorrectionData.renormalizationMatrix, kernelGradient)

        stride = wp.int32(flatOutputShape / dim)
        if kernelProperties.gradientMode == wp.static(GradientScheme.Naive.value): # Naive
            out += curlProduct(fj * apparentVolume, kernelGradient, outputValue, stride, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += curlProduct(jPtcl.mass * iPtcl.density * (fi / iPow(iPtcl.density,2) + fj / iPow(jPtcl.density,2)), kernelGradient, outputValue, stride, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Difference.value): # Difference
            out += curlProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, stride, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Summation.value): # Summation
            out += curlProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, stride, flatInputShape, flatOutputShape)

    return out


@wp.func
def computeSPHCurlTensor_Func_Adjacency(
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

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValue: Any, referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    fi = queryValue[i]
    if correctionData.useGradHTerms:
        fi = queryValue[i] / iCorrectionData.omega

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHCurlTensor_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
            numDims, flatInputShape, flatOutputShape,
            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHCurlTensor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHCurlTensor_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def _computeSPHCurl_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    gradientMode: GradientScheme,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None
    queryValues: torch.Tensor, referenceValues: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
):
    with record_function("warpSPH[Curl]"):
        with record_function("warpSPH[Curl] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]
            D = queryPositions.shape[1]

            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for d in inputShape:
                flatInputShape *= d

            if D == 3:
                outputShape = inputShape
            elif D == 2:
                outputShape = inputShape[:-1]
                if outputShape == ():
                    outputShape = [1] # vector field in 2D -> scalar output, represented as shape [1]
            else: # 1D curl is identically zero, so it's a scalar
                outputShape = []
            flatOutputShape = 1
            for d in outputShape:
                flatOutputShape *= d
            numDims = len(inputShape)

            outputDtype = _get_warp_vector_dtype(flatOutputShape, queryValues.dtype)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Curl,
                gradientMode=gradientMode,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Curl] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHCurlTensor_Kernel,
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
