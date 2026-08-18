import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
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

@wp.func
def computeSPHDivergenceTensor_Func_i(
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
    consistentDivergence: wp.bool,
    flatInputShape: wp.int32, flatOutputShape: wp.int32,
    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    dotMode = kernelProperties.divergenceMode
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
        if consistentDivergence:
            apparentVolume = jPtcl.mass / iPtcl.density if not correctionData.useVolume else correctionData.referenceVolumes[j] * jPtcl.density / iPtcl.density

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

        if kernelProperties.gradientMode == wp.static(GradientScheme.Naive.value): # Naive
            out += divergenceProduct(fj * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += divergenceProduct(jPtcl.mass * iPtcl.density * (fi / iPow(iPtcl.density,2) + fj / iPow(jPtcl.density,2)), kernelGradient, outputValue, flatOutputShape, dotMode)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Difference.value): # Difference
            out += divergenceProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Summation.value): # Summation
            out += divergenceProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)

    return out


@wp.func
def computeSPHDivergenceTensor_Func_Adjacency(
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
    consistentDivergence: wp.bool,
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

        out += computeSPHDivergenceTensor_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
            consistentDivergence,
            flatInputShape, flatOutputShape,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHDivergenceTensor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    consistentDivergence: wp.bool,

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHDivergenceTensor_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        consistentDivergence,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def _divergenceOutputDtype(ctx, extras):
    return _get_warp_vector_dtype(int(extras["flatOutputShape"]), extras["queryValuesFlat"].dtype)


_DIVERGENCE_SPEC = OperatorSpec(
    kernel=computeSPHDivergenceTensor_Kernel,
    outputs=(OutputSpec(dtype=_divergenceOutputDtype),),
    extras=(
        ExtraSpec("consistentDivergence", ExtraKind.SCALAR),
        ExtraSpec("numDims", ExtraKind.SCALAR),
        ExtraSpec("flatInputShape", ExtraKind.SCALAR),
        ExtraSpec("flatOutputShape", ExtraKind.SCALAR),
        ExtraSpec("queryValuesFlat", ExtraKind.TENSOR),
        ExtraSpec("referenceValuesFlat", ExtraKind.TENSOR),
    ),
)


def _computeSPHDivergence_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    gradientMode: GradientScheme,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None
    queryValues: torch.Tensor, referenceValues: torch.Tensor,
    consistentDivergence: bool = False,
    dotMode: bool = False,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
):
    with record_function("warpSPH[Divergence]"):
        with record_function("warpSPH[Divergence] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]

            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for d in inputShape:
                flatInputShape *= d

            outputShape = inputShape[1:] if dotMode else inputShape[:-1]
            flatOutputShape = 1
            for d in outputShape:
                flatOutputShape *= d
            numDims = len(inputShape)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Divergence,
                gradientMode=gradientMode,
                supportMode=mode,
                operationMode=operationMode,
                divergenceDotMode=dotMode,
            )

        with record_function("warpSPH[Divergence] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles,
                properties=operationProperties,
                domain=domain,
                adjacency=adjacency,
                reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            result = launchOperator(
                _DIVERGENCE_SPEC, ctx,
                consistentDivergence=wp.bool(consistentDivergence),
                numDims=wp.int32(numDims),
                flatInputShape=wp.int32(flatInputShape),
                flatOutputShape=wp.int32(flatOutputShape),
                queryValuesFlat=queryValues.view(-1, flatInputShape),
                referenceValuesFlat=referenceValues.view(-1, flatInputShape),
            )

    return result.view(outputSize, *outputShape)
