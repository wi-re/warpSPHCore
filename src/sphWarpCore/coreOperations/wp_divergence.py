import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
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
from ..autograd import *

from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified Divergence kernel: same design as the unified Gradient/Interpolate/Density
# kernels (see warpier_core.md's "Working Prototype -> Production" section). Divergence
# shares essentially all of Gradient's correction-path machinery (CRK, grad-h, volume,
# renormalization) and differs only in the neighbor-level contraction: `divergenceProduct`
# (contracts the input's last/first flattened axis against the kernel gradient) in place
# of Gradient's `outerTensorProduct` (which appends a new axis instead).

# In dot mode we compute torch.einsum('nd..., nd -> n...', q, k)
# Otherwise we compute torch.einsum('n...d, nd -> n...', q, k)
# the inputs are the flattened versions of the original tensors, i.e.,
# if q initially was of shape [n, d, d, d] it is now [n, d^3]
# The output is always of shape [n, d^(N-1)] where N is the rank of the input tensor q. So if q was originally [n, d, d, d] the output will be [n, d^2]
# The kernelGradient is always of shape [n, d]
# this also allows us to overload the function based on d! We can then use the numDims parameter to do the correct indexing inside the kernel without having to write separate kernels for different dimensions.
@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=3), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.

    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]

    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=2), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.

    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]

    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=1), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    # in 1D the divergence product is just a simple multiplication
    res[0] = fij[0] * kernelGradient[0]
    return res


@wp.func
def computeSPHDivergenceTensor_Func_i(
    i: wp.int32, dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t), # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    correctionData: Any,

    consistentDivergence: wp.bool,

    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    dotMode = divergenceMode_int != 0
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
        if consistentDivergence:
            apparentVolume = mj / rhoi if not useVolume else referenceVolumes[j] * rhoj / rhoi

        # Explicit if/else, not a ternary: see docs/lessons_learned.md and the note in
        # computeSPHGradientTensor_Func_i (wp_gradient.py) -- both branches here would
        # read referenceValues[j], which silently zeroes that array's adjoint if written
        # as a ternary.
        fj = referenceValues[j]
        if useGradHTerms:
            fj = referenceValues[j] / referenceOmegas[j]

        kernelGradient = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)

        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            out += divergenceProduct(fj * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += divergenceProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)), kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            out += divergenceProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            out += divergenceProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)

    return out


@wp.func
def computeSPHDivergenceTensor_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,

    consistentDivergence: wp.bool,

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
    if useGradHTerms:
        fi = queryValue[i] / omega_i

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

        out += computeSPHDivergenceTensor_Func_i(
            i, dim, numDims, flatInputShape, flatOutputShape,
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

            consistentDivergence,

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

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
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
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt,
        consistentDivergence,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
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

            outputDtype = _get_warp_vector_dtype(flatOutputShape, queryValues.dtype)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Divergence,
                gradientMode=gradientMode,
                supportMode=mode,
                operationMode=operationMode,
                divergenceDotMode=dotMode,
            )

        with record_function("warpSPH[Divergence] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHDivergenceTensor_Kernel,
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
                    wp.bool(consistentDivergence),
                    wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                    queryValues.view(-1, flatInputShape), referenceValues.view(-1, flatInputShape),
                ),
            )

    return result.view(outputSize, *outputShape)
