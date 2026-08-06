import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch

from sphWarpCore.radiusSearch.grid_util import checkOffset
from sphWarpCore.state import GradHState, RenormalizationState
from sphWarpCore.utils.wp_autograd import *


from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.math import *
from sphWarpCore.kernels.wp_kernel import *
from sphWarpCore.utils.wp_util import _get_warp_matrix_dtype, _get_warp_vector_dtype, getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *


# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.

@wp.func
def computeGradient_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,
    
    # Operation specific parameters
    
            
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


    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore
    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue# out * scalar_t(0.0)
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        xij = -computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)

        fj   = zero_like_warp(referenceValues)

        if useGradHTerms: # pre scatter quantities is not supported anymore, it wasnt ever used and caused issues with autograd
            fj = referenceValues[j] / referenceOmegas[j]
        else:
            fj = referenceValues[j]
    
        kernelGradient = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        kernel = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi  
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
        # out += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        # out[1]+=fi[0]
            
    return out


@wp.func
def computeGradient_Func_Adjacency(
    i : wp.int32, dim: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32, 

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValue: Any, referenceValues: Any, # type: ignore
    
    outputValue : Any, # type: ignore
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

    out = type(outputValue)() * scalar_t(0.0)
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
        
        out += computeGradient_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, 

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            correctionData,

            numDims, flatInputShape, flatOutputShape,
            fi, referenceValues,
            

            outputValue,
        )
    return out

@wp.kernel
def computeGradient_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeGradient_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int,  laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt, #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        # zero_like_warp on the array itself only has overloads for output
        # lengths 1-3 (wp_util.py) and silently breaks for longer flattened
        # outputs (vector/matrix-valued fields). Index into the array instead.
        zero_like_warp(outputValues[i]),
    )

from sphWarpCore.enumTypes import *
from typing import Optional



from sphWarpCore.math import outerTensorProduct

import torch
from torch.profiler import record_function


from sphWarpCore.warp_state_util import warpWrapper2

def get_gradient_dtype(
    dim: int,
    inputTensor: torch.Tensor,
):
    if len(inputTensor.shape) == 1:
        return _get_warp_vector_dtype(dim, 1, inputTensor.dtype)
    elif len(inputTensor.shape) == 2:
        return _get_warp_matrix_dtype(dim, dim, inputTensor.dtype)
    elif len(inputTensor.shape) > 2:
        raise ValueError(f"Input tensor has more than 2 dimensions, which is not supported for gradient computation. Input shape: {inputTensor.shape}")
    

def computeGradient(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryValues: torch.Tensor, referenceValues: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeGradient]"):
        with record_function("warpSPH[computeGradient] - Preprocessing"):

            outputSize = queryParticles.positions.shape[0]
            outputDtype = castTorchToWarpAsBuiltins(queryParticles.densities).dtype

            queryPositions = queryParticles.positions
            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            
            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for dim in inputShape:
                flatInputShape *= dim

            outputShape = inputShape + (queryPositions.shape[1],) # add an extra dimension for the gradient
            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            numDims = len(inputShape)

            D = queryPositions.shape[1]

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            referenceValues = referenceValues if referenceValues is not None else queryValues

            # outputDtype = get_gradient_dtype(dim, queryValues)
            outputDtype = _get_warp_vector_dtype(flatOutputShape, queryValues.dtype)

            print(f"computeGradient: outputSize={outputSize}, outputDtype={outputDtype}, numDims={numDims}, flatInputShape={flatInputShape}, flatOutputShape={flatOutputShape}")

        with record_function("warpSPH[computeGradient] - Kernel Execution"):
            return warpWrapper2(
                launcher = launch_kernel,
                kernel   = computeGradient_Kernel,
                outputSizes  = outputSize,
                outputDtypes = outputDtype,
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
            ).view(queryPositions.shape[0], *outputShape)

