import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity

# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.
@wp.func
def computeSPHCovariance_Func(
    i : wp.int32,
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, # type: ignore
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, preScatteredQuantities: wp. bool,
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    L: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), renormalize: wp.bool, # type: ignore
    opInt: wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    outputValue: vector(length=Any, dtype=wp.float32) # type: ignore
):
    dim = wp.int32(xi.length)
    grad_f_interpolated = type(outputValue)(0.0)

    Li = type(L[0])()
    if renormalize:
        Li = L[i]
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj

        fij = -computeDistanceVec(xi, positions[j], periodicity, domainMin, domainMax)
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        if renormalize:
            kernelGradient = matmul(Li, kernelGradient)        

        grad_f_interpolated += outerTensorProduct(fij * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHCovariance_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), preScatteredQuantities: wp. bool,# type: ignore
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    L: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), renormalize: wp.bool, # type: ignore
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    
    outputValues[i] = computeSPHCovariance_Func(
        i,
        xi, hi, mi, rhoi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, 
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        preScatteredQuantities,
        numDims, flatInputShape, flatOutputShape,
        L, renormalize,
        opInt, referenceKinds,
        type(outputValues[i])(0.0))
    


from ..enumTypes import *
from typing import Optional

def computeSPHCovariance_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,
    renormalizationMatrices: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[Covariance]"):
        with record_function("warpSPH[Covariance] - Preprocessing"):
            # Preprocessing and input validation
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = convertModeToUint(mode.name)
            kernel_int = kernel.value
            gradientMode_int = 0
            opInt = wp.int32(operationMode.value)

            preScatteredQuantities = False # Indicates if the input quantities have already been scattered to the neighbor level (e.g. mass/density products), which can save some redundant computations if they are needed for multiple operations. This can also help with some custom kernels where we want to pre-compute certain quantities at the neighbor level on the Python side and pass them in as additional fields to avoid redundant computations in the kernel. 

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            inputShape = queryPositions.shape[1:]
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
            L = renormalizationMatrices if renormalizationMatrices is not None else getCachedDummyTensor((1, D, D), dtype=torch.float32, device=queryPositions.device)
            renormalize = renormalizationMatrices is not None

        with record_function("warpSPH[Covariance] - Kernel Execution"):
            warp_result = warpWrapper(
                launch_kernel, computeSPHCovariance_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int,
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, preScatteredQuantities,
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                L, wp.bool(renormalize),
                opInt, queryKinds, referenceKinds,
            )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension