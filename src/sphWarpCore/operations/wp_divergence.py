import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from torch.profiler import profile, record_function, ProfilerActivity

# In dot mode we compute torch.einsum('nd..., nd -> n...', q, k) 
# Otherwise we compute torch.einsum('n...d, nd -> n...', q, k)
# the inputs are the flattened versions of the original tensors, i.e.,
# if q initially was of shape [n, d, d, d] it is now [n, d^3]
# The output is always of shape [n, d^(N-1)] where N is the rank of the input tensor q. So if q was originally [n, d, d, d] the output will be [n, d^2]
# The kernelGradient is always of shape [n, d]
# this also allows us to overload the function based on d! We can then use the numDims parameter to do the correct indexing inside the kernel without having to write separate kernels for different dimensions.
@wp.func
def divergenceProduct(
    fij: vector(dtype = wp.float32, length=Any), 
    kernelGradient: vector(dtype = wp.float32, length=3),
    output: vector(dtype = wp.float32, length=Any),
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # res is now of shape [d^(N-1)] where N is the rank of the original input tensor.
    # we need to do a product between fij which is of shape [d^N] and kernelGradient which is of shape [d] to get an output of shape [d^(N-1)]
    
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
    fij: vector(dtype = wp.float32, length=Any), 
    kernelGradient: vector(dtype = wp.float32, length=2),
    output: vector(dtype = wp.float32, length=Any),
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # res is now of shape [d^(N-1)] where N is the rank of the original input tensor.
    # we need to do a product between fij which is of shape [d^N] and kernelGradient which is of shape [d] to get an output of shape [d^(N-1)]
    
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.
    # wp.printf("divergenceProduct: dim=%d, outputElements=%d, inputElements=%d\n", dim, outputElements, dim * outputElements)
    
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
    fij: vector(dtype = wp.float32, length=Any), 
    kernelGradient: vector(dtype = wp.float32, length=1),
    output: vector(dtype = wp.float32, length=Any),
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # in 1D the divergence product is just a simple multiplication
    res[0] = fij[0] * kernelGradient[0]
    return res 

@wp.func
def computeSPHDivergenceTensor_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any),
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)),
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, consistentDivergence: wp.bool,
    
    neighborList: wp.array(dtype = wp.int64),
    neighborOffset : wp.int64, numNeighs: wp.int32, preScatteredQuantities: wp.bool,
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32, dotMode: wp.bool,
    
    outputValue: vector(dtype = wp.float32, length=Any)
):
    grad_f_interpolated = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + wp.int64(neighborIndex)
        j = wp.int32(neighborList[jj])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        if consistentDivergence:
            apparentVolume = mj / rhoi
        fj = type(fi)(0.0)
        if preScatteredQuantities:
            fj = values[jj]
        else:
            fj = values[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        if gradientMode_int == 1: # Naive
            grad_f_interpolated += divergenceProduct(fj * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += divergenceProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += divergenceProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += divergenceProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHDivergenceTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), 
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),
    queryValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, consistentDivergence: wp.bool,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64), preScatteredQuantities: wp.bool,
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32, dotMode: wp.bool,
    
    outputValues : wp.array(dtype = vector(length = Any, dtype = wp.float32))
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]
    
    outputValues[i] = computeSPHDivergenceTensor_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int, consistentDivergence,
        neighborList, neighborListRowOffsets[i], wp.int32(numNeighbors[i]), 
        preScatteredQuantities,
        numDims, flatInputShape, flatOutputShape, dotMode,
        type(outputValues[i])(0.0))
    
from ..enumTypes import *

def computeSPHDivergence_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    adjacency: AdjacencyListWarp,
    consistentDivergence: bool = False,
    dotMode: bool = False, # if true compute the divergence based on torch.einsum('nd..., nd -> n...', q, k) instead of the normal div torch.einsum('n...d, nd -> n...', q, k)
    scatteredQuantities: Optional[torch.Tensor] = None,
):
    with record_function("warpSPH[Divergence]"):
        with record_function("warpSPH[Divergence] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = convertModeToUint(mode.name)
            kernel_int = kernel.value
            gradientMode_int = gradientMode.value

            preScatteredQuantities = False
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the divergence computation.")
                preScatteredQuantities = True
                qV = scatteredQuantities
                rV = scatteredQuantities
            else:
                qV = queryValues
                rV = referenceValues

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            inputShape = qV.shape[1:]
            flatInputShape = 1
            for dim in inputShape:
                flatInputShape *= dim
                
            outputShape = inputShape[1:] if dotMode else inputShape[:-1]
            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            numDims = len(inputShape)
            
    # print(f"computeSPHDivergenceTensor_warpBackend: inputShape={inputShape}, flatInputShape={flatInputShape}, outputShape={outputShape}, flatOutputShape={flatOutputShape}, numDims={numDims}")
        with record_function("warpSPH[Divergence] - Kernel Execution"):
            warp_result = warpWrapper(
                launch_kernel, computeSPHDivergenceTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int, wp.bool(consistentDivergence),
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, wp.bool(preScatteredQuantities),
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape), wp.bool(dotMode),
            )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension