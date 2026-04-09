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

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = wp.float32, length=Any), # type: ignore
    vec : vector(dtype = wp.float32, length=3), # type: ignore
    out : vector(dtype = wp.float32, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    dim = wp.int32(3) # hardcoded as this is only implemented for 3D vectors currently.
    
    # the output is stored as a flattened vector, so we need to compute the correct index for accumulation
    res = type(out)(0.0)
    for i in range(flatInputShape): # loop over elements of input tensor
        for j in range(dim): # loop over dimensions of output gradient
            outIndex = j * flatInputShape + i # compute flattened index for output
            res[outIndex] += vec[j] * tensor[i] # accumulate outer product into output
            
    return res

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = wp.float32, length=Any), # type: ignore
    vec : vector(dtype = wp.float32, length=2), # type: ignore
    out : vector(dtype = wp.float32, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    dim = wp.int32(2) # hardcoded as this is only implemented for 2D vectors currently.
    
    # the output is stored as a flattened vector, so we need to compute the correct index for accumulation
    res = type(out)(0.0)
    for i in range(flatInputShape): # loop over elements of input tensor
        for j in range(dim): # loop over dimensions of output gradient
            outIndex = j  + i * dim# compute flattened index for output
            res[outIndex] += vec[j] * tensor[i] # accumulate outer product into output
            
    return res

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = wp.float32, length=Any), # type: ignore
    vec : vector(dtype = wp.float32, length=1), # type: ignore
    out : vector(dtype = wp.float32, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    # for 1D vectors the outer product is just a scalar multiplication, so we can skip the indexing logic
    res = type(out)(0.0)
    for i in range(flatInputShape):
        res[i] += vec[0] * tensor[i]
    return res
    
                
# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.
@wp.func
def computeSPHGradientTensor_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, # type: ignore
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int64, numNeighs: wp.int32, preScatteredQuantities: wp. bool,
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValue: vector(length=Any, dtype=wp.float32) # type: ignore
):
    dim = wp.int32(xi.length)
    grad_f_interpolated = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + wp.int64(neighborIndex)
        j = wp.int32(neighborList[jj])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        fj = type(fi)(0.0)
        if preScatteredQuantities:
            fj = values[jj]
        else:
            fj = values[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        if gradientMode_int == 1: # Naive
            grad_f_interpolated += outerTensorProduct(fj * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += outerTensorProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHGradientTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64), preScatteredQuantities: wp. bool,# type: ignore
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]
    
    outputValues[i] = computeSPHGradientTensor_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], wp.int32(numNeighbors[i]), 
        preScatteredQuantities,
        numDims, flatInputShape, flatOutputShape,
        type(outputValues[i])(0.0))
    


from ..enumTypes import *
from typing import Optional

def computeSPHGradient_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    adjacency: AdjacencyListWarp,
    scatteredQuantities: Optional[torch.Tensor] = None
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    mode_uint = convertModeToUint(mode.name)
    kernel_int = kernel.value
    gradientMode_int = gradientMode.value

    preScatteredQuantities = False # Indicates if the input quantities have already been scattered to the neighbor level (e.g. mass/density products), which can save some redundant computations if they are needed for multiple operations. This can also help with some custom kernels where we want to pre-compute certain quantities at the neighbor level on the Python side and pass them in as additional fields to avoid redundant computations in the kernel. 
    if queryValues is None and referenceValues is None:
        if scatteredQuantities is None:
            raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the gradient computation.")
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

    outputShape = inputShape + (queryPositions.shape[1],) # add an extra dimension for the gradient
    flatOutputShape = 1
    for dim in outputShape:
        flatOutputShape *= dim
    # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
    numDims = len(inputShape)

    warp_result = warpWrapper(
        launch_kernel, computeSPHGradientTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
        domainMin, domainMax, periodicity,
        mode_uint, kernel_int, gradientMode_int,
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, preScatteredQuantities,
        wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape)
    )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension