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
def getStride(
    outputElements: wp.int32, numDims: wp.int32,
    V: vector(dtype = wp.float32, length=3)
):
    return wp.int32(outputElements / 3)
@wp.func
def getStride(
    outputElements: wp.int32, numDims: wp.int32,
    V: vector(dtype = wp.float32, length=2)
):
    return wp.int32(outputElements / 2)
@wp.func
def getStride(
    outputElements: wp.int32, numDims: wp.int32,
    V: vector(dtype = wp.float32, length=1)
):
    return wp.int32(outputElements)


# 
@wp.func
def curlProduct(
    T: vector(dtype = wp.float32, length=Any), 
    V: vector(dtype = wp.float32, length=3),
    output: vector(dtype = wp.float32, length=Any),
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(0.0)
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.
    # stride = wp.int32(outputElements / dim) # this is the number of elements in each dimension of the input tensor fij. So if fij is of shape [d^N] and output is of shape [d^(N-1)] then stride is d^(N-1).
    # // Loop over all possible combinations of the 'trailing' indices
    for s in range(stride):
        # // We focus on the first index of the tensor (k) 
        # // and the vector index (j) to produce the result index (i)
        
        # // Flattened locations for T[0][s], T[1][s], T[2][s]
        k0 = wp.int32(s) # this is the location of T[0][s] in the flattened fij
        k1 = wp.int32(s + stride) # this is the location of T[1][s] in the flattened fij
        k2 = wp.int32(s + 2 * stride) # this is the location of T[2][s] in the flattened fij
        # // Apply Levi-Civita / Cross Product logic:
        # // Result index i=0: V1*T2 - V2*T1
        R[0 * stride + s] = V[1] * T[k2] - V[2] * T[k1];

        # // Result index i=1: V2*T0 - V0*T2
        R[1 * stride + s] = V[2] * T[k0] - V[0] * T[k2];

        # // Result index i=2: V0*T1 - V1*T0
        R[2 * stride + s] = V[0] * T[k1] - V[1] * T[k0];
    return -R # the negative sign is needed to match the right hand rule convention for the curl operator.
# 
@wp.func
def curlProduct(
    T: vector(dtype = wp.float32, length=Any), 
    V: vector(dtype = wp.float32, length=2),
    output: vector(dtype = wp.float32, length=Any),
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(0.0)
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.
    # stride = wp.int32(outputElements / dim) # this is the number of elements in each dimension of the input tensor fij. So if fij is of shape [d^N] and output is of shape [d^(N-1)] then stride is d^(N-1).
    # // Loop over all possible combinations of the 'trailing' indices
    for s in range(stride+1): # we loop to stride+1 because in 2D the output has one less dimension than the input so we need to compute one more element to account for this.
        # // We focus on the first index of the tensor (k) 
        # // and the vector index (j) to produce the result index (i)
        
        # // Flattened locations for T[0][s], T[1][s], T[2][s]
        k0 = wp.int32(s) # this is the location of T[0][s] in the flattened fij
        k1 = wp.int32(s + stride+1) # this is the location of T[1][s] in the flattened fij
        # // 2D Cross Product logic: V0*T1 - V1*T0
        # // This collapses the first dimension of T and the dimension of V
        R[s] = V[0] * T[k1] - V[1] * T[k0];
    return R
        
@wp.func
def curlProduct(
    T: vector(dtype = wp.float32, length=Any), 
    V: vector(dtype = wp.float32, length=1),
    output: vector(dtype = wp.float32, length=Any),
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(0.0)
    # in 1D the curl product is just a simple multiplication
    R[0] = 0
    return R


@wp.func
def computeSPHCurlTensor_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any),
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)),
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, 
    
    neighborList: wp.array(dtype = wp.int64),
    neighborOffset : wp.int64, numNeighs: wp.int32,
    dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValue: vector(dtype = wp.float32, length=Any)
):
    grad_f_interpolated = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        j = wp.int32(neighborList[neighborOffset + wp.int64(neighborIndex)])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        fj = values[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        if gradientMode_int == 1: # Naive
            grad_f_interpolated += curlProduct(fj * apparentVolume, kernelGradient, outputValue, wp.int32(flatOutputShape/dim), flatInputShape, flatOutputShape)
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += curlProduct(rhoj * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, outputValue, wp.int32(flatOutputShape/dim), flatInputShape, flatOutputShape)
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += curlProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, wp.int32(flatOutputShape/dim), flatInputShape, flatOutputShape)
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += curlProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, wp.int32(flatOutputShape/dim), flatInputShape, flatOutputShape)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHCurlTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), 
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),
    queryValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64),
    
    dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32, 
    
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
    
    outputValues[i] = computeSPHCurlTensor_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], wp.int32(numNeighbors[i]), 
        dim, numDims, flatInputShape, flatOutputShape,
        type(outputValues[i])(0.0))
    
from ..enumTypes import *

def computeSPHCurl_warpBackend(
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
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    mode_uint = convertModeToUint(mode.name)
    kernel_int = kernel.value
    gradientMode_int = gradientMode.value

    # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
    outputSize = (queryPositions.shape[0])
    
    inputShape = queryValues.shape[1:]
    flatInputShape = 1
    for dim in inputShape:
        flatInputShape *= dim
        
    if queryPositions.shape[1] == 3:
        outputShape = inputShape
    elif queryPositions.shape[1] == 2:
        outputShape = inputShape[:-1]
        if outputShape == ():
            outputShape = [1] # if the input is a vector field, the output is a scalar field so we set the output shape to [1] to represent this.
    else: # 1D curl is just 0 so we can return a scalar
        outputShape = []
    flatOutputShape = 1
    for dim in outputShape:
        flatOutputShape *= dim
    numDims = len(inputShape)
    
    
    print(f"computeSPHDivergenceTensor_warpBackend: inputShape={inputShape}, flatInputShape={flatInputShape}, outputShape={outputShape}, flatOutputShape={flatOutputShape}, numDims={numDims}")

    warp_result = warpWrapper(
        launch_kernel, computeSPHCurlTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues.view(-1, flatInputShape), referenceValues.view(-1, flatInputShape),
        domainMin, domainMax, periodicity,
        mode_uint, kernel_int, gradientMode_int,
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
        wp.int32(queryPositions.shape[1]), wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape), 
    )

    return warp_result.view(queryValues.shape[0], *outputShape) # reshape back to original shape with new gradient dimension