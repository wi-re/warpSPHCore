import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from ...utils.wp_autograd import *
from ...radiusSearch.radius_util import convertModeToUint

from ...radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ...mathutil.wp_math import *
from ...kernels.wp_kernel import *

@wp.func
def computeSPHGradientScalar_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : wp.float32,
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = wp.float32),
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32,
    
    neighborList: wp.array(dtype = wp.int64),
    neighborOffset : wp.int32, numNeighs: wp.int32,
    
    outputValue: vector(length=Any, dtype=wp.float32)
):
    grad_f_interpolated = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        j = wp.int32(neighborList[neighborOffset + neighborIndex])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        fj = values[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        if gradientMode_int == 1: # Naive
            grad_f_interpolated += fj * apparentVolume * kernelGradient
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += rhoj * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume * kernelGradient
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += (fj - fi) * apparentVolume * kernelGradient
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += (fj + fi) * apparentVolume * kernelGradient
            
    return grad_f_interpolated

@wp.kernel
def computeSPHGradientScalar_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), 
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),
    queryValues: wp.array(dtype = wp.float32), referenceValues: wp.array(dtype = wp.float32),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32),
    
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32))
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]
    
    outputValues[i] = computeSPHGradientScalar_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        type(outputValues[i])(0.0))
    


from ...enumTypes import *

def computeSPHGradientScalar_warpBackend(
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
    outputShape = (queryPositions.shape[0])

    warp_result = warpWrapper(
        launch_kernel, computeSPHGradientScalar_Kernel, outputShape, vector(length=queryPositions.shape[1], dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues,
        domainMin, domainMax, periodicity,
        mode_uint, kernel_int, gradientMode_int,
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
    )

    return warp_result