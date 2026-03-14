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
def getVecLength(vec: vector(dtype = wp.float32, length=1)):
    return wp.int32(1)
@wp.func
def getVecLength(vec: vector(dtype = wp.float32, length=2)):
    return wp.int32(2)
@wp.func
def getVecLength(vec: vector(dtype = wp.float32, length=3)):
    return wp.int32(3)

@wp.func
def getMatrixShape(mat: matrix(dtype = wp.float32, shape=(1, 1))):
    return (wp.int32(1), wp.int32(1))
@wp.func
def getMatrixShape(mat: matrix(dtype = wp.float32, shape=(2, 2))):
    return (wp.int32(2), wp.int32(2))
@wp.func
def getMatrixShape(mat: matrix(dtype = wp.float32, shape=(3, 3))):
    return (wp.int32(3), wp.int32(3))



# this implements a manual outer product between a matrix and a vector
# in torch terms we are doing out += torch.einsum('..., d -> ...d', mat, vec)
@wp.func
def outerMatrixVectorFlattenedAccumulate(
    mat : matrix(dtype = wp.float32, shape=(1, 1)),
    vec : vector(dtype = wp.float32, length=1),
    out : vector(dtype = wp.float32, length=1)
):
    res = type(out)(0.0)
    
    for i in range(1):
        for j in range(1):
            for k in range(1):
                res[k + j * 1 + i * 1 * 1] = vec[k] * mat[i, j]
                
    return res

@wp.func
def outerMatrixVectorFlattenedAccumulate(
    mat : matrix(dtype = wp.float32, shape=(2, 2)),
    vec : vector(dtype = wp.float32, length=2),
    out : vector(dtype = wp.float32, length=8)
):
    res = type(out)(0.0)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                res[k + j * 2 + i * 2 * 2] = vec[k] * mat[i, j]
                
    return res

@wp.func
def outerMatrixVectorFlattenedAccumulate(
    mat : matrix(dtype = wp.float32, shape=(3, 3)),
    vec : vector(dtype = wp.float32, length=3),
    out : vector(dtype = wp.float32, length=27)
):
    res = type(out)(0.0)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                res[k + j * 3 + i * 3 * 3] = vec[k] * mat[i, j]
                
    return res
                
    

# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.
@wp.func
def computeSPHGradientMatrix_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : matrix(dtype = wp.float32, shape=(Any, Any)),
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = matrix(dtype = wp.float32, shape=(Any, Any))),
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32,
    
    neighborList: wp.array(dtype = wp.int64),
    neighborOffset : wp.int64, numNeighs: wp.int32,
    
    outputValue: vector(length=Any, dtype=wp.float32)
):
    dim = wp.int32(xi.length)
    grad_f_interpolated = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        j = wp.int32(neighborList[neighborOffset + wp.int64(neighborIndex)])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        fj = values[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        if gradientMode_int == 1: # Naive
            grad_f_interpolated += outerMatrixVectorFlattenedAccumulate(fj * apparentVolume, kernelGradient, grad_f_interpolated)
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += outerMatrixVectorFlattenedAccumulate(rhoj * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, grad_f_interpolated)
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += outerMatrixVectorFlattenedAccumulate((fj - fi) * apparentVolume, kernelGradient, grad_f_interpolated)
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += outerMatrixVectorFlattenedAccumulate((fj + fi) * apparentVolume, kernelGradient, grad_f_interpolated)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHGradientMatrix_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), 
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),
    queryValues: wp.array(dtype =matrix(dtype = wp.float32, shape=(Any, Any))), referenceValues: wp.array(dtype = matrix(dtype = wp.float32, shape=(Any, Any))),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64),
    
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
    
    outputValues[i] = computeSPHGradientMatrix_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], wp.int32(numNeighbors[i]), 
        type(outputValues[i])(0.0))
    


from ..enumTypes import *

def computeSPHGradientMatrix_warpBackend(
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
        launch_kernel, computeSPHGradientMatrix_Kernel, outputShape, vector(length=queryPositions.shape[1]**3, dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues,
        domainMin, domainMax, periodicity,
        mode_uint, kernel_int, gradientMode_int,
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
    )

    return warp_result.view(queryPositions.shape[0], queryPositions.shape[1], queryPositions.shape[1], queryPositions.shape[1])