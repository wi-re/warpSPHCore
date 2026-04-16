import warp as wp
from warp.types import vector, matrix
@wp.func
def mod_distance(
    x : float, y: float, minDomain: float, maxDomain: float, periodic: bool
):
    if periodic:
        dx = x - y
        domain_size = maxDomain - minDomain
        if wp.abs(dx) > domain_size / 2.0:
            dx = wp.sign(dx) * (wp.abs(dx) - domain_size)
    else:
        dx = x - y
    
    return dx

# @wp.kernel
def computeCartesianDistance_2d(
    x: wp.vec2f,
    y: wp.vec2f,
    minDomain: wp.vec2f,
    maxDomain: wp.vec2f,
    periodic: wp.array(dtype=wp.bool)
):
    dist_sq = float(0.0)
    dx = mod_distance(x[0], y[0], minDomain[0], maxDomain[0], periodic[0])
    dy = mod_distance(x[1], y[1], minDomain[1], maxDomain[1], periodic[1])
    dist_sq = dx * dx + dy * dy
    return wp.sqrt(dist_sq)
    
    
@wp.func
def computeCartesianDistance(
    x: wp.array(dtype=wp.float32),  # Shape (D,)
    y: wp.array(dtype=wp.float32),  # Shape (D,)
    minDomain: wp.array(dtype=wp.float32),  # Shape (D,)
    maxDomain: wp.array(dtype=wp.float32),  # Shape (D,)
    periodic: wp.array(dtype=wp.bool)        # Shape (D,)
):
    dist_sq = float(0.0)
    for d in range(wp.len(x)):
        dx = mod_distance(x[d], y[d], minDomain[d], maxDomain[d], periodic[d])
        dist_sq += dx * dx
    return wp.sqrt(dist_sq)


@wp.func 
def mod_warp(x : wp.float32, min: wp.float32, max: wp.float32):
    h = max - min
    return ((x + h / 2.0) - wp.floor((x + h / 2.0) / h) * h) - h / 2.0

@wp.func
def moduloDistanceWarp(xij:wp.array(dtype = wp.float32), periodicity: wp.array(dtype = wp.bool), min: wp.array(dtype = wp.float32), max: wp.array(dtype = wp.float32)):
    result = wp.zeros_like(xij)
    for i in range(periodicity.shape[0]):
        if periodicity[i]:
            result[i] = mod_warp(xij[i], min[i], max[i])
        else:
            result[i] = xij[i]
    return result
@wp.func
def minimumImageDistanceWarp(x: wp.array(dtype = wp.float32), y: wp.array(dtype = wp.float32), min: wp.array(dtype = wp.float32), max: wp.array(dtype = wp.float32), periodicity: wp.array(dtype = wp.bool)):
    x_projected = wp.zeros_like(x)
    y_projected = wp.zeros_like(y)
    for i in range(periodicity.shape[0]):
        if periodicity[i]:
            x_projected[i] = wp.remainder(x[i] - min[i], max[i] - min[i]) + min[i]
            y_projected[i] = wp.remainder(y[i] - min[i], max[i] - min[i]) + min[i]
        else:
            x_projected[i] = x[i]
            y_projected[i] = y[i]
    xij = x_projected - y_projected
    return moduloDistanceWarp(xij, periodicity, min, max)

@wp.func 
def computeDistance(x: wp.array(dtype = wp.float32), y: wp.array(dtype = wp.float32), min: wp.array(dtype = wp.float32), max: wp.array(dtype = wp.float32), periodicity: wp.array(dtype = wp.bool)):
    vectorDistance = minimumImageDistanceWarp(x, y, min, max, periodicity)
    length = wp.sqrt(wp.sum(vectorDistance * vectorDistance))
    return length


from warp.types import vector

@wp.func
def safe_sqrt(x: float):
    return wp.sqrt(x)
@wp.func_grad(safe_sqrt)
def adj_safe_sqrt(x: float, adj_ret: float):
    if x > 0.0:
        wp.adjoint[x] += 1.0 / (2.0 * wp.sqrt(x)) * adj_ret

@wp.func 
def mod_warp(x : wp.float32, min: wp.float32, max: wp.float32):
    h = max - min
    return ((x + h / 2.0) - wp.floor((x + h / 2.0) / h) * h) - h / 2.0

@wp.func
def project_mod(x : wp.float32, min: wp.float32, max: wp.float32):
    # return torch.remainder(x - min, max - min) + min
    # we need to implement this manually because torch.remainder does not work in warp and warp has no remainder
    h = max - min
    return x - h * wp.floor((x - min) / h)

@wp.func
def moduloDistanceComponent(
    x: wp.float32, y: wp.float32, periodic: bool, min: wp.float32, max: wp.float32
):
    if periodic:
        x_projected = project_mod(x, min, max)
        y_projected = project_mod(y, min, max)
        xij = x_projected - y_projected
        return mod_warp(xij, min, max)
    else:
        return x - y

from typing import Any
@wp.func
def minimumImageDistance(
    x: vector(dtype=wp.float32, length=Any),
    y: vector(dtype=wp.float32, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=wp.float32),
    max: wp.array(dtype=wp.float32),
    D: wp.int32
):
    retVal = wp.vector(0.0, length = x.length, dtype = wp.float32)
    for i in range(D):
        retVal[i] = moduloDistanceComponent(x[i], y[i], periodicity[i], min[i], max[i])
    return retVal
    
    
    # return wp.vector(
    #     [moduloDistanceComponent(x[i], y[i], periodicity[i], min[i], max[i]) for i in range(x.length)],
    #     dtype=wp.float32
    # )
    
@wp.func
def computeDistance(
    x: vector(dtype=wp.float32, length=Any),
    y: vector(dtype=wp.float32, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=wp.float32),
    max: wp.array(dtype=wp.float32)
):
    distVec = minimumImageDistance(x, y, periodicity, min, max, wp.int32(x.length))
    # distVec = x-y
    return safe_sqrt(wp.dot(distVec, distVec))
    
@wp.func
def computeDistanceVec(
    x: vector(dtype=wp.float32, length=Any),
    y: vector(dtype=wp.float32, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=wp.float32),
    max: wp.array(dtype=wp.float32)
):
    distVec = minimumImageDistance(x, y, periodicity, min, max, wp.int32(x.length))
    return distVec
    # distVec = x-y
    # return safe_sqrt(wp.dot(distVec, distVec))

# @wp.func 
# def computeDistance(x: vector(dtype = wp.float32), y: vector(dtype = wp.float32), periodicity: wp.array(dtype = wp.bool), min: wp.array(dtype = wp.float32), max: wp.array(dtype = wp.float32)):
#     vectorDistance = minimumImageDistanceWarp(x, y, periodicity, min, max)
#     length = wp.sqrt(wp.sum(vectorDistance * vectorDistance))
#     return length


         
@wp.func
def matmul(
    mat: matrix(shape=(1, 1), dtype=wp.float32), # type: ignore
    vec: vector(dtype = wp.float32, length=1), # type: ignore
):
    numRows = 1
    numCols = 1
    
    res = type(vec)(0.0)
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res

@wp.func
def matmul(
    mat: matrix(shape=(2, 2), dtype=wp.float32), # type: ignore
    vec: vector(dtype = wp.float32, length=2), # type: ignore
):
    numRows = 2
    numCols = 2
    
    res = type(vec)(0.0)
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res

@wp.func
def matmul(
    mat: matrix(shape=(3, 3), dtype=wp.float32), # type: ignore
    vec: vector(dtype = wp.float32, length=3), # type: ignore
):
    numRows = 3
    numCols = 3
    
    res = type(vec)(0.0)
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res


from warp.types import vector, matrix

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
    
import numpy as np

import torch
@torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : int, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)
    

@wp.func
def volumeToSupport_warp(volume : wp.float32, targetNeighbors : wp.int32, dim : wp.int32):
    if dim == 1:
        return targetNeighbors * volume / 2.0
    elif dim == 2:
        return safe_sqrt(targetNeighbors * volume / np.pi)
    else:
        return wp.pow(targetNeighbors * volume / np.pi * 3.0 /4.0, 1.0/3.0)