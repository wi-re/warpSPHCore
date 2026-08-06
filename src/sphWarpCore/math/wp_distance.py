import warp as wp
from warp.types import vector, matrix
from ..types import *
from .wp_sqrt import *
from typing import Optional, Any, Union, List, Tuple

@wp.func
def mod_distance(
    x : scalar_t, y: scalar_t, minDomain: scalar_t, maxDomain: scalar_t, periodic: bool
):
    if periodic:
        dx = x - y
        domain_size = maxDomain - minDomain
        if wp.abs(dx) > domain_size / scalar_t(2.0):
            dx = wp.sign(dx) * (wp.abs(dx) - domain_size)
    else:
        dx = x - y
    
    return dx

# # @wp.kernel
# def computeCartesianDistance_2d(
#     x: wp.vec2f,
#     y: wp.vec2f,
#     minDomain: wp.vec2f,
#     maxDomain: wp.vec2f,
#     periodic: wp.array(dtype=wp.bool)
# ):
#     dist_sq = float(0.0)
#     dx = mod_distance(x[0], y[0], minDomain[0], maxDomain[0], periodic[0])
#     dy = mod_distance(x[1], y[1], minDomain[1], maxDomain[1], periodic[1])
#     dist_sq = dx * dx + dy * dy
#     return wp.sqrt(dist_sq)
    
    
@wp.func
def computeCartesianDistance(
    x: wp.array(dtype=scalar_t),  # Shape (D,)
    y: wp.array(dtype=scalar_t),  # Shape (D,)
    minDomain: wp.array(dtype=scalar_t),  # Shape (D,)
    maxDomain: wp.array(dtype=scalar_t),  # Shape (D,)
    periodic: wp.array(dtype=wp.bool)        # Shape (D,)
):
    dist_sq = scalar_t(0.0)
    for d in range(wp.len(x)):
        dx = mod_distance(x[d], y[d], minDomain[d], maxDomain[d], periodic[d])
        dist_sq += dx * dx
    return wp.sqrt(dist_sq)


@wp.func 
def mod_warp(x : scalar_t, min: scalar_t, max: scalar_t):
    h = max - min
    return ((x + h / scalar(2.0)) - wp.floor((x + h / scalar(2.0)) / h) * h) - h / scalar(2.0)

@wp.func
def moduloDistanceWarp(xij:wp.array(dtype = scalar_t), periodicity: wp.array(dtype = wp.bool), min: wp.array(dtype = scalar_t), max: wp.array(dtype = scalar_t)):
    result = wp.zeros_like(xij)
    for i in range(periodicity.shape[0]):
        if periodicity[i]:
            result[i] = mod_warp(xij[i], min[i], max[i])
        else:
            result[i] = xij[i]
    return result
@wp.func
def minimumImageDistanceWarp(x: wp.array(dtype = scalar_t), y: wp.array(dtype = scalar_t), min: wp.array(dtype = scalar_t), max: wp.array(dtype = scalar_t), periodicity: wp.array(dtype = wp.bool)):
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
def computeDistance(x: wp.array(dtype = scalar_t), y: wp.array(dtype = scalar_t), min: wp.array(dtype = scalar_t), max: wp.array(dtype = scalar_t), periodicity: wp.array(dtype = wp.bool)):
    vectorDistance = minimumImageDistanceWarp(x, y, min, max, periodicity)
    length = wp.sqrt(wp.sum(vectorDistance * vectorDistance))
    return length


@wp.func 
def mod_warp(x : scalar_t, min: scalar_t, max: scalar_t):
    h = max - min
    return ((x + h / scalar(2.0)) - wp.floor((x + h / scalar(2.0)) / h) * h) - h / scalar(2.0)

@wp.func
def project_mod(x : scalar_t, min: scalar_t, max: scalar_t):
    # return torch.remainder(x - min, max - min) + min
    # we need to implement this manually because torch.remainder does not work in warp and warp has no remainder
    h = max - min
    return x - h * wp.floor((x - min) / h)

@wp.func
def moduloDistanceComponent(
    x: scalar_t, y: scalar_t, periodic: bool, min: scalar_t, max: scalar_t
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
    x: vector(dtype=scalar_t, length=Any),
    y: vector(dtype=scalar_t, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=scalar_t),
    max: wp.array(dtype=scalar_t),
    D: wp.int32
):
    retVal = wp.vector(scalar(0.0), length = x.length, dtype = scalar_t)
    for i in range(D):
        retVal[i] = moduloDistanceComponent(x[i], y[i], periodicity[i], min[i], max[i])
    return retVal
    
    
    # return wp.vector(
    #     [moduloDistanceComponent(x[i], y[i], periodicity[i], min[i], max[i]) for i in range(x.length)],
    #     dtype=scalar_t
    # )
    
@wp.func
def computeDistance(
    x: vector(dtype=scalar_t, length=Any),
    y: vector(dtype=scalar_t, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=scalar_t),
    max: wp.array(dtype=scalar_t)
):
    distVec = minimumImageDistance(x, y, periodicity, min, max, wp.int32(x.length))
    # distVec = x-y
    return safe_sqrt(wp.dot(distVec, distVec))
    
@wp.func
def computeDistanceVec(
    x: vector(dtype=scalar_t, length=Any),
    y: vector(dtype=scalar_t, length=Any),
    periodicity: wp.array(dtype=wp.bool),
    min: wp.array(dtype=scalar_t),
    max: wp.array(dtype=scalar_t)
):
    distVec = minimumImageDistance(x, y, periodicity, min, max, wp.int32(x.length))
    return distVec
    # distVec = x-y
    # return safe_sqrt(wp.dot(distVec, distVec))

# @wp.func 
# def computeDistance(x: vector(dtype = scalar_t), y: vector(dtype = scalar_t), periodicity: wp.array(dtype = wp.bool), min: wp.array(dtype = scalar_t), max: wp.array(dtype = scalar_t)):
#     vectorDistance = minimumImageDistanceWarp(x, y, periodicity, min, max)
#     length = wp.sqrt(wp.sum(vectorDistance * vectorDistance))
#     return length
