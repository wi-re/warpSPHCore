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

# @wp.func
# def zeroInit(
#     referenceValues: Any
# ):
#     return type(referenceValues)()

# @wp.overload
# def zeroInit(
#     referenceValues: wp.float32
# ):
#     return wp.float32(0.0)



@wp.func
def computeSPHInterpolation_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32,
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32),
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32,
    
    neighborList: wp.array(dtype = wp.int64),
    neighborOffset : wp.int64, numNeighs: wp.int32,
    
    fi: Any,
    fieldValues: wp.array(dtype = Any)
):
    f_interpolated = type(fi)(0.0)
    
    for neighborIndex in range(numNeighs):
        j = wp.int32(neighborList[neighborOffset + wp.int64(neighborIndex)])
        
        
        xj= positions[j]
        hj= supports[j]
        
        pairwiseDistance = computeDistance(xi, xj, periodicity, domainMin, domainMax)
        pairwiseSupport = computePairwiseSupport(hi, hj, mode_uint)
        if pairwiseDistance <= pairwiseSupport:
            f_interpolated += fieldValues[j] * masses[j] * wendland4_2d(pairwiseDistance, pairwiseSupport, 2) / densities[j]
            
    return f_interpolated


@wp.kernel
def computeSPHInterpolation_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), 
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),
    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64),
    
    outputValues : wp.array(dtype = Any)
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]
    
    neighborOffset = neighborListRowOffsets[i]
    numNeighs = numNeighbors[i]
    
    outputValues[i] = computeSPHInterpolation_Func(
        xi, hi, mi, rhoi,
        referencePositions, referenceSupports, referenceMasses, referenceDensities,
        periodicity, domainMin, domainMax, mode_uint,
        neighborList, neighborOffset, wp.int32(numNeighs),
        fi, referenceValues
    )
    
def warp_sphInterpolation(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    
    domainMin, domainMax, periodicity,
    mode_uint,
    
    neighborList, neighborListRowOffsets, numNeighbors
):
    inputs = [
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        domainMin, domainMax, periodicity,
        mode_uint,
        neighborList, neighborListRowOffsets, numNeighbors
    ]
    requires_grad = any(input.requires_grad for input in inputs if hasattr(input, 'requires_grad'))
    
    output = wp.zeros(queryPositions.shape[0], dtype=queryValues.dtype, device=queryPositions.device)
    output.requires_grad = requires_grad
    
    wp.launch(
        computeSPHInterpolation_Kernel,
        dim = queryPositions.shape[0],
        inputs = [
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            
            domainMin, domainMax, periodicity,
            
            wp.uint32(mode_uint),
            neighborList, neighborListRowOffsets, numNeighbors,
            
            output
        ],
        device = queryPositions.device
    )
    
    return output


def computeSPHInterpolant_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    domain: DomainDescription,
    mode: str,    
    adjacency
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
    # For higher-rank inputs (e.g. shape (n, p, m, d)) we flatten the field
    # dimensions to a single vector dimension, interpolate, then restore the
    # original shape.  Rank <= 3 inputs (scalar / vector / matrix per particle)
    # pass through unchanged.
    field_shape = queryValues.shape[1:]   # all dims after the particle batch dim
    needs_flatten = queryValues.dim() > 3
    if needs_flatten:
        flat_len = queryValues[0].numel()
        queryValues    = queryValues.reshape(queryValues.shape[0],    flat_len).contiguous()
        referenceValues = referenceValues.reshape(referenceValues.shape[0], flat_len).contiguous()

    warp_interpolation = warpWrapper(
        warp_sphInterpolation,
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues,
        
        domainMin, domainMax, periodicity,
        convertModeToUint(mode),
        
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors
    )

    if needs_flatten:
        warp_interpolation = warp_interpolation.reshape(
            warp_interpolation.shape[0], *field_shape
        )

    return warp_interpolation