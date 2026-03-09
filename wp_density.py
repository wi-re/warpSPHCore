import warp as wp
from warp.types import vector
from typing import Any
import torch
from wp_autograd import *
from radius_util import convertModeToUint

from radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from wp_math import *
from wp_kernel import *

@wp.func
def computeDensityWarp(
    xi : vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32,
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32),
    mode_uint: wp.uint32,
    
    
    neighborList: wp.array(dtype = wp.int64), 
    neighborOffset : wp.int64, numNeighs: wp.int32,
):
    rho = wp.float32(0.0)
    # pairwiseDistance = computeDistance(xi, xi, periodicity, domainMin, domainMax)
    # pairwiseSupport = computePairwiseSupport(hi, hi, mode_uint)
    # rho = mi * ( 1.0 - pairwiseDistance) #wendland4_2d(pairwiseDistance, pairwiseSupport, 2)
    
    for neighborIndex in range(numNeighs):
        j = wp.int32(neighborList[neighborOffset + wp.int64(neighborIndex)])
        
        
        xj= referencePositions[j]
        hj= referenceSupports[j]
        
        pairwiseDistance = computeDistance(xi, xj, periodicity, domainMin, domainMax)
        pairwiseSupport = computePairwiseSupport(hi, hj, mode_uint)
        # if pairwiseDistance <= pairwiseSupport:
        rho += referenceMasses[j] * wendland4_2d(pairwiseDistance, pairwiseSupport, 2)
    return rho

@wp.kernel
def sphDensity_warp(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)),
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32),
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool),
    
    mode_uint: wp.uint32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64),
    
    densities : wp.array(dtype = wp.float32)                                                                                                               
                                                                                  
):
    N = queryPositions.shape[0]
    M = referencePositions.shape[0]
    D = queryPositions.dtype._length_
    
    i = wp.tid()
    if i >= N:
        return
    
    xi= queryPositions[i]
    hi= querySupports[i]
    mi = queryMasses[i]
    numNeighs = wp.int32(numNeighbors[i])
    neighborOffset = wp.int64(neighborListRowOffsets[i])
    
    rho = computeDensityWarp(xi, hi, mi, referencePositions, referenceSupports, referenceMasses, periodicity, domainMin, domainMax, mode_uint, neighborList, neighborOffset, numNeighs)
        
    densities[i] = rho
        
def warp_sphDensityFunction(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    domainMin, domainMax, periodicity,
    mode_uint,
    neighborList, neighborListRowOffsets, numNeighbors
):
    inputs = [
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        domainMin, domainMax, periodicity,
        mode_uint,
        neighborList, neighborListRowOffsets, numNeighbors
    ]
    requires_grad = any(input.requires_grad for input in inputs if hasattr(input, 'requires_grad'))
    
    # for i, input in enumerate(inputs):
        # print(f'Input {i:02d}: type: {type(input)}')
    
    densities = wp.zeros(queryPositions.shape[0], dtype=querySupports.dtype, device=queryPositions.device)
    densities.requires_grad = requires_grad
    
    # print(f'Output: 0: type: {type(densities)} [dtype: {densities.dtype}, device: {densities.device}, shape: {densities.shape}, requires_grad: {densities.requires_grad}]')
    
    wp.launch(
        sphDensity_warp,
        dim = queryPositions.shape[0],
        inputs = [
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            
            domainMin, domainMax, periodicity,
            
            wp.uint32(mode_uint),
            neighborList, neighborListRowOffsets, numNeighbors,
            
            densities
        ],
        device = queryPositions.device
    )
    
    return densities


def computeDensity_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    domain: DomainDescription,
    mode: str,    
    adjacency
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    warp_sphDensity = warpWrapper(
        warp_sphDensityFunction,
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        domainMin, domainMax, periodicity,
        convertModeToUint(mode),
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors
    )
    
    return warp_sphDensity