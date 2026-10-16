import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *

@wp.func
def updateNeighborsVerletFunc(
    # General Shape Parameters and indices
    i : wp.int32,

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = scalar_t, length=Any)), querySupports: wp.array(dtype = scalar_t), # type: ignore
    # SPH properties for the reference set (indexed by j)
    referencePositions: wp.array(dtype=vector(dtype = scalar_t, length=Any)), referenceSupports: wp.array(dtype = scalar_t), # type: ignore

    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    mode_uint: wp.uint32,

    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    neighborList: wp.array(dtype = index_t), # type: ignore
    neighborOffset : offset_t, numNeighs: wp.int32,

    # New Neighbor list data
    newNeighborOffset: offset_t, newNumNeighs: wp.int32,

    # Outputs
    edge_i: wp.array1d(dtype=index_t),  # shape [total_edges] # type: ignore
    edge_j: wp.array1d(dtype=index_t)   # shape [total_edges] # type: ignore
):
    xi = queryPositions[i]
    hi = querySupports[i]

    domainState = domainData()
    domainState.domainMin = domainMin
    domainState.domainMax = domainMax
    domainState.periodicity = periodicity
    domainState.dim = wp.int32(domainMin.shape[0])

    counter = wp.int32(0)
    # Loop over neighbors to compute the gradient contribution from each neighbor
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + offset_t(neighborIndex)
        j  = wp.int32(neighborList[jj])

        xj = referencePositions[j]
        hj = referenceSupports[j]

        x_ij = computeDistanceVec(xi, xj, domainState)
        hij = computePairwiseSupport(hi, hj, mode_uint)
        
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        edge_index = newNeighborOffset + offset_t(counter)
        if r_ij <= hij:
            edge_i[edge_index] = index_t(i)
            edge_j[edge_index] = index_t(j)
            counter += 1

    return counter


@wp.kernel
def updateNeighborsVerlet(
    queryPositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), referencePositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), # type: ignore
    querySupports: wp.array(dtype=scalar_t), referenceSupports: wp.array(dtype=scalar_t), # type: ignore

    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore

    mode_uint: wp.uint32, 
    neighborList: wp.array(dtype = index_t), neighborListRowOffsets: wp.array(dtype = offset_t), numNeighbors: wp.array(dtype = wp.int32), # type: ignore

    # Output
    newNeighborListRowOffsets: wp.array(dtype = offset_t), newNeighborCounter: wp.array(dtype = wp.int32), # type: ignore

    edge_i: wp.array1d(dtype=index_t),  # shape [total_edges] # type: ignore
    edge_j: wp.array1d(dtype=index_t)   # shape [total_edges] # type: ignore
):                                                                            
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    updateNeighborsVerletFunc(
        i, 
        queryPositions, querySupports, 
        referencePositions, referenceSupports, 
        periodicity, domainMin, domainMax, 
        mode_uint, 
        neighborList, neighborListRowOffsets[i], numNeighbors[i],
        newNeighborListRowOffsets[i], newNeighborCounter[i],
        edge_i, edge_j
    )
    
