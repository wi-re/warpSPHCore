from .wp_compactHash import *


# Verlet lists store not just the required neighbors but also some extra neighbors to avoid rebuilding the neighbor list every step.


@torch.jit.script
def _minimum_image_delta(
        current: torch.Tensor,
        previous: torch.Tensor,
        periodicity: torch.Tensor,
        domainMin: torch.Tensor,
        domainMax: torch.Tensor):
    delta = current - previous
    domainSize = domainMax - domainMin
    for d in range(delta.shape[1]):
        if bool(periodicity[d].item()):
            L = domainSize[d]
            # Shift into [-L/2, L/2) so boundary-crossing motion is measured correctly.
            delta_d = torch.remainder(delta[:, d] + L / 2, L) - L / 2
            delta[:, d] = delta_d
    return delta


@torch.jit.script
def _verlet_validity_metrics(
        queryPositions: torch.Tensor,
        referencePositions: torch.Tensor,
        priorQueryPositions: torch.Tensor,
        priorReferencePositions: torch.Tensor,
        querySupports: torch.Tensor,
        referenceSupports: torch.Tensor,
        supports_a: torch.Tensor,
        supports_b: torch.Tensor,
    periodicity: torch.Tensor,
    domainMin: torch.Tensor,
    domainMax: torch.Tensor,
        verletScale: float,
        support_case: int):
    # Stored supports in the Verlet adjacency were scaled by `verletScale` during build.
    priorQuerySupports = supports_a / verletScale
    priorReferenceSupports = supports_b / verletScale

    delta_a = _minimum_image_delta(queryPositions, priorQueryPositions, periodicity, domainMin, domainMax)
    delta_b = _minimum_image_delta(referencePositions, priorReferencePositions, periodicity, domainMin, domainMax)
    distance_a_max = torch.linalg.vector_norm(delta_a, dim=-1).amax()
    distance_b_max = torch.linalg.vector_norm(delta_b, dim=-1).amax()
    maxDistance = distance_a_max + distance_b_max

    querySupportDeltaMax = torch.abs(priorQuerySupports - querySupports).amax()
    referenceSupportDeltaMax = torch.abs(priorReferenceSupports - referenceSupports).amax()

    queryMinSupport = torch.minimum(priorQuerySupports.amin(), querySupports.amin())
    referenceMinSupport = torch.minimum(priorReferenceSupports.amin(), referenceSupports.amin())

    if support_case == 0:
        supportFactor = querySupportDeltaMax
        minSupport = queryMinSupport
    elif support_case == 1:
        supportFactor = referenceSupportDeltaMax
        minSupport = referenceMinSupport
    else:
        supportFactor = torch.maximum(querySupportDeltaMax, referenceSupportDeltaMax)
        minSupport = torch.minimum(queryMinSupport, referenceMinSupport)

    supportBuffer = (verletScale - 1.0) * minSupport
    # Motion and support drift both consume the same Verlet buffer budget.
    budgetUse = maxDistance + supportFactor
    shouldRebuild = budgetUse > supportBuffer
    return shouldRebuild, maxDistance, supportFactor, minSupport, supportBuffer


# @torch.jit.script
def buildVerletList_(
        queryPositions: torch.Tensor, referencePositions: torch.Tensor,
        querySupports: torch.Tensor, referenceSupports: torch.Tensor,
        domain : DomainDescription, verletScale : float = 1.0, 
        mode : SupportScheme = SupportScheme.SuperSymmetric,
        priorNeighborhood : Optional[AdjacencyList] = None, 
        verbose : bool = False):
    with record_function(f"[Verlet] {mode}"):
        if priorNeighborhood is None or (priorNeighborhood is not None and priorNeighborhood.queryPositions is None) or (priorNeighborhood is not None and priorNeighborhood.referencePositions is None):
            if verbose:
                print('Building neighborhood from scratch [no prior neighborhood]')
            with record_function(f"[Verlet] no prior"):
                adjacency = radiusSearchCompactHashMap_(
                    queryPositions, referencePositions,
                    querySupports * verletScale, referenceSupports * verletScale,
                    domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                )
        else:
            if queryPositions.shape[0] != priorNeighborhood.queryPositions.shape[0] or referencePositions.shape[0] != priorNeighborhood.referencePositions.shape[0]:
                if verbose:
                    print('Building neighborhood from scratch [different number of particles]')
                with record_function(f"[Verlet] mismatch"):
                    adjacency = radiusSearchCompactHashMap_(
                        queryPositions, referencePositions,
                        querySupports * verletScale, referenceSupports * verletScale,
                        domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                    )
            else:
                with record_function(f"[Verlet] Checking validity of prior neighborhood"):
                    supports_a = priorNeighborhood.querySupports
                    supports_b = priorNeighborhood.referenceSupports

                    if mode == SupportScheme.Gather:
                        support_case = 0
                    elif mode == SupportScheme.Scatter:
                        support_case = 1
                    else:
                        support_case = 2

                    shouldRebuild_t, maxDistance, supportFactor, minSupport, supportBuffer = _verlet_validity_metrics(
                        queryPositions,
                        referencePositions,
                        priorNeighborhood.queryPositions,
                        priorNeighborhood.referencePositions,
                        querySupports,
                        referenceSupports,
                        supports_a,
                        supports_b,
                        domain.periodic,
                        domain.min,
                        domain.max,
                        verletScale,
                        support_case,
                    )
                    shouldRebuild = bool(shouldRebuild_t.item())

                    if verbose:
                        distance_a = torch.linalg.vector_norm(
                            _minimum_image_delta(
                                queryPositions,
                                priorNeighborhood.queryPositions,
                                domain.periodic,
                                domain.min,
                                domain.max,
                            ),
                            dim=-1,
                        )
                        distance_b = torch.linalg.vector_norm(
                            _minimum_image_delta(
                                referencePositions,
                                priorNeighborhood.referencePositions,
                                domain.periodic,
                                domain.min,
                                domain.max,
                            ),
                            dim=-1,
                        )
                        priorQuerySupports = supports_a / verletScale
                        priorReferenceSupports = supports_b / verletScale
                        print(f'Distance a: min: {distance_a.min()}, max: {distance_a.max()}, avg: {distance_a.mean()}')
                        print(f'Distance b: min: {distance_b.min()}, max: {distance_b.max()}, avg: {distance_b.mean()}')
                        print(f'Support a (prior): min: {priorQuerySupports.min()}, max: {priorQuerySupports.max()}, avg: {priorQuerySupports.mean()}')
                        print(f'Support b (prior): min: {priorReferenceSupports.min()}, max: {priorReferenceSupports.max()}, avg: {priorReferenceSupports.mean()}')
                        print(f'Support a (current): min: {querySupports.min()}, max: {querySupports.max()}, avg: {querySupports.mean()}')
                        print(f'Support b (current): min: {referenceSupports.min()}, max: {referenceSupports.max()}, avg: {referenceSupports.mean()}')
                        print(f'Max |Δh|: {supportFactor}, Support buffer: {supportBuffer}, Minimum support: {minSupport}')
                        print(f'Max Distance: {maxDistance}')
                        print(f'Distance Factor: {maxDistance / minSupport}')
                        print(f'Distance Threshold Ratio: {1 + maxDistance / minSupport}, Verlet Scale: {verletScale}')



                    # print(maxDistance, minSupport * verletScale)
                    if shouldRebuild:
                        if verbose:
                    # if verbose:
                            print('Support Factor: ', supportFactor / minSupport, 'Minimum Support: ', minSupport)
                            print('Distance Factor: ', maxDistance / minSupport)
                            print('Threshold exceeded by: ', torch.maximum(maxDistance, supportFactor) / supportBuffer)
                            print('Building neighborhood from scratch [verlet buffer exceeded]')
                        with record_function(f"[Verlet] Rebuilding Invalid"):
                            adjacency = radiusSearchCompactHashMap_(
                                queryPositions, referencePositions,
                                querySupports * verletScale, referenceSupports * verletScale,
                                domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                            )
                            
                    else:
                        if verbose:
                            print('Reusing neighborhood')
                        adjacency = priorNeighborhood
        return adjacency
    
from ..state import *

def buildVerletList(
        queryParticles: ParticleState, 
        domain: DomainDescription, 
        verletScale : float = 1.0,
        supportMode: SupportScheme = SupportScheme.Gather,
        priorNeighborhood : Optional[AdjacencyList] = None,
        verbose : bool = False,
        referenceParticles: Optional[ParticleState] = None
):
    with record_function(f"[warpSPH] - buildVerletList"):
        if referenceParticles is None:
            referenceParticles = queryParticles

        return buildVerletList_(
            queryParticles.positions, referenceParticles.positions,
            queryParticles.supports, referenceParticles.supports,
            domain, verletScale, supportMode, 
            priorNeighborhood, verbose
        )

from sphWarpCore.utils import computePairwiseSupport

@wp.func
def countNeighborsVerletFunc(
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
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
):
    xi = queryPositions[i]
    hi = querySupports[i]

    counter = wp.int32(0)
    # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j  = wp.int32(neighborList[jj])

        xj = referencePositions[j]
        hj = referenceSupports[j]
        
        x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        hij = computePairwiseSupport(hi, hj, mode_uint)
        
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        if r_ij <= hij:
            counter += 1

    return counter

@wp.kernel
def countNeighborsVerlet(
    queryPositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), referencePositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), # type: ignore
    querySupports: wp.array(dtype=scalar_t), referenceSupports: wp.array(dtype=scalar_t), # type: ignore

    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore

    mode_uint: wp.uint32, 
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore

    # Output
    neighborCounter: wp.array(dtype = wp.int32), # type: ignore    
):                                                                            
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    neighborCounter[i] = countNeighborsVerletFunc(
        i, 
        queryPositions, querySupports, 
        referencePositions, referenceSupports, 
        periodicity, domainMin, domainMax, 
        mode_uint, 
        neighborList, neighborListRowOffsets[i], numNeighbors[i]
    )
    
    
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
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 

    # New Neighbor list data
    newNeighborOffset: wp.int32, newNumNeighs: wp.int32, 
    
    # Outputs
    edge_i: wp.array1d(dtype=wp.int64),  # shape [total_edges] # type: ignore
    edge_j: wp.array1d(dtype=wp.int64)   # shape [total_edges] # type: ignore
):
    xi = queryPositions[i]
    hi = querySupports[i]

    counter = wp.int32(0)
    # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j  = wp.int32(neighborList[jj])

        xj = referencePositions[j]
        hj = referenceSupports[j]
        
        x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        hij = computePairwiseSupport(hi, hj, mode_uint)
        
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        edge_index = newNeighborOffset + counter
        if r_ij <= hij:
            edge_i[edge_index] = wp.int64(i)
            edge_j[edge_index] = wp.int64(j)
            counter += 1

    return counter


@wp.kernel
def updateNeighborsVerlet(
    queryPositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), referencePositions: wp.array(dtype=vector(dtype=scalar_t, length = Any)), # type: ignore
    querySupports: wp.array(dtype=scalar_t), referenceSupports: wp.array(dtype=scalar_t), # type: ignore

    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore

    mode_uint: wp.uint32, 
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore

    # Output
    newNeighborListRowOffsets: wp.array(dtype = wp.int32), newNeighborCounter: wp.array(dtype = wp.int32), # type: ignore  

    edge_i: wp.array1d(dtype=wp.int64),  # shape [total_edges] # type: ignore
    edge_j: wp.array1d(dtype=wp.int64)   # shape [total_edges] # type: ignore  
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
    

# This function filters the Verlet list to produce the actual neighbor list for the current positions and supports, which may have changed since the Verlet list was built. 
def filterVerletList_(
        queryPositions: torch.Tensor, referencePositions: torch.Tensor, 
        querySupports: torch.Tensor, referenceSupports: torch.Tensor,
        domain: DomainDescription, adjacency: AdjacencyList, 
        mode : str = 'symmetric',
):                   
    
    edge_count = torch.zeros_like(adjacency.numNeighbors)
    wp.launch(
        countNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.j), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        castTorchToWarp(edge_count)]

    )
    warpDevice = castTorchToWarp(queryPositions).device
    # Convert counts to host (only the counts, not the main data)
    edge_count_t = edge_count
    total_edges = torch.sum(edge_count_t).cpu().item()

    N = queryPositions.shape[0]
    # Compute cumulative offsets
    edge_offsets = torch.zeros(N, dtype=torch.int32, device = queryPositions.device)
    edge_offsets[1:] = torch.cumsum(edge_count_t[:-1], dim=0)
    edge_offsets_warp = wp.from_torch(edge_offsets)

    # Allocate output arrays on GPU
    edge_i = wp.zeros(total_edges, dtype=wp.int64, device=warpDevice)
    edge_j = wp.zeros(total_edges, dtype=wp.int64, device=warpDevice)

    wp.launch(
        updateNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.j), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        edge_offsets_warp, castTorchToWarp(edge_count),

        edge_i, edge_j
        ]
    )
    i_torch = wp.to_torch(edge_i)
    j_torch = wp.to_torch(edge_j)

    return AdjacencyList(
        i = i_torch, j = j_torch,
        numNeighbors=edge_count_t, edgeOffsets=edge_offsets,
        numRows=queryPositions.shape[0], numCols=referencePositions.shape[0],
        queryPositions = queryPositions, referencePositions = referencePositions,
        querySupports = querySupports, referenceSupports = referenceSupports
    )

def filterVerletList(
        queryParticles: ParticleState, 
        domain: DomainDescription, 
        adjacency: AdjacencyList, 
        supportMode: SupportScheme = SupportScheme.Gather,
        referenceParticles: Optional[ParticleState] = None, 
):
    return filterVerletList_(
        queryParticles.positions, referenceParticles.positions if referenceParticles is not None else queryParticles.positions, 
        queryParticles.supports, referenceParticles.supports if referenceParticles is not None else queryParticles.supports,
        domain, adjacency, supportMode
    )