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
import numpy as np    
from ...profiling import record_function
from ...dataTypes import *

# For some operations we want to be able to run the operations directly on the hash map, e.g., to do a particle to grid transfer. In this case we only access the neighborhood once per query set. In this case, it is not worth it to build the full neighbor list as the overhead of building the neighbor list is larger than the cost of just doing the search.
# For this purpose we want to be able to wrap the hash map information into a datastructure that allows us to do the search directly on the hash map without building the full neighbor list. This is what the CompactHashMap class is for. It contains all the information about the hash map and the cell table, as well as the sorted positions and supports, and allows us to do the search directly on this information without building the full neighbor list.


from .buildHashmap import buildCompactHashMap
from .search import radiusSearchOnCompactHashMap
    

def radiusSearchCompactHashMap_(
    queryPositions: torch.Tensor,
    referencePositions: torch.Tensor,
    querySupports: torch.Tensor,
    referenceSupports: torch.Tensor,
    periodicity: torch.Tensor,
    domainDescription: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLength: int = 4096,
    returnCompactHashMap: bool = False
):
    with record_function("warpNeighborSearch - radiusSearchCompactHashMap"):
        datastructure = buildCompactHashMap(
            queryPositions,
            referencePositions,
            querySupports,
            referenceSupports,
            periodicity,
            domainDescription,
            mode,
            hashMapLength
        ) 
        adjacencyCH = radiusSearchOnCompactHashMap(
            datastructure,
            queryPositions,
            referencePositions,
            querySupports,
            referenceSupports,
            periodicity,
            domainDescription,
            mode,
            hashMapLength
        )
        
        return adjacencyCH if not returnCompactHashMap else (adjacencyCH, datastructure)

    

def radiusSearchCompactHashMap(
    queryParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLengthMode: HashMapLengthMode = HashMapLengthMode.NextPrime,
    fixedHashMapLength: int = 4096,
    returnCompactHashMap: bool = False,
    referenceParticles: Optional[ParticleState] = None
):
    referenceParticles = queryParticles if referenceParticles is None else referenceParticles
    queryPositions = queryParticles.positions
    referencePositions = referenceParticles.positions
    querySupports = queryParticles.supports if queryParticles.supports is not None else torch.zeros(queryParticles.positions.shape[0], device=queryParticles.positions.device)
    referenceSupports = referenceParticles.supports if referenceParticles.supports is not None else torch.zeros(referenceParticles.positions.shape[0], device=referenceParticles.positions.device)
    periodicity = torch.tensor(domain.periodic if domain.periodic is not None else [False] * queryParticles.positions.shape[1], device=queryParticles.positions.device) if not isinstance(domain.periodic, torch.Tensor) else domain.periodic.clone().to(queryParticles.positions.device)

    if hashMapLengthMode == HashMapLengthMode.Fixed:
        hashMapLength = fixedHashMapLength
    elif hashMapLengthMode == HashMapLengthMode.NumberOfParticles:
        hashMapLength = queryPositions.shape[0]
        if hashMapLength % 2 == 0:
            hashMapLength += 1  # Ensure it's odd to reduce collisions
    elif hashMapLengthMode == HashMapLengthMode.NextPrime:
        hashMapLength = getNextPrime(queryPositions.shape[0])
    else:
        raise ValueError("Invalid hash map length mode")

    return radiusSearchCompactHashMap_(
        queryPositions,
        referencePositions,
        querySupports,
        referenceSupports,
        periodicity,
        domain,
        mode,
        hashMapLength,
        returnCompactHashMap
    )