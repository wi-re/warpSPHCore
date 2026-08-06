
from ..dataTypes import CompactHashMap

from .crk_density import computeCRKDensityWarp
from .crk_terms import computeCRKTermsWarp
from .crk_volume import computeCRKVolumeWarp
from .crk_moments import computeCRKMomentsWarp
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Union
import torch
from sphWarpCore.autograd import *

from ..dataTypes import *
from sphWarpCore.math import *
from sphWarpCore.kernels import *
from ..util import *
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.autograd.arg_check import *
from typing import Optional



def computeCRKFactors_(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,

    domain: DomainDescription,
    supportMode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,
    queryKinds: Optional[torch.Tensor] = None, referenceKinds: Optional[torch.Tensor] = None,    
):
    # if supportMode != SupportScheme.Gather:
        # raise NotImplementedError("Currently only Gather support mode is implemented for CRK factors computation.")
        
    apparentArea = computeCRKVolumeWarp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        domain = domain, adjacency = adjacency, 
        operationMode = operationMode,
        kernel = kernel, supportMode = SupportScheme.Gather,
        queryKinds = queryKinds, referenceKinds = referenceKinds
    )

        
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = computeCRKMomentsWarp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        useVolume=True, queryVolumes = apparentArea, referenceVolumes = apparentArea,
        domain = domain, adjacency = adjacency, 
        operationMode = operationMode,
        kernel =  kernel, supportMode = supportMode,
        queryKinds = queryKinds, referenceKinds = referenceKinds
    )

    A, B, gradA, gradB = computeCRKTermsWarp(
        m_0, m_1, m_2,
        dm_0dgamma, dm_1dgamma, dm_2dgamma,
        num_nbrs = adjacency.numNeighbors, supports = querySupports
    )

    crk_density = computeCRKDensityWarp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        useVolume=True, queryVolumes = apparentArea, referenceVolumes = apparentArea,
        domain = domain, adjacency = adjacency, 
        operationMode = operationMode,
        kernel = kernel, supportMode = SupportScheme.Scatter,
        useCRK=True, crk_A = A, crk_B = B, crk_gradA = gradA, crk_gradB = gradB,
        queryKinds = queryKinds, referenceKinds = referenceKinds
    )

    return apparentArea, crk_density, A, B, gradA, gradB

from ..dataTypes import *

def computeCRKFactors(
  queryParticles: ParticleState,
  domain: DomainDescription,
  kernel: KernelFunctions,
  operationMode: OperationDirection = OperationDirection.AllToAll,
  adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,   
  referenceState: Optional[ParticleState] = None,   
):
    if referenceState is None:
        referenceState = queryParticles

    if adjacency is None or isinstance(adjacency, CompactHashMap):
        raise NotImplementedError("Adjacency list must be provided for CRK factors computation. Building a compact hash map and using it as adjacency is not currently supported for this operation.")
    
    apparentArea, crk_density, A, B, gradA, gradB = computeCRKFactors_(
        queryParticles.positions, referenceState.positions,
        queryParticles.supports, referenceState.supports,
        queryParticles.masses, referenceState.masses,

        domain = domain, 
        adjacency = adjacency,
        
        operationMode = operationMode, kernel = kernel, 
        supportMode = SupportScheme.Scatter, # Currently only Gather support mode is implemented for CRK factors computation.
        queryKinds = queryParticles.kinds, referenceKinds = referenceState.kinds
    )

    return apparentArea, crk_density, CRKState(A=A, B=B, gradA=gradA, gradB=gradB)