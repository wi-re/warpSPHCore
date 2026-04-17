
from .crk_density import computeCRKDensityWarp
from .crk_terms import computeCRKTermsWarp
from .crk_volume import computeCRKVolumeWarp
from .crk_moments import computeCRKMomentsWarp
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.utils.wp_autograd import *
from sphWarpCore.radiusSearch.radius_util import convertModeToUint

from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.mathutil.wp_math import *
from sphWarpCore.kernels.wp_kernel import *
from sphWarpCore.utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.utils.arg_check import *
from typing import Optional



def computeCRKFactors(
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
        
    apparentArea = computeCRKVolumeWarp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        domain = domain, adjacency = adjacency, 
        operationMode = OperationDirection.AllToAll,
        kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather,
    )

        
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = computeCRKMomentsWarp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        useVolume=True, queryVolumes = apparentArea, referenceVolumes = apparentArea,
        domain = domain, adjacency = adjacency, 
        operationMode = OperationDirection.AllToAll,
        kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather,
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
        operationMode = OperationDirection.AllToAll,
        kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather,
        useCRK=True, crk_A = A, crk_B = B, crk_gradA = gradA, crk_gradB = gradB
    )

    return apparentArea, crk_density, A, B, gradA, gradB