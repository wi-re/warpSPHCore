
from ..dataTypes import CompactHashMap

from .crk_density import _computeCRKDensity_stateBackend
from .crk_terms import computeCRKTermsWarp
from .crk_volume import _computeCRKVolume_stateBackend
from .crk_moments import _computeCRKMoments_stateBackend
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


def computeCRKFactors(
  queryParticles: ParticleState,
  domain: DomainDescription,
  kernel: KernelFunctions,
  operationMode: OperationDirection = OperationDirection.AllToAll,
  adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
  referenceState: Optional[ParticleState] = None,
):
    """Solves for the CRK correction terms (A, B, gradA, gradB) for every query
    particle, along with the apparent-volume estimate and a CRK-corrected consistency
    density used as diagnostics. Both traversal modes are supported: an explicit
    AdjacencyList/AdjacencyListWarp neighbor list, a CompactHashMap, or adjacency=None
    (in which case each underlying kernel builds its own CompactHashMap on the fly, see
    extractStateInfo) -- there is no longer a neighbor-list-only restriction here, since
    computeCRKMoments/_computeCRKVolume_stateBackend/_computeCRKDensity_stateBackend are
    all dual-path (adjacency + grid) kernels.
    """
    volumeProperties = OperationProperties(
        kernel=kernel,
        supportMode=SupportScheme.Gather,
        operationMode=operationMode,
    )
    apparentArea = _computeCRKVolume_stateBackend(
        queryParticles, volumeProperties, domain,
        adjacency=adjacency, referenceParticles=referenceState,
    )

    momentsProperties = OperationProperties(
        kernel=kernel,
        supportMode=SupportScheme.Scatter,
        operationMode=operationMode,
    )
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors = _computeCRKMoments_stateBackend(
        queryParticles, momentsProperties, domain,
        queryVolumes=apparentArea, referenceVolumes=apparentArea,
        adjacency=adjacency, referenceParticles=referenceState,
    )

    A, B, gradA, gradB = computeCRKTermsWarp(
        m_0, m_1, m_2,
        dm_0dgamma, dm_1dgamma, dm_2dgamma,
        num_nbrs=numNeighbors, supports=queryParticles.supports
    )

    densityProperties = OperationProperties(
        kernel=kernel,
        supportMode=SupportScheme.Scatter,
        operationMode=operationMode,
    )
    crk_density = _computeCRKDensity_stateBackend(
        queryParticles, densityProperties, domain,
        crkState=CRKState(A=A, B=B, gradA=gradA, gradB=gradB),
        queryVolumes=apparentArea, referenceVolumes=apparentArea,
        adjacency=adjacency, referenceParticles=referenceState,
    )

    return apparentArea, crk_density, CRKState(A=A, B=B, gradA=gradA, gradB=gradB)