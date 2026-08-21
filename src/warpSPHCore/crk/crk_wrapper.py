
from ..dataTypes import CompactHashMap

from .crk_density import _computeCRKDensity_stateBackend
from .crk_terms import computeCRKTermsWarp
from .crk_volume import _computeCRKVolume_stateBackend
from .crk_volume_jvp import computeCRKVolumeGeometryJVP
from .crk_moments import _computeCRKMoments_stateBackend
from .crk_moments_jvp import computeCRKMomentsGeometryJVP
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Union
import torch
from warpSPHCore.autograd import *

from ..dataTypes import *
from warpSPHCore.math import *
from warpSPHCore.kernels import *
from ..util import *
from torch.profiler import profile, record_function, ProfilerActivity
from warpSPHCore.enumTypes import *
from warpSPHCore.autograd.arg_check import *
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


def computeCRKFactorsJVP(
  queryParticles: ParticleState,
  domain: DomainDescription,
  kernel: KernelFunctions,
  queryTangentState: 'ParticleTangentState',
  operationMode: OperationDirection = OperationDirection.AllToAll,
  adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
  referenceState: Optional[ParticleState] = None,
  referenceTangentState: Optional['ParticleTangentState'] = None,
):
    """JVP counterpart to `computeCRKFactors`
    (`warpier_tier2_correction_jvp_plan.md` phase (c), Stages 1-3): given
    `queryTangentState` (position/support tangent), chains Stage 1's
    apparent-volume tangent (`computeCRKVolumeGeometryJVP`) into Stage 2's
    moment tangents (`computeCRKMomentsGeometryJVP`) into Stage 3's
    `computeCRKTermsWarp`, whose own JVP is obtained via
    `torch.autograd.functional.jvp` directly -- valid here (unlike every
    other Warp-kernel-backed piece of this pipeline) because
    `computeCRKTermsWarp` has no Warp call anywhere in it, so double-backward
    through it is exact (see `scripts/spike_forward_mode_tier2_crk.py`'s
    module docstring for the full justification, and its `crk_terms_jvp` for
    the pattern this mirrors). `create_graph=True` is required here (unlike
    the spike, which never differentiates the assembled JVP a second time):
    production callers need `dA/dB/dgradA/dgradB` to stay differentiable back
    through `m_0..dm_2dgamma` to `positions`/`supports`, for
    `torch.autograd.gradcheck` on the whole pipeline to see a nonzero
    gradient through this hop (`warpier_tier2_correction_jvp_plan.md` phase
    (a2)'s "hard requirement", the same failure class documented in
    `docs/lessons_learned.md` for any tensor that bypasses this kind of
    bridge).

    Returns `(apparentArea, dApparentArea, crkState, crkTangentState)`. Does
    not compute `crk_density`/its tangent -- that diagnostic has no consumer
    anywhere in the JVP path.
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
    dApparentArea = computeCRKVolumeGeometryJVP(
        queryParticles, domain, kernel, adjacency, apparentArea,
        queryTangentState=queryTangentState,
        referenceParticles=referenceState, referenceTangentState=referenceTangentState,
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
    dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma = computeCRKMomentsGeometryJVP(
        queryParticles, domain, kernel, adjacency,
        referenceVolumes=apparentArea, tangentReferenceVolumes=dApparentArea,
        queryTangentState=queryTangentState,
        referenceParticles=referenceState, referenceTangentState=referenceTangentState,
    )

    primals = (m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    tangents = (dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma)

    def f(m0, m1, m2, dm0g, dm1g, dm2g):
        return computeCRKTermsWarp(m0, m1, m2, dm0g, dm1g, dm2g, numNeighbors, queryParticles.supports)

    (A, B, gradA, gradB), (dA, dB, dgradA, dgradB) = torch.autograd.functional.jvp(
        f, primals, tangents, create_graph=True,
    )

    return (
        apparentArea, dApparentArea,
        CRKState(A=A, B=B, gradA=gradA, gradB=gradB),
        CRKTangentState(A=dA, B=dB, gradA=dgradA, gradB=dgradB),
    )