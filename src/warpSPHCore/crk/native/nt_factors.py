"""Warp-free CRK preparation with explicit query/reference population roles."""
from dataclasses import dataclass, replace
import torch
from ..._models import CRKState, OperationProperties
from ...enumTypes import OperationDirection, SupportScheme
from ...native._inputs import prepare_adjacency
from ...native._topology import CSRTopology
from ..crk_terms import computeCRKTerms
from .nt_volume import computeCRKVolume
from .nt_moments import computeCRKMoments
from .nt_density import computeCRKDensity


@dataclass(frozen=True)
class CRKPreparation:
    """Outputs in the caller's particle order, ready for corrected pair operators.

    Volumes and coefficients are state-dependent. Recompute after positions,
    supports, kinds or interaction modes change, even if CSR topology is reused.
    All arrays retain their input population order; no hidden permutation or
    topology cache is owned by this result.
    """
    queryVolumes: torch.Tensor
    referenceVolumes: torch.Tensor
    consistencyDensity: torch.Tensor
    crkState: CRKState


def prepareCRK(queryParticles, domain, kernel,
               operationMode=OperationDirection.AllToAll, adjacency=None,
               referenceState=None, *, referenceVolumeMode=None,
               referenceAdjacency=None):
    """Build volumes, moments, factors and the consistency diagnostic.

    ``adjacency`` joins queries to references. For distinct populations,
    ``referenceAdjacency`` joins references to themselves; it must never be
    replaced by the query-to-reference list. Both may be omitted to use native
    compact-hash search. Supplied CSR must cover every pair used by the Gather
    volume and Scatter moment/density passes (a union/SuperSymmetric search is
    suitable for unequal supports).
    """
    reference = queryParticles if referenceState is None else referenceState
    volume_properties = OperationProperties(kernel, supportMode=SupportScheme.Gather,
                                             operationMode=operationMode)
    moment_properties = replace(volume_properties, supportMode=SupportScheme.Scatter)
    # Reuse a union graph for all query/reference passes. Kernel evaluation
    # still applies each pass's own support rule, as on the Warp path.
    with torch.no_grad():
        graph = prepare_adjacency(queryParticles, reference, domain,
            replace(volume_properties, supportMode=SupportScheme.SuperSymmetric), adjacency)
        graph = adjacency if isinstance(adjacency, CSRTopology) else CSRTopology(graph)
    query_volumes = computeCRKVolume(queryParticles, volume_properties, domain,
                                     graph, reference)
    reference_volumes = query_volumes
    if reference is not queryParticles or referenceVolumeMode is not None:
        reference_properties = replace(volume_properties, operationMode=(
            operationMode if referenceVolumeMode is None else referenceVolumeMode))
        reference_volumes = computeCRKVolume(reference, reference_properties, domain,
                                             referenceAdjacency)
    elif referenceAdjacency is not None:
        raise ValueError('referenceAdjacency requires a distinct referenceState or referenceVolumeMode')
    moments = computeCRKMoments(queryParticles, moment_properties, domain,
                                query_volumes, reference_volumes, graph, reference)
    A, B, gradA, gradB = computeCRKTerms(*moments, queryParticles.supports)
    state = CRKState(*(value.contiguous() for value in (A, B, gradA, gradB)))
    density = computeCRKDensity(queryParticles, moment_properties, domain, state,
                                query_volumes, reference_volumes, graph, reference)
    return CRKPreparation(query_volumes, reference_volumes, density, state)


def computeCRKFactors(queryParticles, domain, kernel,
                      operationMode=OperationDirection.AllToAll, adjacency=None,
                      referenceState=None, *, referenceVolumeMode=None,
                      referenceAdjacency=None):
    """Compatibility tuple: query apparent volumes, consistency density, CRKState.

    Use ``prepareCRK`` when the caller also needs distinct reference volumes.
    """
    result = prepareCRK(queryParticles, domain, kernel, operationMode, adjacency,
                        referenceState, referenceVolumeMode=referenceVolumeMode,
                        referenceAdjacency=referenceAdjacency)
    return result.queryVolumes, result.consistencyDensity, result.crkState
