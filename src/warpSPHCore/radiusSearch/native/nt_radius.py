"""Native radius-search entry point; the primal row code lives in nt_radius_forward.*."""
from ..._models import ParticleState, DomainDescription, AdjacencyList
from ...enumTypes import SupportScheme


def radiusSearch(queryParticles: ParticleState, domain: DomainDescription,
                 referenceParticles=None, mode=SupportScheme.Gather):
    import torch
    if queryParticles.positions.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError('Native radius search allocates variable-sized lists; supply a fixed adjacency during graph capture')
    from ...native._build import build
    from ...native._inputs import prepare_geometry
    reference = referenceParticles if referenceParticles is not None else queryParticles
    args = prepare_geometry(queryParticles, reference, domain, allow_reverse=True)
    if not isinstance(mode, SupportScheme):
        raise TypeError('mode must be a SupportScheme')
    neighbors, counts, offsets = build().radius(*args, mode.value)
    return AdjacencyList(None, neighbors, counts, offsets,
                         queryParticles.positions.shape[0], reference.positions.shape[0],
                         numEdges=neighbors.numel())
