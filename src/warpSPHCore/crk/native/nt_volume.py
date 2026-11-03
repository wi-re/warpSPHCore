"""Apparent volume: reciprocal kernel sum, with zero for empty/masked rows."""
from .nt_preparation import prepare


def computeCRKVolume(queryParticles, operationProperties, domain, adjacency=None,
                     referenceParticles=None):
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from ...native._row_dispatch import preparation
        reference=queryParticles if referenceParticles is None else referenceParticles
        return preparation(8,computeCRKVolume,queryParticles,reference,domain,operationProperties,adjacency)
    inputs, options, topology = prepare(queryParticles, referenceParticles, domain,
                              operationProperties, adjacency)
    from ...native._differential import apply_preparation
    return apply_preparation(8, inputs, options, topology)
