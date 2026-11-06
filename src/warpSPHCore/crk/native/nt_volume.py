"""Apparent volume: reciprocal kernel sum, with zero for empty/masked rows."""
from dataclasses import replace
from .nt_preparation import prepare


def computeCRKVolume(queryParticles, operationProperties, domain, adjacency=None,
                     referenceParticles=None):
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from ...native._row_dispatch import preparation
        reference=queryParticles if referenceParticles is None else referenceParticles
        return preparation(8,computeCRKVolume,queryParticles,reference,domain,operationProperties,adjacency)
    from ...native._options import normalization_factor
    factor=normalization_factor(operationProperties,domain.dim)
    properties=replace(operationProperties,calibrateNormalization=False,positiveDivergence=False)
    if type(adjacency).__name__ in ('CompactHashMap','SearchIndex'):adjacency=None
    reference=queryParticles if referenceParticles is None else referenceParticles
    inputs, options, topology = prepare(queryParticles, reference, domain,properties,adjacency)
    from .nt_volume_backward import apply
    result=apply(inputs,options,topology,(queryParticles.masses,reference.masses,
                                       queryParticles.densities,reference.densities))
    return result/factor if factor!=1. else result
