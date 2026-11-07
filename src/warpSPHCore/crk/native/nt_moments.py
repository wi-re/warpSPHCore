"""Raw CRK moments and spatial derivatives in the existing tensor layout."""
from dataclasses import replace
from .nt_preparation import prepare


def computeCRKMoments(queryParticles, operationProperties, domain, queryVolumes,
                      referenceVolumes, adjacency=None, referenceParticles=None):
    # The reference-volume role takes priority, then the shared query-volume
    # fallback. Neither role present means zero weights, with counts retained.
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from ...native._row_dispatch import preparation
        reference=queryParticles if referenceParticles is None else referenceParticles
        return preparation(9,computeCRKMoments,queryParticles,reference,domain,operationProperties,
                           adjacency,queryVolumes,referenceVolumes)
    from ...native._options import normalization_factor
    factor = normalization_factor(operationProperties, domain.dim)
    properties = replace(operationProperties, calibrateNormalization=False, positiveDivergence=False)
    reference = queryParticles if referenceParticles is None else referenceParticles
    volumes = referenceVolumes if referenceVolumes is not None else queryVolumes
    if volumes is None:
        # getVolume_j returns zero when neither correction-volume role exists.
        # Neighbor counts still include every allowed CSR edge.
        volumes = queryParticles.positions.new_zeros((len(reference.positions),))
    if type(adjacency).__name__ in ('CompactHashMap', 'SearchIndex'):
        adjacency = None
    inputs, options, topology = prepare(queryParticles, referenceParticles, domain,
                              properties, adjacency, volumes)
    from .nt_moments_backward import apply
    ignored = (queryVolumes, queryParticles.masses, queryParticles.densities, reference.densities)
    outputs = apply(inputs, options, topology, volumes, ignored)
    return (*[value * factor for value in outputs[:6]], outputs[-1]) if factor != 1. else outputs
