"""Raw CRK moments and spatial derivatives in the existing tensor layout."""
from .nt_preparation import prepare


def computeCRKMoments(queryParticles, operationProperties, domain, queryVolumes,
                      referenceVolumes, adjacency=None, referenceParticles=None):
    # queryVolumes remains part of the shared module ABI. These directed
    # moment equations consume only the reference population's volumes.
    inputs, options = prepare(queryParticles, referenceParticles, domain,
                              operationProperties, adjacency, referenceVolumes)
    if referenceVolumes is None:
        raise ValueError('CRK moments require referenceVolumes')
    from ...native._build import build
    return tuple(build().crk_moments_forward(inputs, referenceVolumes, *options))
