"""Apparent volume: reciprocal kernel sum, with zero for empty/masked rows."""
from .nt_preparation import prepare


def computeCRKVolume(queryParticles, operationProperties, domain, adjacency=None,
                     referenceParticles=None):
    inputs, options = prepare(queryParticles, referenceParticles, domain,
                              operationProperties, adjacency)
    from ...native._build import build
    return build().crk_volume_forward(inputs, *options)
