"""Apparent volume: reciprocal kernel sum, with zero for empty/masked rows."""
from .nt_preparation import prepare


def computeCRKVolume(queryParticles, operationProperties, domain, adjacency=None,
                     referenceParticles=None):
    inputs, options, topology = prepare(queryParticles, referenceParticles, domain,
                              operationProperties, adjacency)
    from ...native._differential import apply_preparation
    return apply_preparation(8, inputs, options, topology)
