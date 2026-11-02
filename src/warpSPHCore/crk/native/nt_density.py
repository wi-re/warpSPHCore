"""CRK consistency density, distinct from the ordinary SPH density operator."""
from dataclasses import replace
from ..._models import CRKState
from ...enumTypes import SupportScheme
from .nt_preparation import prepare


def computeCRKDensity(queryParticles, operationProperties, domain, crkState,
                      queryVolumes, referenceVolumes, adjacency=None,
                      referenceParticles=None):
    if not isinstance(crkState, CRKState):
        raise TypeError('crkState must be a CRKState')
    if referenceVolumes is None or crkState.A is None or crkState.B is None:
        raise ValueError('CRK density requires referenceVolumes, A and B')
    reference = queryParticles if referenceParticles is None else referenceParticles
    operationProperties = replace(operationProperties, supportMode=SupportScheme.Scatter)
    inputs, options, topology = prepare(queryParticles, reference, domain, operationProperties,
                              adjacency, referenceVolumes, reference.masses, crkState.A, crkState.B)
    from ...native._differential import apply_preparation
    return apply_preparation(10, inputs, options, topology, referenceVolumes, crkState.A, crkState.B)
