"""CRK consistency density, distinct from the ordinary SPH density operator."""
from dataclasses import replace
from ..._models import CRKState
from ...enumTypes import SupportScheme
from .nt_preparation import prepare


def computeCRKDensity(queryParticles, operationProperties, domain, crkState,
                      queryVolumes, referenceVolumes, adjacency=None,
                      referenceParticles=None):
    import torch
    states = crkState if isinstance(crkState,tuple) and len(crkState)==2 else (crkState,)
    state = states[0]
    if state is not None and not isinstance(state, CRKState):
        raise TypeError('crkState must be a CRKState')
    if state is not None and (state.A is None or state.B is None):
        raise ValueError('CRK density requires A and B when a CRK state is supplied')
    reference = queryParticles if referenceParticles is None else referenceParticles
    operationProperties = replace(operationProperties, supportMode=SupportScheme.Scatter,
                                  calibrateNormalization=False,positiveDivergence=False)
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from ...native._row_dispatch import preparation
        return preparation(10,computeCRKDensity,queryParticles,reference,domain,operationProperties,
                           adjacency,queryVolumes,referenceVolumes,crkState)
    if type(adjacency).__name__ in ('CompactHashMap','SearchIndex'): adjacency=None
    volumes = referenceVolumes if referenceVolumes is not None else queryVolumes
    if volumes is None: volumes=queryParticles.positions.new_zeros((len(reference.positions),))
    A = queryParticles.positions.new_zeros((len(queryParticles.positions),)) if state is None else state.A
    B = torch.zeros_like(queryParticles.positions) if state is None else state.B
    inputs, options, topology = prepare(queryParticles, reference, domain, operationProperties,
                              adjacency, volumes, reference.masses, A, B)
    from .nt_density_backward import apply
    ignored = (queryVolumes,queryParticles.masses,queryParticles.densities,reference.densities,
               *(getattr(s,name,None) for s in states for name in ('gradA','gradB')),
               *(getattr(s,name,None) for s in states[1:] for name in ('A','B')))
    return apply(inputs,options,topology,volumes,A,B,ignored)
