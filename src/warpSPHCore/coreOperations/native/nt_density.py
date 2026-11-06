"""Native density entry point: validation, discrete topology and product dispatch.

The primal calculation lives in csrc_density/nt_density_forward.*; Torch
backward orchestration calls the separately qualified native adjoint.
"""
from dataclasses import replace
from ...enumTypes import WarpOperation
from ...native._inputs import prepare_operator_inputs


def computeDensity(queryParticles, referenceParticles, domain, operationProperties, adjacency=None):
    from ...native._options import normalization_factor
    factor = normalization_factor(operationProperties, domain.dim)
    properties = replace(operationProperties, operation=WarpOperation.Density,
                         calibrateNormalization=False)
    # Grid membership is discrete. Rebuild native CSR in the caller's original
    # population order instead of differentiating reference-grid sorting.
    if adjacency is not None and type(adjacency).__name__ in ('CompactHashMap', 'SearchIndex'):
        adjacency = None
    inputs = prepare_operator_inputs(queryParticles, referenceParticles, domain, properties, adjacency)
    from .nt_density_backward import apply
    inputs[15] = (referenceParticles.densities if referenceParticles.densities is not None
                  else queryParticles.positions.new_empty(0))
    result = apply(inputs, properties, queryParticles.masses)
    return result * factor if factor != 1. else result
