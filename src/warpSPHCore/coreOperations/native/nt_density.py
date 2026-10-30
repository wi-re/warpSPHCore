"""Native density entry point: validation, dispatch and future autograd policy.

The primal calculation lives in nt_density_forward.*. The native pilot rejects
AD until an explicit backward bridge and numerical adjoint are qualified.
"""
from dataclasses import replace
from ...enumTypes import WarpOperation
from ...native._inputs import prepare_operator_inputs


def computeDensity(queryParticles, referenceParticles, domain, operationProperties, adjacency=None):
    from ...native._build import build
    properties = replace(operationProperties, operation=WarpOperation.Density)
    inputs = prepare_operator_inputs(queryParticles, referenceParticles, domain, properties, adjacency)
    return build().density_forward(inputs, properties.supportMode.value, properties.operationMode.value)
