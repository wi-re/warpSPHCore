"""Native density entry point: validation, dispatch and future autograd policy.

The primal calculation lives in csrc_density/nt_density_forward.*; Torch
backward orchestration calls the separately qualified native adjoint.
"""
from dataclasses import replace
from ...enumTypes import WarpOperation
from ...native._inputs import prepare_operator_inputs


def computeDensity(queryParticles, referenceParticles, domain, operationProperties, adjacency=None):
    properties = replace(operationProperties, operation=WarpOperation.Density)
    inputs = prepare_operator_inputs(queryParticles, referenceParticles, domain, properties, adjacency)
    from .nt_density_backward import apply
    return apply(inputs, properties)
