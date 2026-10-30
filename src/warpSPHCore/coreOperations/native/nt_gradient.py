"""Native scalar/vector/matrix gradient entry point and first-order AD boundary."""
from dataclasses import replace
from ...enumTypes import WarpOperation
from ...native._inputs import prepare_operator_inputs


def computeGradient(queryParticles, referenceParticles, domain, operationProperties, adjacency,
                    queryValues, referenceValues, queryVolumes=None, referenceVolumes=None,
                    crkState=None, gradHState=None, renormalizationState=None):
    properties = replace(operationProperties, operation=WarpOperation.Gradient)
    if queryValues is None or referenceValues is None or referenceParticles.densities is None:
        raise ValueError('Gradients require query/reference values and reference densities')
    inputs = prepare_operator_inputs(
        queryParticles, referenceParticles, domain, properties, adjacency,
        fields=(queryValues, referenceValues, referenceParticles.densities),
        corrections=(queryVolumes, referenceVolumes, crkState, gradHState, renormalizationState))
    from .nt_gradient_backward import apply
    return apply(inputs, properties)
