"""Native scalar difference-gradient entry point and future autograd boundary."""
from dataclasses import replace
from ...enumTypes import WarpOperation, GradientScheme
from ...native._inputs import prepare_operator_inputs


def computeGradient(queryParticles, referenceParticles, domain, operationProperties, adjacency,
                    queryValues, referenceValues, queryVolumes=None, referenceVolumes=None,
                    crkState=None, gradHState=None, renormalizationState=None):
    from ...native._build import build
    properties = replace(operationProperties, operation=WarpOperation.Gradient)
    if properties.gradientMode != GradientScheme.Difference:
        raise NotImplementedError('Native pilot supports Difference gradients only')
    if queryValues is None or referenceValues is None or referenceParticles.densities is None:
        raise ValueError('Difference gradients require query/reference values and reference densities')
    inputs = prepare_operator_inputs(
        queryParticles, referenceParticles, domain, properties, adjacency,
        fields=(queryValues, referenceValues, referenceParticles.densities),
        corrections=(queryVolumes, referenceVolumes, crkState, gradHState, renormalizationState))
    return build().gradient_forward(inputs, properties.supportMode.value, properties.operationMode.value)
