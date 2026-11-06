"""Native gradient boundary, including packed fields and supplied search grids."""
from dataclasses import replace
from ...enumTypes import WarpOperation
from ...native._options import normalization_factor


def computeGradient(queryParticles,referenceParticles,domain,operationProperties,adjacency,
                    queryValues,referenceValues,queryVolumes=None,referenceVolumes=None,
                    crkState=None,gradHState=None,renormalizationState=None):
    from .nt_corrected import compute
    factor=normalization_factor(operationProperties,domain.dim)
    properties=replace(operationProperties,operation=WarpOperation.Gradient,calibrateNormalization=False)
    result=compute(queryParticles,referenceParticles,domain,properties,adjacency,
        queryValues,referenceValues,queryVolumes,referenceVolumes,crkState,gradHState,renormalizationState)
    return result*factor if factor!=1 else result
