from .wp_divergence_vector import computeSPHDivergenceVector_warpBackend
from ..wp_divergence import computeSPHDivergence_warpBackend

import warp as wp
import torch

from typing import Any
from ...utils.wp_autograd import *


from ...radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ...mathutil.wp_math import *
from ...kernels.wp_kernel import *


def computeSPHDivergence_warpBackend_manual(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    adjacency: AdjacencyListWarp,
    consistentDivergence: bool = False, dotMode: bool = False  
):
    if len(queryValues.shape) == 1:
        raise NotImplementedError("Scalar divergence is not meaningfully defined.")
    elif len(queryValues.shape) == 2 and queryValues.shape[1] == queryPositions.shape[1]:
        return computeSPHDivergenceVector_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain, mode, kernel, gradientMode, adjacency, consistentDivergence
        )
    # elif len(queryValues.shape) > 2 and queryValues.shape[1] == queryPositions.shape[1] and queryValues.shape[2] == queryPositions.shape[1]:
    else:
        return computeSPHDivergence_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain, mode, kernel, gradientMode, adjacency, consistentDivergence, dotMode
        ).view(queryValues.shape[0], *queryValues.shape[1:-1]) # reshape back to original shape
    # else:
    #     raise ValueError("Unsupported value shape for SPH gradient computation. Expected (N,), (N, D), or (N, D, D) where N is the number of particles and D is the spatial dimension.")