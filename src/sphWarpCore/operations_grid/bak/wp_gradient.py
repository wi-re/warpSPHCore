from .wp_gradient_scalar import computeSPHGradientScalar_warpBackend
from .wp_gradient_matrix import computeSPHGradientMatrix_warpBackend
from .wp_gradient_vector import computeSPHGradientVector_warpBackend

import warp as wp
import torch

from typing import Any
from ...utils.wp_autograd import *
from ...radiusSearch.radius_util import convertModeToUint

from ...radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ...mathutil.wp_math import *
from ...kernels.wp_kernel import *


def computeSPHGradient_warpBackend_manual(
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
):
    if len(queryValues.shape) == 1:
        return computeSPHGradientScalar_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain, mode, kernel, gradientMode, adjacency
        )
    elif len(queryValues.shape) == 2 and queryValues.shape[1] == queryPositions.shape[1]:
        return computeSPHGradientVector_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain, mode, kernel, gradientMode, adjacency
        )
    elif len(queryValues.shape) == 3 and queryValues.shape[1] == queryPositions.shape[1] and queryValues.shape[2] == queryPositions.shape[1]:
        return computeSPHGradientMatrix_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain, mode, kernel, gradientMode, adjacency
        )
    else:
        raise ValueError("Unsupported value shape for SPH gradient computation. Expected (N,), (N, D), or (N, D, D) where N is the number of particles and D is the spatial dimension.")