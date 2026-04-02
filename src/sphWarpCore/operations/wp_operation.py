import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *


from .wp_curl import computeSPHCurl_warpBackend
from .wp_laplacian import computeSPHLaplacian_warpBackend
from .wp_gradient import computeSPHGradient_warpBackend
from .wp_divergence import computeSPHDivergence_warpBackend
from .wp_interpolate import computeSPHInterpolant_warpBackend

from ..enumTypes import *



def sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    domain: DomainDescription,
    adjacency: AdjacencyListWarp ,
    operation: WarpOperation,
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    supportMode: SupportScheme = SupportScheme.Gather,    
    gradientMode: GradientScheme = GradientScheme.Naive,
    laplacianMode: LaplacianScheme = LaplacianScheme.Default,
    positiveDivergence: bool = False
):
    if operation == WarpOperation.Interpolate:
        return computeSPHInterpolant_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, adjacency=adjacency, 
            kernel = kernel, mode = supportMode
        )
    elif operation == WarpOperation.Gradient:
        return computeSPHGradient_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, adjacency= adjacency, 
            kernel = kernel, mode = supportMode, 
            gradientMode= gradientMode
        )
    elif operation == WarpOperation.Divergence:
        return computeSPHDivergence_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, adjacency= adjacency, 
            kernel = kernel, mode = supportMode, 
            gradientMode= gradientMode            
        )
    elif operation == WarpOperation.Curl:
        return computeSPHCurl_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, adjacency= adjacency, 
            kernel = kernel, mode = supportMode, 
            gradientMode= gradientMode            
        )
    elif operation == WarpOperation.Laplacian:
        return computeSPHLaplacian_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, adjacency= adjacency, 
            kernel = kernel, mode = supportMode, 
            gradientMode= gradientMode, 
            laplacianMode= laplacianMode, positiveDivergence=positiveDivergence
            
        )
    elif operation == WarpOperation.Density:
        return computeSPHInterpolant_warpBackend(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            torch.ones_like(queryMasses), torch.ones_like(referenceMasses),
            torch.ones_like(queryMasses), torch.ones_like(referenceMasses),
            domain = domain, adjacency= adjacency, 
            kernel = kernel, mode = supportMode, 
        )