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
from typing import Optional


def sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    adjacency: AdjacencyListWarp ,
    operation: WarpOperation,
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    supportMode: SupportScheme = SupportScheme.Gather,    
    gradientMode: GradientScheme = GradientScheme.Naive,
    laplacianMode: LaplacianScheme = LaplacianScheme.Default,
    positiveDivergence: bool = False,
    preScatteredQuantities: Optional[torch.Tensor] = None
):
    if operation == WarpOperation.Density:
        # For density estimation, we can just use the interpolation kernel with the input values set to 1, which will give us the standard SPH density summation. This is more efficient than having a separate kernel for density estimation since we can reuse the same interpolation kernel and just change the input values. 
        queryValues = torch.ones_like(queryMasses) if queryValues is None else queryValues
        referenceValues = torch.ones_like(referenceMasses) if referenceValues is None else referenceValues    
    if queryValues is None and referenceValues is None:
        if preScatteredQuantities is None:
            raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the SPH operation.")
    if queryValues is not None and referenceValues is not None and preScatteredQuantities is not None:
        raise ValueError("Pre-scattered quantities should not be provided if queryValues and referenceValues are already provided, as they are redundant in this case.")
    

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
            gradientMode= gradientMode,
            scatteredQuantities= preScatteredQuantities
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