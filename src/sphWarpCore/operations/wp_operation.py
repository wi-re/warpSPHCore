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
from .wp_density import computeSPHDensity_warpBackend

from ..enumTypes import *
from typing import Optional
from torch.profiler import profile, record_function, ProfilerActivity


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
    operationMode: OperationDirection = OperationDirection.AllToAll,
    positiveDivergence: bool = False,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    renormalizationMatrices: Optional[torch.Tensor] = None,
    queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    queryKinds: Optional[torch.Tensor] = None, referenceKinds: Optional[torch.Tensor] = None
):
    if operationMode != OperationDirection.AllToAll and (queryKinds is None or referenceKinds is None):
        raise ValueError("Query and reference kinds must be provided for non AllToAll operation modes. Operation mode: {}, queryKinds is None: {}, referenceKinds is None: {}".format(operationMode, queryKinds is None, referenceKinds is None))
    if operationMode == OperationDirection.AllToAll and (queryKinds is None):
        queryKinds = adjacency.numNeighbors # This is safe to do as the kinds are never checked in this case
    if operationMode == OperationDirection.AllToAll and (referenceKinds is None):
        referenceKinds = adjacency.numNeighbors # This is safe to do as the kinds are never checked in this case


    with record_function(f"warpSPH - Operation"):
        if operation == WarpOperation.Density:
            return computeSPHDensity_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency=adjacency, 
                kernel = kernel, mode = supportMode,
                operationMode = operationMode, 
            )  
        
        if queryValues is None and referenceValues is None:
            if preScatteredQuantities is None:
                raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the SPH operation.")
        if queryValues is not None and referenceValues is not None and preScatteredQuantities is not None:
            raise ValueError("Pre-scattered quantities should not be provided if queryValues and referenceValues are already provided, as they are redundant in this case.")
        if preScatteredQuantities is not None and gradientMode != GradientScheme.Naive:
            raise ValueError("Pre-scattered quantities only support the naive scheme as they are meant to provide pre-computed neighbor-level quantities for custom kernels that may not be compatible with the standard gradient schemes. If using pre-scattered quantities, the gradientMode must be set to Naive.")
    
        if operation == WarpOperation.Interpolate:
            return computeSPHInterpolant_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency=adjacency, 
                kernel = kernel, mode = supportMode,
                operationMode = operationMode, 
                scatteredQuantities= preScatteredQuantities
            )
        elif operation == WarpOperation.Gradient:
            return computeSPHGradient_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency= adjacency, 
                kernel = kernel, mode = supportMode, 
                operationMode = operationMode, 
                gradientMode= gradientMode,
                scatteredQuantities= preScatteredQuantities,
                renormalizationMatrices= renormalizationMatrices,
                queryOmegas= queryOmegas, referenceOmegas= referenceOmegas
            )
        elif operation == WarpOperation.Divergence:
            return computeSPHDivergence_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency= adjacency, 
                kernel = kernel, mode = supportMode, 
                operationMode = operationMode, 
                gradientMode= gradientMode,
                scatteredQuantities= preScatteredQuantities
            )
        elif operation == WarpOperation.Curl:
            return computeSPHCurl_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency= adjacency, 
                kernel = kernel, mode = supportMode, 
                operationMode = operationMode, 
                gradientMode= gradientMode,
                scatteredQuantities= preScatteredQuantities
            )
        elif operation == WarpOperation.Laplacian:
            return computeSPHLaplacian_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                queryKinds= queryKinds, referenceKinds= referenceKinds,
                domain = domain, adjacency= adjacency, 
                kernel = kernel, mode = supportMode, 
                operationMode = operationMode, 
                gradientMode= gradientMode, 
                laplacianMode= laplacianMode, positiveDivergence=positiveDivergence,
                scatteredQuantities= preScatteredQuantities
                
            )
        else:
            raise ValueError("Unsupported SPH operation: {}".format(operation))