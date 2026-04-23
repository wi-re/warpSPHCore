import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch

from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..radiusSearch.wp_compactHash import CompactHashMap, computeGridSupport, getDomainExtents
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *


# from .wp_curl_grid import computeSPHCurl_warpBackend
# from .wp_laplacian_grid import computeSPHLaplacian_warpBackend
# from .wp_gradient_grid import computeSPHGradient_warpBackend
# from .wp_divergence_grid import computeSPHDivergence_warpBackend
from .wp_interpolate_grid import computeSPHInterpolant_grid_warpBackend
# from .wp_density_grid import computeSPHDensity_warpBackend

from ..enumTypes import *
from typing import Optional
from torch.profiler import profile, record_function, ProfilerActivity

from ..utils.wp_util import getCachedDummyTensor
from sphWarpCore.radiusSearch.verlet import *
from ..radiusSearch.wp_compactHash import buildCompactHashMap

def sphOperation_warp_grid(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    datastructure: CompactHashMap ,
    operation: WarpOperation,
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    supportMode: SupportScheme = SupportScheme.Gather,    
    gradientMode: GradientScheme = GradientScheme.Naive,
    laplacianMode: LaplacianScheme = LaplacianScheme.Default,
    operationMode: OperationDirection = OperationDirection.AllToAll,
    positiveDivergence: bool = False,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    queryKinds: Optional[torch.Tensor] = None, referenceKinds: Optional[torch.Tensor] = None,

    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    if datastructure is None:
        datastructure = buildCompactHashMap(
            queryPositions, referencePositions, 
            querySupports, referenceSupports,
            periodicity = domain.periodic,
            domainDescription = domain,
            mode = 'superSymmetric',
        )


    if operationMode != OperationDirection.AllToAll and (queryKinds is None or referenceKinds is None):
        raise ValueError("Query and reference kinds must be provided for non AllToAll operation modes. Operation mode: {}, queryKinds is None: {}, referenceKinds is None: {}".format(operationMode, queryKinds is None, referenceKinds is None))
    if operationMode == OperationDirection.AllToAll and (queryKinds is None):
        queryKinds = getCachedDummyTensor((1,), dtype = torch.int32, device = queryPositions.device) # This is safe to do as the kinds are never checked in this case
    if operationMode == OperationDirection.AllToAll and (referenceKinds is None):
        referenceKinds = getCachedDummyTensor((1,), dtype = torch.int32, device = referencePositions.device) # This is safe to do as the kinds are never checked in this case

    if useGradientRenormalization and renormalizationMatrices is None:
        raise ValueError("Renormalization matrices must be provided if useGradientRenormalization is True.")
    if useGradHTerms and (queryOmegas is None or referenceOmegas is None):
        raise ValueError("Omegas must be provided if useGradHTerms is True.")
    if useVolume and (queryVolumes is None or referenceVolumes is None):
        raise ValueError("Volumes must be provided if useVolume is True.")
    if useCRK and (crk_A is None or crk_B is None):
        raise ValueError("CRK correction A and B tensors must be provided if useCRK is True.")
    if useCRK and (crk_gradA is None or crk_gradB is None) and operation in [WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl]:
        raise ValueError("CRK gradient correction A and B tensors must be provided if useCRK is True and the operation is Gradient, Divergence, or Curl.")
    if renormalizationMatrices is None:
        renormalizationMatrices = getCachedDummyTensor((1,queryPositions.shape[1], queryPositions.shape[1]), dtype=queryPositions.dtype, device=queryPositions.device)
    if queryOmegas is None:
        queryOmegas = getCachedDummyTensor((1,), dtype=queryPositions.dtype, device=queryPositions.device)
    if referenceOmegas is None:
        referenceOmegas = getCachedDummyTensor((1,), dtype=queryPositions.dtype, device=queryPositions.device)
    if queryVolumes is None:
        queryVolumes = getCachedDummyTensor((1,), dtype=queryPositions.dtype, device=queryPositions.device)
    if referenceVolumes is None:
        referenceVolumes = getCachedDummyTensor((1,), dtype=queryPositions.dtype, device=queryPositions.device)
    if crk_A is None:
        crk_A = getCachedDummyTensor((1,), dtype=queryPositions.dtype, device=queryPositions.device)
    if crk_B is None:
        crk_B = getCachedDummyTensor((1, queryPositions.shape[1]), dtype=queryPositions.dtype, device=queryPositions.device)
    if crk_gradA is None:
        crk_gradA = getCachedDummyTensor((1,queryPositions.shape[1]), dtype=queryPositions.dtype, device=queryPositions.device)
    if crk_gradB is None:
        crk_gradB = getCachedDummyTensor((1, queryPositions.shape[1], queryPositions.shape[1]), dtype=queryPositions.dtype, device=queryPositions.device)

    minDomain = domain.min if domain.min is not None else None
    maxDomain = domain.max if domain.max is not None else None
    periodicity = domain.periodic if domain.periodic is not None else [False] * referencePositions.shape[1]
    minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)

    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT

    with record_function(f"warpSPH - Operation"):
        # if operation == WarpOperation.Density:
        #     return computeSPHDensity_grid_warpBackend(
        #         queryPositions, referencePositions,
        #         querySupports, referenceSupports,
        #         queryMasses, referenceMasses,
        #         queryKinds= queryKinds, referenceKinds= referenceKinds,
        #         domain = domain, datastructure=datastructure, 
        #         kernel = kernel, mode = supportMode,
        #         operationMode = operationMode, 
        #     )  
        
        if queryValues is None and referenceValues is None:
            if preScatteredQuantities is None:
                raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the SPH operation.")
        if queryValues is not None and referenceValues is not None and preScatteredQuantities is not None:
            raise ValueError("Pre-scattered quantities should not be provided if queryValues and referenceValues are already provided, as they are redundant in this case.")
        if preScatteredQuantities is not None and gradientMode != GradientScheme.Naive:
            raise ValueError("Pre-scattered quantities only support the naive scheme as they are meant to provide pre-computed neighbor-level quantities for custom kernels that may not be compatible with the standard gradient schemes. If using pre-scattered quantities, the gradientMode must be set to Naive.")
    
        if operation == WarpOperation.Interpolate:
            return computeSPHInterpolant_grid_warpBackend(
                y, x,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                operationMode = operationMode, queryKinds= queryKinds, referenceKinds= referenceKinds,

                domain = domain, datastructure=datastructure, 
                kernel = kernel, mode = supportMode,           

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B,
            )
        # elif operation == WarpOperation.Gradient:
        #     return computeSPHGradient_grid_warpBackend(
        #         queryPositions, referencePositions,
        #         querySupports, referenceSupports,
        #         queryMasses, referenceMasses,
        #         queryDensities, referenceDensities,
        #         queryValues, referenceValues,
        #         operationMode = operationMode, queryKinds= queryKinds, referenceKinds= referenceKinds,

        #         domain = domain, datastructure=datastructure, 
        #         kernel = kernel, mode = supportMode, 
                
        #         gradientMode= gradientMode,

        #         scatteredQuantities= preScatteredQuantities,

        #         useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
        #         useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
        #         useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
        #         useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,                
        #     )
        # elif operation == WarpOperation.Divergence:
        #     return computeSPHDivergence_grid_warpBackend(
        #         queryPositions, referencePositions,
        #         querySupports, referenceSupports,
        #         queryMasses, referenceMasses,
        #         queryDensities, referenceDensities,
        #         queryValues, referenceValues,
        #         queryKinds= queryKinds, referenceKinds= referenceKinds,
        #         domain = domain, datastructure=datastructure, 
        #         kernel = kernel, mode = supportMode, 
        #         operationMode = operationMode, 
        #         gradientMode= gradientMode,
        #         scatteredQuantities= preScatteredQuantities,

        #         useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
        #         useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
        #         useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
        #         useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
        #     )
        # elif operation == WarpOperation.Curl:
        #     return computeSPHCurl_grid_warpBackend(
        #         queryPositions, referencePositions,
        #         querySupports, referenceSupports,
        #         queryMasses, referenceMasses,
        #         queryDensities, referenceDensities,
        #         queryValues, referenceValues,
        #         queryKinds= queryKinds, referenceKinds= referenceKinds,
        #         domain = domain, datastructure=datastructure, 
        #         kernel = kernel, mode = supportMode, 
        #         operationMode = operationMode, 
        #         gradientMode= gradientMode,
        #         scatteredQuantities= preScatteredQuantities,

        #         useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
        #         useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
        #         useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
        #         useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
        #     )
        # elif operation == WarpOperation.Laplacian:
        #     return computeSPHLaplacian_grid_warpBackend(
        #         queryPositions, referencePositions,
        #         querySupports, referenceSupports,
        #         queryMasses, referenceMasses,
        #         queryDensities, referenceDensities,
        #         queryValues, referenceValues,
        #         queryKinds= queryKinds, referenceKinds= referenceKinds,
        #         domain = domain, datastructure=datastructure, 
        #         kernel = kernel, mode = supportMode, 
        #         operationMode = operationMode, 
        #         gradientMode= gradientMode, 
        #         laplacianMode= laplacianMode, positiveDivergence=positiveDivergence,
        #         scatteredQuantities= preScatteredQuantities,

        #         useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
        #         useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
        #         useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
        #         useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
        #     )
        else:
            raise ValueError("Unsupported SPH operation: {}".format(operation))