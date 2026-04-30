from ..radiusSearch.wp_compactHash import CompactHashMap
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Union
import torch

from ..utils.wp_autograd import *


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
from ..operations_grid import sphOperation_warp_grid

from ..utils.wp_util import getCachedDummyTensor

def sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    operation: WarpOperation = WarpOperation.Interpolate,
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    supportMode: SupportScheme = SupportScheme.Gather,    
    gradientMode: GradientScheme = GradientScheme.Naive,
    laplacianMode: LaplacianScheme = LaplacianScheme.Default,
    operationMode: OperationDirection = OperationDirection.AllToAll,
    positiveDivergence: bool = False,
    consistentDivergence: bool = False,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    queryKinds: Optional[torch.Tensor] = None, referenceKinds: Optional[torch.Tensor] = None,

    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    if adjacency is None or isinstance(adjacency, CompactHashMap):
        return sphOperation_warp_grid(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            queryDensities, referenceDensities,
            queryValues, referenceValues,
            domain = domain, datastructure=adjacency, 
            kernel = kernel, supportMode = supportMode,
            operation = operation, operationMode = operationMode, 
            gradientMode= gradientMode, laplacianMode= laplacianMode, positiveDivergence=positiveDivergence,
            consistentDivergence= consistentDivergence,
            preScatteredQuantities= preScatteredQuantities, queryKinds= queryKinds, referenceKinds= referenceKinds,
            useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
            useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,
            useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
            useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
        )
    if operationMode != OperationDirection.AllToAll and (queryKinds is None or referenceKinds is None):
        raise ValueError("Query and reference kinds must be provided for non AllToAll operation modes. Operation mode: {}, queryKinds is None: {}, referenceKinds is None: {}".format(operationMode, queryKinds is None, referenceKinds is None))
    if operationMode == OperationDirection.AllToAll and (queryKinds is None):
        queryKinds = adjacency.numNeighbors # This is safe to do as the kinds are never checked in this case
    if operationMode == OperationDirection.AllToAll and (referenceKinds is None):
        referenceKinds = adjacency.numNeighbors # This is safe to do as the kinds are never checked in this case

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
                operationMode = operationMode, queryKinds= queryKinds, referenceKinds= referenceKinds,

                domain = domain, adjacency=adjacency, 
                kernel = kernel, mode = supportMode,           

                scatteredQuantities= preScatteredQuantities,

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B,
            )
        elif operation == WarpOperation.Gradient:
            return computeSPHGradient_warpBackend(
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                queryValues, referenceValues,
                operationMode = operationMode, queryKinds= queryKinds, referenceKinds= referenceKinds,

                domain = domain, adjacency= adjacency, 
                kernel = kernel, mode = supportMode, 
                
                gradientMode= gradientMode,

                scatteredQuantities= preScatteredQuantities,

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
                useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
                useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,                
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
                scatteredQuantities= preScatteredQuantities,
                consistentDivergence = consistentDivergence,

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
                useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
                useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
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
                scatteredQuantities= preScatteredQuantities,

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
                useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
                useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
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
                scatteredQuantities= preScatteredQuantities,

                useVolume= useVolume, queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,
                useCRK= useCRK, crk_A= crk_A, crk_B= crk_B, crk_gradA= crk_gradA, crk_gradB= crk_gradB,
                useGradientRenormalization= useGradientRenormalization, renormalizationMatrices= renormalizationMatrices,
                useGradHTerms= useGradHTerms, queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,           
                
            )
        else:
            raise ValueError("Unsupported SPH operation: {}".format(operation))
        

from ..state import *

def warpOperation(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryValues : Optional[torch.Tensor] = None, referenceValues : Optional[torch.Tensor] = None,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
    consistentDivergence: bool = False,
):
    queryPositions = queryParticles.positions
    referencePositions = queryParticles.positions if referenceParticles is None else referenceParticles.positions
    querySupports = queryParticles.supports
    referenceSupports = queryParticles.supports if referenceParticles is None else referenceParticles.supports
    queryMasses = queryParticles.masses
    referenceMasses = queryParticles.masses if referenceParticles is None else referenceParticles.masses
    queryDensities = queryParticles.densities #if hasattr(queryParticles, 'densities') else None
    referenceDensities = queryDensities if referenceParticles is None else referenceParticles.densities
    # referenceDensities = queryDensities if referenceParticles is None or hasattr(referenceParticles,'densities') else referenceParticles.densities
    queryKinds = queryParticles.kinds if hasattr(queryParticles, 'kinds') else None
    referenceKinds = queryKinds if referenceParticles is None or not hasattr(referenceParticles, 'kinds') else referenceParticles.kinds

    if gradHState is not None:
        if isinstance(gradHState, GradHState):
            queryOmegas = gradHState.queryOmegas
            referenceOmegas = gradHState.referenceOmegas
        elif isinstance(gradHState, tuple) and len(gradHState) == 2:
            queryOmegas, referenceOmegas = gradHState
        elif isinstance(gradHState, torch.Tensor):
            queryOmegas = gradHState
            referenceOmegas = gradHState
        else:
            raise ValueError("Invalid type for gradHState: {}. Must be either GradHState, a tuple of two tensors, or a single tensor.".format(type(gradHState)))
        
        if referenceOmegas is None:
            referenceOmegas = queryOmegas
    else:
        queryOmegas = None
        referenceOmegas = None

    if renormalizationState is not None:
        if isinstance(renormalizationState, RenormalizationState):
            renormalizationMatrices = renormalizationState.renormalizationMatrices
        elif isinstance(renormalizationState, torch.Tensor):
            renormalizationMatrices = renormalizationState
        else:
            raise ValueError("Invalid type for renormalizationState: {}. Must be either RenormalizationState or torch.Tensor.".format(type(renormalizationState)))
    else:
        renormalizationMatrices = None

    referenceVolumes = referenceVolumes if referenceVolumes is not None else (queryVolumes if queryVolumes is not None else None)

    

    return sphOperation_warp(
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues if referenceValues is not None else queryValues,
        domain = domain, adjacency=adjacency,
        operation = operationProperties.operation, supportMode = operationProperties.supportMode,
        kernel = operationProperties.kernel, operationMode = operationProperties.operationMode,
        gradientMode= operationProperties.gradientMode, laplacianMode= operationProperties.laplacianMode, positiveDivergence= operationProperties.positiveDivergence,
        preScatteredQuantities= preScatteredQuantities, 
        queryKinds= queryKinds, referenceKinds= referenceKinds,

        useGradHTerms= gradHState is not None, 
        queryOmegas= queryOmegas, referenceOmegas= referenceOmegas,

        useVolume = queryVolumes is not None,
        queryVolumes= queryVolumes, referenceVolumes= referenceVolumes,

        useCRK= crkState is not None, 
        crk_A= crkState.A if crkState is not None else None, crk_B= crkState.B if crkState is not None else None, crk_gradA= crkState.gradA if crkState is not None else None, crk_gradB= crkState.gradB if crkState is not None else None,

        useGradientRenormalization= renormalizationState is not None, renormalizationMatrices= renormalizationMatrices,
        consistentDivergence= consistentDivergence,
    )

