
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.utils.wp_autograd import *

from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.mathutil.wp_math import *
from sphWarpCore.kernels.wp_kernel import *
from sphWarpCore.utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from sphWarpCore.enumTypes import *
from typing import Optional


def checkInputRenormalization( dim: int, device: torch.device,
    useGradientRenormalization: bool, renormalizationMatrices: Optional[torch.Tensor],
):
    if useGradientRenormalization and renormalizationMatrices is None:
        raise ValueError("Gradient renormalization is enabled but no renormalization matrices were provided.")
    if not useGradientRenormalization and renormalizationMatrices is None:
        return getCachedDummyTensor((1, dim, dim), dtype=torch.float32, device= device)
    return renormalizationMatrices

def checkInputGradHTerms( dim: int, device: torch.device,
    useGradHTerms: bool, queryOmegas: Optional[torch.Tensor], referenceOmegas: Optional[torch.Tensor]
):  
    if useGradHTerms and (queryOmegas is None or referenceOmegas is None):
        raise ValueError("Grad-h correction is enabled but query and/or reference omegas were not provided.")
    if not useGradHTerms and (queryOmegas is None or referenceOmegas is None):
        dummyQueryOmegas = getCachedDummyTensor((1,), dtype=torch.float32, device=device)
        dummyReferenceOmegas = getCachedDummyTensor((1,), dtype=torch.float32, device=device)
        return dummyQueryOmegas, dummyReferenceOmegas
    return queryOmegas, referenceOmegas

def checkInputVolume( dim: int, device: torch.device,
    useVolume: bool, queryVolumes: Optional[torch.Tensor], referenceVolumes: Optional[torch.Tensor]
):
    if useVolume and (queryVolumes is None or referenceVolumes is None):
        raise ValueError("Using actual volume is enabled but query and/or reference volumes were not provided.")
    if not useVolume and (queryVolumes is None or referenceVolumes is None):
        dummyQueryVolumes = getCachedDummyTensor((1,), dtype=torch.float32, device=device)
        dummyReferenceVolumes = getCachedDummyTensor((1,), dtype=torch.float32, device=device)
        return dummyQueryVolumes, dummyReferenceVolumes
    return queryVolumes, referenceVolumes

def checkInputCRK( dim: int, device: torch.device,
    useCRK: bool, crk_A: Optional[torch.Tensor], crk_B: Optional[torch.Tensor], crk_gradA: Optional[torch.Tensor], crk_gradB: Optional[torch.Tensor]
):
    if useCRK and (crk_A is None or crk_B is None or crk_gradA is None or crk_gradB is None):
        raise ValueError("Using CRK correction is enabled but CRK correction terms were not fully provided.")
    if not useCRK and (crk_A is None or crk_B is None or crk_gradA is None or crk_gradB is None):
        dummy_crk_A = getCachedDummyTensor((1,), dtype=torch.float32, device=device)
        dummy_crk_B = getCachedDummyTensor((1, dim), dtype=torch.float32, device=device)
        dummy_crk_gradA = getCachedDummyTensor((1, dim), dtype=torch.float32, device=device)
        dummy_crk_gradB = getCachedDummyTensor((1, dim, dim), dtype=torch.float32, device=device)
        return dummy_crk_A, dummy_crk_B, dummy_crk_gradA, dummy_crk_gradB
    return crk_A, crk_B, crk_gradA, crk_gradB

def checkQV(
    queryValues: Optional[torch.Tensor], referenceValues: Optional[torch.Tensor],
    scatteredQuantities: Optional[torch.Tensor]
):
    if queryValues is None and referenceValues is None:
        if scatteredQuantities is None:
            raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the gradient computation.")
        return True, scatteredQuantities, scatteredQuantities
    return False, queryValues, referenceValues

def checkKinds(
    operationMode: OperationDirection, device: torch.device, queryKinds: Optional[torch.Tensor], referenceKinds: Optional[torch.Tensor]):
    if operationMode == OperationDirection.AllToAll:
        return getCachedDummyTensor((1,), dtype=torch.int32, device=device) if queryKinds is None else queryKinds, getCachedDummyTensor((1,), dtype=torch.int32, device=device) if referenceKinds is None else referenceKinds
    else:
        if queryKinds is None or referenceKinds is None:
            raise ValueError("For directional operations, query and reference kinds must be provided to determine interaction masking.")
        return queryKinds, referenceKinds
