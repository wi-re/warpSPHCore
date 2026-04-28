import warp as wp
from warp.types import vector, matrix
from typing import Any, Union
import torch

from .warp_state import *

# from sphWarpCore import ParticleState
from .state import ParticleState, OperationProperties, GradHState, RenormalizationState, CRKState
from .radiusSearch.wp_compactHash import CompactHashMap, AdjacencyListWarp, buildCompactHashMap
from typing import Tuple, Union, Optional

from .utils.wp_autograd import *
from .radiusSearch.radius_util import convertModeToUint

from .radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from .mathutil.wp_math import *
from .kernels.wp_kernel import *
from .utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from .utils.wp_util import castTorchToWarpAsBuiltins, castWarpToTorch
# from torch.profiler import profile, record_function, ProfilerActivity
from .enumTypes import *
from .utils.arg_check import *

def parseArguments(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
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

    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    supportMode = operationProperties.supportMode
    kernel = operationProperties.kernel
    operationMode = operationProperties.operationMode

    mode_uint = convertModeToUint(supportMode.name)
    kernel_int = kernel.value
    gradientMode_int = 0
    opInt = wp.int32(operationMode.value)

    device = queryPositions.device
    dim = queryPositions.shape[1]

    qK, rK = checkKinds(operationMode, device, queryKinds, referenceKinds)

    useCRK = crkState is not None
    useVolumes = queryVolumes is not None or referenceVolumes is not None
    useGradHTerms = gradHState is not None
    useGradientRenormalization = renormalizationState is not None

    dim = queryPositions.shape[1]

    renormalizationMatrices_ = renormalizationState.renormalizationMatrices if useGradientRenormalization else getCachedDummyTensor((1, dim, dim), device= device, dtype=torch.float32)
    queryOmegas_, referenceOmegas_ = None, None
    if useGradHTerms:
        if isinstance(gradHState, GradHState):
            queryOmegas_ = gradHState.queryOmegas
            referenceOmegas_ = gradHState.referenceOmegas
        elif isinstance(gradHState, tuple) and len(gradHState) == 2:
            queryOmegas_, referenceOmegas_ = gradHState
        elif isinstance(gradHState, torch.Tensor):
            queryOmegas_ = gradHState
            referenceOmegas_ = gradHState
    else:
        queryOmegas_ = getCachedDummyTensor((1,), device=device, dtype=torch.float32)
        referenceOmegas_ = getCachedDummyTensor((1,), device=device, dtype=torch.float32)

    queryVolumes_, referenceVolumes_ = (queryVolumes, referenceVolumes) if useVolumes else (None, None)
    if referenceVolumes_ is None and queryVolumes_ is not None:
        referenceVolumes_ = queryVolumes_
    if not useVolumes:
        queryVolumes_ = getCachedDummyTensor((1,), device=device, dtype=torch.float32)
        referenceVolumes_ = getCachedDummyTensor((1,), device=device, dtype=torch.float32)

    crk_A_, crk_B_, crk_gradA_, crk_gradB_ = (crkState.A, crkState.B, crkState.gradA, crkState.gradB) if useCRK else (\
        getCachedDummyTensor((1,),device= device, dtype = torch.float32), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch.float32), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch.float32), 
        getCachedDummyTensor((1, dim, dim), device=device, dtype = torch.float32)
    )

    adj = None
    if adjacency is None:
        adjacency = buildCompactHashMap(
            queryPositions, referencePositions, 
            querySupports, referenceSupports,
            periodicity = domain.periodic,
            domainDescription = domain,
            mode = 'superSymmetric',
        )

    
    adjacencyState = adjacencyData()
    gridState = gridData()
    if not isinstance(adjacency, CompactHashMap):
        adj = True #, (adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
                #      getCachedDummyTensor((1,), device=device, dtype = torch.int64), 
                #      domainMin, domainMax, wp.float32(0.0), # periodicity is handled within the kernel, so we can just pass a dummy value here
                #      getCachedDummyTensor((1,), device=device, dtype = torch.int32),  # numCells
                #      getCachedDummyTensor((1, 2), device=device, dtype = torch.int32), # hashTable
                #      getCachedDummyTensor((1, 3), device=device, dtype = torch.int64), # cellTable
                #      dim, 0, getCachedDummyTensor((1, 3), device=device, dtype = torch.int32) # cellOffsets
                # )
        adjacencyState.neighborList = castTorchToWarpAsBuiltins(adjacency.j)
        adjacencyState.neighborOffsets = castTorchToWarpAsBuiltins(adjacency.edgeOffsets)
        adjacencyState.numNeighbors = castTorchToWarpAsBuiltins(adjacency.numNeighbors)
        
        gridState.sortIndex = castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int64))
        gridState.qMin = castTorchToWarpAsBuiltins(domainMin)
        gridState.qMax = castTorchToWarpAsBuiltins(domainMax)
        gridState.hCell = 0.0 # not used in adjacency mode
        gridState.numCells = castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int32))
        gridState.hashTable = castTorchToWarpAsBuiltins(getCachedDummyTensor((1, 2), device=device, dtype = torch.int32))
        gridState.cellTable = castTorchToWarpAsBuiltins(getCachedDummyTensor((1, 3), device=device, dtype = torch.int64))
        gridState.D = dim
        gridState.numOffsets = 0
        gridState.cellOffsets = castTorchToWarpAsBuiltins(getCachedDummyTensor((1, 3), device=device, dtype = torch.int32))
        
    else:
        adj = False #, (
        #     getCachedDummyTensor((1,), device, dtype = torch.int64), getCachedDummyTensor((1,), device=device, dtype = torch.int32), getCachedDummyTensor((1,), device=device, dtype = torch.int32),
        #     adjacency.sortIndex,
        #     adjacency.qMin, adjacency.qMax, adjacency.hCell,
        #     adjacency.numCells, adjacency.hashTable, adjacency.sortedCellTable, dim,
        #     adjacency.numOffsets, adjacency.cellOffsets,
        # )
        adjacencyState.neighborList = castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int64))
        adjacencyState.neighborOffsets = castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int32))
        adjacencyState.numNeighbors =castTorchToWarpAsBuiltins( getCachedDummyTensor((1,), device=device, dtype = torch.int32))
        gridState.sortIndex = castTorchToWarpAsBuiltins(adjacency.sortIndex)
        gridState.qMin = castTorchToWarpAsBuiltins(adjacency.qMin)
        gridState.qMax = castTorchToWarpAsBuiltins(adjacency.qMax)
        gridState.hCell = adjacency.hCell
        gridState.numCells = castTorchToWarpAsBuiltins(adjacency.numCells)
        gridState.hashTable = castTorchToWarpAsBuiltins(adjacency.hashTable)
        gridState.cellTable = castTorchToWarpAsBuiltins(adjacency.sortedCellTable)
        gridState.D = dim
        gridState.numOffsets = castTorchToWarpAsBuiltins(adjacency.numOffsets)
        gridState.cellOffsets = castTorchToWarpAsBuiltins(adjacency.cellOffsets)

    queryParticles = particleDataSoA_1() if dim == 1 else (particleDataSoA_2() if dim == 2 else particleDataSoA_3())
    queryParticles.positions = castTorchToWarpAsBuiltins(queryPositions)
    queryParticles.supports = castTorchToWarpAsBuiltins(querySupports)
    queryParticles.masses = castTorchToWarpAsBuiltins(queryMasses)
    queryParticles.densities = castTorchToWarpAsBuiltins(queryDensities)
    queryParticles.kinds = castTorchToWarpAsBuiltins(qK) if qK is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int32))
    referenceParticles = particleDataSoA_1() if dim == 1 else (particleDataSoA_2() if dim == 2 else particleDataSoA_3())
    referenceParticles.positions = castTorchToWarpAsBuiltins(referencePositions)
    referenceParticles.supports = castTorchToWarpAsBuiltins(referenceSupports)
    referenceParticles.masses = castTorchToWarpAsBuiltins(referenceMasses)
    referenceParticles.densities = castTorchToWarpAsBuiltins(referenceDensities)
    referenceParticles.kinds = castTorchToWarpAsBuiltins(rK) if rK is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.int32))

    domainState = domainData()
    domainState.domainMin = castTorchToWarpAsBuiltins(domainMin)
    domainState.domainMax = castTorchToWarpAsBuiltins(domainMax)
    domainState.periodicity = castTorchToWarpAsBuiltins(periodicity)
    domainState.dim = dim

    correctionState = correctionData_1() if dim == 1 else (correctionData_2() if dim == 2 else correctionData_3())
    correctionState.useGradientRenormalization = useGradientRenormalization
    correctionState.useGradHTerms = useGradHTerms
    correctionState.useVolume = useVolumes
    correctionState.useCRK = useCRK
    correctionState.renormalizationMatrices = castTorchToWarpAsBuiltins(renormalizationMatrices_)
    correctionState.queryOmegas = castTorchToWarpAsBuiltins(queryOmegas_)
    correctionState.referenceOmegas = castTorchToWarpAsBuiltins(referenceOmegas_)
    correctionState.queryVolumes = castTorchToWarpAsBuiltins(queryVolumes_) if queryVolumes_ is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.float32))
    correctionState.referenceVolumes = castTorchToWarpAsBuiltins(referenceVolumes_) if referenceVolumes_ is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch.float32))
    correctionState.queryA = castTorchToWarpAsBuiltins(crk_A_)
    correctionState.queryB = castTorchToWarpAsBuiltins(crk_B_)
    correctionState.queryGradA = castTorchToWarpAsBuiltins(crk_gradA_)
    correctionState.queryGradB = castTorchToWarpAsBuiltins(crk_gradB_)

    # print("Parsed Arguments for computeDeltaShift_Warp:")
    # print("useGradientRenormalization:", useGradientRenormalization)
    # print("useGradHTerms:", useGradHTerms)
    # print("useVolume:", useVolumes)
    # print("useCRK:", useCRK)


    return (
        queryParticles, referenceParticles, domainState,
        adj, adjacencyState, gridState,
        correctionState, #
        mode_uint, kernel_int, gradientMode_int, opInt
    ), device, dim

    