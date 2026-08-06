import warp as wp
from warp.types import vector, matrix
from typing import Any, Union
import torch

from .radiusSearch.wp_compactHash import CompactHashMap, AdjacencyListWarp, buildCompactHashMap
from typing import Tuple, Union, Optional

from .utils.wp_autograd import *

from .dataTypes import *
from .math import *
from .kernels import *
from .utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from .utils.wp_util import castTorchToWarpAsBuiltins, castWarpToTorch
# from torch.profiler import profile, record_function, ProfilerActivity
from .enumTypes import *
from .utils.arg_check import *

from .types import *

def parseArguments(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[Union[Tuple[CRKState,CRKState], CRKState]] = None,
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

    mode_uint = supportSchemeToUint(supportMode)
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

    torch_t = get_torch_precision()

    renormalizationMatrices_ = renormalizationState.renormalizationMatrices if useGradientRenormalization else getCachedDummyTensor((1, dim, dim), device= device, dtype=torch_t)
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
        queryOmegas_ = getCachedDummyTensor((1,), device=device, dtype=torch_t)
        referenceOmegas_ = getCachedDummyTensor((1,), device=device, dtype=torch_t)

    queryVolumes_, referenceVolumes_ = (queryVolumes, referenceVolumes) if useVolumes else (None, None)
    if referenceVolumes_ is None and queryVolumes_ is not None:
        referenceVolumes_ = queryVolumes_
    if not useVolumes:
        queryVolumes_ = getCachedDummyTensor((1,), device=device, dtype=torch_t)
        referenceVolumes_ = getCachedDummyTensor((1,), device=device, dtype=torch_t)

    qcrkState, rcrkState = None, None
    if isinstance(crkState, tuple) and len(crkState) == 2:
        qcrkState, rcrkState = crkState
    elif isinstance(crkState, CRKState):
        qcrkState = crkState
        rcrkState = crkState
    else:
        pass

    qcrk_A_, qcrk_B_, qcrk_gradA_, qcrk_gradB_ = (qcrkState.A, qcrkState.B, qcrkState.gradA, qcrkState.gradB) if useCRK else (\
        getCachedDummyTensor((1,),device= device, dtype = torch_t), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch_t), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch_t), 
        getCachedDummyTensor((1, dim, dim), device=device, dtype = torch_t)
    )
    rcrk_A_, rcrk_B_, rcrk_gradA_, rcrk_gradB_ = (rcrkState.A, rcrkState.B, rcrkState.gradA, rcrkState.gradB) if useCRK else (\
        getCachedDummyTensor((1,), device=device, dtype = torch_t), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch_t), 
        getCachedDummyTensor((1, dim), device=device, dtype = torch_t), 
        getCachedDummyTensor((1, dim, dim), device=device, dtype = torch_t)
    )

    adj = None
    if adjacency is None:
        adjacency = buildCompactHashMap(
            queryPositions, referencePositions, 
            querySupports, referenceSupports,
            periodicity = domain.periodic,
            domainDescription = domain,
            mode = SupportScheme.SuperSymmetric,
        )

    
    adjacencyState = adjacencyData()
    gridState = gridData()
    if not isinstance(adjacency, CompactHashMap):
        adj = True #, (adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
                #      getCachedDummyTensor((1,), device=device, dtype = torch.int64), 
                #      domainMin, domainMax, wp.float32(scalar_t(0.0)), # periodicity is handled within the kernel, so we can just pass a dummy value here
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
        gridState.hCell = scalar_t(scalar_t(0.0)) # not used in adjacency mode
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
        gridState.numOffsets = adjacency.numOffsets
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
    correctionState.queryVolumes = castTorchToWarpAsBuiltins(queryVolumes_) if queryVolumes_ is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch_t))
    correctionState.referenceVolumes = castTorchToWarpAsBuiltins(referenceVolumes_) if referenceVolumes_ is not None else castTorchToWarpAsBuiltins(getCachedDummyTensor((1,), device=device, dtype = torch_t))
    correctionState.queryA = castTorchToWarpAsBuiltins(qcrk_A_)
    correctionState.queryB = castTorchToWarpAsBuiltins(qcrk_B_)
    correctionState.queryGradA = castTorchToWarpAsBuiltins(qcrk_gradA_)
    correctionState.queryGradB = castTorchToWarpAsBuiltins(qcrk_gradB_)
    correctionState.referenceA = castTorchToWarpAsBuiltins(rcrk_A_)
    correctionState.referenceB = castTorchToWarpAsBuiltins(rcrk_B_)
    correctionState.referenceGradA = castTorchToWarpAsBuiltins(rcrk_gradA_)
    correctionState.referenceGradB = castTorchToWarpAsBuiltins(rcrk_gradB_)

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


# ---------------------------------------------------------------------------
# State-aware autograd wrapper
# ---------------------------------------------------------------------------

from torch.profiler import record_function

def extractStateInfo(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryVolumes: Optional[torch.Tensor] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[Union[Tuple[CRKState, CRKState], CRKState]] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor, RenormalizationState]] = None,
):
    """
    Extract every torch.Tensor embedded in the structured state arguments into a
    single, deterministically ordered flat list and return a *build_fn* closure
    that can reconstruct all Warp kernel structs from a parallel list of Warp arrays.

    This is the extraction half of what ``parseArguments`` previously did in one
    shot; the conversion half is deferred to ``StateAwareWarpFunction.forward`` so
    that the original torch tensors can be saved for autograd.

    Flat tensor index layout (36 entries):
        0  qPos          6  qDen         12  rOmega       18  qcrk_gradB
        1  rPos          7  rDen         13  qVol         19  rcrk_A
        2  qSup          8  qK           14  rVol         20  rcrk_B
        3  rSup          9  rK           15  qcrk_A       21  rcrk_gradA
        4  qMas         10  renormMat    16  qcrk_B       22  rcrk_gradB
        5  rMas         11  qOmega       17  qcrk_gradA   23  adj_neighborList
       24  adj_neighborOffsets           30  grid_hashTable
       25  adj_numNeighbors              31  grid_cellTable
       26  grid_sortIndex                32  grid_cellOffsets
       27  grid_qMin                     33  domainMin
       28  grid_qMax                     34  domainMax
       29  grid_numCells                 35  periodicity

    Returns:
        flat_tensors  : List[torch.Tensor]
        build_fn      : Callable[[List[wp.array]], tuple]  – returns full kernel args
        device        : torch.device
        dim           : int
    """
    with record_function("extractStateInfo [ESI]"):
        with record_function("[ESI] 1. resolve particle fields"):
            # ------------------------------------------------------------------ #
            # 1.  Resolve particle fields
            # ------------------------------------------------------------------ #
            qPos = queryParticles.positions
            rPos = qPos if referenceParticles is None else referenceParticles.positions
            qSup = queryParticles.supports
            rSup = qSup if referenceParticles is None else referenceParticles.supports
            qMas = queryParticles.masses
            rMas = qMas if referenceParticles is None else referenceParticles.masses
            qDen = queryParticles.densities
            rDen = qDen if referenceParticles is None else referenceParticles.densities

            device = qPos.device
            dim    = qPos.shape[1]

            torch_t = get_torch_precision()
            _d1f   = getCachedDummyTensor((1,),          dtype=torch_t, device=device)
            _d1i   = getCachedDummyTensor((1,),          dtype=torch.int32,   device=device)
            _d1Df  = getCachedDummyTensor((1, dim),      dtype=torch_t, device=device)
            _d1DDf = getCachedDummyTensor((1, dim, dim), dtype=torch_t, device=device)

            # Replace None densities with a dummy so the flat list is never sparse
            if qDen is None:
                qDen = _d1f
            if rDen is None:
                rDen = _d1f

            # Kinds
            queryKinds    = queryParticles.kinds if hasattr(queryParticles, 'kinds') else None
            referenceKinds = (
                queryKinds if referenceParticles is None or not hasattr(referenceParticles, 'kinds')
                else referenceParticles.kinds
            )
            operationMode = operationProperties.operationMode
            qK, rK = checkKinds(operationMode, device, queryKinds, referenceKinds)
        with record_function("[ESI] 2. resolve correction states"):
            # ------------------------------------------------------------------ #
            # 2.  Resolve correction states
            # ------------------------------------------------------------------ #
            useGradientRenormalization = renormalizationState is not None
            useGradHTerms              = gradHState is not None
            useVolumes                 = queryVolumes is not None or referenceVolumes is not None
            useCRK                     = crkState is not None

            # Renormalization matrices
            if useGradientRenormalization:
                renormMat = (
                    renormalizationState.renormalizationMatrices
                    if isinstance(renormalizationState, RenormalizationState)
                    else renormalizationState
                )
            else:
                renormMat = _d1DDf

            # Grad-h omegas
            if useGradHTerms:
                if isinstance(gradHState, GradHState):
                    qOmega, rOmega = gradHState.queryOmegas, gradHState.referenceOmegas
                elif isinstance(gradHState, tuple) and len(gradHState) == 2:
                    qOmega, rOmega = gradHState
                elif isinstance(gradHState, torch.Tensor):
                    qOmega = rOmega = gradHState
                else:
                    raise ValueError(
                        "Invalid gradHState type: {}".format(type(gradHState))
                    )
                if rOmega is None:
                    rOmega = qOmega
            else:
                qOmega = rOmega = _d1f

            # Volumes
            if useVolumes:
                qVol = queryVolumes    if queryVolumes    is not None else _d1f
                rVol = referenceVolumes if referenceVolumes is not None else qVol
            else:
                qVol = rVol = _d1f

            # CRK correction terms
            if useCRK:
                if isinstance(crkState, tuple) and len(crkState) == 2:
                    qcrkSt, rcrkSt = crkState
                elif isinstance(crkState, CRKState):
                    qcrkSt = rcrkSt = crkState
                else:
                    raise ValueError("Invalid crkState type: {}".format(type(crkState)))
                qcrk_A, qcrk_B, qcrk_gradA, qcrk_gradB = qcrkSt.A, qcrkSt.B, qcrkSt.gradA, qcrkSt.gradB
                rcrk_A, rcrk_B, rcrk_gradA, rcrk_gradB = rcrkSt.A, rcrkSt.B, rcrkSt.gradA, rcrkSt.gradB
            else:
                qcrk_A  = rcrk_A  = _d1f
                qcrk_B  = rcrk_B  = _d1Df
                qcrk_gradA  = rcrk_gradA  = _d1Df
                qcrk_gradB  = rcrk_gradB  = _d1DDf
        with record_function("[ESI] 3. resolve domain tensors"):
            # ------------------------------------------------------------------ #
            # 3.  Domain tensors
            # ------------------------------------------------------------------ #
            domainMin  = domain.min
            domainMax  = domain.max
            periodicity = domain.periodic

        with record_function("[ESI] 4. build/unpack adjacency structure"):
            # ------------------------------------------------------------------ #
            # 4.  Build / unpack adjacency structure
            # ------------------------------------------------------------------ #
            if adjacency is None:
                adjacency = buildCompactHashMap(
                    qPos, rPos,
                    qSup, rSup,
                    periodicity=domain.periodic,
                    domainDescription=domain,
                    mode=SupportScheme.SuperSymmetric,
                )

            useAdjacency = not isinstance(adjacency, CompactHashMap)

            if useAdjacency:
                adj_neighborList    = adjacency.j
                adj_neighborOffsets = adjacency.edgeOffsets
                adj_numNeighbors    = adjacency.numNeighbors
                grid_sortIndex  = getCachedDummyTensor((1,),    dtype=torch.int64,  device=device)
                grid_qMin       = domainMin
                grid_qMax       = domainMax
                grid_hCell      = scalar_t(scalar_t(0.0))
                grid_numCells   = getCachedDummyTensor((1,),    dtype=torch.int32,  device=device)
                grid_hashTable  = getCachedDummyTensor((1, 2),  dtype=torch.int32,  device=device)
                grid_cellTable  = getCachedDummyTensor((1, 3),  dtype=torch.int64,  device=device)
                grid_numOffsets = 0
                grid_cellOffsets = getCachedDummyTensor((1, 3), dtype=torch.int32,  device=device)
            else:
                adj_neighborList    = getCachedDummyTensor((1,), dtype=torch.int64,  device=device)
                adj_neighborOffsets = getCachedDummyTensor((1,), dtype=torch.int32,  device=device)
                adj_numNeighbors    = getCachedDummyTensor((1,), dtype=torch.int32,  device=device)
                grid_sortIndex   = adjacency.sortIndex
                grid_qMin        = adjacency.qMin
                grid_qMax        = adjacency.qMax
                grid_hCell       = adjacency.hCell
                grid_numCells    = adjacency.numCells
                grid_hashTable   = adjacency.hashTable
                grid_cellTable   = adjacency.sortedCellTable
                grid_numOffsets  = adjacency.numOffsets
                grid_cellOffsets = adjacency.cellOffsets
        with record_function("[ESI] 5. resolve operation properties"):
            # ------------------------------------------------------------------ #
            # 5.  Operation scalars (non-tensor config captured by build_fn)
            # ------------------------------------------------------------------ #
            supportMode    = operationProperties.supportMode
            mode_uint      = supportSchemeToUint(supportMode)
            kernel_int     = operationProperties.kernel.value
            gradientMode_int = operationProperties.gradientMode.value if operationProperties.gradientMode is not None else 0
            laplacianMode_int = operationProperties.laplacianMode.value if operationProperties.laplacianMode is not None else 0
            positiveDivergence = operationProperties.positiveDivergence if operationProperties.positiveDivergence is not None else False
            divergenceMode = operationProperties.divergenceDotMode if operationProperties.divergenceDotMode is not None else False

            opInt          = wp.int32(operationMode.value)

            cfg = {
                'dim':                      dim,
                'useAdjacency':             useAdjacency,
                'grid_hCell':               grid_hCell,
                'grid_numOffsets':          grid_numOffsets,
                'useGradientRenormalization': useGradientRenormalization,
                'useGradHTerms':            useGradHTerms,
                'useVolumes':               useVolumes,
                'useCRK':                   useCRK,
                'mode_uint':                mode_uint,
                'kernel_int':               kernel_int,
                'gradientMode_int':         gradientMode_int,
                'laplacianMode_int':        laplacianMode_int,
                'positiveDivergence_int':   1 if positiveDivergence else 0,
                'divergenceMode_int':       1 if divergenceMode else 0,
                'opInt':                    opInt,
            }
        with record_function("[ESI] 6. assemble flat tensor list"):
            # ------------------------------------------------------------------ #
            # 6.  Assemble flat tensor list (see index layout in docstring)
            # ------------------------------------------------------------------ #
            flat_tensors = [
                # particle fields
                qPos, rPos, qSup, rSup, qMas, rMas, qDen, rDen,      # 0-7
                qK, rK,                                                # 8-9
                # correction
                renormMat, qOmega, rOmega, qVol, rVol,                # 10-14
                qcrk_A, qcrk_B, qcrk_gradA, qcrk_gradB,              # 15-18
                rcrk_A, rcrk_B, rcrk_gradA, rcrk_gradB,              # 19-22
                # adjacency
                adj_neighborList, adj_neighborOffsets, adj_numNeighbors,  # 23-25
                grid_sortIndex, grid_qMin, grid_qMax,                  # 26-28
                grid_numCells, grid_hashTable, grid_cellTable,         # 29-31
                grid_cellOffsets,                                      # 32
                # domain
                domainMin, domainMax, periodicity,                     # 33-35
            ]

    # ------------------------------------------------------------------ #
    # 7.  build_fn: reconstruct Warp structs from a parallel list of arrays.
    #     Pre-resolve every type and scalar constant into the closure so that
    #     the hot path (called every kernel launch) contains no branches or
    #     dict lookups — only attribute assignments.
    # ------------------------------------------------------------------ #

    # Pre-resolve concrete types once based on dim
    _dim            = cfg['dim']
    _ParticleSoA    = particleDataSoA_1 if _dim == 1 else (particleDataSoA_2 if _dim == 2 else particleDataSoA_3)
    _CorrData       = correctionData_1  if _dim == 1 else (correctionData_2  if _dim == 2 else correctionData_3)

    # Capture all scalars that never change
    _grid_hCell                  = cfg['grid_hCell']
    _grid_numOffsets             = cfg['grid_numOffsets']
    _useAdjacency                = cfg['useAdjacency']
    _useGradientRenormalization  = cfg['useGradientRenormalization']
    _useGradHTerms               = cfg['useGradHTerms']
    _useVolumes                  = cfg['useVolumes']
    _useCRK                      = cfg['useCRK']
    _mode_uint                   = cfg['mode_uint']
    _kernel_int                  = cfg['kernel_int']
    _gradientMode_int            = cfg['gradientMode_int']
    _laplacianMode_int           = cfg['laplacianMode_int']
    _positiveDivergence_int      = cfg['positiveDivergence_int']
    _divergenceMode_int          = cfg['divergenceMode_int']
    _opInt                       = cfg['opInt']

    def build_fn(wa: list) -> tuple:
        # Particle structs (no branch — types resolved at closure build time)
        qPart = _ParticleSoA()
        qPart.positions = wa[0]
        qPart.supports  = wa[2]
        qPart.masses    = wa[4]
        qPart.densities = wa[6]
        qPart.kinds     = wa[8]

        rPart = _ParticleSoA()
        rPart.positions = wa[1]
        rPart.supports  = wa[3]
        rPart.masses    = wa[5]
        rPart.densities = wa[7]
        rPart.kinds     = wa[9]

        # Adjacency structs
        adjState = adjacencyData()
        adjState.neighborList    = wa[23]
        adjState.neighborOffsets = wa[24]
        adjState.numNeighbors    = wa[25]

        gState = gridData()
        gState.sortIndex   = wa[26]
        gState.qMin        = wa[27]
        gState.qMax        = wa[28]
        gState.hCell       = _grid_hCell
        gState.numCells    = wa[29]
        gState.hashTable   = wa[30]
        gState.cellTable   = wa[31]
        gState.D           = _dim
        gState.numOffsets  = _grid_numOffsets
        gState.cellOffsets = wa[32]

        # Domain struct
        domState = domainData()
        domState.domainMin   = wa[33]
        domState.domainMax   = wa[34]
        domState.periodicity = wa[35]
        domState.dim         = _dim

        # Correction struct (no branch)
        corrState = _CorrData()
        corrState.useGradientRenormalization = _useGradientRenormalization
        corrState.useGradHTerms              = _useGradHTerms
        corrState.useVolume                  = _useVolumes
        corrState.useCRK                     = _useCRK
        corrState.renormalizationMatrices    = wa[10]
        corrState.queryOmegas                = wa[11]
        corrState.referenceOmegas            = wa[12]
        corrState.queryVolumes               = wa[13]
        corrState.referenceVolumes           = wa[14]
        corrState.queryA                     = wa[15]
        corrState.queryB                     = wa[16]
        corrState.queryGradA                 = wa[17]
        corrState.queryGradB                 = wa[18]
        corrState.referenceA                 = wa[19]
        corrState.referenceB                 = wa[20]
        corrState.referenceGradA             = wa[21]
        corrState.referenceGradB             = wa[22]

        return (
            qPart, rPart, domState,
            _useAdjacency, adjState, gState,
            corrState,
            _mode_uint, _kernel_int,
            _gradientMode_int, _laplacianMode_int, _positiveDivergence_int, _divergenceMode_int, _opInt,
        )

    return flat_tensors, build_fn, device, dim

from .utils.wp_autograd import StateAwareWarpFunction

def warpWrapper2(
    launcher,
    kernel,
    outputSizes,
    outputDtypes,
    defaultStateArguments: tuple,
    additionalArguments: tuple = (),
    numThreads: Optional[int] = None,
):
    """
    State-aware autograd wrapper for SPH kernels.

    Unlike the flat-tensor ``warpWrapper``, this variant accepts high-level
    structured state objects (``ParticleState``, ``CRKState``, etc.) directly.
    It extracts all torch.Tensors from those structures, routes them through
    ``StateAwareWarpFunction`` so that gradients are properly tracked, and
    defers all Warp struct assembly to the autograd forward pass.

    Args:
        launcher:               Kernel launcher, e.g. ``launch_kernel``.
        kernel:                 The ``wp.kernel`` to execute.
        outputSizes:            Output shape passed to the launcher.
        outputDtypes:           Output Warp dtype(s) passed to the launcher.
        defaultStateArguments:  Tuple in the same order as ``parseArguments``:
                                    (queryParticles, operationProperties, domain,
                                     queryVolumes, referenceVolumes, adjacency,
                                     referenceParticles, crkState,
                                     gradHState, renormalizationState)
        additionalArguments:    Extra per-kernel arguments appended after the
                                standard struct args.  Any ``torch.Tensor`` entries
                                will be tracked for gradients; plain Python scalars
                                and ints are forwarded unchanged.
        numThreads:             Explicit thread count for wp.launch(). If None,
                                defaults to outputSizes. Use this when the number
                                of threads should differ from output size.

    Returns:
        torch.Tensor or tuple of torch.Tensor – kernel output(s).
    """
    with record_function("warpWrapper2 [WW2]"):
        # --- extract state tensors and the struct-building closure ---
        flat_state_tensors, state_build_fn, device, dim = extractStateInfo(
            *defaultStateArguments
        )
        n_state = len(flat_state_tensors)

        # --- split additionalArguments into tensors and non-tensors ---
        add_tensor_pos = []   # (original_index, tensor)
        add_scalar_map = {}   # original_index -> scalar value
        for i, arg in enumerate(additionalArguments):
            if isinstance(arg, torch.Tensor):
                add_tensor_pos.append((i, arg))
            else:
                add_scalar_map[i] = arg

        add_tensors = [t for _, t in add_tensor_pos]
        n_add       = len(additionalArguments)

        # --- unified flat tensor list (state first, then additional tensors) ---
        flat_tensors = flat_state_tensors + add_tensors

        # --- build_fn combines struct args + reconstructed additional args ---
        def build_fn(wa: list) -> tuple:
            struct_args = state_build_fn(wa[:n_state])

            # Reconstruct additional args preserving original order
            reconstructed = [None] * n_add
            for pos, (orig_idx, _) in enumerate(add_tensor_pos):
                reconstructed[orig_idx] = wa[n_state + pos]
            for orig_idx, val in add_scalar_map.items():
                reconstructed[orig_idx] = val

            return struct_args + tuple(reconstructed)

        # Wrap launcher to inject numThreads if provided
        if numThreads is not None:
            original_launcher = launcher
            def launcher_with_threads(kernel, output_shape, output_dtype, *args):
                return original_launcher(kernel, output_shape, output_dtype, *args, numThreads=numThreads)
            launcher = launcher_with_threads

        return StateAwareWarpFunction.apply(
            build_fn, launcher, kernel, outputSizes, outputDtypes,
            *flat_tensors,
        )
