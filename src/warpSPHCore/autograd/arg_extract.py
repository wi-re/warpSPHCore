import torch
import warp as wp
from ..util import *
from .cache import *
from ..dataTypes import *
from typing import Optional, Union, Tuple
from .arg_check import *
from ..radiusSearch import buildCompactHashMap
from .stateBundle import getStateBundle

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

    This is the extraction half of what a single-shot flat-tensor builder used to
    do; the conversion half is deferred to ``StateAwareWarpFunction.forward`` so
    that the original torch tensors can be saved for autograd.

    Disabled correction paths (no CRK, no grad-h, no volumes, no renormalization,
    adjacency-vs-grid whichever is unused) are filled with a permanent ``Field``
    from the null-field registry (``util.fieldRegistry.nullField``,
    warpier_fields.md Step C) rather than a fresh dummy tensor -- the entry is
    never re-converted, since ``StateAwareWarpFunction.forward`` reads a Field's
    view directly instead of running it through ``castTorchToWarpAsBuiltins``.
    The flat list is therefore heterogeneous: ``torch.Tensor`` for real inputs,
    ``Field`` for disabled/null slots.

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
        flat_tensors  : List[torch.Tensor | Field]
        build_fn      : Callable[[List[wp.array]], tuple]  - returns full kernel args
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
            _d1f   = nullField(FieldKind.SCALAR, dim, device, dtype=torch_t)
            _d1Df  = nullField(FieldKind.VECTOR, dim, device, dtype=torch_t)
            _d1DDf = nullField(FieldKind.MATRIX, dim, device, dtype=torch_t)

            # Replace None densities with a dummy so the flat list is never sparse
            if qDen is None:
                qDen = _d1f
            if rDen is None:
                rDen = _d1f

            # Kinds -- required on ParticleState (warpier_fields.md Section 2.5);
            # no more hasattr probe or None fallback.
            queryKinds    = queryParticles.kinds
            referenceKinds = queryKinds if referenceParticles is None else referenceParticles.kinds
            operationMode = operationProperties.operationMode
            qK, rK = checkKinds(queryKinds, referenceKinds)
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
                grid_sortIndex  = nullField(FieldKind.INT64, dim, device)
                grid_qMin       = domainMin
                grid_qMax       = domainMax
                grid_hCell      = scalar_t(scalar_t(0.0))
                grid_numCells   = nullField(FieldKind.INT32, dim, device)
                grid_hashTable  = nullField(FieldKind.VEC2I, dim, device)
                grid_cellTable  = nullField(FieldKind.VEC3L, dim, device)
                grid_numOffsets = 0
                grid_cellOffsets = nullField(FieldKind.VEC3I, dim, device)
            else:
                adj_neighborList    = nullField(FieldKind.INT64, dim, device)
                adj_neighborOffsets = nullField(FieldKind.INT32, dim, device)
                adj_numNeighbors    = nullField(FieldKind.INT32, dim, device)
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
                'positiveDivergence':       positiveDivergence,
                'divergenceMode':           divergenceMode,
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
    #
    #     use_bundle=True (warpier_fields.md Step F) skips this fresh
    #     per-call construction entirely and instead refreshes a persistent
    #     StateBundle in place -- but ONLY when the caller has established
    #     nothing in this call requires grad. This is not a gating nicety:
    #     wp.Tape holds a live reference to whatever struct object a launch
    #     was given and re-reads its fields lazily at backward() time (does
    #     not snapshot values at launch time, verified directly against
    #     warp 1.16.0 -- see stateBundle.py's module docstring). Reusing a
    #     mutable struct across grad-requiring calls would silently corrupt
    #     an earlier call's gradient the moment its backward is deferred
    #     past a later call that refreshes the same bundle -- ordinary
    #     PyTorch usage, not an edge case. StateAwareWarpFunction.forward is
    #     the only caller that sets use_bundle=True, and only when
    #     `not ctx.any_requires_grad`.
    # ------------------------------------------------------------------ #

    # Pre-resolve concrete types once based on dim (Requirement 3, Section
    # 3.6: struct types come from structFor's table, never a hard-coded
    # dim==1/2/3 ternary -- Phase 6 registers *_dual rows into the same
    # table instead of every extractor needing a rewrite).
    _dim            = cfg['dim']
    _mode           = ExecutionMode.REVERSE
    _ParticleSoA    = structFor("particleDataSoA", _dim, _mode)
    _CorrData       = structFor("correctionData", _dim, _mode)

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
    _positiveDivergence          = cfg['positiveDivergence']
    _divergenceMode              = cfg['divergenceMode']
    _opInt                       = cfg['opInt']

    def build_fn(wa: list, use_bundle: bool = False) -> tuple:
        if use_bundle:
            bundle = getStateBundle(_dim, _mode)
            bundle.refresh(wa, cfg)
            return (
                bundle.queryParticle, bundle.referenceParticle, bundle.domain,
                _useAdjacency, bundle.adjacency, bundle.grid,
                bundle.correction,
                bundle.kernelProperties,
            )
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

        # Kernel-properties struct — replaces the seven flat scalars
        # (mode_uint/kernel_int/gradientMode_int/laplacianMode_int/
        # positiveDivergence_int/divergenceMode_int/opInt) that used to be
        # threaded individually through every operator kernel's ABI. See
        # dataTypes/kernelState_t.py and warpier_core.md.
        kernProps = kernelState()
        kernProps.kernelFunction        = _kernel_int
        kernProps.supportMode           = _mode_uint
        kernProps.gradientMode          = _gradientMode_int
        kernProps.laplacianMode         = _laplacianMode_int
        kernProps.positiveDivergenceMode = _positiveDivergence
        kernProps.divergenceMode        = _divergenceMode
        kernProps.operationMode         = _opInt

        return (
            qPart, rPart, domState,
            _useAdjacency, adjState, gState,
            corrState,
            kernProps,
        )

    return flat_tensors, build_fn, device, dim
