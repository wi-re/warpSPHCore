"""Shared boilerplate for the Tier-2 JVP kernels
(`warpier_tier2_operators_plan.md` Steps 0/3, `warpier_tier2_jvp_csr_backend_plan.md`):
SoA/domain/kernel-state builders shared by every `wp_<op>JVP.py`, the
CSR-port launch helpers (`buildAdjacencyOrGridState`/`buildNullCorrectionData`/
`gradientWeightsJVP`), and `launchGeometryJVP`
(`warpier_tier2_jvp_reverse_mode_plan.md`) -- the autograd-bridged launcher
that replaced every `wp_<op>JVP.py`'s own bare `wp.launch` once the CSR port
made the Tier-2 JVP kernels' ABI match `extractStateInfo`'s own
per-query-particle convention closely enough that a dedicated `build_fn`
(rather than a generalization of `extractStateInfo` itself, which stays
untouched) was all reverse-mode differentiability needed.
"""

from typing import Any, Optional, Union
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, castTorchToWarpAsBuiltins, allocateTorchWarp
from ..enumTypes import supportSchemeToUint
from ..autograd import StateAwareWarpFunction, launch_kernel

__all__ = [
    'buildParticleSoA', 'buildDomainState', 'buildKernelState',
    'buildAdjacencyOrGridState', 'buildNullCorrectionData', 'gradientWeightsJVP',
    'launchGeometryJVP',
]

_SoA_BY_DIM = {1: particleDataSoA_1, 2: particleDataSoA_2, 3: particleDataSoA_3}


def buildParticleSoA(
    dim: int,
    positions: torch.Tensor,
    supports: torch.Tensor,
    masses: torch.Tensor,
    densities: Optional[torch.Tensor] = None,
):
    SoA = _SoA_BY_DIM[dim]()
    SoA.positions = castTorchToWarpAsBuiltins(positions.contiguous())
    SoA.supports = castTorchToWarp(supports.contiguous())
    SoA.masses = castTorchToWarp(masses.contiguous())
    n = positions.shape[0]
    if densities is None:
        densities = torch.zeros(n, device=positions.device, dtype=positions.dtype)
    SoA.densities = castTorchToWarp(densities.contiguous())
    SoA.kinds = castTorchToWarp(torch.zeros(n, device=positions.device, dtype=torch.int32))
    return SoA


def buildDomainState(domain: DomainDescription) -> domainData:
    d = domainData()
    d.domainMin = castTorchToWarp(domain.min)
    d.domainMax = castTorchToWarp(domain.max)
    d.periodicity = castTorchToWarp(domain.periodic)
    d.dim = domain.dim
    return d


def buildKernelState(
    kernel: KernelFunctions, supportMode: SupportScheme,
    gradientMode: Optional[GradientScheme] = None,
    laplacianMode: Optional[LaplacianScheme] = None,
) -> kernelState:
    k = kernelState()
    k.kernelFunction = kernel.value
    k.supportMode = supportSchemeToUint(supportMode)
    # gradientMode/laplacianMode default to 0, which matches neither GradientScheme
    # nor LaplacianScheme's enum values (both start at 1) -- CSR JVP kernels that
    # branch on kernelProperties.gradientMode/.laplacianMode (Gradient/Divergence/
    # Curl/Laplacian) need these actually set; the COO path never reads them from
    # this struct at all (its coefficient math is pure torch), so this was never
    # needed before the CSR port (warpier_tier2_jvp_csr_backend_plan.md).
    if gradientMode is not None:
        k.gradientMode = gradientMode.value
    if laplacianMode is not None:
        k.laplacianMode = laplacianMode.value
    return k


@wp.func
def gradientWeightsJVP(
    massJ: scalar_t, densityI: scalar_t, densityJ: scalar_t,
    dMassJ: scalar_t, dDensityI: scalar_t, dDensityJ: scalar_t,
    gradientMode: wp.int32,
):
    """`coeff_ij = fi*A_ij + fj*B_ij` (`fi`/`fj` frozen -- Tier 1 territory,
    contribute no term of their own here), `A`/`B`/`dA`/`dB` per
    `warpier_adjoint.md` Tier 2.2 -- the CSR JVP kernels' shared `(A, B, dA,
    dB)` building block (`warpier_tier2_jvp_csr_backend_plan.md` Step 3: "the
    shared piece that survives is the per-pair building block ... called
    from inside seven distinct per-query kernels"), used by
    Gradient/Divergence/Curl/Laplacian(Brookshaw/Naive)'s CSR kernels to
    combine with their own operator-specific `fi`/`fj` weighting.
    """
    Vj = massJ / densityJ
    dVj = dMassJ / densityJ - massJ * dDensityJ / (densityJ * densityJ)

    A = scalar_t(0.0)
    dA = scalar_t(0.0)
    B = scalar_t(0.0)
    dB = scalar_t(0.0)
    if gradientMode == wp.static(GradientScheme.Naive.value):
        B = Vj
        dB = dVj
    elif gradientMode == wp.static(GradientScheme.Difference.value):
        A = -Vj
        dA = -dVj
        B = Vj
        dB = dVj
    elif gradientMode == wp.static(GradientScheme.Summation.value):
        A = Vj
        dA = dVj
        B = Vj
        dB = dVj
    elif gradientMode == wp.static(GradientScheme.Symmetric.value):
        A = massJ / densityI
        dA = dMassJ / densityI - massJ * dDensityI / (densityI * densityI)
        B = massJ * densityI / (densityJ * densityJ)
        dB = (dMassJ * densityI / (densityJ * densityJ)
              + massJ * dDensityI / (densityJ * densityJ)
              - scalar_t(2.0) * massJ * densityI * dDensityJ / (densityJ * densityJ * densityJ))
    return A, B, dA, dB


_CORRECTION_BY_DIM = {1: correctionData_1, 2: correctionData_2, 3: correctionData_3}


def buildNullCorrectionData(dim: int, device: torch.device) -> Any:
    """A `correctionData_{dim}` struct with every `useX` flag `False` and
    size-1 zero-filled arrays behind them -- the CSR JVP kernels' `_Func_i`/
    `_Func_Adjacency` accept `correctionData`/`iCorrectionData` purely to
    match the canonical structured kernel ABI's parameter list (Tier-2 JVP
    has no CRK/renorm/grad-h support, enforced centrally in
    `operations.py`'s `warpOperationJVP`), so the arrays behind the flags
    are never read -- mirrors `extractStateInfo`'s own disabled-correction
    path (`nullField`) without pulling in that machinery's tensor-identity
    caching, which a one-shot manual launch like this has no use for.
    """
    torchDtype = get_torch_precision()
    zeroScalar = lambda: castTorchToWarpAsBuiltins(torch.zeros(1, device=device, dtype=torchDtype))
    zeroVec = lambda: castTorchToWarpAsBuiltins(torch.zeros((1, dim), device=device, dtype=torchDtype))
    zeroMat = lambda: castTorchToWarpAsBuiltins(torch.zeros((1, dim, dim), device=device, dtype=torchDtype))

    corrState = _CORRECTION_BY_DIM[dim]()
    corrState.useGradientRenormalization = False
    corrState.renormalizationMatrices = zeroMat()
    corrState.useVolume = False
    corrState.queryVolumes = zeroScalar()
    corrState.referenceVolumes = zeroScalar()
    corrState.useGradHTerms = False
    corrState.queryOmegas = zeroScalar()
    corrState.referenceOmegas = zeroScalar()
    corrState.useCRK = False
    corrState.queryA = zeroScalar()
    corrState.queryB = zeroVec()
    corrState.queryGradA = zeroVec()
    corrState.queryGradB = zeroMat()
    corrState.referenceA = zeroScalar()
    corrState.referenceB = zeroVec()
    corrState.referenceGradA = zeroVec()
    corrState.referenceGradB = zeroMat()
    return corrState


def buildAdjacencyOrGridState(
    adjacency: Union[AdjacencyList, CompactHashMap],
    domain: 'DomainDescription',
):
    """Convert a torch-facing `AdjacencyList` (CSR-capable already --
    `.j`/`.edgeOffsets`/`.numNeighbors`, see `dataTypes/adjacency_t.py`'s own
    docstring) or `CompactHashMap` into the warp-side `(adjacencyData,
    gridData)` struct pair `getIndexRange` consumes, plus the `useAdjacency`/
    `numOffsets` pair driving `_Func_Adjacency`'s dispatch -- the same
    conversion `extractStateInfo` performs (`autograd/arg_extract.py`
    Section 4) for `launchOperator`, done by hand here since a hand-launched
    JVP kernel bypasses that machinery entirely (see this module's
    docstring). The branch not in use is filled with size-1 zero dummies
    (adjacency's own `domain.min`/`domain.max` stand in for the unused grid
    qMin/qMax, matching `extractStateInfo`'s own choice) -- never read, since
    `getIndexRange` only touches one branch depending on `useAdjacency`.
    """
    device = domain.min.device
    dim = domain.dim

    adjState = adjacencyData()
    gState = gridData()

    if isinstance(adjacency, CompactHashMap):
        useAdjacency = False
        adjState.neighborList = castTorchToWarp(torch.zeros(1, device=device, dtype=torch.int64))
        adjState.neighborOffsets = castTorchToWarp(torch.zeros(1, device=device, dtype=torch.int32))
        adjState.numNeighbors = castTorchToWarp(torch.zeros(1, device=device, dtype=torch.int32))

        gState.sortIndex = castTorchToWarp(adjacency.sortIndex)
        gState.qMin = castTorchToWarpAsBuiltins(adjacency.qMin)
        gState.qMax = castTorchToWarpAsBuiltins(adjacency.qMax)
        gState.hCell = scalar_t(adjacency.hCell)
        gState.numCells = castTorchToWarp(adjacency.numCells)
        gState.hashTable = castTorchToWarpAsBuiltins(adjacency.hashTable)
        gState.cellTable = castTorchToWarpAsBuiltins(adjacency.sortedCellTable)
        gState.D = adjacency.D
        gState.numOffsets = adjacency.numOffsets
        gState.cellOffsets = castTorchToWarpAsBuiltins(adjacency.cellOffsets)
        numOffsets = adjacency.numOffsets
    elif isinstance(adjacency, AdjacencyList):
        useAdjacency = True
        adjState.neighborList = castTorchToWarp(adjacency.j)
        adjState.neighborOffsets = castTorchToWarp(adjacency.edgeOffsets)
        adjState.numNeighbors = castTorchToWarp(adjacency.numNeighbors)

        gState.sortIndex = castTorchToWarp(torch.zeros(1, device=device, dtype=torch.int64))
        gState.qMin = castTorchToWarpAsBuiltins(domain.min)
        gState.qMax = castTorchToWarpAsBuiltins(domain.max)
        gState.hCell = scalar_t(0.0)
        gState.numCells = castTorchToWarp(torch.zeros(dim, device=device, dtype=torch.int32))
        gState.hashTable = castTorchToWarpAsBuiltins(torch.zeros((1, 2), device=device, dtype=torch.int32))
        gState.cellTable = castTorchToWarpAsBuiltins(torch.zeros((1, 3), device=device, dtype=torch.int64))
        gState.D = dim
        gState.numOffsets = 0
        gState.cellOffsets = castTorchToWarpAsBuiltins(torch.zeros((1, 3), device=device, dtype=torch.int32))
        numOffsets = 1
    else:
        raise NotImplementedError(
            "buildAdjacencyOrGridState: adjacency must be an AdjacencyList or "
            f"CompactHashMap, got {type(adjacency)}."
        )
    return useAdjacency, adjState, gState, numOffsets


def launchGeometryJVP(
    kernel: Any,
    domain: 'DomainDescription',
    kernelFn: 'KernelFunctions',
    supportMode: 'SupportScheme',
    adjacency: Union['AdjacencyList', 'CompactHashMap'],

    queryPositions: torch.Tensor, querySupports: torch.Tensor, queryMasses: torch.Tensor,
    referencePositions: torch.Tensor, referenceSupports: torch.Tensor, referenceMasses: torch.Tensor,

    tangentQueryPositions: torch.Tensor, tangentQuerySupports: torch.Tensor, tangentQueryMasses: torch.Tensor,
    tangentReferencePositions: torch.Tensor, tangentReferenceSupports: torch.Tensor, tangentReferenceMasses: torch.Tensor,

    outputShape: int,
    outputDtype: Any,

    queryDensities: Optional[torch.Tensor] = None,
    referenceDensities: Optional[torch.Tensor] = None,
    tangentQueryDensities: Optional[torch.Tensor] = None,
    tangentReferenceDensities: Optional[torch.Tensor] = None,

    gradientMode: Optional['GradientScheme'] = None,
    laplacianMode: Optional['LaplacianScheme'] = None,

    extraTensors: tuple = (),
) -> torch.Tensor:
    """Autograd-bridged launcher for every Tier-2 JVP kernel's shared CSR
    argument order (`queryState, referenceState, queryTangentState,
    referenceTangentState, domainState, useAdjacency, adjacencyState,
    gridState, correctionData, kernelProperties[, *extraTensors],
    outputValues`) -- routes through `StateAwareWarpFunction` (the same
    bridge every primal operator reaches via `launchOperator`/`_launch`)
    instead of a bare `wp.launch`, so gradients flow back through
    `positions`/`supports`/`masses`/`densities` on both the query and
    reference roles, their tangent counterparts, and any `extraTensors`
    (e.g. Gradient/Divergence/Curl/Laplacian's frozen `queryValues`/
    `referenceValues`) -- closing the reverse-mode gap
    `warpier_tier2_jvp_reverse_mode_plan.md` documents.

    `domainState`/`correctionData`/`adjacencyState`/`gridState`/
    `kernelProperties`/particle `kinds` are built once per call and closed
    over by `build_fn` rather than threaded through `flat_tensors`: none of
    them are differentiable (indices, domain bounds, disabled-correction
    flags), so there is nothing for the bridge to track for them -- only the
    16 particle-state tensors (8 primal + 8 tangent) and any `extraTensors`
    participate in `StateAwareWarpFunction`'s autograd bookkeeping.
    """
    device = queryPositions.device
    dtype = queryPositions.dtype
    dim = domain.dim
    nQuery = queryPositions.shape[0]
    nRef = referencePositions.shape[0]

    if queryDensities is None:
        queryDensities = torch.zeros(nQuery, device=device, dtype=dtype)
    if referenceDensities is None:
        referenceDensities = torch.zeros(nRef, device=device, dtype=dtype)
    if tangentQueryDensities is None:
        tangentQueryDensities = torch.zeros(nQuery, device=device, dtype=dtype)
    if tangentReferenceDensities is None:
        tangentReferenceDensities = torch.zeros(nRef, device=device, dtype=dtype)

    domainState = buildDomainState(domain)
    kernelProperties = buildKernelState(kernelFn, supportMode, gradientMode=gradientMode, laplacianMode=laplacianMode)
    correctionData = buildNullCorrectionData(dim, device)
    useAdjacency, adjacencyState, gridState, _numOffsets = buildAdjacencyOrGridState(adjacency, domain)

    qKinds = castTorchToWarp(torch.zeros(nQuery, device=device, dtype=torch.int32))
    rKinds = castTorchToWarp(torch.zeros(nRef, device=device, dtype=torch.int32))

    _ParticleSoA = _SoA_BY_DIM[dim]

    flat_tensors = [
        queryPositions, referencePositions,               # 0-1
        querySupports, referenceSupports,                  # 2-3
        queryMasses, referenceMasses,                       # 4-5
        queryDensities, referenceDensities,                 # 6-7
        tangentQueryPositions, tangentReferencePositions,   # 8-9
        tangentQuerySupports, tangentReferenceSupports,     # 10-11
        tangentQueryMasses, tangentReferenceMasses,         # 12-13
        tangentQueryDensities, tangentReferenceDensities,   # 14-15
    ] + list(extraTensors)
    n_extra = len(extraTensors)

    def build_fn(wa: list, use_bundle: bool = False) -> tuple:
        qPart = _ParticleSoA()
        qPart.positions, qPart.supports = wa[0], wa[2]
        qPart.masses, qPart.densities, qPart.kinds = wa[4], wa[6], qKinds

        rPart = _ParticleSoA()
        rPart.positions, rPart.supports = wa[1], wa[3]
        rPart.masses, rPart.densities, rPart.kinds = wa[5], wa[7], rKinds

        qTangent = _ParticleSoA()
        qTangent.positions, qTangent.supports = wa[8], wa[10]
        qTangent.masses, qTangent.densities, qTangent.kinds = wa[12], wa[14], qKinds

        rTangent = _ParticleSoA()
        rTangent.positions, rTangent.supports = wa[9], wa[11]
        rTangent.masses, rTangent.densities, rTangent.kinds = wa[13], wa[15], rKinds

        extra = tuple(wa[16 + i] for i in range(n_extra))

        return (
            qPart, rPart, qTangent, rTangent, domainState,
            useAdjacency, adjacencyState, gridState,
            correctionData, kernelProperties,
        ) + extra

    return StateAwareWarpFunction.apply(
        build_fn, launch_kernel, kernel, outputShape, outputDtype, *flat_tensors,
    )
