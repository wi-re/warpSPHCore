"""Geometry-tangent JVP of the Laplacian
operator's Brookshaw and Naive schemes (`warpier_tier2_operators_plan.md`
Steps 7/8, `warpier_adjoint.md` Tiers 2.2/2.3). Dot/Default `laplacianMode`s
are out of scope, enforced centrally in `operations.py`.

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`
Steps 3/4):

**Brookshaw** (`computeSPHLaplacianBrookshawGeometryJVP`): `q_ij =
(fj-fi)*B_ij` (`B` from `_jvpCommon.gradientWeightsJVP` -- literally the same
coefficient as Gradient's `B` term, not re-derived), `D_ij = r_ij +
eps*h_ij` (`eps=1e-8`, matching `wp_laplacian.py`'s literal constant),
`n_ij = x_ij/D_ij`, `L_ij = -2*q_ij*dot(G_ij,n_ij)/D_ij`; the
regularized-distance chain (`dr_ij`, `dD_ij`, `dn_ij`, `dL_ij`) is ordinary
calculus on top of `kernels.kernelJVP.sphKernelGradientJVP`'s `(G_ij,
dG_ij)`, transcribed to warp scalar/vector ops using the same
already-validated building blocks `wp_laplacian.py`'s own primal kernel uses
(`computeDistanceVec`/`safe_sqrt`/`computePairwiseSupport`), plus
`computePairwiseSupportJVP` for the tangent.

**Naive** (`computeSPHLaplacianNaiveGeometryJVP`): `q_ij` is the exact same
`B_ij` again (`wp_laplacian.py`'s `q_ij` depends only on `gradientMode`,
never `laplacianMode` -- Tier 2.2's finding, re-confirmed by Tier 2.3 under
this scheme), but `L_ij`/`dL_ij` come from `sphKernelLaplacianJVP`
(`kernels/kernelJVP.py`, the actual analytic second-derivative-of-r
estimator's own JVP) instead of Brookshaw's gradient-based `n_ij/D_ij`
estimator, `L = sum_j q_ij*L_ij`, `dL = sum_j (dq_ij*L_ij + q_ij*dL_ij)`.

`computeSPHLaplacianGeometryJVP` is the thin dispatcher `operations.py`
actually registers (`_GEOMETRY_JVP_DISPATCH[WarpOperation.Laplacian]`),
picking between the two by `laplacianMode`.

Both schemes also support grid (`CompactHashMap`) traversal, via the same
`_Func_Adjacency` dispatch every primal operator already uses. Replaced the
original pair-indexed (COO) implementations once proven numerically
equivalent to float32 round-off; see git history around 2026-08-20 for those
implementations and their own equivalence tests if reference is ever needed.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp, safe_sqrt
from ..util import allocateTorchWarp, castTorchToWarpAsBuiltins
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i
from ..util.support import computePairwiseSupport, computePairwiseSupportJVP
from ..math.wp_distance import computeDistanceVec
from ..radiusSearch.grid_util import getIndexRange
from ._jvpCommon import (
    buildParticleSoA as _buildParticleSoA,
    buildDomainState as _buildDomainState,
    buildKernelState as _buildKernelState,
    gradientWeightsJVP as _gradientWeightsJVP,
    buildAdjacencyOrGridState as _buildAdjacencyOrGridState,
    buildNullCorrectionData as _buildNullCorrectionData,
)
from ..kernels.kernelJVP import sphKernelGradientJVP, sphKernelLaplacianJVP

__all__ = [
    'computeSPHLaplacianGeometryJVP',
    'computeSPHLaplacianBrookshawGeometryJVP',
    'computeSPHLaplacianNaiveGeometryJVP',
]

_LAPLACIAN_EPS = 1e-8  # matches wp_laplacian.py's literal constant, not get_epsilon(r)


# ---------------------------------------------------------------------------
# Brookshaw CSR (per-query-particle) launch shape.
# ---------------------------------------------------------------------------

@wp.func
def computeSPHLaplacianBrookshawJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,

    fi: scalar_t, referenceValues: wp.array(dtype = scalar_t), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        jTangentPtcl = getParticleData(referenceTangentState, j)

        G, dG = sphKernelGradientJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        _A, B, _dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
        )

        fj = referenceValues[j]
        q = (fj - fi) * B
        dq = (fj - fi) * dB

        x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        dx_ij = iTangentPtcl.position - jTangentPtcl.position
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        if r_ij > scalar_t(0.0):
            dr_ij = wp.dot(x_ij, dx_ij) / r_ij
        else:
            dr_ij = scalar_t(0.0)

        h_ij = computePairwiseSupport(iPtcl.support, jPtcl.support, kernelProperties.supportMode)
        dh_ij = computePairwiseSupportJVP(
            iPtcl.support, jPtcl.support, iTangentPtcl.support, jTangentPtcl.support, kernelProperties.supportMode,
        )

        eps = scalar_t(1e-8)  # matches wp_laplacian.py's literal constant (this file's own _LAPLACIAN_EPS)
        D_ij = r_ij + eps * h_ij
        dD_ij = dr_ij + eps * dh_ij
        n_ij = x_ij / D_ij
        dn_ij = (dx_ij - n_ij * dD_ij) / D_ij

        dot_Gn = wp.dot(G, n_ij)
        d_dot_Gn = wp.dot(dG, n_ij) + wp.dot(G, dn_ij)
        P = dot_Gn / D_ij
        dP = d_dot_Gn / D_ij - dot_Gn * dD_ij / (D_ij * D_ij)

        out += -scalar_t(2.0) * (dq * P + q * dP)

    return out


@wp.func
def computeSPHLaplacianBrookshawJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHLaplacianBrookshawJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHLaplacianBrookshawJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHLaplacianBrookshawJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHLaplacianBrookshawGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
    tangentQuerySupports: Optional[torch.Tensor] = None,
    tangentReferenceSupports: Optional[torch.Tensor] = None,
    tangentReferenceMasses: Optional[torch.Tensor] = None,
    tangentQueryDensities: Optional[torch.Tensor] = None,
    tangentReferenceDensities: Optional[torch.Tensor] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dLaplacian_i`, shape `[numParticles]`, Brookshaw scheme specifically
    (see `computeSPHLaplacianNaiveGeometryJVP` for Naive; `computeSPHLaplacianGeometryJVP`
    is the dispatcher between the two that `operations.py` actually calls).

    This is the geometry/mass/density-tangent **partial** contribution to
    Laplacian's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).

    `queryValues`/`referenceValues` (`fi`/`fj`, scalar fields) are required
    and frozen here. `queryParticles.densities`/`referenceParticles.densities`
    must already hold real values, same requirement as
    `computeSPHGradientGeometryJVP`. `adjacency` is an `AdjacencyList` or
    `CompactHashMap`.
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianBrookshawGeometryJVP: queryValues and "
            "referenceValues (frozen fi/fj) are both required."
        )

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    tangentReferencePositions = tangentReferencePositions if tangentReferencePositions is not None else zerosVec(nRef)
    tangentQuerySupports = tangentQuerySupports if tangentQuerySupports is not None else zerosScalar(nQuery)
    tangentReferenceSupports = tangentReferenceSupports if tangentReferenceSupports is not None else zerosScalar(nRef)
    tangentReferenceMasses = tangentReferenceMasses if tangentReferenceMasses is not None else zerosScalar(nRef)
    tangentQueryDensities = tangentQueryDensities if tangentQueryDensities is not None else zerosScalar(nQuery)
    tangentReferenceDensities = tangentReferenceDensities if tangentReferenceDensities is not None else zerosScalar(nRef)

    queryState = _buildParticleSoA(
        dim, queryParticles.positions, queryParticles.supports, queryParticles.masses, queryParticles.densities,
    )
    referenceState = _buildParticleSoA(
        dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        referenceParticles.densities,
    )
    queryTangentState = _buildParticleSoA(
        dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery), tangentQueryDensities,
    )
    referenceTangentState = _buildParticleSoA(
        dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses, tangentReferenceDensities,
    )
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode, gradientMode=gradientMode)
    correctionData = _buildNullCorrectionData(dim, device)

    useAdjacency, adjacencyState, gridState, _numOffsets = _buildAdjacencyOrGridState(adjacency, domain)

    queryValuesWarp = castTorchToWarpAsBuiltins(queryValues.contiguous())
    referenceValuesWarp = castTorchToWarpAsBuiltins(referenceValues.contiguous())
    warpDevice = queryState.positions.device
    dLaplacian_t, dLaplacian_w = allocateTorchWarp(nQuery, queryState.masses.dtype, warpDevice)

    wp.launch(
        computeSPHLaplacianBrookshawJVP_Kernel,
        dim=nQuery,
        inputs=[
            queryState, referenceState,
            queryTangentState, referenceTangentState,
            domainState,
            useAdjacency, adjacencyState, gridState,
            correctionData,
            kernelProperties,
            queryValuesWarp, referenceValuesWarp,
            dLaplacian_w,
        ],
        device=warpDevice,
    )
    return dLaplacian_t


# ---------------------------------------------------------------------------
# Naive CSR (per-query-particle) launch shape. Reuses sphKernelLaplacianJVP
# (the analytic second-derivative-of-r estimator's own JVP) directly instead
# of Brookshaw's gradient-based n_ij/D_ij estimator -- q_ij is the same
# B-only gradientWeightsJVP coefficient as Brookshaw, per this file's own
# module docstring finding.
# ---------------------------------------------------------------------------


@wp.func
def computeSPHLaplacianNaiveJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,

    fi: scalar_t, referenceValues: wp.array(dtype = scalar_t), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        jTangentPtcl = getParticleData(referenceTangentState, j)

        L, dL = sphKernelLaplacianJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        _A, B, _dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
        )

        fj = referenceValues[j]
        q = (fj - fi) * B
        dq = (fj - fi) * dB

        out += dq * L + q * dL

    return out


@wp.func
def computeSPHLaplacianNaiveJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHLaplacianNaiveJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHLaplacianNaiveJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = scalar_t), # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHLaplacianNaiveJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHLaplacianNaiveGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
    tangentQuerySupports: Optional[torch.Tensor] = None,
    tangentReferenceSupports: Optional[torch.Tensor] = None,
    tangentReferenceMasses: Optional[torch.Tensor] = None,
    tangentQueryDensities: Optional[torch.Tensor] = None,
    tangentReferenceDensities: Optional[torch.Tensor] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dLaplacian_i`, shape `[numParticles]`, Naive scheme
    (`warpier_tier2_operators_plan.md` Step 8, `warpier_adjoint.md` Tier
    2.3): `q_ij = (fj-fi)*B_ij` (same `B` as Brookshaw, `gradientMode`-
    dispatched, not `laplacianMode`-dispatched -- Tier 2.2's finding,
    re-confirmed here), `L = sum_j q_ij*L_ij`, `dL = sum_j (dq_ij*L_ij +
    q_ij*dL_ij)`, `(L_ij, dL_ij)` from `sphKernelLaplacianJVP`
    (`kernels/kernelJVP.py`). `adjacency` is an `AdjacencyList` or
    `CompactHashMap`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Laplacian's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`)."""
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianNaiveGeometryJVP: queryValues and "
            "referenceValues (frozen fi/fj) are both required."
        )

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    tangentReferencePositions = tangentReferencePositions if tangentReferencePositions is not None else zerosVec(nRef)
    tangentQuerySupports = tangentQuerySupports if tangentQuerySupports is not None else zerosScalar(nQuery)
    tangentReferenceSupports = tangentReferenceSupports if tangentReferenceSupports is not None else zerosScalar(nRef)
    tangentReferenceMasses = tangentReferenceMasses if tangentReferenceMasses is not None else zerosScalar(nRef)
    tangentQueryDensities = tangentQueryDensities if tangentQueryDensities is not None else zerosScalar(nQuery)
    tangentReferenceDensities = tangentReferenceDensities if tangentReferenceDensities is not None else zerosScalar(nRef)

    queryState = _buildParticleSoA(
        dim, queryParticles.positions, queryParticles.supports, queryParticles.masses, queryParticles.densities,
    )
    referenceState = _buildParticleSoA(
        dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        referenceParticles.densities,
    )
    queryTangentState = _buildParticleSoA(
        dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery), tangentQueryDensities,
    )
    referenceTangentState = _buildParticleSoA(
        dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses, tangentReferenceDensities,
    )
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode, gradientMode=gradientMode)
    correctionData = _buildNullCorrectionData(dim, device)

    useAdjacency, adjacencyState, gridState, _numOffsets = _buildAdjacencyOrGridState(adjacency, domain)

    queryValuesWarp = castTorchToWarpAsBuiltins(queryValues.contiguous())
    referenceValuesWarp = castTorchToWarpAsBuiltins(referenceValues.contiguous())
    warpDevice = queryState.positions.device
    dLaplacian_t, dLaplacian_w = allocateTorchWarp(nQuery, queryState.masses.dtype, warpDevice)

    wp.launch(
        computeSPHLaplacianNaiveJVP_Kernel,
        dim=nQuery,
        inputs=[
            queryState, referenceState,
            queryTangentState, referenceTangentState,
            domainState,
            useAdjacency, adjacencyState, gridState,
            correctionData,
            kernelProperties,
            queryValuesWarp, referenceValuesWarp,
            dLaplacian_w,
        ],
        device=warpDevice,
    )
    return dLaplacian_t


def computeSPHLaplacianGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    tangentQueryPositions: torch.Tensor,
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw,
    **kwargs,
) -> torch.Tensor:
    """Dispatcher `operations.py` actually registers in `_GEOMETRY_JVP_DISPATCH`
    -- routes to `computeSPHLaplacianBrookshawGeometryJVP`/
    `computeSPHLaplacianNaiveGeometryJVP` by `laplacianMode`. Dot/Default are
    rejected before reaching here (`operations.py`'s own centralized scope
    check); any other value is a defensive fallback, not expected to be
    reachable."""
    if laplacianMode is LaplacianScheme.Brookshaw:
        return computeSPHLaplacianBrookshawGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, tangentQueryPositions, **kwargs,
        )
    elif laplacianMode is LaplacianScheme.Naive:
        return computeSPHLaplacianNaiveGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, tangentQueryPositions, **kwargs,
        )
    raise NotImplementedError(
        f"computeSPHLaplacianGeometryJVP: geometry JVP laplacianMode={laplacianMode} is not "
        "implemented -- only Brookshaw and Naive are (warpier_tier2_operators_plan.md "
        "Steps 7/8)."
    )
