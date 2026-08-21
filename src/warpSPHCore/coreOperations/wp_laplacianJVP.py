"""Geometry-tangent JVP of the Laplacian
operator's Brookshaw, Naive, Dot, and Default schemes (`warpier_tier2_operators_plan.md`
Steps 7/8, `warpier_adjoint.md` Tiers 2.2/2.3, and Tier 2.2's own "Dot/Default deferred"
note -- since resolved, see `computeSPHLaplacianDotGeometryJVP`/
`computeSPHLaplacianDefaultGeometryJVP` below).

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

**Dot** (`computeSPHLaplacianDotGeometryJVP`, `math/wp_laplaciandot.py`'s
`computeLaplacianDot2`, DJ Price SPH/MHD eq 96): unlike Brookshaw/Naive
(scalar-field only, `queryValues`/`referenceValues` are plain `scalar_t`
arrays), Dot's own forward formula only makes sense for a field whose
flattened per-particle size is a multiple of the spatial dimension (it
projects `dim`-sized blocks of the field against `n_ij`), enforced by
`wp_laplacian.py`'s own `ValueError` for scalar fields in >1D -- so this
scheme's `queryValues`/`referenceValues`/output are generic `Any` (vector- or
scalar-dtype, following `computeSPHInterpolateGeometryJVP`'s precedent, not
Brookshaw/Naive's fixed `scalar_t`), and the same shape restriction is
re-enforced here. Reuses `_laplacianGeometryChainJVP`/`_laplacianPJVP` below
for `(G, dG, n_ij, dn_ij, D_ij, dD_ij)` and `F_ab = P` (Dot's own `r_eps`
division is bit-for-bit Brookshaw's `P = dot(G,n_ij)/D_ij`, both built from
the same `n_ij`/`D_ij` with `eps=1e-8`) -- the only genuinely new math is the
per-`dim`-block projection `proj_b = dot(q_ij[block], n_ij)` and its tangent
`dproj_b`, differentiated by ordinary product rule alongside `F_ab`/`dF_ab`.

**Reverse-mode adjoint bug -- FOUND AND FIXED 2026-08-20, same day.**
`computeSPHLaplacianDotJVP_Func_i`'s own automatic (`wp.Tape`) reverse-mode
adjoint was initially wrong -- caught by `torch.autograd.gradcheck` failing
on `computeSPHLaplacianDotGeometryJVP` specifically. Root cause: `proj =
dot(q_ij[block], n_ij)`, a loop-accumulated local, was then consumed by a
further non-linear op (`proj * n_ij[d]`) in the *same* function body -- a
known Warp code-generation footgun in this repo (`docs/lessons_learned.md`'s
"Warp kernel authoring gotchas"), also hit by `math/wp_laplaciandot.py`'s
`computeLaplacianDot2` (the forward/primal Dot kernel this JVP
differentiates). Fixed the established way: the reduction loop was moved
into its own `@wp.func` (`_laplacianDotProjJVP` below) that *returns* the
accumulated `(proj, dproj)` rather than leaving them as locals reused later
in the caller -- confirmed fixed by `torch.autograd.gradcheck` (now green,
`scripts/gradcheck_tier2_jvp_laplacian.py`) and by finite differences
agreeing with the jacobian-based test reference
(`tests/operations/test_forward_mode_geometry_jvp_laplacian_dot.py`).

**Default** (`computeSPHLaplacianDefaultGeometryJVP`, `computeDotLaplacian`):
also generic `Any`-typed (no `dim`-block restriction -- `computeDotLaplacian`
has no block indexing at all, it's a plain elementwise scalar-times-vector
form), and turns out to be Brookshaw's exact `-2*q_ij*P`-shaped formula one
quotient-rule level deeper: `computeDotLaplacian` internally re-divides the
already-regularized `n_ij` by a *second* regularized distance `D2_ij = r_ij +
1e-12*h_ij` (a different literal `eps` than `D_ij`'s `1e-8`, matching
`math/wp_laplaciandot.py`'s own constant exactly rather than approximating
it away), broadcasting the resulting scalar `dot(n_ij2, G)` across every
field component. `dn_ij2`/`dD2_ij` follow the identical quotient-rule pattern
`_laplacianGeometryChainJVP` already uses one level up for `n_ij`/`D_ij`.

`computeSPHLaplacianGeometryJVP` is the thin dispatcher `operations.py`
actually registers (`_GEOMETRY_JVP_DISPATCH[WarpOperation.Laplacian]`),
picking between the four by `laplacianMode`.

All four schemes also support grid (`CompactHashMap`) traversal, via the same
`_Func_Adjacency` dispatch every primal operator already uses. Brookshaw/Naive
replaced their original pair-indexed (COO) implementations once proven
numerically equivalent to float32 round-off; see git history around
2026-08-20 for those implementations and their own equivalence tests if
reference is ever needed.
"""

from typing import Any, Optional
import torch
import warp as wp
from warp.types import vector

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp, safe_sqrt
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i, getParticleCorrectionTangentData_i
from ..util import castTorchToWarpAsBuiltins
from ..util.support import computePairwiseSupport, computePairwiseSupportJVP
from ..math.wp_distance import computeDistanceVec
from ..radiusSearch.grid_util import getIndexRange
from ._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)
from ..kernels.kernelJVP import sphKernelGradientJVP, sphKernelLaplacianJVP

__all__ = [
    'computeSPHLaplacianGeometryJVP',
    'computeSPHLaplacianBrookshawGeometryJVP',
    'computeSPHLaplacianNaiveGeometryJVP',
    'computeSPHLaplacianDotGeometryJVP',
    'computeSPHLaplacianDefaultGeometryJVP',
]

_LAPLACIAN_EPS = 1e-8  # matches wp_laplacian.py's literal constant, not get_epsilon(r)
_LAPLACIAN_DOT2_EPS = 1e-12  # matches math/wp_laplaciandot.py's computeDotLaplacian's n_ij2 divisor


@wp.func
def _laplacianGeometryChainJVP(
    iPtcl: Any, jPtcl: Any, iTangentPtcl: Any, jTangentPtcl: Any,
    kernelProperties: kernelState, domainState: domainData,
):
    """Shared `(G, dG, n_ij, dn_ij, D_ij, dD_ij, r_ij, dr_ij, h_ij, dh_ij)`
    regularized-distance chain -- `wp_laplacian.py`'s own comment: "the
    Brookshaw/Dot/Default laplacian schemes below all take the form Sum_j
    K_ij*q_ij" where `K_ij` is built from this same `(G, n_ij, D_ij)` triple,
    just combined differently per scheme. Derived once here (was inlined in
    Brookshaw's `_Func_i` before Dot/Default needed the identical chain) --
    same math, zero behavior change for Brookshaw."""
    G, dG = sphKernelGradientJVP(
        iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
        iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
        kernelProperties, domainState,
    )

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

    return G, dG, n_ij, dn_ij, D_ij, dD_ij, r_ij, dr_ij, h_ij, dh_ij


@wp.func
def _laplacianPJVP(
    G: vector(dtype=scalar_t, length=dim_t), dG: vector(dtype=scalar_t, length=dim_t), # type: ignore
    n_ij: vector(dtype=scalar_t, length=dim_t), dn_ij: vector(dtype=scalar_t, length=dim_t), # type: ignore
    D_ij: scalar_t, dD_ij: scalar_t,
):
    """`P = dot(G,n_ij)/D_ij`, `dP` by the ordinary quotient rule -- Brookshaw's
    own `L_ij = -2*q_ij*P` weighting, and (bit-for-bit, since both are built
    from the same `n_ij`/`D_ij` with `eps=1e-8`) Dot's `F_ab` (`math/
    wp_laplaciandot.py`'s `computeLaplacianDot2`)."""
    dot_Gn = wp.dot(G, n_ij)
    d_dot_Gn = wp.dot(dG, n_ij) + wp.dot(G, dn_ij)
    P = dot_Gn / D_ij
    dP = d_dot_Gn / D_ij - dot_Gn * dD_ij / (D_ij * D_ij)
    return P, dP


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
    iCorrectionTangentData: Any, correctionTangentData: Any,

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

        G, dG, n_ij, dn_ij, D_ij, dD_ij, _r_ij, _dr_ij, _h_ij, _dh_ij = _laplacianGeometryChainJVP(
            iPtcl, jPtcl, iTangentPtcl, jTangentPtcl, kernelProperties, domainState,
        )
        P, dP = _laplacianPJVP(G, dG, n_ij, dn_ij, D_ij, dD_ij)

        _A, B, _dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        fj = referenceValues[j]
        q = (fj - fi) * B
        dq = (fj - fi) * dB

        out += -scalar_t(2.0) * (dq * P + q * dP)

    return out


@wp.func
def computeSPHLaplacianBrookshawJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
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
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

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
            iCorrectionTangentData, correctionTangentData,

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
    correctionData: Any, correctionTangentData: Any,

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
        correctionData, correctionTangentData, domainState,
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
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
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

    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )

    return _launchGeometryJVP(
        computeSPHLaplacianBrookshawJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=scalar_t,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        extraTensors=(queryValues, referenceValues),
    )


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
    iCorrectionTangentData: Any, correctionTangentData: Any,

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
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
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
    correctionData: Any, correctionTangentData: Any,
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
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

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
            iCorrectionTangentData, correctionTangentData,

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
    correctionData: Any, correctionTangentData: Any,

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
        correctionData, correctionTangentData, domainState,
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
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
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

    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )

    return _launchGeometryJVP(
        computeSPHLaplacianNaiveJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=scalar_t,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        extraTensors=(queryValues, referenceValues),
    )


# ---------------------------------------------------------------------------
# Dot CSR (per-query-particle) launch shape. Generic `Any`-typed fields
# (unlike Brookshaw/Naive's fixed scalar_t) -- Dot's own forward formula
# (math/wp_laplaciandot.py's computeLaplacianDot2) needs a field whose
# flattened per-particle size is a multiple of the spatial dimension, so a
# genuinely scalar field is already out of scope for this scheme in >1D
# (wp_laplacian.py's own ValueError), re-enforced in the public wrapper below.
# ---------------------------------------------------------------------------

@wp.func
def _laplacianDotProjJVP(
    q_ij: vector(dtype = scalar_t, length=Any), dq_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    n_ij: vector(dtype=scalar_t, length=dim_t), dn_ij: vector(dtype=scalar_t, length=dim_t), # type: ignore
    dim: wp.int32, base: wp.int32,
):
    """`proj_b`/`dproj_b`'s own reduction loop, factored into a dedicated
    `@wp.func` that *returns* the accumulated value rather than leaving it as
    a local later multiplied against `n_ij`/`dn_ij` again in the caller's own
    body -- a loop-accumulated value consumed by a further non-linear op
    (here, `proj * n_ij[d]`) in the *same* function silently drops part of
    its reverse-mode adjoint contribution in this Warp version. This exact
    class of bug has surfaced before in this repo (`docs/lessons_learned.md`'s
    "Warp kernel authoring gotchas") and moving the loop into its own
    returning function is the established fix -- confirmed here too: forward
    value unchanged, reverse-mode adjoint now matches finite differences
    (previously off by a clean factor of ~2, both here and in
    `math/wp_laplaciandot.py`'s `computeLaplacianDot2`, which has the same
    fix applied for the same reason)."""
    proj = scalar_t(0.0)
    dproj = scalar_t(0.0)
    for c in range(dim):
        proj += q_ij[base + c] * n_ij[c]
        dproj += dq_ij[base + c] * n_ij[c] + q_ij[base + c] * dn_ij[c]
    return proj, dproj


@wp.func
def computeSPHLaplacianDotJVP_Func_i(
    i: wp.int32, dim: wp.int32, flatInputShape: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

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

        G, dG, n_ij, dn_ij, D_ij, dD_ij, _r_ij, _dr_ij, _h_ij, _dh_ij = _laplacianGeometryChainJVP(
            iPtcl, jPtcl, iTangentPtcl, jTangentPtcl, kernelProperties, domainState,
        )
        # F_ab is computeLaplacianDot2's own name for this quantity -- bit-for-bit
        # Brookshaw's P (same n_ij/D_ij, same eps=1e-8).
        F_ab, dF_ab = _laplacianPJVP(G, dG, n_ij, dn_ij, D_ij, dD_ij)

        _A, B, _dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        fj = referenceValues[j]
        q_ij = (fj - fi) * B
        dq_ij = (fj - fi) * dB

        # computeLaplacianDot2's two accumulation loops collapse to
        # output[k] = -left*F_ab + q_ij[k]*F_ab, left = (dim+2)*proj_b*n_ij[d],
        # proj_b = dot(q_ij[block b], n_ij), d = k%dim, b = k//dim -- see
        # math/wp_laplaciandot.py's own derivation comment (DJ Price eq 96).
        # flatInputShape is threaded explicitly rather than read via q_ij.length
        # as the loop bound -- see launchGeometryJVP's extraScalars docstring
        # (math/wp_distance.py's minimumImageDistance has the same guard, for
        # the same documented Warp .length-as-loop-bound footgun).
        contribution = zero_like_warp(outputValue)
        for k in range(flatInputShape):
            d = k % dim
            b = k // dim
            base = b * dim

            proj, dproj = _laplacianDotProjJVP(q_ij, dq_ij, n_ij, dn_ij, dim, base)

            left = scalar_t(dim + 2) * proj * n_ij[d]
            dleft = scalar_t(dim + 2) * (dproj * n_ij[d] + proj * dn_ij[d])

            contribution[k] += -dleft * F_ab - left * dF_ab + dq_ij[k] * F_ab + q_ij[k] * dF_ab

        out += contribution

    return out


@wp.func
def computeSPHLaplacianDotJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32, flatInputShape: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHLaplacianDotJVP_Func_i(
            i, dim, flatInputShape,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            iCorrectionTangentData, correctionTangentData,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHLaplacianDotJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore
    flatInputShape: wp.int32,

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHLaplacianDotJVP_Func_Adjacency(
        i, domainState.dim, flatInputShape,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHLaplacianDotGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dLaplacian_i`, shape `queryValues.shape`, Dot scheme
    (`math/wp_laplaciandot.py`'s `computeLaplacianDot2`, DJ Price SPH/MHD eq
    96): `q_ij = (fj-fi)*B_ij` (same `B` as Brookshaw/Naive), `F_ab =
    dot(n_ij,G_ij)/D_ij` (bit-for-bit Brookshaw's `P`), `proj_b =
    dot(q_ij[block b], n_ij)` for each `dim`-sized block `b` of the
    (flattened) field, `Laplacian_i[k] = -(dim+2)*proj_b*n_ij[k%dim]*F_ab +
    q_ij[k]*F_ab`; `dLaplacian_i` by the ordinary product rule through every
    factor. `queryValues`/`referenceValues` (`fi`/`fj`) must have a flattened
    per-particle size that is a multiple of `domain.dim` when `domain.dim >
    1` (matching `wp_laplacian.py`'s own restriction for this scheme -- a
    plain scalar field does not have blocks to project). `adjacency` is an
    `AdjacencyList` or `CompactHashMap`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Laplacian's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianDotGeometryJVP: queryValues and "
            "referenceValues (frozen fi/fj) are both required."
        )

    dim = domain.dim
    inputShape = queryValues.shape[1:]
    flatInputShape = 1
    for d in inputShape:
        flatInputShape *= d
    if dim > 1 and flatInputShape % dim != 0:
        raise ValueError(
            f"computeSPHLaplacianDotGeometryJVP: LaplacianScheme.Dot assumes the field's "
            f"flattened size is a multiple of the spatial dimension ({dim}) -- it indexes "
            f"q_ij[block*dim + k] for k in range(dim), which reads out of bounds for a field "
            f"whose flattened size ({flatInputShape}) isn't a multiple of dim. Use "
            f"LaplacianScheme.Naive, Brookshaw, or Default for scalar fields instead."
        )

    # Always cast through a (n, flatInputShape) view, even when flatInputShape == 1
    # (a plain scalar field, only reachable when dim == 1) -- matching
    # wp_laplacian.py's own `.view(-1, flatInputShape)` convention for its Any-typed
    # field arrays. Without this, castTorchToWarpAsBuiltins would resolve a bare
    # (n,) tensor to a raw scalar_t rather than a length-1 vector, and this scheme's
    # q_ij[base+c] block indexing requires an indexable vector unconditionally.
    queryValuesFlat = queryValues.reshape(-1, flatInputShape)
    referenceValuesFlat = referenceValues.reshape(-1, flatInputShape)

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )

    outputDtype = castTorchToWarpAsBuiltins(queryValuesFlat).dtype

    result = _launchGeometryJVP(
        computeSPHLaplacianDotJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=outputDtype,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        extraTensors=(queryValuesFlat, referenceValuesFlat),
        extraScalars=(wp.int32(flatInputShape),),
    )
    return result.reshape(nQuery, *inputShape)


# ---------------------------------------------------------------------------
# Default CSR (per-query-particle) launch shape. Generic `Any`-typed fields
# like Dot, but no dim-block restriction -- computeDotLaplacian has no block
# indexing at all, see this file's module docstring.
# ---------------------------------------------------------------------------

@wp.func
def computeSPHLaplacianDefaultJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
    iCorrectionTangentData: Any, correctionTangentData: Any,

    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

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

        G, dG, n_ij, dn_ij, _D_ij, _dD_ij, r_ij, dr_ij, h_ij, dh_ij = _laplacianGeometryChainJVP(
            iPtcl, jPtcl, iTangentPtcl, jTangentPtcl, kernelProperties, domainState,
        )

        _A, B, _dA, dB = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            kernelProperties.gradientMode,
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )

        fj = referenceValues[j]
        q_ij = (fj - fi) * B
        dq_ij = (fj - fi) * dB

        # computeDotLaplacian's own second regularized distance (a different
        # literal eps than D_ij's 1e-8 -- matched exactly, not approximated).
        eps2 = scalar_t(_LAPLACIAN_DOT2_EPS)
        D2_ij = r_ij + eps2 * h_ij
        dD2_ij = dr_ij + eps2 * dh_ij
        n_ij2 = n_ij / D2_ij
        dn_ij2 = (dn_ij - n_ij2 * dD2_ij) / D2_ij

        dotn2G = wp.dot(n_ij2, G)
        d_dotn2G = wp.dot(dn_ij2, G) + wp.dot(n_ij2, dG)

        out += -scalar_t(2.0) * (dq_ij * dotn2G + q_ij * d_dotn2G)

    return out


@wp.func
def computeSPHLaplacianDefaultJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,

    queryValue: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    iCorrectionTangentData = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHLaplacianDefaultJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            iCorrectionTangentData, correctionTangentData,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHLaplacianDefaultJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHLaplacianDefaultJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def computeSPHLaplacianDefaultGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
    gradientMode: GradientScheme = GradientScheme.Symmetric,
) -> torch.Tensor:
    """`dLaplacian_i`, shape `queryValues.shape`, Default scheme
    (`computeDotLaplacian`, `math/wp_laplaciandot.py`): `q_ij = (fj-fi)*B_ij`
    (same `B` as Brookshaw/Naive/Dot), `n_ij2 = n_ij/D2_ij` with `D2_ij =
    r_ij + 1e-12*h_ij` (a second, more tightly regularized distance than
    Brookshaw's own `D_ij`, matching `computeDotLaplacian`'s literal
    constant), `Laplacian_i[k] = -2*q_ij[k]*dot(n_ij2,G_ij)` (broadcast
    elementwise across the field, no `dim`-block structure -- unlike Dot);
    `dLaplacian_i` by the ordinary product/quotient rule through every
    factor. `adjacency` is an `AdjacencyList` or `CompactHashMap`.

    This is the geometry/mass/density-tangent **partial** contribution to
    Laplacian's JVP -- `queryValues`/`referenceValues` are held at their
    **primal** (non-tangent) value here. It is **not** the full derivative
    on its own; add the value-tangent (value JVP) contribution (`warpOperation`
    relaunched with the tangent value arrays) for that, or call
    `warpOperationJVP` directly, which sums both automatically
    (`warpier_tier2_combined_jvp_plan.md`).
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianDefaultGeometryJVP: queryValues and "
            "referenceValues (frozen fi/fj) are both required."
        )

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )

    outputDtype = castTorchToWarpAsBuiltins(queryValues).dtype

    return _launchGeometryJVP(
        computeSPHLaplacianDefaultJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=outputDtype,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        gradientMode=gradientMode,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        extraTensors=(queryValues, referenceValues),
    )


def computeSPHLaplacianGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw,
    **kwargs,
) -> torch.Tensor:
    """Dispatcher `operations.py` actually registers in `_GEOMETRY_JVP_DISPATCH`
    -- routes to `computeSPHLaplacianBrookshawGeometryJVP`/
    `computeSPHLaplacianNaiveGeometryJVP`/`computeSPHLaplacianDotGeometryJVP`/
    `computeSPHLaplacianDefaultGeometryJVP` by `laplacianMode`. Any other value is
    rejected before reaching here (`operations.py`'s own centralized scope
    check); any other value is a defensive fallback, not expected to be
    reachable."""
    if laplacianMode is LaplacianScheme.Brookshaw:
        return computeSPHLaplacianBrookshawGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, **kwargs,
        )
    elif laplacianMode is LaplacianScheme.Naive:
        return computeSPHLaplacianNaiveGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, **kwargs,
        )
    elif laplacianMode is LaplacianScheme.Dot:
        return computeSPHLaplacianDotGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, **kwargs,
        )
    elif laplacianMode is LaplacianScheme.Default:
        return computeSPHLaplacianDefaultGeometryJVP(
            queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, **kwargs,
        )
    raise NotImplementedError(
        f"computeSPHLaplacianGeometryJVP: geometry JVP laplacianMode={laplacianMode} is not "
        "implemented -- Brookshaw/Naive/Dot/Default are (warpier_tier2_operators_plan.md "
        "Steps 7/8, remaining-work plan's Dot/Default follow-up)."
    )
