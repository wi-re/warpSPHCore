"""Tier-2 position/support/mass/density-tangent JVP of the Laplacian
operator's Brookshaw and Naive schemes (`warpier_tier2_operators_plan.md`
Steps 7/8, `warpier_adjoint.md` Tiers 2.2/2.3). Dot/Default `laplacianMode`s
are out of scope, enforced centrally in `operations.py`.

**Brookshaw** (`computeSPHLaplacianBrookshawPositionJVP`): `q_ij =
(fj-fi)*B_ij` (`B` from `_jvpCommon.gradientWeights` -- literally the same
coefficient as Gradient's `B` term, not re-derived), `D_ij = r_ij +
eps*h_ij` (`eps=1e-8`, matching `wp_laplacian.py`'s literal constant),
`n_ij = x_ij/D_ij`, `L_ij = -2*q_ij*dot(G_ij,n_ij)/D_ij`; the
regularized-distance chain (`dr_ij`, `dD_ij`, `dn_ij`, `dL_ij`) is ordinary
calculus on top of `wp_kernelGradientJVP`'s shared `(G_ij, dG_ij)` pair
kernel.

**Naive** (`computeSPHLaplacianNaivePositionJVP`): `q_ij` is the exact same
`B_ij` again (`wp_laplacian.py`'s `q_ij` depends only on `gradientMode`,
never `laplacianMode` -- Tier 2.2's finding, re-confirmed by Tier 2.3 under
this scheme), but `L_ij`/`dL_ij` come from `sphKernelLaplacianJVP`
(`kernels/kernelJVP.py`, the actual analytic second-derivative-of-r
estimator's own JVP) instead of Brookshaw's gradient-based `n_ij/D_ij`
estimator -- a genuinely different per-pair kernel launch, `L = sum_j
q_ij*L_ij`, `dL = sum_j (dq_ij*L_ij + q_ij*dL_ij)`.

`computeSPHLaplacianPositionJVP` is the thin dispatcher `operations.py`
actually registers (`_TIER2_VALUE_DISPATCH[WarpOperation.Laplacian]`),
picking between the two by `laplacianMode`.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, allocateTorchWarp
from ._jvpCommon import (
    buildParticleSoA as _buildParticleSoA,
    buildDomainState as _buildDomainState,
    buildKernelState as _buildKernelState,
    gradientWeights as _gradientWeights,
)
from .wp_kernelGradientJVP import launchPairKernelGradientJVP as _launchPairKernelGradientJVP
from ..kernels.kernelJVP import sphKernelLaplacianJVP
from ..util.stateUtil import getParticle

__all__ = [
    'computeSPHLaplacianPositionJVP',
    'computeSPHLaplacianBrookshawPositionJVP',
    'computeSPHLaplacianNaivePositionJVP',
]

_LAPLACIAN_EPS = 1e-8  # matches wp_laplacian.py's literal constant, not get_epsilon(r)


def _pairwiseSupportAndTangent(hi, hj, dhi, dhj, mode: SupportScheme):
    """`computePairwiseSupport`'s dispatch, evaluated in torch on flat
    `[numPairs]` tensors (mirrors `scripts/spike_forward_mode_tier2_gradient.py`'s
    `_h_ij_and_tangent`, minus the dense-grid broadcasting) -- used only for
    Laplacian's `eps` regularization term, independent of which branch
    `sphKernelGradientJVP` itself took for `G_ij`/`dG_ij`. `mode` here is
    already the `SupportScheme` enum (not the raw `.value` int the spike's
    helper had to coerce back from), since this module passes it straight
    through from the Python-level `supportMode` argument."""
    if mode == SupportScheme.Gather:
        return hi, dhi
    elif mode == SupportScheme.Scatter:
        return hj, dhj
    elif mode == SupportScheme.MeanSymmetric:
        return (hi + hj) / 2.0, (dhi + dhj) / 2.0
    else:
        hij = torch.maximum(hi, hj)
        dhij = torch.where(hi >= hj, dhi, dhj)
        return hij, dhij


def computeSPHLaplacianBrookshawPositionJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: AdjacencyList,
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
    (see `computeSPHLaplacianNaivePositionJVP` for Naive; `computeSPHLaplacianPositionJVP`
    is the dispatcher between the two that `operations.py` actually calls).
    `queryValues`/`referenceValues` (`fi`/`fj`, scalar fields) are required
    and frozen. `queryParticles.densities`/`referenceParticles.densities`
    must already hold real values, same requirement as
    `computeSPHGradientPositionJVP`.
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianBrookshawPositionJVP: queryValues and "
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

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    queryTangentState = _buildParticleSoA(dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery))
    referenceTangentState = _buildParticleSoA(dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)

    G_t, dG_t = _launchPairKernelGradientJVP(
        queryState, referenceState, queryTangentState, referenceTangentState,
        domainState, kernelProperties, edgeI, edgeJ, dim, device, dtype,
    )

    iIdx = adjacency.i.long()
    jIdx = adjacency.j.long()
    massJ = referenceParticles.masses[jIdx]
    densityI = queryParticles.densities[iIdx]
    densityJ = referenceParticles.densities[jIdx]
    dMassJ = tangentReferenceMasses[jIdx]
    dDensityI = tangentQueryDensities[iIdx]
    dDensityJ = tangentReferenceDensities[jIdx]

    _, B, _, dB = _gradientWeights(massJ, densityI, densityJ, dMassJ, dDensityI, dDensityJ, gradientMode)

    fi = queryValues[iIdx]
    fj = referenceValues[jIdx]
    q = (fj - fi) * B
    dq = (fj - fi) * dB

    xI = queryParticles.positions[iIdx]
    xJ = referenceParticles.positions[jIdx]
    dxI = tangentQueryPositions[iIdx]
    dxJ = tangentReferencePositions[jIdx]
    x_ij = xI - xJ
    dx_ij = dxI - dxJ
    r_ij = x_ij.norm(dim=-1)
    r_ij_safe = torch.where(r_ij > 0, r_ij, torch.ones_like(r_ij))
    # safe_sqrt's custom adjoint (math/wp_sqrt.py) contributes 0 when its
    # argument is <=0, i.e. production defines dr_ij=0 exactly at r_ij=0
    # (self-pairs) rather than 0/0 -- matched here, not just NaN-avoided
    # (warpier_adjoint.md Tier 2.2's own note on this).
    dr_ij = torch.where(r_ij > 0, (x_ij * dx_ij).sum(-1) / r_ij_safe, torch.zeros_like(r_ij))

    hI = queryParticles.supports[iIdx]
    hJ = referenceParticles.supports[jIdx]
    dhI = tangentQuerySupports[iIdx]
    dhJ = tangentReferenceSupports[jIdx]
    h_ij, dh_ij = _pairwiseSupportAndTangent(hI, hJ, dhI, dhJ, supportMode)

    D_ij = r_ij + _LAPLACIAN_EPS * h_ij
    dD_ij = dr_ij + _LAPLACIAN_EPS * dh_ij
    n_ij = x_ij / D_ij.unsqueeze(-1)
    dn_ij = (dx_ij - n_ij * dD_ij.unsqueeze(-1)) / D_ij.unsqueeze(-1)

    dot_Gn = (G_t * n_ij).sum(-1)
    d_dot_Gn = (dG_t * n_ij).sum(-1) + (G_t * dn_ij).sum(-1)
    P = dot_Gn / D_ij
    dP = d_dot_Gn / D_ij - dot_Gn * dD_ij / D_ij ** 2

    dL = -2.0 * (dq * P + q * dP)

    dLaplacian = torch.zeros(nQuery, device=device, dtype=dtype)
    dLaplacian.index_add_(0, iIdx, dL)
    return dLaplacian


@wp.kernel
def _sphKernelLaplacianJVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outL: wp.array(dtype=scalar_t),
    outDL: wp.array(dtype=scalar_t),
):
    e = wp.tid()
    if e >= edgeI.shape[0]:
        return
    i = wp.int32(edgeI[e])
    j = wp.int32(edgeJ[e])

    xi, hi, _mi, _rhoi, _ki = getParticle(queryState, i)
    xj, hj, _mj, _rhoj, _kj = getParticle(referenceState, j)
    dxi, dhi, _dmi, _drhoi, _dki = getParticle(queryTangentState, i)
    dxj, dhj, _dmj, _drhoj, _dkj = getParticle(referenceTangentState, j)

    L, dL = sphKernelLaplacianJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)
    outL[e] = L
    outDL[e] = dL


def _launchPairKernelLaplacianJVP(
    queryState, referenceState, queryTangentState, referenceTangentState,
    domainState: domainData, kernelProperties: kernelState,
    edgeI, edgeJ,
):
    """One thread per adjacency pair, producing the pairwise `(L_ij, dL_ij)`
    JVP (`d(sphKernelLaplacian)/d{x,h}`) as flat `[numPairs]` torch tensors
    -- Naive's own per-pair kernel launch (distinct from Brookshaw's shared
    `(G_ij, dG_ij)` launch in `wp_kernelGradientJVP.py`, since Naive uses the
    analytic second-derivative-of-r estimator directly rather than a
    gradient-based one). Structurally identical to `_jvpCommon.launchPairKernelJVP`
    (both produce a `[numPairs]` scalar pair), just calling `sphKernelLaplacianJVP`
    instead of `sphKernelJVP` -- kept local to this file rather than promoted
    to `_jvpCommon.py` since Naive is its only consumer.
    """
    numPairs = edgeI.shape[0]
    L_t, L_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)
    dL_t, dL_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)

    wp.launch(
        _sphKernelLaplacianJVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, queryTangentState, referenceTangentState,
                domainState, kernelProperties, edgeI, edgeJ, L_w, dL_w],
        device=edgeI.device,
    )
    return L_t, dL_t


def computeSPHLaplacianNaivePositionJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: AdjacencyList,
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
    (`kernels/kernelJVP.py`)."""
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHLaplacianNaivePositionJVP: queryValues and "
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

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    queryTangentState = _buildParticleSoA(dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery))
    referenceTangentState = _buildParticleSoA(dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)

    L_t, dL_t = _launchPairKernelLaplacianJVP(
        queryState, referenceState, queryTangentState, referenceTangentState,
        domainState, kernelProperties, edgeI, edgeJ,
    )

    iIdx = adjacency.i.long()
    jIdx = adjacency.j.long()
    massJ = referenceParticles.masses[jIdx]
    densityI = queryParticles.densities[iIdx]
    densityJ = referenceParticles.densities[jIdx]
    dMassJ = tangentReferenceMasses[jIdx]
    dDensityI = tangentQueryDensities[iIdx]
    dDensityJ = tangentReferenceDensities[jIdx]

    _, B, _, dB = _gradientWeights(massJ, densityI, densityJ, dMassJ, dDensityI, dDensityJ, gradientMode)

    fi = queryValues[iIdx]
    fj = referenceValues[jIdx]
    q = (fj - fi) * B
    dq = (fj - fi) * dB

    pairContribution = dq * L_t + q * dL_t

    dLaplacian = torch.zeros(nQuery, device=device, dtype=dtype)
    dLaplacian.index_add_(0, iIdx, pairContribution)
    return dLaplacian


def computeSPHLaplacianPositionJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: AdjacencyList,
    tangentQueryPositions: torch.Tensor,
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw,
    **kwargs,
) -> torch.Tensor:
    """Dispatcher `operations.py` actually registers in `_TIER2_VALUE_DISPATCH`
    -- routes to `computeSPHLaplacianBrookshawPositionJVP`/
    `computeSPHLaplacianNaivePositionJVP` by `laplacianMode`. Dot/Default are
    rejected before reaching here (`operations.py`'s own centralized scope
    check); any other value is a defensive fallback, not expected to be
    reachable."""
    if laplacianMode is LaplacianScheme.Brookshaw:
        return computeSPHLaplacianBrookshawPositionJVP(
            queryParticles, domain, kernel, supportMode, adjacency, tangentQueryPositions, **kwargs,
        )
    elif laplacianMode is LaplacianScheme.Naive:
        return computeSPHLaplacianNaivePositionJVP(
            queryParticles, domain, kernel, supportMode, adjacency, tangentQueryPositions, **kwargs,
        )
    raise NotImplementedError(
        f"computeSPHLaplacianPositionJVP: Tier-2 laplacianMode={laplacianMode} is not "
        "implemented -- only Brookshaw and Naive are (warpier_tier2_operators_plan.md "
        "Steps 7/8)."
    )
