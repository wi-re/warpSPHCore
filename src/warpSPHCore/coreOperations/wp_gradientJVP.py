"""Tier-2 position/support/mass/density-tangent JVP of the Gradient operator
(`warpier_tier2_operators_plan.md` Step 4, `warpier_adjoint.md` Tier 2.2):
`dGradient_i = sum_j [ dcoeff_ij*G_ij + coeff_ij*dG_ij ]`, `coeff_ij =
fi*A_ij + fj*B_ij` (`fi`/`fj` = frozen `queryValues`/`referenceValues`),
`(G_ij, dG_ij)` from `wp_kernelGradientJVP.launchPairKernelGradientJVP`,
`(A, B, dA, dB)` from `_jvpCommon.gradientWeights`.

Same pair-indexed-kernel-plus-torch-scatter shape as `wp_densityJVP.py`/
`wp_interpolateJVP.py` -- no new `@wp.kernel` here, this is pure-torch
coefficient assembly on top of Step 3's shared pair kernel.
"""

from typing import Optional
import torch

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp
from ._jvpCommon import (
    buildParticleSoA as _buildParticleSoA,
    buildDomainState as _buildDomainState,
    buildKernelState as _buildKernelState,
    gradientWeights as _gradientWeights,
)
from .wp_kernelGradientJVP import launchPairKernelGradientJVP as _launchPairKernelGradientJVP

__all__ = ['computeSPHGradientPositionJVP']


def computeSPHGradientPositionJVP(
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
    """`dGradient_i`, shape `[numParticles, dim]`. `queryValues`/
    `referenceValues` (`fi`/`fj`, scalar fields) are required and frozen (no
    tangent on them -- that would be Tier 1, never derived combined with
    Tier 2). `queryParticles.densities`/`referenceParticles.densities` must
    already hold real (non-dummy) values -- `coeff_ij` depends on them
    directly, same requirement as `computeSPHInterpolatePositionJVP`.
    """
    if queryValues is None or referenceValues is None:
        raise ValueError(
            "computeSPHGradientPositionJVP: queryValues and referenceValues "
            "(frozen fi/fj) are both required."
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

    A, B, dA, dB = _gradientWeights(massJ, densityI, densityJ, dMassJ, dDensityI, dDensityJ, gradientMode)

    fi = queryValues[iIdx]
    fj = referenceValues[jIdx]
    coeff = fi * A + fj * B
    dcoeff = fi * dA + fj * dB

    pairContribution = dcoeff.unsqueeze(-1) * G_t + coeff.unsqueeze(-1) * dG_t

    dGradient = torch.zeros((nQuery, dim), device=device, dtype=dtype)
    dGradient.index_add_(0, iIdx, pairContribution)
    return dGradient
