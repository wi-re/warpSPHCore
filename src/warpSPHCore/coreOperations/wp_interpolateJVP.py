"""Tier-2 position/support/mass/density-tangent JVP of the Interpolate
operator (`warpier_tier2_operators_plan.md` Step 2, `warpier_adjoint.md`
Tier 2.1): `dInterpolate_i = sum_j fj * (dVj * W_ij + Vj * dW_ij)`, with
`Vj = mass_j / density_j`, `dVj = dmass_j/density_j - mass_j*ddensity_j/density_j^2`,
`fj` (`referenceValues`) held **frozen** -- combined Tier-1 (value tangent)
+ Tier-2 was never derived, matching every other Tier-2 operator's scope.

Reuses `_jvpCommon.launchPairKernelJVP`'s pair-indexed `(W_ij, dW_ij)`
kernel (same one `wp_densityJVP.py` launches) -- no new `@wp.kernel` here,
this is the plan's proof that the shared per-pair kernel and the new
`queryValues`/`referenceValues` plumbing in `warpOperationJVP` compose
correctly, on the cheapest operator that needs both.
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
    launchPairKernelJVP as _launchPairKernelJVP,
)

__all__ = ['computeSPHInterpolatePositionJVP']


def computeSPHInterpolatePositionJVP(
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
    tangentReferenceDensities: Optional[torch.Tensor] = None,
    queryValues: Optional[torch.Tensor] = None,
    referenceValues: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`dInterpolate_i`, shape `[numParticles, *referenceValues.shape[1:]]`.
    `referenceValues` (`fj`) is required and frozen (no tangent on it --
    that would be Tier 1). `queryValues` (`fi`) is not part of Interpolate's
    formula at all (mirroring `_computeSPHInterpolant_stateBackend`, which
    never reads a query-side field either) and must not be provided.
    """
    if referenceValues is None:
        raise ValueError("computeSPHInterpolatePositionJVP: referenceValues (frozen fj) is required.")
    if queryValues is not None:
        raise ValueError(
            "computeSPHInterpolatePositionJVP: Interpolate has no queryValues (fi) term "
            "-- only referenceValues (fj) is used, matching warpOperation(Interpolate)."
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
    tangentReferenceDensities = tangentReferenceDensities if tangentReferenceDensities is not None else zerosScalar(nRef)

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    queryTangentState = _buildParticleSoA(dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery))
    referenceTangentState = _buildParticleSoA(dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)

    W_t, dW_t = _launchPairKernelJVP(
        queryState, referenceState, queryTangentState, referenceTangentState,
        domainState, kernelProperties, edgeI, edgeJ,
    )

    iIdx = adjacency.i.long()
    jIdx = adjacency.j.long()
    massJ = referenceParticles.masses[jIdx]
    densityJ = referenceParticles.densities[jIdx]
    dMassJ = tangentReferenceMasses[jIdx]
    dDensityJ = tangentReferenceDensities[jIdx]

    Vj = massJ / densityJ
    dVj = dMassJ / densityJ - massJ * dDensityJ / densityJ ** 2
    coeff = dVj * W_t + Vj * dW_t  # [numPairs]

    fj = referenceValues[jIdx]  # [numPairs, *fieldShape]
    coeff = coeff.reshape((coeff.shape[0],) + (1,) * (fj.dim() - 1))
    pairContribution = fj * coeff

    dInterpolate = torch.zeros((nQuery,) + tuple(fj.shape[1:]), device=device, dtype=dtype)
    dInterpolate.index_add_(0, iIdx, pairContribution)
    return dInterpolate
