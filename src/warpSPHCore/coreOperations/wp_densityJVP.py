"""Tier-2 position/support/mass-tangent JVP of the Density operator
(`warpier_forward_mode_plan.md` Phase 4, `warpier_adjoint.md` Tier 2.1):
`dDensity_i = sum_j [ dm_j * W_ij + m_j * dW_ij ]`, `dW_ij` from
`kernels.kernelJVP.sphKernelJVP`.

Launches one thread per neighbor *pair* `(i, j)`, not one per query particle
-- like `wp_implicitShifting.computeShiftingPairTerms` in the `warpSPH`
frontend (which this mirrors field-for-field), because the result feeds a
`scatter_sum`-based assembly (Phase 4's `grad C`/matvec use), not a
per-particle-only consumer. `OperatorSpec`/`launchOperator` only supports
per-query-particle thread counts, so this bypasses that machinery the same
way `computeShiftingPairTerms` already does, rather than duplicating its
reasoning.

Scope: position and support tangents on both query and reference roles, plus
a reference-side mass tangent (`dm_j`) -- the terms Tier 2.1's formula
actually has. No query-side mass tangent (`Density_i` has no `m_i` term) and
no density tangent (Density has no `queryValues`/`referenceValues`/density
input at all). Field-value tangents are Tier 1's `warpOperationJVP`, not
this.
"""

from typing import Any, Optional
import torch
import warp as wp
from warp.types import vector, matrix

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, castTorchToWarpAsBuiltins, allocateTorchWarp
from ..enumTypes import supportSchemeToUint
from ..kernels.kernelJVP import sphKernelJVP
from ..util.stateUtil import getParticle

__all__ = ['computeSPHDensityPositionJVP']

_SoA_BY_DIM = {1: particleDataSoA_1, 2: particleDataSoA_2, 3: particleDataSoA_3}


@wp.kernel
def _computeSPHDensityJVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outW: wp.array(dtype=scalar_t),
    outDW: wp.array(dtype=scalar_t),
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

    W, dW = sphKernelJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)
    outW[e] = W
    outDW[e] = dW


def _buildParticleSoA(dim: int, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor):
    SoA = _SoA_BY_DIM[dim]()
    SoA.positions = castTorchToWarpAsBuiltins(positions.contiguous())
    SoA.supports = castTorchToWarp(supports.contiguous())
    SoA.masses = castTorchToWarp(masses.contiguous())
    n = positions.shape[0]
    dummy = torch.zeros(n, device=positions.device, dtype=positions.dtype)
    SoA.densities = castTorchToWarp(dummy)
    SoA.kinds = castTorchToWarp(torch.zeros(n, device=positions.device, dtype=torch.int32))
    return SoA


def _buildDomainState(domain: DomainDescription) -> domainData:
    d = domainData()
    d.domainMin = castTorchToWarp(domain.min)
    d.domainMax = castTorchToWarp(domain.max)
    d.periodicity = castTorchToWarp(domain.periodic)
    d.dim = domain.dim
    return d


def _buildKernelState(kernel: KernelFunctions, supportMode: SupportScheme) -> kernelState:
    k = kernelState()
    k.kernelFunction = kernel.value
    k.supportMode = supportSchemeToUint(supportMode)
    return k


def computeSPHDensityPositionJVP(
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
) -> torch.Tensor:
    """`dDensity_i`, shape `[numParticles]`. `adjacency` is the torch-facing
    `AdjacencyList` (`.i`/`.j` flat neighbor pairs -- what `buildVerletList`
    returns and what every `warpOperation` call in this codebase already
    threads through), not the warp-side struct `launchOperator` builds
    internally.
    """
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

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    queryTangentState = _buildParticleSoA(dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery))
    referenceTangentState = _buildParticleSoA(dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)
    numPairs = adjacency.i.shape[0]

    W_t, W_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)
    dW_t, dW_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)

    wp.launch(
        _computeSPHDensityJVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, queryTangentState, referenceTangentState,
                domainState, kernelProperties, edgeI, edgeJ, W_w, dW_w],
        device=edgeI.device,
    )

    massJ = referenceParticles.masses[adjacency.j.long()]
    dMassJ = tangentReferenceMasses[adjacency.j.long()]
    pairContribution = dMassJ * W_t + massJ * dW_t

    dDensity = torch.zeros(nQuery, device=device, dtype=dtype)
    dDensity.index_add_(0, adjacency.i.long(), pairContribution)
    return dDensity
