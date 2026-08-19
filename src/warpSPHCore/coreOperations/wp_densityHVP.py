"""Hessian-vector product of the Density operator w.r.t. positions
(`warpier_forward_mode_plan.md` Phase 4 Step 3, "`Hess C . v` is a JVP of
that JVP"): differentiate `computeSPHDensityPositionJVP`'s own position
tangent once more, in the same direction. Concretely this reduces to
`HVP_i = sum_j m_j * H_ij @ (v_i - v_j)`, `H_ij` the pairwise kernel Hessian
`kernels.hessian.sphKernelHessian` already computes -- **no new kernel math**,
the same callback Tier 2.1's own derivation made for the first-order JVP
(`warpier_adjoint.md` Tier 2.1 "Building blocks needed").

Restricted to the position-only case (support/mass tangents frozen, i.e.
`dh=0` on both differentiation orders) to match `sphKernelHessian`'s own
scope, so this is directly comparable term-for-term to
`warpSPH/modules/shifting/wp_implicitShifting.py`'s hand-built `H` output --
the comparison baseline this exists to validate against
(`warpSPH/tests/test_implicitShiftingHessianJVP.py`).

**Composing this via generic torch machinery does not work, tried first.**
`torch.func.jvp` applied twice, or `torch.autograd.forward_ad.make_dual`/
`dual_level` nested, over `computeSPHDensityPositionJVP` cannot propagate a
second tangent through a `wp.launch`-backed function in this codebase:
`computeSPHDensityPositionJVP` is not wrapped in a `torch.autograd.Function`
at all (unlike `warpOperation`'s `StateAwareWarpFunction`, which has no
`jvp()` registered either -- see `scripts/spike_forward_mode_tier1.py`'s own
finding for the *first*-order case: `torch.autograd.forward_ad` silently
returns `tangent=None` there already). Empirically, `torch.func.jvp` errors
immediately (`RuntimeError: Cannot access data pointer of Tensor that
doesn't have storage` -- functorch's dual tensors have no real storage for
`wp.from_torch` to view), and `torch.autograd.forward_ad.make_dual` runs but
silently drops the tangent, same failure mode as the first-order case one
level down. This module is the "small explicit second-order helper" the plan
flagged as the fallback if composition didn't work -- it didn't.
"""

from typing import Any, Optional
import torch
import warp as wp
from warp.types import vector, matrix

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, allocateTorchWarp
from ..kernels.hessian import sphKernelHessian
from ..util.stateUtil import getParticle
from .wp_densityJVP import _buildParticleSoA, _buildDomainState, _buildKernelState

__all__ = ['computeSPHDensityPositionHVP']


@wp.kernel
def _computeSPHDensityHVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outH: wp.array(dtype=Any),
):
    e = wp.tid()
    if e >= edgeI.shape[0]:
        return
    i = wp.int32(edgeI[e])
    j = wp.int32(edgeJ[e])

    xi, hi, _mi, _rhoi, _ki = getParticle(queryState, i)
    xj, hj, _mj, _rhoj, _kj = getParticle(referenceState, j)

    outH[e] = sphKernelHessian(xi, xj, hi, hj, kernelProperties, domainState)


def computeSPHDensityPositionHVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: AdjacencyList,
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`Hess(Density)_i @ v`, shape `[numQuery, dim]` -- the Hessian-vector
    product a Newton-Krylov matvec needs (Phase 4 Steps 3-4), with `v` =
    `tangentQueryPositions`/`tangentReferencePositions` (the same
    perturbation field in both roles, when query and reference are the same
    particle population, matching implicit shifting's own usage).

    Mirrors `wp_implicitShifting._multiplyLaplacianBlock`'s math exactly
    (`sum_j m_j * H_ij @ (v_i - v_j)`) but is assembled from warpSPHCore's
    own Tier-2.0 `sphKernelHessian` building block through this package's
    adjacency/domain/particle-state conventions, rather than warpSPH's
    hand-rolled per-pair kernel.
    """
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    tangentReferencePositions = (
        tangentReferencePositions if tangentReferencePositions is not None
        else torch.zeros((nRef, dim), device=device, dtype=dtype)
    )

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)
    numPairs = adjacency.i.shape[0]

    H_t, H_w = allocateTorchWarp(numPairs, matrix(shape=(dim, dim), dtype=scalar_t), edgeI.device)

    wp.launch(
        _computeSPHDensityHVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, domainState, kernelProperties, edgeI, edgeJ, H_w],
        device=edgeI.device,
    )

    ii, jj = adjacency.i.long(), adjacency.j.long()
    # Self-pairs (i == j) contribute zero analytically -- a particle's
    # distance to itself is identically zero regardless of where it moves --
    # but sphKernelHessian's near-origin regularization branch is
    # numerically unstable there rather than exactly zero. Same hazard
    # `wp_implicitShifting.computeImplicitShift`'s own docstring documents;
    # dropped before assembly for the same reason, not a new finding.
    pairMask = ii != jj
    ii, jj, H_t = ii[pairMask], jj[pairMask], H_t[pairMask]

    massJ = referenceParticles.masses[jj]
    relTangent = tangentQueryPositions[ii] - tangentReferencePositions[jj]
    pairHVP = massJ[:, None] * torch.einsum('nab,nb->na', H_t, relTangent)

    HVP = torch.zeros((nQuery, dim), device=device, dtype=dtype)
    HVP.index_add_(0, ii, pairHVP)
    return HVP
