"""Verifies the self-pair-exclusion claim `wp_densityHVP.py`'s module
docstring makes: dropping self-pairs (`i == j`) from `computeSPHDensityPositionHVP`
is an exact identity (translation invariance forces the self term's true
contribution to `d^2 C_i/dx_i^2` to be zero for *any* finite pairwise kernel
Hessian, not because `sphKernelHessian` is unstable at `r=0` -- it isn't; see
the module docstring and the root `wp_kernels.ipynb` notebook for the direct
numeric check that it's a well-defined, finite curvature value there), not a
numerical-safety fallback copied uncritically from
`warpSPH/modules/shifting/wp_implicitShifting.py`'s own (previously
mischaracterized) self-pair drop.

Two checks, both by calling the pair-kernel machinery directly with the
self-pair mask forced on/off, rather than trusting the production function's
own (always-on) masking:

1. **Shared tangent (the actual Phase 4 matvec usage)**: with
   `tangentQueryPositions == tangentReferencePositions` (the same particle
   moving in both roles), keeping vs. dropping self-pairs must be a bitwise
   no-op -- the self term's own `(v_i - v_i) = 0` factor already annihilates
   it regardless of `H_ii`'s value.
2. **Asymmetric tangent (the diagBlock-isolation usage
   `implicitShiftingAutomatic.py` needs)**: with `tangentQueryPositions` a
   constant coordinate-basis field and `tangentReferencePositions = 0`,
   dropping self-pairs is *required* -- keeping them changes the result,
   confirming the drop is load-bearing there, not an inert simplification.
"""

from __future__ import annotations

import torch
import warp as wp
from warp.types import matrix

from warpSPHCore import (
    DomainDescription, OperationProperties, ParticleState,
    radiusSearchCompactHashMap,
)
from warpSPHCore.coreOperations.wp_densityHVP import _computeSPHDensityHVP_PairKernel
from warpSPHCore.coreOperations.wp_densityJVP import _buildParticleSoA, _buildDomainState, _buildKernelState
from warpSPHCore.enumTypes import SupportScheme, KernelFunctions
from warpSPHCore.type_config import scalar_t
from warpSPHCore.util import castTorchToWarp, allocateTorchWarp

DEVICE = torch.device("cpu")
DTYPE = torch.float32
KERNEL = KernelFunctions.Wendland2


def _grid_case_2d(n_per_side: int = 3, spacing: float = 0.4):
    coords = torch.linspace(-(n_per_side - 1) / 2 * spacing, (n_per_side - 1) / 2 * spacing,
                             n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    n = positions.shape[0]
    h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE) * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    return positions, supports, masses


def _raw_hvp(positions, supports, masses, domain, adjacency, tangentQuery, tangentReference, dropSelf: bool):
    """Bypasses `computeSPHDensityPositionHVP`'s own (always-on) masking to
    let the test control it directly."""
    dim = domain.dim
    queryState = _buildParticleSoA(dim, positions, supports, masses)
    referenceState = _buildParticleSoA(dim, positions, supports, masses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(KERNEL, SupportScheme.Gather)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)
    numPairs = adjacency.i.shape[0]
    H_t, H_w = allocateTorchWarp(numPairs, matrix(shape=(dim, dim), dtype=scalar_t), edgeI.device)
    wp.launch(_computeSPHDensityHVP_PairKernel, dim=numPairs,
              inputs=[queryState, referenceState, domainState, kernelProperties, edgeI, edgeJ, H_w],
              device=edgeI.device)

    ii, jj = adjacency.i.long(), adjacency.j.long()
    if dropSelf:
        mask = ii != jj
        ii, jj, H_use = ii[mask], jj[mask], H_t[mask]
    else:
        H_use = H_t

    massJ = masses[jj]
    relTangent = tangentQuery[ii] - tangentReference[jj]
    pairHVP = massJ[:, None] * torch.einsum('nab,nb->na', H_use, relTangent)
    n = positions.shape[0]
    out = torch.zeros(n, dim, device=positions.device, dtype=positions.dtype)
    out.index_add_(0, ii, pairHVP)
    return out


def _setup():
    positions, supports, masses = _grid_case_2d()
    domain = DomainDescription(
        min=torch.tensor([-10.0, -10.0], dtype=DTYPE, device=DEVICE),
        max=torch.tensor([10.0, 10.0], dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False, False], device=DEVICE), dim=2,
    )
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    return positions, supports, masses, domain, adjacency


def test_selfPairDrop_isNoOp_forSharedTangent():
    positions, supports, masses, domain, adjacency = _setup()
    n, dim = positions.shape

    torch.manual_seed(0)
    v = torch.randn(n, dim, dtype=DTYPE)

    withSelf = _raw_hvp(positions, supports, masses, domain, adjacency, v, v, dropSelf=False)
    noSelf = _raw_hvp(positions, supports, masses, domain, adjacency, v, v, dropSelf=True)

    torch.testing.assert_close(withSelf, noSelf, rtol=0, atol=0)


def test_selfPairDrop_isRequired_forAsymmetricTangent_toMatchDiagBlockConvention():
    positions, supports, masses, domain, adjacency = _setup()
    n, dim = positions.shape

    e0 = torch.zeros(n, dim, dtype=DTYPE)
    e0[:, 0] = 1.0
    zero = torch.zeros(n, dim, dtype=DTYPE)

    withSelf = _raw_hvp(positions, supports, masses, domain, adjacency, e0, zero, dropSelf=False)
    noSelf = _raw_hvp(positions, supports, masses, domain, adjacency, e0, zero, dropSelf=True)

    assert not torch.allclose(withSelf, noSelf), (
        "keeping self-pairs should change the diagBlock-isolation result -- "
        "if this starts passing, either the case no longer exercises a "
        "nonzero self-Hessian or the identity this test protects has changed"
    )
