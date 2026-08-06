"""Shared helpers for the scripts/gradcheck_*.py family.

Not a standalone entrypoint (leading underscore) -- imported by the
per-operator gradcheck scripts. See any gradcheck_*.py script's own
docstring for the precision/reentrancy background this builds on; in short:

  * SPHWARPCORE_PRECISION must be set to "float64" by the entry script
    *before* it imports this module (which imports sphWarpCore) -- that
    setting is baked into every @wp.kernel/@wp.func at import time and
    cannot be changed afterward in the same process.
  * As of 2026-08-05, WarpFunctionWrapper.backward / StateAwareWarpFunction
    .backward (src/sphWarpCore/utils/wp_autograd.py) were fixed to clone
    the gradient read out of Warp and zero the tape afterward, so
    torch.autograd.gradcheck can now be called directly against
    warpOperation -- no manual Jacobian workaround needed for new scripts.
    See warpier_core.md's "Backward-Mode (Reverse AD) Findings".

All cases here are deliberately 1D and small (single particle / a short
line), so failures are easy to reason about by hand and self- vs.
non-self-particle gradient contributions are easy to separate out.
"""

from __future__ import annotations

import torch

from sphWarpCore import ParticleState, radiusSearchCompactHashMap
from sphWarpCore.enumTypes import KernelFunctions, SupportScheme
from sphWarpCore.dataTypes import DomainDescription
from sphWarpCore.radiusSearch.wp_compactHash import buildCompactHashMap

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2


# --------------------------------------------------------------------------
# Domain / particle-case construction
# --------------------------------------------------------------------------

def make_domain(dim: int = 1, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


def single_particle_case(h: float = 1.0):
    """One particle at the origin. Isolates the self-interaction term:
    no neighbors, so any nonzero d(output)/d(position) gradient here would
    be a bug -- a symmetric kernel's gradient at r=0 is exactly zero."""
    positions = torch.tensor([[0.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.tensor([h], dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.tensor([1.0], dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def line_case(n: int, xmin: float = -1.0, xmax: float = 1.0, h: float | None = None):
    """n particles evenly spaced on [xmin, xmax]. Small and regular enough
    to inspect self vs. non-self gradient terms entry-by-entry."""
    x = torch.linspace(xmin, xmax, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    positions = x.detach().clone().requires_grad_(True)  # fresh leaf tensor
    if h is None:
        spacing = (xmax - xmin) / max(n - 1, 1)
        h = max(2.5 * spacing, 1e-3)  # a few particle spacings so neighborhoods overlap
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def grid_case_2d(n_per_side: int = 3, spacing: float = 0.4, h: float | None = None):
    """n_per_side x n_per_side particles on a regular 2D grid centered at the
    origin. Used by operators that need a genuine 2D domain (Divergence's
    matrix-field/dotMode paths, Curl) rather than the degenerate 1D case
    line_case gives -- kept small since gradcheck's numerical Jacobian cost
    grows with total element count across all differentiable inputs."""
    coords = torch.linspace(-(n_per_side - 1) / 2 * spacing, (n_per_side - 1) / 2 * spacing, n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    positions = x.detach().clone().requires_grad_(True)  # fresh leaf tensor
    n = positions.shape[0]
    if h is None:
        h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def build_adjacency(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, domain: DomainDescription, mode=SupportScheme.Gather):
    """Adjacency is treated as non-differentiable and frozen: built once
    from detached positions and reused across every forward call in a
    gradcheck, rather than rebuilt per-call. This matches the standard SPH
    modeling assumption (no contribution from the neighbor search itself to
    the gradient) and keeps the function gradcheck evaluates numerically
    smooth -- rebuilding the neighbor list under finite-difference
    perturbation risks discontinuities right at the support-radius boundary."""
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=mode)
    return adjacency, kinds


def build_grid_adjacency(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, domain: DomainDescription, mode=SupportScheme.Gather):
    """Same non-differentiable/frozen contract as build_adjacency, but returns a genuine
    CompactHashMap (grid traversal, useAdjacency=False) instead of the CSR AdjacencyList
    radiusSearchCompactHashMap returns by default despite its name -- confusingly, that
    default return type means every gradcheck_*_native.py script using build_adjacency
    exercises the neighbor-list traversal branch, not the grid one. Use this helper
    instead of build_adjacency when a script specifically needs to exercise the grid/
    compact-hash-map traversal branch (e.g. as a dual-path regression guard)."""
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    grid = buildCompactHashMap(
        positions.detach(), positions.detach(),
        supports.detach(), supports.detach(),
        periodicity=domain.periodic,
        domainDescription=domain,
        mode=mode,
    )
    return grid, kinds


# --------------------------------------------------------------------------
# Pure-PyTorch reference Wendland2 kernel (no Warp involved at all -- an
# independent implementation of the same formula sphKernel_ uses, dim=1):
#   W(r, h) = C_d/h * k(r/h),  k=0 for q=r/h > 1
# See src/sphWarpCore/kernels/kernelFunctions/wendland2.py and
# src/sphWarpCore/kernels/wp_kernel.py:sphKernel_ for the formula this
# mirrors. Reusable across operators as a ground truth kernel; each
# operator's own reference formula (density sum, gradient sum, ...) is
# built on top of this per-script.
# --------------------------------------------------------------------------
WENDLAND2_C_D_1D = 5.0 / 4.0


def wendland2_kernel_1d(r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    q = r.abs() / h
    inside = q <= 1.0
    val = (1.0 - q).clamp(min=0.0) ** 3 * (1.0 + 3.0 * q)
    k = torch.where(inside, val, torch.zeros_like(q))
    return WENDLAND2_C_D_1D / h * k
