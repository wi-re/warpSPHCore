#!/usr/bin/env python3
"""Particle sets for the convergence harness.

Wraps the shipped `warpSPHCore.sampling.sampleDensestLattice` (1D uniform,
2D hexagonal, 3D FCC -- densest packings, with optional uniform jitter) into
ready-to-probe `Case` objects:

* positions/supports/masses as a `ParticleState` on the target device,
  densities computed with the shipped `warpOperation(Density)`;
* a `DomainDescription` -- periodic on the achieved lattice box, or open
  (non-periodic) with a margin that contains the jitter;
* the compact-hash adjacency used by every probe;
* the interior / boundary-band masks (open domains only). In a periodic
  domain every particle is interior (the band is None).

Two refinement parameterizations are exposed (the parent plan's
"decoupled refinement modes"):

* resolution: fix `target_neighbors` (h/Δx) and vary N -- dx and h both
  shrink as N^(-1/dim);
* smoothing: fix N (and jitter seed, so the lattice is identical) and vary
  `target_neighbors` -- h/Δx varies at fixed discretization.

Both are just `build_case` called with different arguments; nothing here
depends on which mode the caller intends.

Masses are the true lattice cell volume per particle, so a uniform lattice
reads mean density 1 (before the kernel's lattice-density bias L(n_h),
which the measurements are expected to show).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from warpSPHCore import (
    DomainDescription,
    OperationProperties,
    ParticleState,
    radiusSearchCompactHashMap,
    warpOperation,
)
from warpSPHCore.enumTypes import (
    KernelFunctions,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from warpSPHCore.sampling import sampleDensestLattice
from warpSPHCore.util import volumeToSupport

__all__ = ["Case", "build_case", "build_case_from_positions", "cell_volume"]


def cell_volume(dim: int, dx: float) -> float:
    """Volume per particle of the densest packing with nearest-neighbour
    distance dx (so masses = cell_volume give mean density 1).

    1D: dx.  2D hexagonal: sqrt(3)/2 dx^2.  3D FCC: dx^3/sqrt(2)
    (FCC conventional cell side b = dx*sqrt(2), 4 particles per cell).
    """
    if dim == 1:
        return float(dx)
    if dim == 2:
        return math.sqrt(3.0) / 2.0 * dx ** 2
    if dim == 3:
        return dx ** 3 / math.sqrt(2.0)
    raise ValueError(f"dim must be 1, 2 or 3 (got {dim})")


@dataclass
class Case:
    particles: ParticleState          # densities already computed
    domain: DomainDescription
    adjacency: object                 # CompactHashMap
    positions: torch.Tensor           # (N, dim), device, unwrapped
    box: np.ndarray                   # (dim,) achieved lattice box
    N: int
    dim: int
    periodic: bool
    jitter: float
    seed: int | None
    dx: float                         # nearest-neighbour distance
    h: float                          # support radius
    target_neighbors: int
    h_over_dx: float
    cell_vol: float
    interior_mask: torch.Tensor | None   # None when periodic (all interior)
    boundary_mask: torch.Tensor | None   # None when periodic
    kernel: KernelFunctions


def build_case(
    n: int,
    dim: int,
    target_neighbors: int,
    jitter: float = 0.0,
    seed: int | None = None,
    periodic: bool = True,
    device: str = "cuda",
    kernel: KernelFunctions = KernelFunctions.Wendland2,
) -> Case:
    """Build one fully-formed probe case (see module docstring for the
    refinement-mode semantics of the arguments)."""
    sample = sampleDensestLattice(n=n, L=1.0, dim=dim,
                                  jitter=jitter, seed=seed)
    pos_np = sample.positions
    dx = float(sample.spacing)
    N = int(sample.count)
    box = np.asarray(sample.box, dtype=float)
    h = float(volumeToSupport(cell_volume(dim, dx), target_neighbors, dim))

    positions = torch.tensor(pos_np, dtype=torch.float64, device=device)
    # Open-domain margin: contain the jitter displacement (half the jitter
    # width, in units of the nearest-neighbour distance).
    margin = 0.5 * jitter * dx
    if periodic:
        dmin = torch.zeros(dim, dtype=torch.float64, device=device)
        dmax = torch.tensor(box, dtype=torch.float64, device=device)
        periodicity = torch.ones(dim, dtype=torch.bool, device=device)
    else:
        dmin = torch.full((dim,), -margin, dtype=torch.float64, device=device)
        dmax = torch.tensor(box, dtype=torch.float64, device=device) + margin
        periodicity = torch.zeros(dim, dtype=torch.bool, device=device)
    domain = DomainDescription(dmin, dmax, periodicity, dim)

    cell_vol = cell_volume(dim, dx)
    particles = ParticleState(
        positions=positions.contiguous(),
        supports=torch.full((N,), h, dtype=torch.float64, device=device),
        masses=torch.full((N,), cell_vol, dtype=torch.float64,
                          device=device),
        densities=None,
        kinds=torch.zeros(N, dtype=torch.int32, device=device),
    )
    adjacency = radiusSearchCompactHashMap(
        particles, domain, mode=SupportScheme.SuperSymmetric)
    densities = warpOperation(
        particles,
        OperationProperties(kernel=kernel, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather,
                            operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    particles.densities = densities

    interior = None
    boundary = None
    if not periodic:
        # Boundary band: within one support of any open wall (truncated
        # kernel support).
        interior = torch.ones(N, dtype=torch.bool, device=device)
        for d in range(dim):
            interior &= positions[:, d] > domain.min[d] + h
            interior &= positions[:, d] < domain.max[d] - h
        boundary = ~interior

    return Case(
        particles=particles,
        domain=domain,
        adjacency=adjacency,
        positions=positions,
        box=box,
        N=N,
        dim=dim,
        periodic=periodic,
        jitter=jitter,
        seed=seed,
        dx=dx,
        h=h,
        target_neighbors=target_neighbors,
        h_over_dx=h / dx,
        cell_vol=cell_vol,
        interior_mask=interior,
        boundary_mask=boundary,
        kernel=kernel,
    )


def build_case_from_positions(
    positions: torch.Tensor,
    masses: torch.Tensor,
    h: float,
    box: np.ndarray,
    dx: float,
    target_neighbors: int,
    periodic: bool = True,
    device: str = "cuda",
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    margin: float = 0.0,
) -> Case:
    """Build a probe case from an *existing* particle distribution (e.g. the
    uncorrected end state of a saved simulation) instead of sampling a
    lattice: same domain/adjacency/density machinery as `build_case`, with
    masses as given and `cell_vol` the mean mass.

    `positions` (N, dim) raw simulation coordinates (for a periodic `box`
    they may lie outside it -- the code does not re-wrap, and the drift
    grows with the simulated time; the periodic neighbourhood search
    handles the wrap, as it does in the simulation), `masses` (N,), `h` the
    uniform support, `dx` the reference spacing (for `h_over_dx`
    reporting), `target_neighbors` the nominal neighbour count (metadata
    only -- h is taken as given).
    """
    positions = torch.as_tensor(positions, dtype=torch.float64, device=device)
    masses = torch.as_tensor(masses, dtype=torch.float64, device=device)
    N, dim = positions.shape
    box = np.asarray(box, dtype=float)
    h = float(h)
    dx = float(dx)

    if periodic:
        dmin = torch.zeros(dim, dtype=torch.float64, device=device)
        dmax = torch.tensor(box, dtype=torch.float64, device=device)
        periodicity = torch.ones(dim, dtype=torch.bool, device=device)
    else:
        dmin = torch.full((dim,), -margin, dtype=torch.float64, device=device)
        dmax = torch.tensor(box, dtype=torch.float64, device=device) + margin
        periodicity = torch.zeros(dim, dtype=torch.bool, device=device)
    domain = DomainDescription(dmin, dmax, periodicity, dim)

    particles = ParticleState(
        positions=positions.contiguous(),
        supports=torch.full((N,), h, dtype=torch.float64, device=device),
        masses=masses.contiguous(),
        densities=None,
        kinds=torch.zeros(N, dtype=torch.int32, device=device),
    )
    adjacency = radiusSearchCompactHashMap(
        particles, domain, mode=SupportScheme.SuperSymmetric)
    densities = warpOperation(
        particles,
        OperationProperties(kernel=kernel, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather,
                            operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    particles.densities = densities

    interior = None
    boundary = None
    if not periodic:
        interior = torch.ones(N, dtype=torch.bool, device=device)
        for d in range(dim):
            interior &= positions[:, d] > domain.min[d] + h
            interior &= positions[:, d] < domain.max[d] - h
        boundary = ~interior

    return Case(
        particles=particles,
        domain=domain,
        adjacency=adjacency,
        positions=positions,
        box=box,
        N=N,
        dim=dim,
        periodic=periodic,
        jitter=0.0,
        seed=None,
        dx=dx,
        h=h,
        target_neighbors=int(target_neighbors),
        h_over_dx=h / dx,
        cell_vol=float(masses.mean().item()),
        interior_mask=interior,
        boundary_mask=boundary,
        kernel=kernel,
    )
