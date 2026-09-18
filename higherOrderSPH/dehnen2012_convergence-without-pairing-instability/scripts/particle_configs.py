#!/usr/bin/env python3
"""Static particle configurations for the D&A (2012) Fig. 3 density
estimation.

Three 3D periodic configurations, all at mean density rho = 1:

* **FCC** (the paper's solid curves): densest-sphere packing from
  `warpSPHCore.sampling.sampleDensestLattice` (a p x p x p grid of
  conventional FCC cells, N = 4 p^3, nearest-neighbour distance
  d_nn = L/(p sqrt(2))). Translational invariance makes every particle's
  environment identical, so its density estimate is the single lattice
  sum -- exact, no statistics needed.
* **glass** (the paper's squares): generated with the warpSPH
  optimized-sampling components, the recipe of
  `modules/shifting/delta.py` -- a jittered regular lattice relaxed by
  delta-shift iterations until the displacement converges. (NOT
  `sample/optimal.sampleOptimal` as shipped: it overwrites its lattice
  start with uniform random points and applies the raw O(1-10)
  shift term without the `-CFL * Ma * 2 * h^2` scaling, so it diverges;
  the scaling below is the one `delta.py` applies.) The relaxation uses
  the Wendland C^2 kernel at N_H = 50 (fixed support) exactly like
  `sampleOptimal` does; only the start and the scaling differ.
* **paired** (the paper's crosses): the paper's fully-paired
  distribution of section 5.1.1 -- each FCC point replaced by two
  coincident particles, the spacing scaled by 2^(1/3) so the mean
  density (but not rho_hat) stays unchanged. For any spherically
  symmetric kernel with finite W(0) the estimate at a pair particle at
  smoothing H equals the FCC estimate at H / 2^(1/3) EXACTLY (the
  partner's self-term compensates the 2^(1/3) kernel rescaling);
  `fig03_density_estimation.py` checks this identity numerically. It is
  also the paper's pairing criterion: pairing is favoured iff
  rho_hat(N_H/2) < rho_hat(N_H), i.e. iff the FCC curve is rising at
  N_H (to the right of its minimum).

The glass is GPU work (warp kernels); it is cached to a .npz so the
figure script does not regenerate it on every run.
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

wp.init()

from warpSPHCore.sampling import sampleDensestLattice

_HERE = Path(__file__).resolve().parent


@dataclass
class ParticleConfig:
    name: str
    positions: np.ndarray  # (N, 3), float64, in [0, box)
    box: float             # cubic periodic box size
    mass: float            # per-particle mass (mean density = 1)
    n: int

    @property
    def density(self) -> float:
        return self.mass * self.n / self.box**3


# ---------------------------------------------------------------------------
# FCC (densest-sphere packing)
# ---------------------------------------------------------------------------

def fccConfig(n: int = 4000, L: float = 1.0) -> ParticleConfig:
    s = sampleDensestLattice(n, L, 3)
    assert s.exact and np.allclose(s.box, [L, L, L])
    assert s.positions.shape[0] == s.count
    return ParticleConfig("fcc", s.positions, float(L),
                          L**3 / s.count, s.count)


# ---------------------------------------------------------------------------
# Paired (section 5.1.1): each FCC point -> two coincident particles,
# spacing x 2^(1/3), mean density unchanged.
# ---------------------------------------------------------------------------

def pairedConfig(fcc: ParticleConfig) -> ParticleConfig:
    g = 2.0**(1.0 / 3.0)
    box = g * fcc.box
    # Each FCC point becomes a coincident pair; the pair lattice is the
    # FCC lattice scaled by 2^(1/3), which tiles the 2^(1/3)-scaled box
    # exactly (scaling commutes with the periodicity).
    pos = np.concatenate([fcc.positions * g, fcc.positions * g], axis=0)
    n = 2 * fcc.n
    # mass per particle unchanged -> rho = 2N m / (2 V) = Nm/V = 1.
    assert abs(n * fcc.mass / box**3 - 1.0) < 1e-12
    return ParticleConfig("paired", pos, float(box), fcc.mass, n)


# ---------------------------------------------------------------------------
# Glass (jittered lattice -> delta-shift relaxation, warpSPH components)
# ---------------------------------------------------------------------------

_GLASS_NH = 50        # Wendland C2 support during relaxation (N_H = 50)
_GLASS_JITTER = 0.1   # Gaussian jitter width in units of the lattice dx
_GLASS_CFL = 0.3
_GLASS_MACH = 0.1     # the delta.py fallback when there are no velocities
_GLASS_MAX_ITERS = 1000
_GLASS_TOL = 1.0e-5   # on the per-particle displacement magnitude (absolute)


def glassConfig(n: int = 4096, L: float = 1.0, seed: int = 42,
                cachePath: Path | str | None = None) -> ParticleConfig:
    import torch

    if cachePath is not None:
        cachePath = Path(cachePath)
        if cachePath.exists():
            d = np.load(cachePath)
            return ParticleConfig("glass", d["positions"], float(L),
                                  float(d["mass"]), int(d["n"]))

    from warpSPH.utils.domain import buildDomainDescription
    from warpSPH.sample.regular import sampleRegularParticles
    from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp
    from warpSPHCore import (
        HashMapLengthMode,
        KernelFunctions,
        OperationProperties,
        SupportScheme,
        WarpOperation,
        radiusSearchCompactHashMap,
        warpOperation,
    )
    from warpSPHCore.dataTypes import ParticleState
    from warpSPHCore.kernels import sphKernelScale

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain = buildDomainDescription(L, 3, periodic=True, device=device,
                                    dtype=torch.float64)
    nx = int(round(n ** (1.0 / 3.0)))
    ps = sampleRegularParticles(nx, domain, targetNeighbors=_GLASS_NH)
    n = ps.positions.shape[0]
    mass = L**3 / n
    dx = ps.masses.pow(1.0 / 3.0).mean().item()

    gen = torch.Generator(device=device).manual_seed(seed)
    pos = ps.positions + _GLASS_JITTER * torch.randn(
        ps.positions.shape, device=device, dtype=torch.float64,
        generator=gen) * dx
    ps = ps._replace(positions=pos % torch.tensor(L, device=device,
                                                  dtype=torch.float64))

    scale3 = float(sphKernelScale(KernelFunctions.Wendland2.value, 3))
    h2 = (float(ps.supports[0]) / scale3) ** 2
    scale = -_GLASS_CFL * _GLASS_MACH * 2.0 * h2

    op = OperationProperties(operation=WarpOperation.Density,
                             kernel=KernelFunctions.Wendland2,
                             supportMode=SupportScheme.Gather)
    kinds = torch.zeros(n, dtype=torch.int32, device=device)
    L_t = torch.tensor(L, device=device, dtype=torch.float64)

    def state():
        return ParticleState(positions=ps.positions, supports=ps.supports,
                             masses=ps.masses, kinds=kinds,
                             densities=ps.densities)

    maxDisp = float("inf")
    for it in range(_GLASS_MAX_ITERS):
        adjacency = radiusSearchCompactHashMap(
            ps, domain, mode=SupportScheme.SuperSymmetric,
            hashMapLengthMode=HashMapLengthMode.Fixed,
            fixedHashMapLength=4096)
        dens = warpOperation(state(), operationProperties=op,
                             domain=domain, adjacency=adjacency)
        ps = ps._replace(densities=dens)
        raw = computeDeltaShiftWarp(
            state(), operationProperties=op, domain=domain,
            CFL=_GLASS_CFL, computeMach=False, c_max=_GLASS_MACH,
            rho0=1.0, dx=dx, adjacency=adjacency)
        ps = ps._replace(positions=(ps.positions + scale * raw) % L_t)
        maxDisp = float((scale * raw).norm(dim=1).max())
        if it % 100 == 0 or it == _GLASS_MAX_ITERS - 1:
            print(f"  glass: iter {it:4d}  max|disp| = {maxDisp:.3e}")
        if it > 50 and maxDisp < _GLASS_TOL:
            break

    positions = ps.positions.detach().cpu().numpy().astype(np.float64)
    assert np.all(positions >= 0.0) and np.all(positions < L)
    if cachePath is not None:
        cachePath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cachePath, positions=positions,
                            mass=np.float64(mass), n=np.int64(n))
        print(f"  glass: cached {cachePath}")
    return ParticleConfig("glass", positions, float(L), mass, n)


if __name__ == "__main__":
    fcc = fccConfig()
    paired = pairedConfig(fcc)
    glass = glassConfig(
        cachePath=_HERE.parent.parent.parent / ".tmp" /
        "glass_N4096_L1.0_seed42.npz")
    for c in (fcc, paired, glass):
        print(f"{c.name:<8} N = {c.n:6d}  box = {c.box:.6f}  "
              f"mass = {c.mass:.6e}  rho = {c.density:.9f}")
