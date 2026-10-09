#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the order-p polynomial fit (PolyFit).

Backward-mode check of `warpSPHCore.polyfit.PolyFit` -- the MLS/RKPM fit
(`constant=True`) and the LABFM form (`constant=False`) -- through the whole
chain: warp moment kernel -> packed matrix -> torch equilibrate / eigvalsh /
pinv -> warp right-hand-side kernel -> coefficient read-out. Differentiable
inputs: positions, supports, masses, densities and the field values, for the
gradient (p = 1, 2) and the Laplacian (p = 2) outputs.

The set is a small *periodic* jittered lattice so every particle has a full
stencil: a rank-deficient row (pseudo-inverse at a rank change) is a
non-differentiable point and not what this script is about. Run-to-run
backward differences of ~1e-13 (atomic scatter order) need ``nondet_tol``.

    python scripts/gradcheck/gradcheck_polyfit_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from warpSPHCore import (DomainDescription, OperationProperties, ParticleState,
                         radiusSearchCompactHashMap, warpOperation)
from warpSPHCore.enumTypes import (KernelFunctions, OperationDirection,
                                   SupportScheme, WarpOperation)
from warpSPHCore.polyfit import PolyFit
from warpSPHCore.util import volumeToSupport

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2


def build_case(n: int = 7, nbrs: int = 14, jitter: float = 0.2, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    ax = torch.arange(n, dtype=DTYPE)
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2) / n
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=DTYPE) - 0.5) * jitter / n
    N = pts.shape[0]
    h = float(volumeToSupport((1.0 / n) ** 2, nbrs, 2))
    domain = DomainDescription(torch.zeros(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE),
                               torch.ones(2, dtype=torch.bool), 2)
    p = ParticleState(positions=pts, supports=torch.full((N,), h, dtype=DTYPE),
                      masses=torch.full((N,), (1.0 / n) ** 2, dtype=DTYPE),
                      densities=None, kinds=torch.zeros(N, dtype=torch.int32))
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.SuperSymmetric)
    rho = warpOperation(p, OperationProperties(
        kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll), domain, adjacency=adjacency).detach()
    return p, domain, adjacency, rho


def run(name, order, constant, output) -> bool:
    p, domain, adjacency, rho = build_case()
    kinds = p.kinds
    pos = p.positions.clone().requires_grad_(True)
    sup = p.supports.clone().requires_grad_(True)
    mas = p.masses.clone().requires_grad_(True)
    den = rho.clone().requires_grad_(True)
    f = (torch.sin(3 * p.positions[:, 0]) * torch.cos(2 * p.positions[:, 1])).clone().requires_grad_(True)

    def fn(pos_, sup_, mas_, den_, f_):
        q = ParticleState(positions=pos_, supports=sup_, masses=mas_, densities=den_, kinds=kinds)
        pf = PolyFit.build(q, domain, KERNEL, order, constant=constant, adjacency=adjacency)
        assert not pf.deficient.any()
        return pf.gradient(f_) if output == "gradient" else pf.laplacian(f_)

    try:
        ok = torch.autograd.gradcheck(fn, (pos, sup, mas, den, f), eps=1e-6, atol=1e-4, rtol=1e-3,
                                      nondet_tol=1e-8)
    except Exception as e:                      # noqa: BLE001
        print(f"FAIL {name}: {str(e)[:600]}")
        return False
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    return bool(ok)


def main() -> int:
    wp.init()
    cases = [
        ("MLS p=1 gradient", 1, True, "gradient"),
        ("MLS p=2 gradient", 2, True, "gradient"),
        ("MLS p=2 laplacian", 2, True, "laplacian"),
        ("LABFM p=2 gradient", 2, False, "gradient"),
        ("LABFM p=2 laplacian", 2, False, "laplacian"),
    ]
    results = [run(*c) for c in cases]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
