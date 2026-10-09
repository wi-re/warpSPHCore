#!/usr/bin/env python3
"""Cost of one MFM step's worth of work (geometry + closure + rates) for the torch
pair-list backend and the warp CSR backend; 2-D jittered periodic lattice, ~30 neighbours.

    python bench_mfm_backends.py --n 16384 65536 262144
"""
from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import time

import torch
import warp as wp

from warpSPHCore import DomainDescription, ParticleState
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessGeometry, MeshlessWarp, mfmRates

G = 5.0 / 3.0


def lattice(n_side, jitter, device):
    dt = torch.float64
    ax = torch.arange(n_side, dtype=dt, device=device)
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2) / n_side
    gen = torch.Generator(device="cpu").manual_seed(1)
    pts = (pts + ((torch.rand(pts.shape, generator=gen, dtype=dt) - 0.5) * jitter / n_side).to(device)) % 1.0
    return pts


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16384, 65536, 262144])
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--nngb", type=float, default=30.0)
    ap.add_argument("--backends", nargs="+", default=["torch", "warp"])
    a = ap.parse_args()
    wp.init(); torch.set_grad_enabled(False)
    dev = "cuda"
    K = KernelFunctions.Wendland4
    dom = DomainDescription(torch.zeros(2, dtype=torch.float64, device=dev), torch.ones(2, dtype=torch.float64, device=dev),
                            torch.ones(2, dtype=torch.bool, device=dev), 2)
    for n in a.n:
        side = int(round(math.sqrt(n)))
        pos = lattice(side, a.jitter, dev)
        N = pos.shape[0]
        h = torch.full((N,), math.sqrt(a.nngb / (math.pi * N)), dtype=torch.float64, device=dev)
        rho = 1 + 0.1 * torch.sin(6.28 * pos[:, 0])
        vel = torch.stack([torch.sin(6.28 * pos[:, 1]), torch.cos(6.28 * pos[:, 0])], 1) * 0.3
        pres = 1 + 0.1 * torch.cos(6.28 * pos[:, 1])
        P = ParticleState(positions=pos, supports=h, masses=torch.ones(N, dtype=torch.float64, device=dev),
                          densities=torch.ones(N, dtype=torch.float64, device=dev), kinds=torch.zeros(N, dtype=torch.int32, device=dev))
        for be in a.backends:
            torch.cuda.reset_peak_memory_stats()
            try:
                if be == "torch":
                    tb, g = timed(lambda: MeshlessGeometry.build(P, dom, K), 1)
                    tr, _ = timed(lambda: mfmRates(g, rho, vel, pres, G, dt=1e-3, mode="MFM"), 1)
                else:
                    tb, g = timed(lambda: MeshlessWarp.build(pos, h, dom, K), 1)
                    tr, _ = timed(lambda: g.rates(rho, vel, pres, G, dt=1e-3, mode="MFM"), 2)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"N={N:8d} {be:5s}: geometry+closure {tb*1e3:9.1f} ms   rates {tr*1e3:9.1f} ms   peak torch mem {mem:6.2f} GiB", flush=True)
            except torch.OutOfMemoryError:
                print(f"N={N:8d} {be:5s}: out of memory", flush=True)
            del_ = None


if __name__ == "__main__":
    main()
