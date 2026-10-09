#!/usr/bin/env python3
"""Matched Gresho-Chan vortex: the core MFM / MFV backend against GIZMO (~/dev/gizmo).

One set of particles (square lattice in the periodic unit box, optional jitter), one set of masses
(effective volumes of the *core* geometry, so that rho = 1 to round-off), the same kernel (Wendland C4,
GIZMO `KERNEL_FUNCTION=7`), the same neighbour number (`DesNumNgb`), gamma = 5/3 and the same error
metrics evaluated on the final snapshots of both codes:

* `L1_gizmo`: GIZMO's own test metric, mean|v_phi - v_phi,exact| / mean|v_phi,exact| over r < 0.4 at the
  final particle positions (test/gresho/test_gresho.py),
* `L1_vol`: the volume-weighted L1 of |v - v_exact| used by `run_mfm_gresho.py`,
* `KE/KE0`.

    python gizmo_gresho.py --gizmo-dir <scratch GIZMO build> --work <dir> --n1d 64 --modes MFM MFV

GIZMO must be built with `test/gresho/Config.sh` (+ `HYDRO_MESHLESS_FINITE_VOLUME` for MFV, `KERNEL_FUNCTION=7`,
`EOS_GAMMA=(5./3.)`); the script rebuilds it per mode when `--build` is given (the build takes ~1 min).
"""
from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from warpSPHCore import DomainDescription                        # noqa: E402
from warpSPHCore.enumTypes import KernelFunctions               # noqa: E402

import mfm_driver as drv                                        # noqa: E402
from run_mfm_gresho import G, profile, vortex_velocity          # noqa: E402

KERNEL = KernelFunctions.Wendland4
BOX = 1.0


def lattice(n1d, jitter, seed):
    dx = BOX / n1d
    g = (np.arange(n1d) + 0.5) * dx
    x = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    if jitter > 0:
        x = (x + (np.random.default_rng(seed).random(x.shape) - 0.5) * jitter * dx) % BOX
    return x


def domain(device):
    return DomainDescription(dim=2, min=torch.zeros(2, dtype=torch.float64, device=device),
                             max=torch.full((2,), BOX, dtype=torch.float64, device=device),
                             periodic=torch.ones(2, dtype=torch.bool, device=device))


def initial_state(x, nngb, device):
    """Core-geometry initial state: consistent h, masses = rho V_eff with rho = 1, the Gresho profile."""
    dom = domain(device)
    N = x.shape[0]
    box = torch.full((2,), BOX, dtype=torch.float64, device=device)
    h = drv.supports_from_volume(torch.full((N,), BOX ** 2 / N, dtype=torch.float64, device=device), nngb, 2)
    st0 = drv.MFMState(pos=x, mass=torch.ones(N, dtype=x.dtype, device=device),
                       Q=torch.zeros(N, 4, dtype=x.dtype, device=device), h=h)
    for _ in range(4):
        g0 = drv.geometry(st0, dom, KERNEL, "project")
        st0.h = drv.supports_from_volume(g0.volume, nngb, 2)
    v, r = vortex_velocity(x, 0.5 * box, box)
    _, P = profile(r)
    rho = torch.ones(N, dtype=x.dtype, device=device)
    mass = rho * g0.volume
    return dom, box, v, P, rho, mass, st0.h


def write_ics(path, x, v, P, mass, gamma=G, rho0=1.0):
    import h5py
    N = x.shape[0]
    pos3 = np.zeros((N, 3)); pos3[:, :2] = x
    vel3 = np.zeros((N, 3)); vel3[:, :2] = v
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"] = np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BOX
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 1
        p = f.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos3)
        p.create_dataset("Velocities", data=vel3)
        p.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint32))
        p.create_dataset("Masses", data=mass)
        p.create_dataset("InternalEnergy", data=P / ((gamma - 1.0) * rho0))


def metrics(x, v, vol=None):
    """(L1 in GIZMO's test metric, volume-weighted L1, KE) at positions x (N,2) with velocities v (N,2)."""
    x = np.asarray(x, dtype=float); v = np.asarray(v, dtype=float)
    d = x - 0.5
    d -= np.round(d / BOX) * BOX
    r = np.hypot(d[:, 0], d[:, 1])
    vphi_ex = np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))
    vphi = (-d[:, 1] * v[:, 0] + d[:, 0] * v[:, 1]) / (r + 1e-30)
    inside = r < 0.4
    l1g = np.mean(np.abs(vphi[inside] - vphi_ex[inside])) / np.mean(np.abs(vphi_ex[inside]))
    vex = np.stack([-vphi_ex * d[:, 1] / (r + 1e-30), vphi_ex * d[:, 0] / (r + 1e-30)], -1)
    err = np.linalg.norm(v - vex, axis=-1)
    l1v = float(np.average(err, weights=vol)) if vol is not None else float(err.mean())
    return float(l1g), l1v, 0.0


def run_mine(x_np, mode, nngb, tend, cfl, device, backend="warp"):
    drv.BACKEND = backend
    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    dom, box, v, P, rho, mass, h = initial_state(x, nngb, device)
    st = drv.init_state(x, mass, rho, v, P, h, G)
    ke0 = float((0.5 * st.Q[:, 1:3].pow(2).sum(-1) / st.Q[:, 0]).sum())
    t0, steps = time.time(), 0
    while st.t < tend - 1e-12:
        st, g, dt = drv.step(st, dom, KERNEL, G, nngb, cfl=cfl, mode=mode, closure="project", order=2, tmax=tend)
        steps += 1
    g = drv.geometry(st, dom, KERNEL, "project")
    _, vel, _ = drv.primitives(st.Q, g.volume, G)
    ke = float((0.5 * st.Q[:, 1:3].pow(2).sum(-1) / st.Q[:, 0]).sum()) / ke0
    l1g, l1v, _ = metrics(st.pos.cpu().numpy(), vel.cpu().numpy(), g.volume.cpu().numpy())
    return dict(L1_gizmo=l1g, L1_vol=l1v, KE=ke, steps=steps, sec=time.time() - t0)


PARAMS = """InitCondFile                       gresho_ics
OutputDir                          output
TimeMax                            {tend}
BoxSize                            1
TimeBetSnapshot                    {tend}
DesNumNgb                          {nngb}
MaxNumNgbDeviation                 0.1
MaxMemSize                         2000
ErrTolIntAccuracy                  0.001
CourantFac                         {cfl}
MaxRMSDisplacementFac              0.125
MaxSizeTimestep                    1.0
ArtCondConstant                    0.25
ViscosityAMin                      0.025
ViscosityAMax                      2
ResubmitOn                         0
ResubmitCommand                    none
ErrTolTheta                        0.7
ErrTolForceAcc                     0.001
TimeBetStatistics                  0.5
"""

CONFIG = ["BOX_PERIODIC", "BOX_SPATIAL_DIMENSION=2", "SELFGRAVITY_OFF", "EOS_GAMMA=(5./3.)", "KERNEL_FUNCTION=7",
          "INPUT_IN_DOUBLEPRECISION", "OUTPUT_IN_DOUBLEPRECISION", "DEVELOPER_MODE"]


def build_gizmo(gdir, mode, extra, config=None, tag=""):
    cfg = (CONFIG if config is None else config) + ["HYDRO_MESHLESS_FINITE_MASS" if mode == "MFM" else "HYDRO_MESHLESS_FINITE_VOLUME"] + extra
    (gdir / "Config.sh").write_text("\n".join(cfg) + "\n")
    subprocess.run("make clean >/dev/null 2>&1; make -j8 > build.log 2>&1", shell=True, cwd=gdir, check=True)
    exe = gdir / f"GIZMO_{mode}{tag}"
    (gdir / "GIZMO").replace(exe)
    return exe


def run_gizmo(exe, work, x_np, mass, v, P, nngb, tend, cfl, ranks):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    write_ics(work / "gresho_ics.hdf5", x_np, v, P, mass)
    (work / "gresho.params").write_text(PARAMS.format(tend=tend, nngb=nngb, cfl=cfl))
    t0 = time.time()
    with open(work / "gizmo.out", "w") as fo, open(work / "gizmo.err", "w") as fe:
        rc = subprocess.run(["mpirun", "--oversubscribe", "-np", str(ranks), str(exe), "gresho.params"], cwd=work,
                            stdout=fo, stderr=fe).returncode
    sec = time.time() - t0
    snaps = sorted(out.glob("snapshot_*.hdf5"))
    if rc != 0 or not snaps:
        raise RuntimeError(f"GIZMO failed (rc {rc}); see {work}/gizmo.err")
    import h5py
    with h5py.File(snaps[-1], "r") as f:
        pos = f["PartType0/Coordinates"][:, :2]
        vel = f["PartType0/Velocities"][:, :2]
        m = f["PartType0/Masses"][:]
        ids = f["PartType0/ParticleIDs"][:]
        rho = f["PartType0/Density"][:] if "PartType0/Density" in f else np.ones_like(m)
        t = float(f["Header"].attrs["Time"])
    # KE relative to the initial (same masses, initial velocities)
    ke0 = 0.5 * np.sum(mass * (v ** 2).sum(-1))
    ke = 0.5 * np.sum(m * (vel ** 2).sum(-1)) / ke0
    l1g, l1v, _ = metrics(pos, vel, m / np.maximum(rho, 1e-30))
    return dict(L1_gizmo=l1g, L1_vol=l1v, KE=float(ke), sec=sec, t=t, snaps=len(snaps))


def gizmo_volumes(exe, work, x_np, mass0, v, P0, gamma, nngb, ranks):
    """GIZMO's own effective volumes V = m / Density of the initial snapshot (a zero-length run). GIZMO iterates h to
    `DesNumNgb +- MaxNumNgbDeviation`, so on a disordered set its h -- and V = 1/omega -- differ from the core's by a few per
    cent; handing it masses from the core's volumes then starts it from a density field with that much noise."""
    pre = work / "pre"
    pre.mkdir(parents=True, exist_ok=True)
    out = pre / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    write_ics(pre / "gresho_ics.hdf5", x_np, v, P0, mass0, gamma=gamma)
    (pre / "gresho.params").write_text(PARAMS.format(tend=0.02, nngb=nngb, cfl=0.2))
    with open(pre / "gizmo.out", "w") as fo, open(pre / "gizmo.err", "w") as fe:
        rc = subprocess.run(["mpirun", "--oversubscribe", "-np", str(ranks), str(exe), "gresho.params"], cwd=pre,
                            stdout=fo, stderr=fe).returncode
    import h5py
    with h5py.File(out / "snapshot_000.hdf5", "r") as f:
        m, rho = f["PartType0/Masses"][:], f["PartType0/Density"][:]
        ids = f["PartType0/ParticleIDs"][:]
    order = np.argsort(ids)
    return (m / rho)[order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gizmo-dir", required=True, help="scratch copy of the GIZMO tree (modified Makefile for the local libs)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--n1d", type=int, nargs="+", default=[64])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--nngb", type=float, default=30.0)
    ap.add_argument("--tend", type=float, default=3.0)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--extra-config", nargs="*", default=[])
    ap.add_argument("--skip-gizmo", action="store_true")
    ap.add_argument("--skip-mine", action="store_true")
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdir, work = Path(a.gizmo_dir), Path(a.work)
    exes = {}
    for mode in a.modes:
        exe = gdir / f"GIZMO_{mode}"
        if a.build or not exe.exists():
            exe = build_gizmo(gdir, mode, a.extra_config)
        exes[mode] = exe
    for n1d in a.n1d:
        x = lattice(n1d, a.jitter, 7)
        xt = torch.tensor(x, dtype=torch.float64, device=device)
        dom, box, v, P, rho, mass, h = initial_state(xt, a.nngb, device)
        for mode in a.modes:
            line = f"{mode} N={n1d}^2 jitter {a.jitter} nngb {a.nngb:g} cfl {a.cfl} t={a.tend:g}"
            if not a.skip_mine:
                r = run_mine(x, mode, a.nngb, a.tend, a.cfl, device)
                line += (f"\n   core : L1_gizmo {r['L1_gizmo']:.4f}  L1_vol {r['L1_vol']:.4f}  KE/KE0 {r['KE']:.4f}  "
                         f"steps {r['steps']} ({r['sec']:.0f}s)")
            if not a.skip_gizmo:
                wdir = work / f"{mode}_{n1d}_j{a.jitter}"
                VG = gizmo_volumes(exes[mode], wdir, x, mass.cpu().numpy(), v.cpu().numpy(), P.cpu().numpy(), G, a.nngb, a.ranks)
                r = run_gizmo(exes[mode], wdir, x, rho.cpu().numpy() * VG, v.cpu().numpy(),
                              P.cpu().numpy(), a.nngb, a.tend, a.cfl, a.ranks)
                line += (f"\n   GIZMO: L1_gizmo {r['L1_gizmo']:.4f}  L1_vol {r['L1_vol']:.4f}  KE/KE0 {r['KE']:.4f}  "
                         f"({r['sec']:.0f}s, t={r['t']:g}, {r['snaps']} snapshots)")
            print(line, flush=True)


if __name__ == "__main__":
    main()
