#!/usr/bin/env python3
"""Disorder probe: static higher-order operator probes on physically
disordered particle distributions.

Runs the harness operator probes (operators.py: interpolate / gradient x
standard / CRK / renorm / renormVal) at fixed N and h/dx on the saved TGV
distributions (data/tgv2d_*.npz, see data/README.md) plus a purely-jittered
baseline, with the analytic TGV velocity at t = 2 as the probe field. This is
the measurement documented in pde/TGV_NOTES.md section 7.5: how the
W2/W4/W6 operators record a smooth field from a distribution of a given
anisotropy. The probe field is the analytic TGV velocity at t = 2 (k = 2,
nu = 0.01, uMag = 1, L = 2 pi -- the tgv-wc case's own parameters) with its
analytic gradient, which is verified against central differences at start-up.
The field is biharmonic (nabla^2 v = 0), so the Laplacian probe has a zero
target and is skipped.

Distributions (all N = 16384, h = 4 dx, box [0, 2 pi]^2 periodic):
  jitter        the case IC minus the shuffle relaxation: the frontend's
                regular lattice (sampleRegularParticles) + the same Gaussian
                jitter shuffleParticles applies (sigma = 1.0 * h), seed 42
                (the real run's global RNG state is unknowable)
  noshift_t0    saved: reference glass (128 iters of 1/8-strength relaxation)
  noshift_t2    saved: uncorrected flow end state (disorder accumulated)
  fullshift_t0  saved: over-mixed lattice (full-strength relaxation)
  fullshift_t2  saved: re-regularised end state (full-strength shifting)

Outputs (git-ignored, regenerable):
  results/disorder_probe_rows.csv          full run (5 dists x 3 kernels)
  results/disorder_probe_rows_smoke.csv    smoke (3 dists x Wendland4)
  results/disorder_probe_verdict.json      smoke verdict (CI gate)

Usage:
    python run_disorder_probe.py                 # full probe (~2-5 min GPU)
    python run_disorder_probe.py --smoke         # CI gate (one kernel)
    python run_disorder_probe.py --dists jitter noshift_t2
    python run_disorder_probe.py --kernels Wendland4 Wendland6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["warpSPHCore_PRECISION"] = "float64"   # must precede warpSPH import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "pde"))             # pde_cases: analytic TGV field

import numpy as np
import torch
import warp as wp

wp.init()
torch.set_grad_enabled(False)

from warpSPHCore import DomainDescription
from warpSPHCore.enumTypes import KernelFunctions

from distribution import anisotropy_summary, distribution_moments
from metrics import error_norms
from operators import CorrectionCache, run_probe
from particle_sets import build_case_from_positions
from pde_cases import tgv_analytic_velocity
from test_fields import Field

DATA = HERE / "data"
RESULTS = HERE / "results"

L = 2.0 * np.pi
NX = 128
DX = L / NX
H = 0.19634954          # the saved uniform support (= 4 dx)
T_PROBE = 2.0
PARAMS = dict(k=2, nu=0.01, uMag=1.0)
KERNELS = ("Wendland2", "Wendland4", "Wendland6")
MODES = ("standard", "crk", "renorm", "renormVal")
PROBES = ("interpolate", "gradient")
TARGET_NEIGHBORS = 40   # metadata only (h is taken as given)
JITTER_SEED = 42
SUMMARY_KEYS = ("first_anisotropy_rms", "second_anisotropy_rms",
                "second_anisotropy_max", "kernel_sum_min", "kernel_sum_max")


# ---------------------------------------------------------------------------
# probe field: analytic TGV velocity at t = T_PROBE, with analytic gradient
# ---------------------------------------------------------------------------

_KTGv = PARAMS["k"] / 2.0
_PHASE = np.pi / 2.0    # the case's even-k rule
_A = np.exp(-2.0 * PARAMS["nu"] * _KTGv ** 2 * T_PROBE) * PARAMS["uMag"]


def _tgv_value(x):
    return tgv_analytic_velocity(x, T_PROBE, PARAMS)


def _tgv_grad(x):
    # u = A cos(x+ph) sin(y+ph), v = -A sin(x+ph) cos(y+ph); div-free.
    sx = torch.sin(x[:, 0] + _PHASE)
    cx = torch.cos(x[:, 0] + _PHASE)
    sy = torch.sin(x[:, 1] + _PHASE)
    cy = torch.cos(x[:, 1] + _PHASE)
    g = torch.empty(x.shape[0], 2, 2, dtype=x.dtype, device=x.device)
    g[:, 0, 0] = -_A * sx * sy
    g[:, 0, 1] = _A * cx * cy
    g[:, 1, 0] = -_A * cx * cy
    g[:, 1, 1] = _A * sx * sy
    return g


FIELD = Field(name=f"tgv_v_t{T_PROBE:g}", kind="smooth", degree=-1,
              is_vector=True, value=_tgv_value, grad=_tgv_grad, lap=None)


def check_analytic_gradient(device: str) -> float:
    # central-difference guard against a sign slip in _tgv_grad
    x = torch.tensor([[0.3, 0.7], [1.9, -0.4]], dtype=torch.float64,
                     device=device)
    eps = 1e-6
    g_num = torch.zeros(x.shape[0], 2, 2, dtype=torch.float64, device=device)
    for d in range(2):
        xp = x.clone()
        xm = x.clone()
        xp[:, d] += eps
        xm[:, d] -= eps
        g_num[:, :, d] = (_tgv_value(xp) - _tgv_value(xm)) / (2.0 * eps)
    diff = (g_num - _tgv_grad(x)).abs().max().item()
    if diff >= 1e-7:
        raise AssertionError(f"analytic gradient mismatch: {diff:.2e}")
    return diff


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

def stored_summary(d, tag: str) -> dict:
    return {k: float(d[f"{tag}_{k}"]) for k in SUMMARY_KEYS}


def load_saved(npz_name: str, tag: str, device: str):
    d = np.load(DATA / npz_name)
    pos = torch.tensor(d[f"{tag}_positions"], dtype=torch.float64,
                       device=device)
    mass = torch.tensor(d[f"{tag}_masses"], dtype=torch.float64, device=device)
    return pos, mass, float(d[f"{tag}_h"]), stored_summary(d, tag)


def baseline_jitter(device: str):
    # The case IC minus the shuffle relaxation: the same regular lattice the
    # frontend sampler builds (half-cell offset, uniform mass = cell volume),
    # plus the same Gaussian jitter shuffleParticles applies (sigma = h).
    from warpSPH.sample.regular import sampleRegularParticles
    domain = DomainDescription(
        torch.zeros(2, dtype=torch.float64, device=device),
        torch.full((2,), L, dtype=torch.float64, device=device),
        torch.ones(2, dtype=torch.bool, device=device), 2)
    pc = sampleRegularParticles(NX, domain, TARGET_NEIGHBORS, jitter=0.0)
    pos = pc.positions
    assert pos.shape[0] == NX * NX, \
        f"lattice gave {pos.shape[0]} particles, want {NX * NX}"
    torch.manual_seed(JITTER_SEED)
    pos = pos + torch.randn_like(pos) * H
    mom = distribution_moments(pos.cpu(), pc.masses.cpu(), h=H,
                               L=torch.tensor([L, L], dtype=torch.float64),
                               dim=2, kernel="wendland4", periodic=True)
    summ = {k: anisotropy_summary(mom)[k] for k in SUMMARY_KEYS}
    return pos, pc.masses, H, summ


def build_dists(names: list[str], device: str) -> list[tuple]:
    dists = []
    for name in names:
        if name == "jitter":
            d = baseline_jitter(device)
        elif name == "noshift_t0":
            d = load_saved("tgv2d_noshift_nx128.npz", "t0", device)
        elif name == "noshift_t2":
            d = load_saved("tgv2d_noshift_nx128.npz", "t2", device)
        elif name == "fullshift_t0":
            d = load_saved("tgv2d_fullshift_nx128.npz", "t0", device)
        elif name == "fullshift_t2":
            d = load_saved("tgv2d_fullshift_nx128.npz", "t2", device)
        else:
            raise ValueError(f"unknown distribution {name!r}")
        dists.append((name,) + d)
    return dists


# ---------------------------------------------------------------------------
# probe loop
# ---------------------------------------------------------------------------

def run_dists(dists, kernels: tuple[str, ...], device: str) -> list[dict]:
    rows = []
    for dist_name, pos, mass, h, summ in dists:
        print(f"[{dist_name}] a1_rms={summ['first_anisotropy_rms']:.3e} "
              f"a2_rms={summ['second_anisotropy_rms']:.3e} "
              f"a2_max={summ['second_anisotropy_max']:.3e} "
              f"Cs=[{summ['kernel_sum_min']:.4f},{summ['kernel_sum_max']:.4f}]",
              flush=True)
        for kn in kernels:
            case = build_case_from_positions(
                pos, mass, h, box=np.array([L, L]), dx=DX,
                target_neighbors=TARGET_NEIGHBORS, periodic=True,
                device=device, kernel=KernelFunctions[kn])
            cache = CorrectionCache(case)
            dens = case.particles.densities
            values = FIELD.value(case.positions)
            analytic = {"interpolate": values,
                        "gradient": FIELD.grad(case.positions)}
            for probe in PROBES:
                for mode in MODES:
                    out = run_probe(case, cache, values, probe, mode)
                    norms = error_norms(out, analytic[probe])
                    rows.append(dict(
                        dist=dist_name, kernel=kn, probe=probe, mode=mode,
                        l1=norms["l1"], l2=norms["l2"], linf=norms["linf"],
                        a1_rms=summ["first_anisotropy_rms"],
                        a2_rms=summ["second_anisotropy_rms"],
                        a2_max=summ["second_anisotropy_max"],
                        density_min=float(dens.min()),
                        density_max=float(dens.max())))
                    print(f"  [{dist_name}/{kn}/{probe}/{mode}] "
                          f"l2={norms['l2']:.4e} linf={norms['linf']:.4e}",
                          flush=True)
            del case, cache, values
            torch.cuda.empty_cache()
    return rows


COLS = ["dist", "kernel", "probe", "mode", "l1", "l2", "linf", "a1_rms",
        "a2_rms", "a2_max", "density_min", "density_max"]


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def print_pivot(rows: list[dict], dists, kernels) -> None:
    by = {(r["dist"], r["kernel"], r["probe"], r["mode"]): r for r in rows}
    for probe in PROBES:
        for mode in MODES:
            print(f"\n## {probe} / {mode} -- l2 (rms) error", flush=True)
            print("dist".ljust(14) + "".join(k.ljust(16) for k in kernels),
                  flush=True)
            for dn, *_ in dists:
                line = dn.ljust(14)
                for kn in kernels:
                    line += f"{by[(dn, kn, probe, mode)]['l2']:.4e}".ljust(16)
                print(line, flush=True)


def smoke_verdict(rows: list[dict], grad_check: float) -> dict:
    """Headline invariants of the section 7.5 result, as pass/fail checks.

    glass = noshift_t0, disordered = fullshift_t0 (both Wendland4):
      * CRK interpolate is distribution-independent (flat to within 2x)
      * the standard operators are disorder-sensitive (>10x on interpolate
        and gradient)
      * CRK interpolate stays at the glass-level floor (< 5e-3)
    """
    def l2(dist, probe, mode):
        r = next(r for r in rows
                 if r["dist"] == dist and r["probe"] == probe
                 and r["mode"] == mode and r["kernel"] == "Wendland4")
        return r["l2"]

    crk_glass, crk_dis = l2("noshift_t0", "interpolate", "crk"), \
        l2("fullshift_t0", "interpolate", "crk")
    std_glass, std_dis = l2("noshift_t0", "interpolate", "standard"), \
        l2("fullshift_t0", "interpolate", "standard")
    grd_glass, grd_dis = l2("noshift_t0", "gradient", "standard"), \
        l2("fullshift_t0", "gradient", "standard")
    checks = {
        "analytic_grad_check_max_diff": grad_check,
        "crk_interp_flat_ratio": max(crk_glass, crk_dis)
                                 / min(crk_glass, crk_dis),
        "std_interp_disorder_ratio": std_dis / std_glass,
        "std_grad_disorder_ratio": grd_dis / grd_glass,
        "crk_interp_glass": crk_glass,
        "crk_interp_disordered": crk_dis,
        "all_errors_finite_positive": all(
            r["l2"] > 0 and np.isfinite(r["l2"]) for r in rows),
    }
    checks["ok"] = bool(
        grad_check < 1e-7
        and checks["crk_interp_flat_ratio"] < 2.0
        and checks["std_interp_disorder_ratio"] > 10.0
        and checks["std_grad_disorder_ratio"] > 10.0
        and crk_glass < 5e-3 and crk_dis < 5e-3
        and checks["all_errors_finite_positive"])
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dists", nargs="+", default=[
        "jitter", "noshift_t0", "noshift_t2", "fullshift_t0", "fullshift_t2"],
        choices=["jitter", "noshift_t0", "noshift_t2",
                 "fullshift_t0", "fullshift_t2"])
    p.add_argument("--kernels", nargs="+", default=list(KERNELS),
                   choices=list(KERNELS))
    p.add_argument("--smoke", action="store_true",
                   help="CI gate: 3 distributions x Wendland4 + verdict")
    args = p.parse_args(argv)

    if args.smoke:
        args.dists = ["jitter", "noshift_t0", "fullshift_t0"]
        args.kernels = ["Wendland4"]

    missing = [f for f in ("tgv2d_noshift_nx128.npz", "tgv2d_fullshift_nx128.npz")
               if not (DATA / f).exists()]
    if missing:
        print(f"error: saved TGV test data missing: {missing} "
              f"(expected in {DATA})", file=sys.stderr)
        return 2

    grad_check = check_analytic_gradient(args.device)
    print(f"analytic gradient check: max |dnum - dexact| = "
          f"{grad_check:.2e}", flush=True)

    dists = build_dists(args.dists, args.device)
    rows = run_dists(dists, tuple(args.kernels), args.device)

    csv_name = "disorder_probe_rows_smoke.csv" if args.smoke \
        else "disorder_probe_rows.csv"
    write_rows(rows, RESULTS / csv_name)
    print(f"\nwrote {RESULTS / csv_name}", flush=True)
    print_pivot(rows, dists, args.kernels)

    if args.smoke:
        verdict = smoke_verdict(rows, grad_check)
        ok = verdict.pop("ok")
        verdict_path = RESULTS / "disorder_probe_verdict.json"
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        # same {ok, checks} shape as run_baseline.py's smoke_verdict.json
        verdict_path.write_text(json.dumps({"ok": ok, "checks": verdict},
                                           indent=2))
        print(f"\nverdict: ok={ok}  ({verdict_path})", flush=True)
        if not ok:
            print(json.dumps(verdict, indent=2), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
