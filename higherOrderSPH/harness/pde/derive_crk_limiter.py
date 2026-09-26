#!/usr/bin/env python3
"""Derive the CRKSPH viscosity-limiter constants (eta_crit, eta_fold) in the
frontend's units from the actual run configuration.

Frontiere et al. 2017, Eqs. (51)-(53): the limiter phi_ij is multiplied by
exp(-((eta_ij - eta_crit)/eta_fold)^2) for eta_ij < eta_crit, with
eta_ij = min(r_ij/h_i, r_ij/h_j) and (eta_crit, eta_fold) = (1/n_h, 0.2) --
eta in units of the paper's smoothing scale h, n_h = points per h. For their
CRKSPH kernel (extent 4h) they use n_h = 1, i.e. h = nominal spacing dx:

* eta_crit * h = h / n_h = dx      -- the threshold sits *at* the nominal
  nearest-neighbour spacing ("in ordinary smooth regions this second
  multiplier should never activate"); this is kernel-independent;
* eta_fold * h = 0.2 h             -- 0.2 dx in their CRKSPH configuration.

The frontend (`modules/crk/limiter.py`) uses eta = r / H with H the support
*radius*, so in its units

    eta_crit = dx / H,      eta_fold = 0.2 * dx / H        ('spacing')

with dx the nominal spacing, dx = (m / rho)^(1/dim). The frontend currently
hard-codes (1/3, 0.2) -- 1/3 is 1/n_h for n_h = 3, but the compressible
cases run n_h = 4 (H = 4 dx), so every nearest-neighbour pair sits below the
threshold (factor ~0.83 in smooth flow) and the fall-off is 4x wider than
the paper's.

The fold is the kernel-convention-sensitive part (the paper's h is defined
per kernel). For reference the script also reports the 'sigma' convention,
eta_fold = 0.2 * (H / ks) / H = 0.2 / ks, with ks the Dehnen & Aly kernel
scale H/sigma (`sphKernelScale`) -- i.e. reading the paper's h as the D&A
smoothing scale instead of the nominal spacing. The 'spacing' convention is
the one that reproduces the paper's CRKSPH setup exactly and is what the
limiter sweep uses.

Measured, not assumed: dx and H are taken from the sampled t=0 particle set
of each case (median over particles), and the median nearest-neighbour eta
is reported so the threshold's position relative to the real spacing is
visible.

Usage:
    python derive_crk_limiter.py                    # all compressible cases
    python derive_crk_limiter.py --cases sod gresho
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import dataclasses
import json

import torch

PAPER_FOLD = 0.2          # Frontiere et al. 2017, eta_fold in units of h
CURRENT = (0.3333333, 0.2)  # frontend buildDefaultCRKViscosityParams

# case name -> (module, case attribute, spec overrides for a cheap build)
CASES = {
    "sod":             ("warpSPH.cases.sod", "sodCase",
                        dict(scheme="CRKSPH", supportMode="KernelMeanSymmetric")),
    "sedov":           ("warpSPH.cases.sedov", "sedovCase", dict(nx=400)),
    "noh":             ("warpSPH.cases.noh", "nohCase", {}),
    "kidder":          ("warpSPH.cases.kidder", "kidderCase", {}),
    "woodwardColella": ("warpSPH.cases.woodwardColella", "woodwardColellaCase", {}),
    "linearWave":      ("warpSPH.cases.linearWave", "linearWaveCase", {}),
    "gresho":          ("warpSPH.cases.greshoVortex", "greshoVortexCase", dict(nx=64)),
    "yeeVortex":       ("warpSPH.cases.yeeVortex", "yeeVortexCase", {}),
    "kelvinHelmholtz": ("warpSPH.cases.kelvinHelmholtz", "kelvinHelmholtzCase", dict(nx=64)),
    "hydrostatic":     ("warpSPH.cases.hydrostatic", "hydrostaticCase", dict(nx=64)),
    "shearingNoh":     ("warpSPH.cases.shearingNoh", "shearingNohCase", dict(nx=64)),
    "triplePoint":     ("warpSPH.cases.triplePoint", "triplePointCase", dict(nx=64)),
    "rayleighTaylor":  ("warpSPH.cases.rayleighTaylor", "rayleighTaylorCase", dict(nx=64)),
    "sod2d":           ("warpSPH.cases.sodND", "sod2dCase",
                        dict(scheme="CRKSPH", supportMode="KernelMeanSymmetric")),
}


def derive(dx_over_H: float, ks: float | None = None) -> dict:
    """Limiter constants in the frontend's r/H units for a configuration with
    nominal-spacing-to-support ratio `dx_over_H` (= 1/n_h for H = n_h dx)."""
    out = {"spacing": (dx_over_H, PAPER_FOLD * dx_over_H)}
    if ks:
        out["sigma"] = (dx_over_H, PAPER_FOLD / ks)
    return out


def load_case(name: str):
    import importlib
    module, attr, _ = CASES[name]
    return getattr(importlib.import_module(module), attr)


def build_spec(name: str, case, **extra):
    from warpSPH.runner import CaseSpec
    _, _, over = CASES[name]
    over = {**dict(over), **extra}          # caller overrides win
    scheme = over.pop("scheme", case.scheme)
    spec = CaseSpec(caseName=case.name, scheme=scheme,
                    params=dict(case.params)).merged(**case.defaults)
    return spec.merged(**over)


def measure(name: str, device: str = "cuda:0") -> dict:
    """Build the case (one step) and measure dx, H, nearest-neighbour eta."""
    import warpSPH.schemes.builder as B
    from warpSPH.runner import run
    from warpSPHCore.kernels.properties import sphKernelScale

    case = load_case(name)
    seen: dict = {}
    orig = B.crkSPH_step

    def wrapped(system, dt, config, schemeConfig, verbose=False):
        out = orig(system, dt, config, schemeConfig, verbose)
        if not seen:
            st = out[2]                  # stage state; adjacency built by now
            adj_out = out[1]
            dim = int(config.domain.dim)
            V = st.masses / st.densities
            dx = V.clamp_min(0) ** (1.0 / dim)
            H = st.supports
            adj = adj_out
            if adj is not None:
                i, j = adj.i.long(), adj.j.long()
                keep = i != j
                i, j = i[keep], j[keep]
                d = st.positions[i] - st.positions[j]
                L = config.domain.max - config.domain.min
                per = torch.as_tensor(config.domain.periodic, device=d.device).bool()
                d = torch.where(per, d - L * torch.round(d / L), d)
                eta = torch.minimum(d.norm(dim=-1) / H[i], d.norm(dim=-1) / H[j])
                nn = torch.full_like(H, float("inf")).scatter_reduce(
                    0, i, eta, reduce="amin")
                seen["nn_eta_median"] = float(nn[torch.isfinite(nn)].median())
            kern = config.kernel
            seen.update(
                dim=dim, kernel=getattr(kern, "name", str(kern)),
                ks=float(sphKernelScale(int(getattr(kern, "value", kern)), dim)),
                dx_median=float(dx.median()), H_median=float(H.median()),
                dx_over_H=float((dx / H).median()),
                n_h_spec=None)
        return out

    B.crkSPH_step = wrapped
    try:
        spec = build_spec(name, case, nSteps=1, tLimit=None, plot=False,
                          show=False, store=False, video=False, progress=False,
                          quiet=True, device=device)
        seen_spec_nh = getattr(spec, "n_h", None)
        run(case, spec)
    finally:
        B.crkSPH_step = orig
    if not seen:
        return {"case": name, "error": "CRKSPH step never called"}
    seen["n_h_spec"] = seen_spec_nh
    d = derive(seen["dx_over_H"], seen["ks"])
    seen.update(case=name, current=CURRENT, derived_spacing=d["spacing"],
                derived_sigma=d.get("sigma"))
    return seen


def main(argv=None):
    import warp as wp
    wp.init()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", default=None, help="write results to this file")
    args = p.parse_args(argv)
    rows = []
    for name in args.cases:
        try:
            r = measure(name, args.device)
        except Exception as e:                       # keep going over cases
            r = {"case": name, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        rows.append(r)
        if "error" in r:
            print(f"{name:16s} ERROR {r['error']}", flush=True)
            continue
        print(f"{name:16s} dim={r['dim']} {r['kernel']:7s} n_h={r['n_h_spec']} "
              f"dx/H={r['dx_over_H']:.4f} nn_eta={r.get('nn_eta_median', float('nan')):.4f} "
              f"ks={r['ks']:.3f} | current={CURRENT} "
              f"derived(spacing)=({r['derived_spacing'][0]:.4f}, {r['derived_spacing'][1]:.4f}) "
              f"derived(sigma)=({r['derived_sigma'][0]:.4f}, {r['derived_sigma'][1]:.4f})",
              flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
