#!/usr/bin/env python3
"""Figure 6 of D&A (2012) — the SPH sound speed c_SPH = omega_k / |k|.

Replicates Fig. 6: horizontal cuts of the top panels of Figs 4-5 (the
longitudinal / sound mode), i.e. c_SPH / c vs wave number |k| d_nn for a set of
fixed neighbour numbers N_H, with a vertical line at the shortest resolved
wavelength lambda = 8 h (h = the paper's resolution scale = H / kernelScale).
c is the continuum sound speed (= 1 at rho = 1, gamma = 5/3).  The P matrix is
the EXACT linearisation (``StabilityOracle.exact_p_matrix``).

For a fixed N_H, h = H/kernelScale and lambda = 8 h corresponds to
|k| = pi / (4 h), i.e. |k| d_nn = pi d_nn kernelScale / (4 H) (a function of
N_H); the vertical line is drawn at that |k| d_nn for each N_H curve.

Usage:
  python fig06_sound_speed.py                    # default kernels
  python fig06_sound_speed.py --kernel cubic_b4
  python fig06_sound_speed.py --kernel cubic_b4,gaussian
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import stability as st

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

NH_CUTS = (50.0, 100.0, 200.0, 400.0)
# log-spaced (not linear): the paper's Fig. 6 x-axis is log |k|d_nn over
# roughly [0.2, 7] (ticks at 0.5, 1, 5); log spacing gives even resolution
# per decade instead of oversampling the large-k end.
KDN_MIN, KDN_MAX = 0.2, 7.0
KDN_GRID = np.logspace(math.log10(KDN_MIN), math.log10(KDN_MAX), 40)
DNAME, KDD = "111", np.array([1, 1, 1.0]) / np.sqrt(3)   # k // (1,1,1)


def sound_speed_curve(name: str, lat: st.Lattice, NH: float):
    """c_SPH / c vs |k| d_nn (the longitudinal mode) for one (kernel, N_H).
    Returns (kdn, csph_over_c, kdn_at_8h)."""
    o = st.StabilityOracle(name, lat, float(NH))
    dnn = lat.dnn
    kdn = np.empty(len(KDN_GRID))
    csph = np.full(len(KDN_GRID), np.nan)
    for j, kd in enumerate(KDN_GRID):
        k = kd * KDD / dnn
        P = o.exact_p_matrix(k)
        w, v = np.linalg.eigh(P)
        kdnrm = np.linalg.norm(k)
        ov = np.array([abs(v[:, a] @ (k / kdnrm)) for a in range(3)])
        wL = w[int(np.argmax(ov))]
        kdn[j] = kd
        csph[j] = (math.sqrt(wL) / kdnrm / math.sqrt(o.c2)
                   if wL > 0 else np.nan)
    # the shortest resolved wavelength lambda = 8 h
    h = o.H / o.kev.scale3
    kdn_8h = math.pi * dnn / (4.0 * h)
    print(f"    N_H = {NH:7.1f}  (lambda=8h at |k|d_nn = {kdn_8h:.3f})")
    return kdn, csph, kdn_8h


def plot_kernel(name: str, lat: st.Lattice) -> None:
    c = st.COLORS[name]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 12})
    fig, a = plt.subplots(figsize=(8.5, 6))
    for NH in NH_CUTS:
        kdn, csph, kdn_8h = sound_speed_curve(name, lat, NH)
        a.plot(kdn, csph, color=c, lw=1.6,
               label=f"$N_H = {NH:g}$")
        if KDN_GRID[0] < kdn_8h < KDN_GRID[-1]:
            a.axvline(kdn_8h, color=c, ls=":", lw=1.0, alpha=0.7)
    a.axhline(1.0, color="k", lw=0.8, alpha=0.6)
    a.set_xlabel(r"wave number $|k|\, d_{nn}$")
    a.set_ylabel(r"SPH sound speed $c_{SPH} / c$")
    a.set_title(f"D&A (2012) Fig. 6 — sound speed, {st.LABELS[name]} "
                r"(k // 111; dotted verticals $\lambda = 8h$)")
    a.legend(fontsize=9)
    a.set_xscale("log")
    a.set_xlim(KDN_MIN, KDN_MAX)
    ticks = [0.2, 0.5, 1, 2, 5]
    a.set_xticks(ticks)
    a.set_xticks([], minor=True)
    a.set_xticklabels([str(t) for t in ticks])
    a.set_ylim(0.0, 1.4)

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / f"fig06_sound_speed_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s), repeatable and/or comma-separated; "
                         f"known: {st.STABILITY_ORDER} "
                         "(default: cubic_b4 + gaussian)")
    args = ap.parse_args()
    kernels = st.parse_kernel_arg(args.kernel,
                                  default=["cubic_b4", "gaussian"])

    common.init()
    lat = st.Lattice(N=4000, L=1.0)
    for name in kernels:
        print(f"\n=== {st.LABELS[name]}  (k // 111, N_H cuts {NH_CUTS}) ===")
        plot_kernel(name, lat)


if __name__ == "__main__":
    main()
