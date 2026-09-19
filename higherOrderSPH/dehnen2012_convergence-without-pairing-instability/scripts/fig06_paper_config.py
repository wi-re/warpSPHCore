#!/usr/bin/env python3
"""Figure 6 of D&A (2012), REPLICA using the paper's own Table 2
kernel/N_H combinations (companion to ``fig06_sound_speed.py``, which
instead sweeps one kernel over several arbitrary N_H).

The paper's Fig. 6 is 3 panels, one per wave direction -- k // (1,0,0),
(1,1,0), (1,1,1) -- and EACH panel overlays the SAME 10 (kernel, N_H)
combinations from Table 2; the legend is just split 4+3+3 across the three
panels for space (confirmed by rendering the paper's actual Fig. 6 at
500dpi: every curve is visible in every panel, only the inline legend
subset differs). This script reproduces that layout and grouping.

GAUSSIAN OMITTED: the paper's Gaussian rows are N_h=10/20, i.e.
N_H = N_h * kernelScale^3 = 5120 / 10240 (16-sigma truncation, kernelScale=8).
``StabilityOracle._build_neighbors`` builds a DENSE (n1, n2, 3) array
(n1 ~ N_H, n2 ~ 8 N_H) once per (kernel, N_H); at N_H=10240 that is
~5e8 elements (~40 GB as complex128) PER k-point -- infeasible on this
machine without rewriting the oracle's O(N_H^2) density-response term to be
sparse/chunked, which is out of scope here. The other 8 Table-2 rows (up to
N_H=442) are all comfortably feasible (largest dense array ~30 MB) and are
run in full. See phase5_stability_log.md entry (i) follow-up.

Usage:
  python fig06_paper_config.py
  python fig06_paper_config.py --include-gaussian   # WARNING: very slow/
                                                      # memory-heavy, see above
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import stability as st

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

KDN_MIN, KDN_MAX = 0.2, 7.0
KDN_GRID = np.logspace(math.log10(KDN_MIN), math.log10(KDN_MAX), 40)

DIRS = [
    ("100", np.array([1.0, 0.0, 0.0])),
    ("110", np.array([1.0, 1.0, 0.0]) / math.sqrt(2)),
    ("111", np.array([1.0, 1.0, 1.0]) / math.sqrt(3)),
]

# (kernel, N_H, label, color, linestyle, legend-panel-index)
# N_H values and grouping/order match the paper's Table 2 / Fig. 6 legend
# exactly (verified against the rendered page image).
SERIES = [
    ("cubic_b4",    42,  r"cubic spline, $N_H=42$",   "#6a3d9a", "-", 0),
    ("cubic_b4",    55,  r"cubic spline, $N_H=55$",   "#1f78b4", "-", 0),
    ("quartic_b5",  60,  r"quartic spline, $N_H=60$", "#00c8c8", "-", 0),
    ("quintic_b6",  180, r"quintic spline, $N_H=180$", "#33a02c", "-", 0),
    ("wendland_C2", 100, r"Wendland $C^2$, $N_H=100$", "#e31a1c", "-", 1),
    ("wendland_C4", 200, r"Wendland $C^4$, $N_H=200$", "#e377c2", "-", 1),
    ("wendland_C6", 400, r"Wendland $C^6$, $N_H=400$", "#a06a4a", "-", 1),
    ("hoct4",       442, r"HOCT4 kernel, $N_H=442$",  "#ff7f00", "-", 2),
]
GAUSSIAN_SERIES = [
    ("gaussian", 5120,  r"Gaussian, $N_h=10$", "black", "-",  2),
    ("gaussian", 10240, r"Gaussian, $N_h=20$", "black", "--", 2),
]


def sound_speed_curve(name: str, NH: float, kdd: np.ndarray):
    """c_SPH / c vs |k| d_nn for one (kernel, N_H, direction)."""
    lat = st.lattice_for_NH(NH)
    o = st.StabilityOracle(name, lat, float(NH))
    dnn = lat.dnn
    csph = np.full(len(KDN_GRID), np.nan)
    for j, kd in enumerate(KDN_GRID):
        k = kd * kdd / dnn
        P = o.exact_p_matrix(k)
        w, v = np.linalg.eigh(P)
        kdnrm = np.linalg.norm(k)
        ov = np.array([abs(v[:, a] @ (k / kdnrm)) for a in range(3)])
        wL = w[int(np.argmax(ov))]
        csph[j] = (math.sqrt(wL) / kdnrm / math.sqrt(o.c2)
                   if wL > 0 else np.nan)
    return csph


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-gaussian", action="store_true",
                    help="also run the paper's Gaussian N_h=10/20 rows "
                         "(N_H=5120/10240) -- WARNING: dense (n1,n2,3) "
                         "arrays of ~1e8-5e8 elements per k-point, likely "
                         "tens of GB and many minutes; see module docstring")
    args = ap.parse_args()

    common.init()
    series = list(SERIES)
    if args.include_gaussian:
        series += GAUSSIAN_SERIES
    else:
        print("NOTE: Gaussian N_h=10/20 rows omitted by default (dense-array "
              "memory blowup at N_H=5120/10240); pass --include-gaussian to "
              "force them. See the module docstring.")

    plt.rcParams.update({"font.size": 10.5})
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), sharex=True)

    curves = {}   # (name, NH, dname) -> csph array
    for name, NH, label, color, ls, group in series:
        for dname, kdd in DIRS:
            t0 = time.time()
            csph = sound_speed_curve(name, NH, kdd)
            curves[(name, NH, dname)] = csph
            print(f"  {label:32s} k//{dname}  done ({time.time()-t0:.1f}s)")

    for panel_idx, (a, (dname, kdd)) in enumerate(zip(axes, DIRS)):
        for name, NH, label, color, ls, group in series:
            csph = curves[(name, NH, dname)]
            show_label = label if group == panel_idx else None
            a.plot(KDN_GRID, csph, color=color, ls=ls, lw=1.4,
                  label=show_label)
        a.axhline(1.0, color="k", lw=0.7, alpha=0.5)
        a.set_xscale("log")
        a.set_xlim(KDN_MIN, KDN_MAX)
        a.set_xticks([0.5, 1, 5])
        a.set_xticks([], minor=True)
        a.set_xticklabels(["0.5", "1", "5"])
        a.set_ylim(0.0, 1.1)
        a.set_ylabel(r"$c_{SPH}/c$")
        a.text(0.98, 0.92, f"$k \\propto ({dname[0]},{dname[1]},{dname[2]})$",
              transform=a.transAxes, ha="right", va="top", fontsize=11)
        a.legend(fontsize=8, loc="lower left")

    axes[-1].set_xlabel(r"$|k|\, d_{nn}$")
    fig.suptitle("D&A (2012) Fig. 6 replica — paper's Table 2 kernel/"
                 r"$N_H$ combinations" +
                 ("" if args.include_gaussian else " (Gaussian rows omitted)"))
    fig.tight_layout()

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "fig06_paper_config.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
