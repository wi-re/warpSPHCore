#!/usr/bin/env python3
"""Closeout item 5: verify the Phase-5 acceptance boundaries from the
high-res sweep's cached fields (results/stability_<kernel>.npz, written
by fig04_fig05_stability_contours.py --grid 60x200).

For each kernel, reports where the omega^2 <= 0 (pairing-unstable)
region sits in (N_H, |k|d_nn):
  - onset: the smallest N_H with any unstable point (any panel);
  - per-row fraction unstable (the 'gradual' measure), with the N_H
    where it crosses 1e-4 / 1e-3 / 1e-2;
  - islands: connected components of the unstable region (>= 3 points),
    with their N_H / h/d_nn / |k|d_nn extents;
  - the documented cubic long-wavelength dip check: any omega^2 < 0 at
    |k|d_nn in [0.3, 0.6] and N_H in [40, 100]. Phase-5 log entry (f)
    recorded this as an open discrepancy vs the paper's 'accessible
    N_H <= 55'; entry (i) identified it as the phase-reference bug, so
    on the corrected fields this check expects it to be GONE.

Expected boundaries (closeout item 5 / phase5_stability_log.md): cubic
unstable only gradually beyond N_H ~55 (island at h/d_nn ~2-2.9,
|k|d_nn ~2-5), quartic onset ~67, quintic ~190 plus a small-N_H island
near 100, Wendland C2 island near 40, HOCT4 island near 150, the rest
clean.

Usage:
  python check_stability_boundaries.py                 # all cached kernels
  python check_stability_boundaries.py --kernel cubic_b4
  python check_stability_boundaries.py --grid 60x200   # grid the cache must match
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import common
import stability as st

_HERE = Path(__file__).resolve().parent
RESDIR = _HERE.parent / "results"


def load(name: str, n_rows: int, n_kdn: int):
    p = RESDIR / f"stability_{name}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if int(d["n_rows"]) != n_rows or int(d["n_kdn"]) != n_kdn:
        print(f"  {name}: cache exists but for a different grid "
              f"({int(d['n_rows'])}x{int(d['n_kdn'])}); expected "
              f"{n_rows}x{n_kdn} -- run the sweep at this grid first")
        return None
    return d


def islands(mask: np.ndarray, kdn_grid: np.ndarray, nh_grid: np.ndarray,
            hdn_grid: np.ndarray) -> list:
    """Connected components (4-neighbour) of the True mask; (kdn, nh)
    shape. Returns [(size, nh0, nh1, kdn0, kdn1, hdn0, hdn1), ...],
    largest first, keeping components of >= 3 points."""
    comp = np.zeros(mask.shape, int)
    out = []
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            if not mask[j, i] or comp[j, i]:
                continue
            size = 0
            stack = [(j, i)]
            comp[j, i] = 1
            js, ii = [], []
            while stack:
                jj, ii_ = stack.pop()
                size += 1
                js.append(jj)
                ii.append(ii_)
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nj, ni = jj + dj, ii_ + di
                    if 0 <= nj < mask.shape[0] and 0 <= ni < mask.shape[1] \
                            and mask[nj, ni] and not comp[nj, ni]:
                        comp[nj, ni] = 1
                        stack.append((nj, ni))
            if size < 3:
                continue
            jmin, jmax = min(js), max(js)
            imin, imax = min(ii), max(ii)
            out.append((size, nh_grid[imin], nh_grid[imax],
                        kdn_grid[jmin], kdn_grid[jmax],
                        hdn_grid[imin], hdn_grid[imax]))
    out.sort(key=lambda t: -t[0])
    return out


def _report_mode(title: str, bad: np.ndarray, kdn, nh, hdn) -> None:
    """Report the unstable (omega^2<0) structure of one mode's mask
    (shape (n_kdn, n_rows)): onset, per-row fraction, islands."""
    n_bad = int(bad.sum())
    print(f"  {title}: {n_bad} unstable points "
          f"({100.0 * n_bad / bad.size:.2f} % of the field)")
    if n_bad == 0:
        print(f"    CLEAN -- no omega^2<0")
        return
    row_any = bad.any(axis=0)
    print(f"    onset: first unstable N_H = {nh[row_any][0]:.0f} "
          f"(h/d_nn = {hdn[row_any][0]:.2f})")
    frac = bad.mean(axis=0)
    for thr in (1e-4, 1e-3, 1e-2):
        over = row_any & (frac >= thr)
        if over.any():
            print(f"    frac >= {thr:g}: first at N_H = {nh[over][0]:.0f} "
                  f"(h/d_nn = {hdn[over][0]:.2f})")
    for size, nh0, nh1, kdn0, kdn1, hdn0, hdn1 in islands(bad, kdn, nh, hdn):
        print(f"    island: {size:5d} pts, N_H {nh0:5.0f}-{nh1:5.0f} "
              f"(h/d_nn {hdn0:.2f}-{hdn1:.2f}), |k|d_nn {kdn0:.2f}-{kdn1:.2f}")


def check(name: str, d) -> None:
    kdn = d["kdn_grid"]
    nh = d["nh_grid"]
    hdn = d["hdn_grid"]
    # TWO distinct modes, both reported (the acceptance boundaries below
    # are the LONGITUDINAL / pairing instability; the transverse shear
    # instability is a generic SPH pathology, present for every kernel,
    # and must NOT be used for the 'accessible N_H' criterion):
    lon_bad = (d["lon111"] < 0) | (d["lon110"] < 0)   # pairing instability
    tr_bad = (d["tr111"] < 0) | (d["tr110"] < 0)      # shear pathology

    print(f"\n=== {st.LABELS[name]}  "
          f"(N={int(d['N'])}, {len(nh)}x{len(kdn)}, "
          f"N_H {nh[0]:.0f}..{nh[-1]:.0f}, h/d_nn {hdn[0]:.2f}..{hdn[-1]:.2f})")
    print("  [longitudinal / pairing instability -- the acceptance metric]")
    _report_mode("lon", lon_bad, kdn, nh, hdn)
    print("  [transverse / shear instability -- generic SPH feature]")
    _report_mode("tr", tr_bad, kdn, nh, hdn)

    # the documented cubic long-wavelength dip box (longitudinal mode only)
    if name == "cubic_b4":
        jm = (kdn >= 0.3) & (kdn <= 0.6)
        im = (nh >= 40) & (nh <= 100)
        dip = int(lon_bad[np.ix_(jm, im)].sum())
        verdict = ("GONE (confirms phase-5 log entry (i): the long-lambda "
                   "dip was the phase-reference bug)"
                   if dip == 0 else
                   f"PRESENT ({dip} unstable points) -- still an open "
                   "discrepancy vs the paper's 'accessible N_H <= 55'")
        print(f"  cubic long-lambda dip box (|k|d_nn 0.3-0.6, N_H 40-100), "
              f"longitudinal: {dip} unstable points -> {verdict}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s), repeatable and/or comma-separated; "
                         f"known: {st.STABILITY_ORDER} "
                         "(default: all with a cache present)")
    ap.add_argument("--grid", type=str, default="60x200",
                    help="ROWSxKDN the cache must match (default 60x200, "
                         "the closeout item-5 high-res sweep)")
    args = ap.parse_args()
    n_rows, n_kdn = (int(v) for v in args.grid.lower().split("x"))
    kernels = st.parse_kernel_arg(args.kernel)

    common.init()
    for name in kernels:
        d = load(name, n_rows, n_kdn)
        if d is None and (RESDIR / f"stability_{name}.npz").exists():
            continue
        if d is None:
            print(f"\n=== {st.LABELS[name]}  (no cache -- run "
                  f"fig04_fig05_stability_contours.py --grid {n_rows}x{n_kdn} "
                  f"first)")
            continue
        check(name, d)


if __name__ == "__main__":
    main()
