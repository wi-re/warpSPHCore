#!/usr/bin/env python3
"""Phase 5: the EXACT P matrix (linearised conservative-SPH force) and its
built-in validation.

The P matrix linearises the conservative-SPH force (paper eq. 3) on one
particle of a plane-wave displacement of the FCC densest-sphere equilibrium:
    P a = omega^2 a ,   P = -Re[M]  (M the traveling-wave matrix, xddot_0 = M a).

P is the EXACT linearisation: real lattice sums at the real equilibrium, no
complex rho^gamma, no branch cut (the complex-step oracle is WRONG for the
non-integer gamma = 5/3; phase5_stability_log.md entry d). It is the deliverable
P matrix for Figs 4-6. See ``StabilityOracle.exact_p_matrix`` for the formula.

Built-in checks (per kernel, per wavevector):
  (1) P is symmetric (max|P - P^T|  ->  0).
  (2) the equilibrium is force-free (|xddot_0(a=0)| ~ 0) and the density is
      the lattice sum (rho_0 ~ 1).
  (3) [ --fd-check ] P agrees with the GROUND-TRUTH real-FD Jacobian of the
      ACTUAL force (max|P_fd - P_ex| = the FD truncation error, a few x 1e0 at
      h = 1e-6).  This is the load-bearing validation (ground_truth.py).

Usage:
  python stability_p_matrix.py                          # all kernels, fast checks
  python stability_p_matrix.py --kernel cubic_b4        # one kernel
  python stability_p_matrix.py --kernel cubic_b4 --fd-check   # + ground truth
"""

from __future__ import annotations

import argparse

import numpy as np

import common
import stability as st
from ground_truth import F0_brute, fd_jacobian

# default N_H per kernel (paper Table 2 "accessible" resolution)
DEFAULT_NH = {
    "cubic_b4": 55, "quartic_b5": 60, "quintic_b6": 180,
    "b7": 100, "b8": 100,
    "wendland_C2": 100, "wendland_C4": 200, "wendland_C6": 400,
    "hoct4": 442, "gaussian": 5120,
}

DIRS = {
    "111": np.array([1, 1, 1.0]) / np.sqrt(3),
    "110": np.array([1, 1.0, 0.0]) / np.sqrt(2),
}
KDNS = (0.5, 1.0, 2.0)


def check_p(oracle, k, fd: bool = False) -> dict:
    """Run the validation checks for one (oracle, k). Returns a dict of
    diagnostics (see the module docstring)."""
    P = oracle.exact_p_matrix(k)
    res = {
        "P": P,
        "symmetric": float(np.abs(P - P.T).max()),
        "eigenvalues": np.linalg.eigvalsh(P),
    }
    # equilibrium (fast: the precomputed neighbour lists, a = 0 is real)
    F0, rho0 = oracle.force0(k, np.zeros(3, complex))
    res["eq_force"] = float(np.abs(F0).max())
    res["eq_rho0"] = float(rho0.real)
    if fd:
        Pfd = -fd_jacobian(oracle, k)
        res["fd_agree"] = float(np.abs(Pfd - P).max())
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s) to check, repeatable and/or comma-separated; "
                         f"known: {st.STABILITY_ORDER} (default: all)")
    ap.add_argument("--nH", type=float, default=None,
                    help="override the default N_H (per kernel, Table 2)")
    ap.add_argument("--fd-check", action="store_true",
                    help="also run the ground-truth real-FD Jacobian check "
                         "(slow: O(N^2) per wavevector)")
    args = ap.parse_args()
    kernels = st.parse_kernel_arg(args.kernel)

    common.init()
    lat = st.Lattice(N=4000, L=1.0)
    dnn = lat.dnn

    for name in kernels:
        NH = args.nH if args.nH is not None else DEFAULT_NH[name]
        o = st.StabilityOracle(name, lat, float(NH))
        print(f"\n=== {st.LABELS[name]}   N_H={NH:g}   H/d_nn={o.H / dnn:.4f} "
              f"===  {'[+ground-truth FD]' if args.fd_check else '[fast checks]'}")
        print(f"{'k//':>4s} {'|k|d_nn':>8s}  {'sym':>10s}  {'eq|F0|':>9s} "
              f"{'rho0':>9s}  {'min omega^2':>12s}  {'max|Pfd-Pex|':>14s}")
        worst_sym = worst_fd = 0.0
        for dname, kdd in DIRS.items():
            for kdn in KDNS:
                k = kdn * kdd / dnn
                r = check_p(o, k, fd=args.fd_check)
                worst_sym = max(worst_sym, r["symmetric"])
                if args.fd_check:
                    worst_fd = max(worst_fd, r["fd_agree"])
                fdcol = f"{r['fd_agree']:14.3f}" if args.fd_check else f"{'-':>14s}"
                print(f"{dname:>4s} {kdn:8.3f}  {r['symmetric']:10.2e}  "
                      f"{r['eq_force']:9.1e}  {r['eq_rho0']:9.6f}  "
                      f"{r['eigenvalues'][0]:12.3f}  {fdcol}")
        verdict = "PASS" if worst_sym < 1e-8 else "FAIL"
        if args.fd_check:
            verdict += f"  (FD agree {worst_fd:.2f})"
        print(f"   -> P symmetric {verdict} (worst sym {worst_sym:.2e})")


if __name__ == "__main__":
    main()
