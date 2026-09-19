#!/usr/bin/env python3
"""GROUND TRUTH for the Phase-5 P matrix: the real finite-difference Jacobian
of the ACTUAL conservative-SPH force (eq. 3) for a real standing-wave
displacement  delta x_n = a cos(k.x_n)  (a: real 3-vector).

No complex a, no rho^gamma branch cut -> unambiguously valid (the complex-step
oracle is WRONG for non-integer gamma = 5/3; see phase5_stability_log.md
entry d). The FD Jacobian J (F0 = J a) gives P_fd = -J, which the analytic
``StabilityOracle.exact_p_matrix`` must reproduce to within the FD truncation
error (max|P_fd - P_ex| = O(h^2 |F'''|), a few x 1e0 at h = 1e-6).

All pair distances use the MINIMUM-IMAGE convention (periodic box L^3). The
densities rho_j are computed at the FULLY displaced positions (no linearised
density), so this is the exact force, not a linearisation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["F0_brute", "fd_jacobian"]


def F0_brute(o, k, a):
    """Actual SPH acceleration on particle 0 for a REAL standing-wave
    displacement delta x_n = a cos(k.x_n).  ``o`` is a StabilityOracle (provides
    H, C3, m, L, gamma, K, kev, lat.x); ``a`` a real (3,) amplitude.
    All pair distances use the minimum-image convention."""
    H, C3, m, L = o.H, o.C3, o.m, o.L
    X = o.lat.x

    def mi(d):
        return d - L * np.round(d / L)

    cosph = np.cos(k @ X.T)
    Xd = X + a * cosph[:, None]
    x0 = Xd[0]
    dv0 = mi(x0[None, :] - Xd)                # (N,3): x0 - Xd_j, minimum image
    r0 = np.sqrt((dv0**2).sum(-1))
    W0, _, _ = o.kev.ff1f2(r0 / H)
    W0 = W0.real
    rho0 = m * np.sum(C3 * W0 / H**3)
    g0 = o.K * rho0**(o.gamma - 2.0)
    F0 = np.zeros(3)
    for j in range(len(X)):
        if j == 0:
            continue
        rj = r0[j]
        if rj >= H:
            continue
        dvj = mi(Xd[j][None, :] - Xd)
        rjv = np.sqrt((dvj**2).sum(-1))
        Wj, _, _ = o.kev.ff1f2(rjv / H)
        Wj = Wj.real
        rhoj = m * np.sum(C3 * Wj / H**3)
        gj = o.K * rhoj**(o.gamma - 2.0)
        f1 = o.kev.ff1f2(rj / H)[1].real
        gradW = (C3 / H**4) * f1 * (dv0[j] / rj)
        F0 -= m * (g0 + gj) * gradW
    return F0


def fd_jacobian(o, k, h: float = 1e-6):
    """Real central-difference Jacobian J (F0 = J a) for the standing wave.
    The P matrix is P_fd = -J (the force is F0 = -P a)."""
    J = np.zeros((3, 3))
    for b in range(3):
        ap = np.zeros(3); ap[b] = h
        am = np.zeros(3); am[b] = -h
        J[:, b] = (F0_brute(o, k, ap) - F0_brute(o, k, am)) / (2 * h)
    return J
