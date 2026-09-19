#!/usr/bin/env python3
"""Phase 5 linear-stability core for the D&A (2012) replication (Figs 4-6,
Appendix A).

What this module is
-------------------
The P matrix (paper eqs. 21/23) linearises the conservative-SPH force (eq. 3)
on one particle of a plane-wave displacement of the densest-sphere (FCC)
equilibrium:

    xddot_0 = M a ,   M = d(xddot_0)/da ,   P = -Re[M] ,   P a = omega^2 a .

P is computed by COMPLEX-STEP differentiation of the ACTUAL force law, so no
reliance is placed on the paper's Appendix-A closed form (eq. 23). This work
found that eq. 23 (and the A9 density response delta_rho_i = i Phi_i (a.t))
is an approximation that drops the REAL part of the density response
(m sum_k grad W(x_k) cos(k.x_k)); the complex-step force Jacobian is the
ground truth and is what Figs 4-6 are generated from. See
``phase5_stability_log.md`` for the full derivation and the discrepancy
analysis.

Notation pin (same as common.py / the reference YAML):
  * the code smoothing length H is the PAPER'S support radius H;
  * the paper's resolution scale h = 2 sigma = H / kernelScale;
  * N_H = (4 pi/3) H^3 (rho/m)  (3D, rho = 1);
  * FCC densest-sphere packing, N = 4000, box 1^3, d_nn = L/(p sqrt2),
    p = (N/4)^(1/3);  rho = 1, m = 1/N, gamma = 5/3, K = 1/gamma,
    c^2 = K gamma rho^(gamma-1) = 1 at rho = 1.

The force law (eq. 3, the PDF text layer garbles the leading sign AND the
mass; both are pinned by the continuum limit xddot -> -grad P / rho):

    xddot_i = - sum_j m_j [ P_hat_i/rho_hat_i^2 grad W(x_i-x_j, h_i)
                            + P_hat_j/rho_hat_j^2 grad W(x_i-x_j, h_j) ]
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import common
from ft_kernels import (bspline_pieces, wendland_pieces, hoct4_pieces,
                        BSPLINE_ORDER, WENDLAND_POLY)


# ---------------------------------------------------------------------------
# Kernel set + the --kernel CLI flag (shared by the Phase-5 scripts).
# Same set and colour coding as fig01/fig02/fig03 (Figs 1-3 of the paper).
# ---------------------------------------------------------------------------

STABILITY_ORDER = [
    "cubic_b4", "quartic_b5", "quintic_b6", "b7", "b8",
    "wendland_C2", "wendland_C4", "wendland_C6",
    "hoct4", "gaussian",
]
LABELS = {
    "cubic_b4": "cubic spline ($b_4$)",
    "quartic_b5": "quartic spline ($b_5$)",
    "quintic_b6": "quintic spline ($b_6$)",
    "b7": "spline $b_7$",
    "b8": "spline $b_8$",
    "wendland_C2": "Wendland C$^2$",
    "wendland_C4": "Wendland C$^4$",
    "wendland_C6": "Wendland C$^6$",
    "hoct4": "HOCT4 (Read 2010)",
    "gaussian": "Gaussian",
}
COLORS = {
    "cubic_b4": "#1f77b4",
    "quartic_b5": "#17becf",
    "quintic_b6": "#9467bd",
    "b7": "#e377c2",
    "b8": "#bcbd22",
    "wendland_C2": "#2ca02c",
    "wendland_C4": "#d62728",
    "wendland_C6": "#8c564b",
    "hoct4": "#ff7f0e",
    "gaussian": "#7f7f7f",
}


def parse_kernel_arg(values, default=None, valid=None):
    """Parse the --kernel flag: repeatable and/or comma-separated shipped
    names. `valid` is the set of accepted names (default STABILITY_ORDER);
    `default` is the selection when the flag is absent (default: the full
    `valid` set). Returns the selected list (in `valid` order)."""
    if valid is None:
        valid = STABILITY_ORDER
    if default is None:
        default = valid
    if not values:
        return list(default)
    names = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                names.append(part)
    seen, out = set(), []
    for n in names:
        if n in valid and n not in seen:
            seen.add(n)
            out.append(n)
    bad = [n for n in names if n not in valid]
    if bad:
        raise SystemExit(f"unknown kernel(s) {bad}; known: {list(valid)}")
    return out


# ---------------------------------------------------------------------------
# Complex-safe kernel shape evaluation (f, f', f'').
#
# The shipped warp functions are real-only, but the complex-step oracle
# needs f(q), f'(q), f''(q) at COMPLEX q. Every compact-support kernel is a
# piecewise polynomial in q, so we evaluate the polynomial of the
# EQUILIBRIUM piece at the complex q (the perturbation is O(eps), so the
# piece index does not change). The (16-sigma-truncated) Gaussian is an
# exponential (entire), complex-safe directly: f = exp(-128 q^2).
# ---------------------------------------------------------------------------

def _poly_coeffs(c, deriv):
    """Ascending-power coefficients of the deriv-th derivative of a
    polynomial with ascending-power coefficients c."""
    n = len(c)
    out = [0.0] * max(0, n - deriv)
    for m in range(deriv, n):
        fall = 1.0
        for k in range(m - deriv + 1, m + 1):
            fall *= k
        out[m - deriv] = c[m] * fall
    return out


class KernelEval:
    """Complex-safe f, f', f'' for a shipped 3D kernel shape."""

    def __init__(self, name: str, ref: dict):
        self.name = name
        self.C3 = common.C_d(3, name)
        self.scale3 = common.kernel_scale(3, name)
        self.is_gaussian = (name == "gaussian")
        if not self.is_gaussian:
            if name in BSPLINE_ORDER:
                pieces = bspline_pieces(BSPLINE_ORDER[name])
            elif name in WENDLAND_POLY:
                ell, poly = WENDLAND_POLY[name]
                pieces = wendland_pieces(ell, poly)
            elif name == "hoct4":
                pieces = hoct4_pieces(ref)
            else:
                raise ValueError(name)
            # compact support: zero piece for q >= 1 (neighbour-neighbour
            # distances reach 2H, i.e. q up to 2)
            pieces = pieces + [(1.0, math.inf, [0.0])]
            # (a, b, coeffs_f, coeffs_f1, coeffs_f2)
            self.pieces = [(a, b, c, _poly_coeffs(c, 1), _poly_coeffs(c, 2))
                           for (a, b, c) in pieces]
        else:
            self.pieces = None

    def _piece_index(self, q_real: np.ndarray) -> np.ndarray:
        if self.is_gaussian:
            return np.zeros(q_real.shape, dtype=int)
        q = np.asarray(q_real, float)
        idx = np.searchsorted(
            np.array([p[0] for p in self.pieces]), q, side="right") - 1
        idx = np.clip(idx, 0, len(self.pieces) - 1)
        return np.where(q < 0, 0, idx)

    def _eval_poly(self, coeffs, q):
        q = np.asarray(q)
        out = np.zeros_like(q, dtype=np.result_type(q, 1.0j))
        for c in reversed(coeffs):
            out = out * q + c
        return out

    def ff1f2(self, q, idx=None):
        """f, f', f'' at (possibly complex) q. idx = precomputed real piece
        indices. Returns 3 arrays of q.shape, complex if q is complex."""
        q = np.asarray(q)
        qreal = np.abs(q.real) if np.iscomplexobj(q) else q
        if idx is None:
            idx = self._piece_index(qreal)
        if self.is_gaussian:
            e = np.exp(-128.0 * q * q)
            f = e
            f1 = -256.0 * q * e
            f2 = (-256.0 + 65536.0 * q * q) * e
            return f, f1, f2
        f = np.zeros_like(q, dtype=np.result_type(q, 1.0j))
        f1 = np.zeros_like(q, dtype=np.result_type(q, 1.0j))
        f2 = np.zeros_like(q, dtype=np.result_type(q, 1.0j))
        for p in range(len(self.pieces)):
            m = idx == p
            if not m.any():
                continue
            qp = q[m]
            _, _, cf, cf1, cf2 = self.pieces[p]
            f[m] = self._eval_poly(cf, qp)
            f1[m] = self._eval_poly(cf1, qp)
            f2[m] = self._eval_poly(cf2, qp)
        return f, f1, f2


# ---------------------------------------------------------------------------
# The FCC lattice (densest-sphere packing), built once.
# ---------------------------------------------------------------------------

class Lattice:
    """FCC densest-sphere packing, box L^3, N particles, rho = 1.

    Precomputes the minimum-image vectors from particle 0 (d0_all) and their
    distances (r0_all) so that the per-H neighbour lists are cheap
    thresholds."""

    def __init__(self, N: int = 4000, L: float = 1.0):
        from particle_configs import fccConfig
        self.N = N
        self.L = L
        cfg = fccConfig(N, L)
        self.x = cfg.positions
        self.m = L**3 / N
        p = (N / 4) ** (1.0 / 3)
        self.dnn = L / (p * math.sqrt(2.0))
        # particle 0 -> every other particle (minimum image)
        d = self.x - self.x[0]
        d -= L * np.round(d / L)
        self.d0_all = d                       # (N, 3)
        self.r0_all = np.sqrt((d**2).sum(-1))  # (N,)
        self.r0_all[0] = 0.0

    def H_of_NH(self, N_H: float) -> float:
        return (N_H * self.m / (4.0 * math.pi / 3.0)) ** (1.0 / 3.0)

    def NH_of_H(self, H: float) -> float:
        return (4.0 * math.pi / 3.0) * H**3 / self.m


def lattice_for_NH(nh_max: float, safety: float = 0.32, min_N: int = 4000,
                   L: float = 1.0) -> "Lattice":
    """A Lattice with enough particles that the largest required cutoff
    (2H at N_H = nh_max) stays a safe fraction of the box (H/L <= safety,
    validated up to H/L ~ 0.29 in phase5_stability_log.md entry (i)):
    N_H = (4pi/3) H^3 (N/L^3)  =>  N >= N_H / ((4pi/3) safety^3).
    Kernels with a large kernelScale (e.g. the Gaussian, whose N_H at a
    given h/d_nn is ~kernelScale^3 times a B-spline's) need a much bigger
    N_H for the same h/d_nn, hence a much bigger lattice; sized per-kernel
    so lighter kernels stay cheap."""
    N_min = nh_max / ((4.0 * math.pi / 3.0) * safety ** 3)
    p = max(10, math.ceil((max(N_min, min_N) / 4.0) ** (1.0 / 3.0)))
    N = 4 * p ** 3
    return Lattice(N=N, L=L)


# ---------------------------------------------------------------------------
# The stability oracle: complex-step force Jacobian (the P matrix).
# ---------------------------------------------------------------------------

class StabilityOracle:
    """P matrix for one (kernel, N_H) at the FCC equilibrium.

    P a = omega^2 a with P = -Re[d(xddot_0)/da] (complex step). The force is
    the pinned eq.-3 conservative SPH law with the adaptive-h / F-factor
    scheme (here at fixed h = H, the paper's linear-stability convention).
    """

    def __init__(self, name: str, lat: Lattice, N_H: float,
                 dim: int = 3, gamma: float = 5.0 / 3.0, K: float | None = None):
        ref = common.load_reference()
        self.name = name
        self.lat = lat
        self.L = lat.L
        self.H = lat.H_of_NH(N_H)
        self.N_H = N_H
        self.m = lat.m
        self.rho0 = 1.0
        self.gamma = gamma
        self.K = K if K is not None else 1.0 / gamma
        self.kev = KernelEval(name, ref)
        self.C3 = self.kev.C3
        self.c2 = self.K * gamma * self.rho0**(gamma - 1.0)   # = 1
        self._build_neighbors()

    # -- neighbour lists -----------------------------------------------------
    def _mi(self, d):
        return d - self.L * np.round(d / self.L)

    def _build_neighbors(self):
        x, L, H = self.lat.x, self.L, self.H
        r = self.lat.r0_all
        # list1 INCLUDES the self term (r = 0): its gradient is 0 (so it adds
        # nothing to the force) but its W(0) is the self-density contribution.
        self.list1 = np.nonzero(r < H)[0]
        self.d0 = self.lat.d0_all[self.list1]              # (n1, 3)
        self.list2 = np.nonzero(r < 2.0 * H)[0]
        # neighbour -> candidate displacements  (n1, n2, 3)
        dll = self._mi(x[self.list2][None, :, :] - self.lat.x[self.list1][:, None, :])
        self.dll = dll
        self.dll_r = np.sqrt((dll**2).sum(-1))             # (n1, n2)
        self.r0 = np.sqrt((self.d0**2).sum(-1))            # (n1,)
        # equilibrium piece indices (pieces live in q = r/H space)
        self.idx0 = self.kev._piece_index(self.r0 / H)
        self.idxll = self.kev._piece_index(self.dll_r / H)

    # -- kernel quantities at (r_vec, H) -------------------------------------
    def _gradW(self, rvec, H, idx):
        r = np.sqrt((rvec**2).sum(-1))
        r = np.where(r == 0, 1.0, r)
        q = r / H
        f, f1, f2 = self.kev.ff1f2(q, idx)
        rhat = rvec / r[..., None]
        return (self.C3 / H**4) * f1[..., None] * rhat

    def _hessW(self, rvec, H, idx):
        r = np.sqrt((rvec**2).sum(-1))
        r = np.where(r == 0, 1.0, r)
        q = r / H
        f, f1, f2 = self.kev.ff1f2(q, idx)
        rhat = rvec / r[..., None]
        rr = rhat[..., :, None] * rhat[..., None, :]
        I3 = np.eye(3)                       # broadcasts to (nk, 3, 3)
        return (self.C3 / H**5) * (
            (f1 / q)[..., None, None] * (I3 - rr)
            + f2[..., None, None] * rr)

    def _density(self, rvec, H, idx):
        r = np.sqrt((rvec**2).sum(-1))
        q = r / H
        f, _, _ = self.kev.ff1f2(q, idx)
        return self.m * np.sum(self.C3 * f / H**3, axis=-1)

    # -- FIXED-h force on particle 0 ----------------------------------------
    def force0(self, k, a):
        """Acceleration xddot_0 and rho_0 for a plane-wave displacement
        x_i -> x_i + a exp(i k.x_i)  (a: (3,) complex, k: (3,) real).

        The phase uses the ABSOLUTE position (Phi_j = exp(i k.x_j)); only the
        kernel (grad W) uses the minimum-image relative vector. See the log
        (the minimum-image phase is WRONG for neighbours across the box)."""
        H = self.H
        phase = np.exp(1j * (k @ self.lat.x.T))
        ph0 = phase[0]
        ph1 = phase[self.list1]
        ph2 = phase[self.list2]

        rvec0 = -self.d0 + a[None, :] * (ph0 - ph1)[:, None]     # (n1,3)
        rho0 = self._density(rvec0, H, self.idx0)
        P0 = self.K * rho0**self.gamma

        rvec_ll = (-self.dll
                   + a[None, None, :] * (ph1[:, None] - ph2[None, :])[:, :, None])
        f_ll, _, _ = self.kev.ff1f2(np.sqrt((rvec_ll**2).sum(-1)) / H, self.idxll)
        rho_l = self.m * np.sum(self.C3 * f_ll / H**3, axis=-1)  # (n1,)
        P_l = self.K * rho_l**self.gamma

        grad0 = self._gradW(rvec0, H, self.idx0)                # (n1,3)
        F0 = -self.m * np.sum(
            (P0 / rho0**2 + P_l / rho_l**2)[..., None] * grad0, axis=0)
        return F0, rho0

    # -- P matrix via complex step (FLAWED for non-integer gamma) -----------
    def p_matrix(self, k, eps: float = 1e-30):
        """(LEGACY / WRONG for gamma = 5/3) P (3x3, real) via complex step.
        The force contains rho^gamma (non-integer gamma), which is not
        holomorphic, so the complex step is INVALID (it returns the real part
        of the wrong analytic continuation). Use ``exact_p_matrix`` instead.
        Kept only for the CS-vs-FD diagnostic in ``self_consistency``."""
        M = np.zeros((3, 3), dtype=complex)
        for alpha in range(3):
            a = np.zeros(3, dtype=complex)
            a[alpha] = 1j * eps
            F0, _ = self.force0(k, a)
            M[:, alpha] = np.imag(F0) / eps
        return -M.real

    # -- THE EXACT P matrix (real lattice sums; the deliverable) ------------
    def exact_p_matrix(self, k):
        """The EXACT P matrix: real lattice sums at the real equilibrium, no
        complex rho^gamma, no branch cut.  P a = omega^2 a with P = -Re[M],
        M the traveling-wave matrix (xdot_0 = M a):

            M = -2 m Bbar sum_j (1 - e^{i k.x_j}) Hess W(d0_j)         [Hessian]
                - m Bbar (gamma - 2)/rho sum_j grad W(d0_j) (x) B_j    [density]
            B_j = -m sum_k grad W(x_j - x_k) e^{i k.x_k}   (exact response)

        Bbar = K rho^{gamma-2} (matches the Hessian term's own prefactor: an
        EARLIER version of this formula multiplied the density term by an
        extra, spurious factor of K -- i.e. K*Bbar instead of Bbar -- AND
        had the wrong overall sign; both are bugs from mis-deriving the
        d(P_j/rho_j^2)/drho_j chain rule, caught by comparing against
        gamma=2 (which kills this whole term identically, isolating the
        Hessian term: agreed with FD to 3e-9) and then grid-searching
        sign/coefficient variants against the FD ground truth at several k
        (agreement went from O(1-30) "truncation error" that DIDN'T shrink
        with the FD step h -- i.e. was never truncation error, just an
        undetected formula bug -- to true O(h^2) truncation error, 0.01-0.2
        at h=1e-6). See phase5_stability_log.md entry (k). The kernel
        (grad W, Hess W) AND the phase both
        use the minimum-image vector d0_j = MI(x_j - x_0) (x_0 = 0 for the
        reference particle, guaranteed by Lattice/fccConfig). The phase must
        be exp(ik.d0_j), NOT exp(ik.x_j) evaluated at the neighbour's raw
        (box-wrapped) array coordinate: list1/list2 are found via minimum
        image, so whenever a neighbour's nearest periodic image required
        wrapping, its stored coordinate is a DIFFERENT periodic image than
        the one actually interacting, and exp(ik.x_stored) picks up a
        spurious exp(ik.(n.L)) phase. This is only invisible for a reference
        particle far from every box face; particle 0 sits at the box CORNER
        [0,0,0], so ~7/8 of its neighbours are wrap-affected -- confirmed by
        the fix restoring exact translation invariance (identical eigenvalues
        for 6 different reference particles) and a sane continuum-limit
        magnitude (was ~50-135x c^2 at |k|d_nn=0.02, now ~1-1.5x). See
        phase5_stability_log.md entry (i) (supersedes the phase conclusion of
        entry (c), which validated the raw-coordinate phase only against a
        ground truth that shares the same convention/bug).
        The (1 - e^{ik.d0_j}) part of the response B_j drops because
        sum_k grad W(x_j - x_k) = 0 at equilibrium. Validated against the
        ground-truth real-FD Jacobian of the actual force (max|P_fd - P_ex| =
        the FD truncation error; phase5_stability_log.md entries e/f)."""
        m, K, g, rho0 = self.m, self.K, self.gamma, self.rho0
        Bbar = K * rho0 ** (g - 2.0)
        H = self.H
        d0, r0, idx0 = self.d0, self.r0, self.idx0
        x = self.lat.x
        # Hessian + gradient at d0 (the self term r=0 contributes 0)
        keep_idx = np.nonzero(r0 > 0)[0]
        d0k, rk, idxk = d0[keep_idx], r0[keep_idx], idx0[keep_idx]
        phk = np.exp(1j * (k @ d0k.T))                      # (nk,)
        grad0 = self._gradW(d0k, H, idxk)                   # (nk, 3)
        hess0 = self._hessW(d0k, H, idxk)                   # (nk, 3, 3)
        M_hess = -2.0 * m * Bbar * np.einsum("a,aij->ij", (1.0 - phk), hess0)
        # Density response B_j = -m sum_k grad W(x_j - x_k) Phi_k (k in H of j),
        # from the oracle's dll (list1 -> list2). Phi_k is likewise the
        # minimum-image phase relative to x_0, not the raw list2 array
        # coordinate. Vectorized over the full (n1, n2) grid: no explicit
        # mask needed since grad W(x_j-x_k) is already exactly 0 for
        # dll_r >= H (the compact-support "zero piece" past q=1), so the
        # out-of-support entries drop out on their own -- this makes the
        # per-neighbour Python loop unnecessary (was O(N_H^2) python-level
        # iterations; this is the same FLOPs done in vectorized numpy).
        d0_list2 = self._mi(x[self.list2] - x[0])
        phase2 = np.exp(1j * (k @ d0_list2.T))              # (n2,)
        g_all = self._gradW(self.dll, H, self.idxll)        # (n1, n2, 3)
        B_all = -m * np.einsum("ijc,j->ic", g_all, phase2)  # (n1, 3) complex
        grad0_all = self._gradW(d0, H, idx0)                # (n1, 3)
        M_rho = np.einsum("ia,ib->ab",
                          grad0_all[keep_idx], B_all[keep_idx])
        M_rho *= -m * Bbar * (g - 2.0) / rho0
        return (-(M_hess + M_rho)).real

    # -- eigenvalues / sound speed ------------------------------------------
    def omega2(self, k):
        """The three omega^2 eigenvalues of the EXACT P (real, symmetric)."""
        P = self.exact_p_matrix(k)
        return np.linalg.eigvalsh(P)

    def sound_speed(self, k):
        """c_SPH = omega_k / |k| (the longitudinal / sound eigenvalue of the
        EXACT P) and the ratio to the continuum c."""
        k = np.asarray(k, float)
        knorm = np.linalg.norm(k)
        if knorm == 0:
            return None
        P = self.exact_p_matrix(k)
        w, v = np.linalg.eigh(P)
        kd = k / knorm
        # the longitudinal mode is the eigenvector most aligned with k
        ov = np.array([abs(v[:, al] @ kd) for al in range(3)])
        iL = int(np.argmax(ov))
        wL = w[iL]
        c_sph = math.sqrt(max(wL, 0.0)) / knorm if wL > 0 else np.nan
        return c_sph, c_sph / math.sqrt(self.c2)


# ---------------------------------------------------------------------------
# Analytic limits for validation (paper eqs. 25-29, A19, footnote 10).
# ---------------------------------------------------------------------------

def continuum_sound_speed(oracle: StabilityOracle, k) -> float:
    """Large-|k| (h|k| >> 1) longitudinal omega^2/k^2 -> c^2 * Wbar. As
    |k| grows Wbar -> 0, so omega^2_k/k^2 -> 0 (short sound waves not
    resolved). The RESOLVED-wave check instead fixes h|k| small (eq. 29)."""
    k = np.asarray(k, float)
    w = oracle.omega2(k)
    knorm = np.linalg.norm(k)
    kd = k / knorm
    P = oracle.p_matrix(k)
    _, v = np.linalg.eigh(P)
    ov = np.array([abs(v[:, al] @ kd) for al in range(3)])
    wL = w[int(np.argmax(ov))]
    return wL / knorm**2


def small_k_sound_speed(oracle: StabilityOracle, k) -> float:
    """Small-|k| longitudinal omega^2_k/k^2 (the SPH sound speed squared at
    finite N_H; -> c^2 = 1 as N_H -> infinity, the continuum)."""
    k = np.asarray(k, float)
    knorm = np.linalg.norm(k)
    P = oracle.p_matrix(k)
    w, v = np.linalg.eigh(P)
    kd = k / knorm
    ov = np.array([abs(v[:, al] @ kd) for al in range(3)])
    wL = w[int(np.argmax(ov))]
    return wL / knorm**2


# ---------------------------------------------------------------------------
# Self-consistency / validation of the oracle.
# ---------------------------------------------------------------------------

def self_consistency(oracle: StabilityOracle, k, eps: float = 1e-30) -> dict:
    """Validate the oracle: (1) complex-step vs central-difference Jacobian
    agree; (2) P is symmetric; (3) the equilibrium force vanishes; (4) the
    equilibrium density is the lattice sum (close to 1)."""
    # 1. CS vs FD Jacobian (column 0, a real central difference)
    h = 1e-7
    a_p = np.zeros(3, complex); a_p[0] = h
    a_m = np.zeros(3, complex); a_m[0] = -h
    Fp, _ = oracle.force0(k, a_p)
    Fm, _ = oracle.force0(k, a_m)
    J_fd = (Fp - Fm) / (2 * h)                       # complex, ~ real
    a_c = np.zeros(3, complex); a_c[0] = 1j * eps
    Fc, _ = oracle.force0(k, a_c)
    J_cs = np.imag(Fc) / eps
    cs_fd = float(np.max(np.abs(J_cs - J_fd.real)))

    # 2. P symmetric
    P = oracle.p_matrix(k)
    sym = float(np.max(np.abs(P - P.T)))

    # 3. equilibrium force
    a0 = np.zeros(3, complex)
    F0, rho0 = oracle.force0(k, a0)
    eq_force = float(np.max(np.abs(F0)))

    return {
        "cs_vs_fd": cs_fd,
        "P_symmetric": sym,
        "equilibrium_force": eq_force,
        "equilibrium_rho0": float(rho0.real),
    }
