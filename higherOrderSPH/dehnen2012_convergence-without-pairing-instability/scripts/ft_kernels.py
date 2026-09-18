#!/usr/bin/env python3
"""3D Fourier transforms of the Fig. 2 kernel set (D&A 2012, eq. 14).

All kernels are represented by their dimensionless 3D shape f(r) on
r in [0,1] with W(x) = C3 f(|x|/H) / H^3, so

    wbar(kappa) = (4 pi / kappa) int_0^1 C3 f(r) r sin(kappa r) dr,
    kappa = |k| H,   wbar(0) = 1.

Two evaluations, cross-checking each other:

* **Numerical** (primary curve of Fig. 2, per eq. 14): composite Simpson
  on r in [0,1]. The shipped kernels' f is evaluated through the actual
  warp functions (common.shape, float64 on CPU) -- no re-transcription.
* **Closed form** (cross-check): every kernel except the Gaussian is a
  piecewise polynomial in r, so int_a^b r^m sin(kappa r) dr is evaluated
  by exact antiderivatives (repeated integration by parts) on each piece.
  The piecewise transcription is validated against the shipped shapes
  before use. This supersedes the paper's eq. 15, whose PDF
  transcription is garbled (see the PLAN.md findings log).

The Gaussian (truncated at 16 sigma) is numerical only: its closed form
involves the imaginary error function, and the truncation error is
e^-128 ~ 3e-56 anyway.
"""

from __future__ import annotations

import math

import numpy as np
import yaml

from common import C_d, init, kernel_scale, load_reference, shape

_HERE = __import__("pathlib").Path(__file__).resolve().parent
REFERENCE_YAML = _HERE.parent / "data" / "da2012_reference.yaml"

# ---------------------------------------------------------------------------
# Piecewise-polynomial representation:  f(r) = sum_m c_m r^m on each [a, b)
# ---------------------------------------------------------------------------


def _expand_1minusr_pow(ell: int) -> list[float]:
    """Coefficients (ascending powers of r) of (1-r)^ell."""
    c = [0.0] * (ell + 1)
    for m in range(ell + 1):
        c[m] = math.comb(ell, m) * (-1.0) ** (ell - m)
    return c


def _convolve(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
    return out


def _expand_t_minus_r(t: float, p: int) -> list[float]:
    """Coefficients (ascending powers of r) of (t - r)^p."""
    c = [0.0] * (p + 1)
    for m in range(p + 1):
        c[m] = math.comb(p, m) * (t ** (p - m)) * (-1.0) ** m
    return c


def _pad(coeffs: list[float], n: int) -> list[float]:
    c = list(coeffs) + [0.0] * (n - len(coeffs))
    return c


def truncated_to_pieces(terms: list[tuple[float, float, int]]) -> list[tuple[float, float, list[float]]]:
    """f(r) = sum_i coef_i (t_i - r)_+^p_i  ->  piecewise polynomial.

    terms: (coef, t, p); the (t-r)_+ part means the term is active for
    r < t. Returns sorted pieces (a, b, coeffs ascending in r).
    """
    knots = sorted({0.0, 1.0, *[t for (_, t, _) in terms]})
    pieces = []
    for k0, k1 in zip(knots[:-1], knots[1:]):
        acc: list[float] | None = None
        for coef, t, p in terms:
            if t <= k0 + 1e-14:
                continue  # term dead throughout [k0, k1]
            c = _expand_t_minus_r(t, p) if t >= k1 - 1e-14 else None
            if c is None:
                raise ValueError(f"knot {t} lies inside piece [{k0}, {k1}]")
            c = [coef * x for x in c]
            acc = c if acc is None else [_a + _b for _a, _b in zip(acc, _pad(c, len(acc)))]
        pieces.append((k0, k1, acc or [0.0]))
    return pieces


def bspline_pieces(n: int) -> list[tuple[float, float, list[float]]]:
    """D&A eq.-11 family b_n (the shipped shapes): truncated-power form
    sum_i (-1)^i C(n,i) ((n-2i)/n - r)_+^(n-1), i = 0..floor((n-1)/2)."""
    terms = [
        ((-1.0) ** i * float(math.comb(n, i)), (n - 2 * i) / n, n - 1)
        for i in range(int((n - 1) // 2) + 1)
    ]
    return truncated_to_pieces(terms)


def wendland_pieces(ell: int, poly: list[float]) -> list[tuple[float, float, list[float]]]:
    """Wendland psi: f(r) = (1-r)^ell * sum_j poly[j] r^j on [0,1]."""
    c = _convolve(_expand_1minusr_pow(ell), poly)
    return [(0.0, 1.0, c)]


def hoct4_pieces(ref: dict) -> list[tuple[float, float, list[float]]]:
    """HOCT4 (read2010 eqs. 46-51) from the verified definition in the YAML.

    Pieces (x = r):
      [0, a]:    P x + Q
      (a, b]:    (1-x)^4 + A (g-x)^4 + B (b-x)^4
      (b, g]:    (1-x)^4 + A (g-x)^4
      (g, 1]:    (1-x)^4
    """
    d = ref["hoct4_definition"]
    A, B = float(d["A"]), float(d["B"])
    a, b, g = float(d["alpha"]), float(d["beta"]), float(d["gamma"])
    P, Q = float(d["P"]), float(d["Q"])
    one4 = _expand_1minusr_pow(4)

    def add(c1: list[float], c2: list[float], f2: float = 1.0) -> list[float]:
        n = max(len(c1), len(c2))
        return [x + f2 * y for x, y in zip(_pad(c1, n), _pad(c2, n))]

    p1 = [Q, P]
    p2 = add(add(one4, _expand_t_minus_r(g, 4), A), _expand_t_minus_r(b, 4), B)
    p3 = add(one4, _expand_t_minus_r(g, 4), A)
    return [(0.0, a, p1), (a, b, p2), (b, g, p3), (g, 1.0, one4)]


def piecewise_eval(pieces: list[tuple[float, float, list[float]]], r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    for a, b, c in pieces:
        m = (r >= a) & (r < b)
        if not m.any():
            continue
        x = r[m]
        y = np.zeros_like(x)
        for cm in reversed(c):
            y = y * x + cm
        out[m] = y
    # r == 1 belongs to the last piece
    m = r >= 1.0 - 1e-15
    a, b, c = pieces[-1]
    x = np.clip(r[m], a, 1.0)
    y = np.zeros_like(x)
    for cm in reversed(c):
        y = y * x + cm
    out[m] = y
    return out


def moment2(pieces: list[tuple[float, float, list[float]]]) -> float:
    """Exact int_0^1 f(r) r^2 dr for a piecewise-polynomial shape."""
    s = 0.0
    for a, b, c in pieces:
        for m, cm in enumerate(c):
            s += cm * (b ** (m + 3) - a ** (m + 3)) / (m + 3)
    return s


# ---------------------------------------------------------------------------
# Closed-form 3D FT
# ---------------------------------------------------------------------------


def _st_m(m: int, r: float, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Antiderivatives (in r) evaluated at r, as functions of k (array):
    S_m(k) = int r^m sin(k r) dr,  T_m(k) = int r^m cos(k r) dr.

    S_m = -r^m cos(k r)/k + (m/k) T_{m-1}
    T_m =  r^m sin(k r)/k - (m/k) S_{m-1}
    S_0 = -cos(k r)/k,  T_0 = sin(k r)/k
    """
    kr = k * r
    S = -np.cos(kr) / k
    T = np.sin(kr) / k
    rp = 1.0
    for j in range(1, m + 1):
        rp *= r
        S, T = -rp * np.cos(kr) / k + (j / k) * T, rp * np.sin(kr) / k - (j / k) * S
    return S, T


def _I_m(m: int, a: float, b: float, k: np.ndarray) -> np.ndarray:
    """Definite integral int_a^b r^m sin(k r) dr, k an array.

    Two exact forms, chosen by k to avoid catastrophic cancellation:

    * small k: Taylor series in k (entire in k, factorial convergence),
      from sin(k r) = sum_j (-1)^j k^(2j+1) r^(2j+1) / (2j+1)!:
      I_m = sum_j (-1)^j k^(2j+1) (b^(m+2j+2) - a^(m+2j+2))
            / ((2j+1)! (m+2j+2)).
      Needed because the antiderivative form subtracts two values of
      magnitude ~ m!/k^(m-1) (12! ~ 5e8 at m = 12, k = 1) to get an O(1)
      result -- the Wendland C6 closed form lost ~1e-4 this way at k = 1.
    * k >= 5: the antiderivative difference (intermediate values ~
      m!/k^(m-1) <= ~10 for m <= 12, no cancellation).
    """
    k = np.atleast_1d(np.asarray(k, float))
    out = np.empty_like(k)
    small = k < 5.0
    if small.any():
        ks = k[small]
        s = np.zeros_like(ks)
        for j in range(80):
            db_j = b ** (m + 2 * j + 2) - a ** (m + 2 * j + 2)
            t = ((-1.0) ** j) * (ks ** (2 * j + 1)) * db_j \
                / (math.factorial(2 * j + 1) * (m + 2 * j + 2))
            s_new = s + t
            if np.all(np.abs(t) < 1e-17 * np.maximum(np.abs(s_new), 1e-300)):
                s = s_new
                break
            s = s_new
        out[small] = s
    if (~small).any():
        Sb, _ = _st_m(m, b, k[~small])
        Sa, _ = _st_m(m, a, k[~small])
        out[~small] = Sb - Sa
    return out


def ft3d_closed(pieces: list[tuple[float, float, list[float]]], C3: float, k: np.ndarray) -> np.ndarray:
    """Exact 3D FT of the piecewise-polynomial shape (see module docstring).

    k must not contain 0 (wbar(0) = 1 is handled separately).
    """
    k = np.atleast_1d(np.asarray(k, float))
    w = np.zeros_like(k)
    for a, b, c in pieces:
        for m, cm in enumerate(c):
            if cm == 0.0:
                continue
            w += cm * _I_m(m + 1, a, b, k)
    return 4.0 * np.pi * C3 * w / k


# ---------------------------------------------------------------------------
# Numerical 3D FT (eq. 14, primary)
# ---------------------------------------------------------------------------

_SIMPSON_N = 200001  # even number of intervals, float64


def _simpson_weights(n: int) -> np.ndarray:
    w = np.full(n, 2.0)
    w[0] = w[-1] = 1.0
    w[1:-1:2] = 4.0
    return w


def ft3d_numeric(f, C3: float, k: np.ndarray, n: int = _SIMPSON_N, chunk: int = 128) -> np.ndarray:
    """Simpson evaluation of eq. 14 for the shape f (vectorised callable)."""
    k = np.atleast_1d(np.asarray(k, float))
    r = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    wts = _simpson_weights(n) * h / 3.0
    fr = np.atleast_1d(f(r)) * r  # shape evaluated once
    out = np.empty_like(k)
    for i0 in range(0, len(k), chunk):
        kk = k[i0:i0 + chunk]
        S = np.sin(np.outer(kk, r))
        out[i0:i0 + chunk] = 4.0 * np.pi * C3 * np.sum(S * fr * wts, axis=1) / kk
    return out


# ---------------------------------------------------------------------------
# The Fig. 2 kernel set
# ---------------------------------------------------------------------------

# name -> shipped code name: f via common.shape (the audited warp
# functions). The whole Fig. 1/2 set is shipped (HOCT4 + Gaussian
# onboarded in Phase 3, 2026-09-18).
SHIPPED_FIG2 = {
    "cubic_b4": "cubic_b4",
    "quartic_b5": "quartic_b5",
    "quintic_b6": "quintic_b6",
    "b7": "b7",
    "b8": "b8",
    "wendland_C2": "wendland_C2",
    "wendland_C4": "wendland_C4",
    "wendland_C6": "wendland_C6",
    "hoct4": "hoct4",
    "gaussian": "gaussian",
}

# closed-form pieces for the shipped B-splines (n of the D&A family) and
# Wendland kernels: (ell, poly coefficients ascending in r). b7/b8 join
# the figure set 2026-09-18 (the D&A Table 1 set lists b4-b6; b7/b8 are
# the same family, shipped as enum B7/B8).
BSPLINE_ORDER = {
    "cubic_b4": 4, "quartic_b5": 5, "quintic_b6": 6,
    "b7": 7, "b8": 8,
}
WENDLAND_POLY = {
    "wendland_C2": (4, [1.0, 4.0]),                    # (1-r)^4 (1+4r)
    "wendland_C4": (6, [1.0, 6.0, 35.0 / 3.0]),        # (1-r)^6 (1+6r+35/3 r^2)
    "wendland_C6": (8, [1.0, 8.0, 25.0, 32.0]),        # (1-r)^8 (1+8r+25r^2+32r^3)
}


def kernel_shapes(ref: dict) -> dict:
    """name -> dict with C3, scale3, shape(r) (numpy), pieces (or None).

    Every kernel is SHIPPED (Phase 3): f through common.shape (the
    audited warp functions, float64 on CPU), C3/scale3 through the
    shipped eval_C_d/eval_kernelScale. The pieces (closed-form FT input)
    are cross-validated against the shipped shape on a dense grid before
    being returned (max|diff| < 1e-12); the Gaussian has no pieces
    (numerical FT only).
    """
    init()
    out = {}
    for name, code_name in SHIPPED_FIG2.items():
        C3 = C_d(3, code_name)
        scale = kernel_scale(3, code_name)

        def f(r, _cn=code_name):
            return shape(np.atleast_1d(r), 3, _cn)

        if name in BSPLINE_ORDER:
            pieces = bspline_pieces(BSPLINE_ORDER[name])
        elif name in WENDLAND_POLY:
            ell, poly = WENDLAND_POLY[name]
            pieces = wendland_pieces(ell, poly)
        elif name == "hoct4":
            pieces = hoct4_pieces(ref)  # closed form, cross-check only
        else:  # gaussian: truncated exponential, numerical FT only
            pieces = None
        if pieces is not None:
            _validate_pieces(pieces, f, name)
        out[name] = dict(C3=C3, scale3=scale, shape=f, pieces=pieces)
    return out


def _validate_pieces(pieces, f, name: str) -> None:
    # Same tolerance as the audit's form check (1e-12): float64 rounding
    # of the expanded polynomial vs the shipped cpow form grows with
    # degree (Wendland C6, degree 11, sits at ~1.6e-13). Any real
    # transcription error is O(1e-2) and would be caught 3+ orders out.
    g = np.linspace(0.0, 1.0, 4001)
    d = np.max(np.abs(piecewise_eval(pieces, g) - np.atleast_1d(f(g))))
    assert d < 1e-12, f"{name}: piecewise transcription deviates from the " \
                      f"shipped shape by {d:.3e}"


if __name__ == "__main__":
    ref = load_reference()
    ks = kernel_shapes(ref)
    k = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 100.0])
    print("name          closed    numeric   max|Δ| (5-pt grid)")
    for name, d in ks.items():
        knum = ft3d_numeric(d["shape"], d["C3"], k)
        row = f"{name:<13} "
        if d["pieces"] is not None:
            kcl = ft3d_closed(d["pieces"], d["C3"], k)
            row += f"{kcl[0]:+.6e}  {knum[0]:+.6e}   {np.max(np.abs(kcl - knum)):.3e}"
        else:
            row += f"{'(numeric only)':>13}  {knum[0]:+.6e}   -"
        print(row)
