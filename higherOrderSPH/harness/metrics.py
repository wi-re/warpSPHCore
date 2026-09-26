#!/usr/bin/env python3
"""Error metrics and observed-order extraction for the convergence harness.

Pure numpy/torch -- no warp, no warpSPHCore -- so it is unit-testable on
CPU without a GPU and without the float64-precision env var dance.

* `error_norms` -- masked L1 / L2 / Linf of (output - analytic). Tensor
  outputs (gradient of a vector field, ...) are Frobenius-reduced per
  particle before norming.
* `observed_order` -- log-log least-squares slope of error vs resolution,
  with a saturation flag: a series whose error stops decreasing (max/min
  < 2) is reported as `saturated`, not fitted as an order (the float64
  noise floor or a discretization plateau looks exactly like that).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["error_norms", "OrderResult", "observed_order",
           "SATURATION_RATIO"]

# A series with max/min error below this is "flat" -- no order to extract.
SATURATION_RATIO = 2.0


def error_norms(out: torch.Tensor, analytic: torch.Tensor,
                mask: torch.Tensor | None = None) -> dict[str, float]:
    """Masked absolute error norms of (out - analytic).

    `out` and `analytic` must have the same shape; `mask` (bool, N,)
    selects the particles (None = all). Extra trailing axes (field
    components) are Frobenius-reduced per particle.
    """
    diff = (out - analytic).abs()
    if diff.ndim > 1:
        # Frobenius reduction over every non-particle axis.
        diff = diff.flatten(start_dim=1).pow(2).sum(dim=1).sqrt()
    if mask is not None:
        diff = diff[mask]
    if diff.numel() == 0:
        return {"l1": float("nan"), "l2": float("nan"), "linf": float("nan")}
    l2 = torch.linalg.vector_norm(diff, ord=2)
    n = diff.numel()
    # L1 / L2 normalized by the particle count so the two are comparable
    # in scale (the slope they produce is identical either way).
    return {
        "l1": float(diff.sum() / n),
        "l2": float(l2 / n ** 0.5),
        "linf": float(diff.max()),
    }


@dataclass
class OrderResult:
    slope: float                 # observed order (d log E / d log x)
    r_squared: float             # goodness of the log-log fit
    pairwise_slopes: list[float] = field(default_factory=list)
    saturated: bool = False      # error series is flat: no order to report
    exact: bool = False          # errors are (all, or some) exactly zero:
                                 # machine-precision reproduction, not a
                                 # convergence -- nothing to fit


def observed_order(x: np.ndarray | list[float],
                   y: np.ndarray | list[float]) -> OrderResult:
    """Observed convergence order of errors `y` vs resolution scale `x`
    (either h or dx -- the slope is with respect to log x).

    Requires at least two strictly positive points; raises ValueError
    otherwise. Returns a saturated result (slope = nan) when the error
    series is flat within SATURATION_RATIO.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if x.size < 2:
        raise ValueError("need at least two points for an order")
    # Collapse duplicate resolution values (two lattice sizes can snap to
    # the same dx): keep the last occurrence. Fewer than two distinct
    # points leaves nothing to fit.
    last = {}
    for xi, yi in zip(x, y):
        last[xi] = yi
    x = np.array(list(last), dtype=float)
    y = np.array([last[xi] for xi in last], dtype=float)
    if x.size < 2:
        return OrderResult(slope=float("nan"), r_squared=float("nan"),
                           pairwise_slopes=[], saturated=True)
    if np.any(x <= 0) or np.any(y < 0):
        raise ValueError("x and y must be non-negative, x strictly positive")
    if np.any(y == 0):
        return OrderResult(slope=float("nan"), r_squared=float("nan"),
                           pairwise_slopes=[], exact=True)

    pairwise = [
        float(np.log(y[i + 1] / y[i]) / np.log(x[i + 1] / x[i]))
        for i in range(x.size - 1)
    ]
    if float(np.max(y) / np.min(y)) < SATURATION_RATIO:
        return OrderResult(slope=float("nan"), r_squared=float("nan"),
                           pairwise_slopes=pairwise, saturated=True)

    lx = np.log(x)
    ly = np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (slope, intercept), residuals, rank, _ = np.linalg.lstsq(A, ly, rcond=None)
    ss_res = float(((A @ [slope, intercept]) - ly) @ ((A @ [slope, intercept]) - ly))
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return OrderResult(slope=float(slope), r_squared=r2,
                       pairwise_slopes=pairwise, saturated=False)


# search range for the finite-reference-corrected order
REF_ORDER_MIN = 0.05
REF_ORDER_MAX = 8.0


def reference_corrected_order(x: np.ndarray | list[float],
                              y: np.ndarray | list[float],
                              x_ref: float) -> OrderResult:
    """Observed order of errors `y` measured against a *finite* reference run
    at resolution `x_ref` (< every x), correcting the self-convergence bias.

    A plain log-log fit of ``|u_h - u_ref|`` vs h over-reads the order when
    the reference is only a factor 1.5-2 finer than the last rung: for a
    systematic error ``e(h) = C (h^p - h_ref^p)`` (same sign structure at
    every level -- true for shock smearing, dissipation, dispersion) a true
    first-order scheme fits ~1.4 on a 200/400/800 vs 1600 ladder and ~2.0 on
    32/48/64 vs 96. This fits that model directly: C is profiled out in
    closed form in log space, p is scanned on a grid and refined.

    The model is wrong for *uncorrelated* errors (e.g. chaotic flows, where
    the reference's error does not cancel), in which case the plain fit is
    the better estimate; report both.

    **Validated against exact solutions (2026-09-26) -- use with care.**
    Sod (CompSPH) L1: plain 1.46, corrected 1.07, exact-solution 0.98 --
    the correction works. Sedov (CRKSPH) L1: plain 0.92, corrected 0.23,
    exact-solution 1.03 -- it badly over-corrects, because the shock
    position error changes sign between rungs, breaking the same-sign
    assumption. Where an exact solution exists, score against it instead
    (`pde_cases.PDECase.exact`); otherwise treat this as a bound on how
    much a plain fit *could* be inflated, not as the order. A fit that lands on a search bound
    (``REF_ORDER_MIN`` / ``REF_ORDER_MAX``) is flagged ``saturated``. At the
    lower bound, read `r_squared`: low means the model does not describe
    the data (the error *decelerates* toward the reference, which the model
    cannot produce); high means the error converges slower than any
    resolvable power law (p -> 0 is log-like: e ~ log(h/h_ref)).
    `pairwise_slopes` are the plain pairwise log-log slopes (coarse ->
    fine), for comparison.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if x.size < 2:
        raise ValueError("need at least two points for an order")
    if not np.all(x > x_ref) or x_ref <= 0:
        raise ValueError("every x must exceed the reference x_ref > 0")
    if np.any(y < 0):
        raise ValueError("errors must be non-negative")
    if np.any(y == 0):
        return OrderResult(slope=float("nan"), r_squared=float("nan"),
                           pairwise_slopes=[], exact=True)
    order = np.argsort(x)[::-1]              # coarse -> fine, as in observed_order
    x, y = x[order], y[order]
    pairwise = [
        float(np.log(y[i + 1] / y[i]) / np.log(x[i + 1] / x[i]))
        for i in range(x.size - 1)
    ]
    ly = np.log(y)

    def sse(p: float) -> float:
        g = np.log(x ** p - x_ref ** p)
        logC = float((ly - g).mean())
        r = ly - g - logC
        return float(r @ r)

    grid = np.linspace(REF_ORDER_MIN, REF_ORDER_MAX, 800)
    vals = np.array([sse(p) for p in grid])
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    for _ in range(60):                      # golden-section refine
        a = hi - 0.618 * (hi - lo)
        b = lo + 0.618 * (hi - lo)
        if sse(a) < sse(b):
            hi = b
        else:
            lo = a
    p = 0.5 * (lo + hi)
    ss_res = sse(p)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    on_bound = (p - REF_ORDER_MIN < 1e-2) or (REF_ORDER_MAX - p < 1e-2)
    return OrderResult(slope=float(p), r_squared=r2,
                       pairwise_slopes=pairwise, saturated=bool(on_bound))
