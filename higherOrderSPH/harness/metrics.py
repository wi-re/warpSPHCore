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
