#!/usr/bin/env python3
"""PDE benchmark case registry for the Pass-2 convergence/dissipation suite.

Each entry points at a `warpSPH` frontend case (imported, never re-implemented
-- the harness convention) plus the lean-budget run parameters and the
convergence metric:

* **analytic** -- the case has a known analytic field; the metric is the L2
  field error vs that field at a fixed final time `t_star` (a clean
  observed-order signal). Used for the smooth cases with closed-form solutions
  (TGV, linearWave).
* **reference** -- the case has no clean analytic solution (or it is a
  shock/dissipation diagnostic); the metric is the L2 field error vs the
  finest-ladder run, both projected onto a common regular grid as mass-weighted
  cell averages (see `field_error.grid_l2_error`). The finest point is the
  reference, so the order is fit on the 3 coarser points. Shocks are expected
  to swamp the order (the error saturates) -- that is the diagnostic.

The `nx_ladder` is 4 points and `t_star` is the case's full simulated time;
the 2D incompressible cases cost ~nx^3 under adaptive dt, so the full run is
an overnight job (~3-4 h). The 1D cases finish in minutes and are the fast
dev/test path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

__all__ = ["PDECase", "CASES", "tgv_analytic_velocity", "linearWave_analytic"]


@dataclass
class PDECase:
    name: str
    module: str                       # import path of the case module
    case_attr: str                    # name of the registered Case object
    dim: int
    metric: str                       # 'analytic' | 'reference'
    nx_ladder: list                   # 4 resolutions (particles across / count)
    t_star: float                     # final simulated time
    # harness-side scheme override (None = the case's own registered scheme).
    # The frontend defaults are deliberately left untouched (they balance
    # other case families); the harness picks the scheme that best represents
    # the method under test.
    scheme: Optional[str] = None
    # harness-side case-parameter overrides (merged over the case's own
    # defaults in run_pde.build_spec) -- e.g. holding the build-time IC
    # fixed while the scheme varies.
    params: dict = field(default_factory=dict)
    ref_nx: Optional[int] = None      # reference resolution (reference metric)
    # analytic field for the 'analytic' metric: (positions, t, params) -> (N, dim)
    analytic: Optional[Callable] = None
    # which state field the metric measures (default 'velocities')
    field: str = "velocities"
    notes: str = ""


# ---------------------------------------------------------------------------
# analytic fields
# ---------------------------------------------------------------------------

def tgv_analytic_velocity(positions: torch.Tensor, t: float,
                          params: dict) -> torch.Tensor:
    """2D Taylor-Green analytic velocity at time `t`.

    Mirrors the case's own initial condition (`cases/tgv.py`), with the
    viscous amplitude decay ``exp(-2 nu kTgv^2 t)`` derived from the
    vorticity equation (``nabla^2 omega = -2 kTgv^2 omega``); the KE decay
    rate ``2 nu k^2`` it implies matches the case's `analyticDecayRate`.
    """
    k = float(params["k"])
    nu = float(params["nu"])
    uMag = float(params["uMag"])
    kTgv = k / 2.0
    phase = np.pi / 2.0 if int(k) % 2 == 0 else 0.0
    decay = torch.exp(torch.tensor(-2.0 * nu * kTgv ** 2 * t,
                                   dtype=positions.dtype, device=positions.device))
    x = positions[:, 0]
    y = positions[:, 1]
    u = uMag * decay * torch.cos(kTgv * x + phase) * torch.sin(kTgv * y + phase)
    v = -uMag * decay * torch.sin(kTgv * x + phase) * torch.cos(kTgv * y + phase)
    return torch.stack([u, v], dim=-1)


def linearWave_analytic(positions: torch.Tensor, t: float,
                        params: dict) -> torch.Tensor:
    """1D linear acoustic wave analytic velocity at time `t`.

    The case (`cases/linearWave.py` / `caseUtils/.../linearWave/wave.py`) seeds
    a small-amplitude sinusoid `delta = A sin(2 pi x / lamda)` and sets the
    linearized velocity `v = c_s * delta`. For a right-travelling acoustic wave
    the density perturbation advects at `c_s`, and the linearized relation is
    `v = (c_s / rho0) * delta`. With the case defaults (rho0 = 1, c_s = 1) this
    is just `A sin(2 pi (x - c_s t) / lamda)`; the general form is kept so the
    metric stays correct if the params are overridden.
    """
    A = float(params["A"])
    lamda = float(params["lamda"])
    c_s = float(params["c_s"])
    rho0 = float(params["rho0"])
    x = positions[:, 0]
    v = (c_s / rho0) * A * torch.sin(2.0 * np.pi * (x - c_s * t) / lamda)
    return v.view(-1, 1)


# ---------------------------------------------------------------------------
# registry (full t_star + 4-resolution ladders; the 2D incompressible cases
# cost ~nx^3 under adaptive dt, so the overnight full run is ~3-4 h)
# ---------------------------------------------------------------------------

CASES: dict[str, PDECase] = {
    "tgv": PDECase(
        name="tgv", module="warpSPH.cases.tgv", case_attr="tgvCase",
        dim=2, metric="analytic",
        nx_ladder=[32, 48, 64, 96], t_star=2.0,
        analytic=tgv_analytic_velocity, field="velocities",
        notes="primary order-of-convergence benchmark (smooth, exact solution)",
    ),
    "tgv-wc": PDECase(
        name="tgv-wc", module="warpSPH.cases.tgvWeaklyCompressible",
        case_attr="tgvWeaklyCompressibleCase",
        dim=2, metric="analytic",
        nx_ladder=[32, 48, 64, 96], t_star=2.0,
        analytic=tgv_analytic_velocity, field="velocities",
        # The shared frontend default shift is 1/8 of Sun 2017 Eq. (7),
        # weakened for free-surface cases; on TGV that is too weak to improve
        # on plain delta-SPH (it "sits on top of" it). The real delta+ is the
        # full-strength shift, so this leg runs sun2017DeltaSPH
        # (TGV_NOTES.md section 6: -30% velocity error, 4.5x less volume
        # drift, same cost).
        scheme="sun2017DeltaSPH",
        # Relax the build-time shuffle at the reference (1/8) shift strength
        # so the IC is comparable across schemes: the full-strength
        # relaxation over-mixes the lattice (a2-rms 0.34 at t=0 vs 8.5e-3
        # for the reference glass), which dominated the coarse-resolution
        # ladder rows (TGV_NOTES.md section 7.3). In-run shifting is
        # unaffected (full strength, via `scheme`).
        params={"shuffleEq7": False},
        notes="delta+-SPH leg of the TGV vortex, full-strength Sun 2017 "
              "Eq.(7) shifting in-run (see `scheme`); build-time shuffle "
              "relaxation held at the reference strength (`params`); same "
              "IC/analytic as tgv; explicit WCSPH so each step is far "
              "cheaper than the incompressible leg; needs the frontend "
              "densityDiffusion eager-scalar coercion for float64 "
              "(TGV_NOTES.md section 4)",
    ),
    "linearWave": PDECase(
        name="linearWave", module="warpSPH.cases.linearWave",
        case_attr="linearWaveCase",
        dim=1, metric="analytic",
        nx_ladder=[100, 200, 400, 800], t_star=1.0,
        analytic=linearWave_analytic, field="velocities",
        notes="1D linear acoustic wave (smooth, exact solution)",
    ),
    "gresho": PDECase(
        name="gresho", module="warpSPH.cases.greshoVortex",
        case_attr="greshoVortexCase",
        dim=2, metric="reference",
        nx_ladder=[32, 48, 64, 96], t_star=3.0,
        field="velocities",
        notes="decaying vortex; no clean analytic -> vs finest-ladder reference",
    ),
    "kelvinHelmholtz": PDECase(
        name="kelvinHelmholtz", module="warpSPH.cases.kelvinHelmholtz",
        case_attr="kelvinHelmholtzCase",
        dim=2, metric="reference",
        nx_ladder=[32, 48, 64, 96], t_star=4.0,
        field="velocities",
        notes="billow roll-up; vs finest-ladder reference",
    ),
    "sod": PDECase(
        name="sod", module="warpSPH.cases.sod", case_attr="sodCase",
        dim=1, metric="reference",
        nx_ladder=[200, 400, 800, 1600], t_star=0.15,
        field="densities",
        notes="shock tube; shock swamps the order (diagnostic); vs reference",
    ),
    "sedov": PDECase(
        name="sedov", module="warpSPH.cases.sedov", case_attr="sedovCase",
        dim=1, metric="reference",
        nx_ladder=[200, 400, 800, 1600], t_star=1.0,
        field="densities",
        notes="blast wave; raw L2 (order 0.47, non-monotonic) is dominated by "
              "the resolution-dependent shock structure, not position (the "
              "alignment shifts are sub-cell) -- the L1 area norm (order 0.92) "
              "and the shift-aligned L2 (order 0.75) are the robust shock "
              "metrics; SEDov_NOTES.md",
    ),
}


def load_case_entry(entry: PDECase):
    """Import the case module and return the registered Case object."""
    import importlib
    module = importlib.import_module(entry.module)
    return getattr(module, entry.case_attr)
