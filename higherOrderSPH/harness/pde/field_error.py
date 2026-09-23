#!/usr/bin/env python3
"""Grid-projection field error for the PDE benchmark's *reference* metric.

For cases without a clean analytic solution (Gresho, Sod, Sedov,
Kelvin--Helmholtz), the convergence metric is the L2 field error against a
high-resolution reference run. Two different particle sets (the coarse run and
the reference run) are compared by projecting each onto a common regular grid
as a **cloud-in-cell (CIC) mass-weighted field average**, then taking the RMS
difference over the cells both sets fill.

CIC spreads each particle's mass (and mass*field) over the ``2^dim``
neighbouring cells with linear weights, so the projected field is a smooth
(cell-averaged) representation: a cell's value is the mass-weighted average of
the field at the particles in it. This is used for every field -- velocity
(vector) and density (scalar) alike. (A ``mass/volume`` density projection is
rejected because it is spiky on a grid finer than the particle spacing: the
mass is concentrated in the particles, so the cell density oscillates between
spikes and zeros and swamps the true field difference.)

This is a post-processing *diagnostic* (like the driver-side conservation
quantities in `conservation.py`), not a re-implementation of an SPH operator --
it reuses no `src/` kernel, consistent with the harness convention. A common
grid sidesteps the fiddly mixed query/reference adjacency, volume and support
conventions of an inter-set SPH interpolation.

Pure torch, so it is unit-testable on CPU without warp or the float64 env.
"""

from __future__ import annotations

import itertools

import torch

__all__ = ["grid_l2_error"]


def _row_major(idx: torch.Tensor, n_grid: int, dim: int) -> torch.Tensor:
    """Row-major linear index from per-dimension cell indices (N, dim)."""
    lin = idx[:, 0].clone()
    for d in range(1, dim):
        lin = lin * n_grid + idx[:, d]
    return lin


def _project(field: torch.Tensor, masses: torch.Tensor,
             pos: torch.Tensor, x0: float, dx_grid: float, n_grid: int,
             periodic: bool, dim: int):
    """CIC-project a particle field onto the common grid as a mass-weighted
    cell average. Returns (cell_field, cell_mass). `field` is scalar `(N,)` or
    vector `(N, D)`. Cells with no mass get a zero field (excluded downstream
    by the shared-cell mask)."""
    n_cells = n_grid ** dim
    device = pos.device
    dtype = masses.dtype
    n = pos.shape[0]

    mass_cell = torch.zeros(n_cells, device=device, dtype=dtype)
    if field.dim() > 1:
        mf_cell = torch.zeros(n_cells, field.shape[1], device=device,
                              dtype=dtype)
    else:
        mf_cell = torch.zeros(n_cells, device=device, dtype=dtype)

    frac = (pos - x0) / dx_grid
    lower = torch.floor(frac).long()
    w_hi = frac - lower.to(dtype)          # weight to the upper cell (f)
    w_lo = 1.0 - w_hi                      # weight to the lower cell (1-f)

    for combo in itertools.product((0, 1), repeat=dim):
        cell = lower.clone()
        weight = torch.ones(n, device=device, dtype=dtype)
        for d in range(dim):
            weight = weight * (w_hi[:, d] if combo[d] else w_lo[:, d])
            if combo[d]:
                cell[:, d] = cell[:, d] + 1
        cell = (cell % n_grid) if periodic else cell.clamp(0, n_grid - 1)
        lin = _row_major(cell, n_grid, dim)
        wmass = masses * weight
        mass_cell.scatter_add_(0, lin, wmass)
        if mf_cell.dim() > 1:
            for c in range(mf_cell.shape[1]):
                mf_cell[:, c].scatter_add_(0, lin, wmass * field[:, c])
        else:
            mf_cell.scatter_add_(0, lin, wmass * field)

    mass_safe = mass_cell.clamp(min=1e-30)
    cell_field = (mf_cell / mass_safe.unsqueeze(1) if mf_cell.dim() > 1
                  else mf_cell / mass_safe)
    return cell_field, mass_cell


def grid_l2_error(field_coarse: torch.Tensor, masses_coarse: torch.Tensor,
                  pos_coarse: torch.Tensor,
                  field_ref: torch.Tensor, masses_ref: torch.Tensor,
                  pos_ref: torch.Tensor,
                  L: float, dim: int, periodic: bool,
                  n_grid: int) -> float:
    """RMS L2 field error between a coarse and a reference particle set, both
    CIC-projected (mass-weighted cell averages) onto a common `n_grid`-per-
    dimension regular grid over `[-L/2, L/2]^dim`. The error is taken over the
    cells both sets fill.

    `field_*` is the state field to compare: scalar `(N,)` (e.g. density) or
    vector `(N, dim)` (e.g. velocity).
    """
    x0 = -L / 2.0
    dx_grid = L / n_grid
    fc, mc = _project(field_coarse, masses_coarse, pos_coarse, x0, dx_grid,
                      n_grid, periodic, dim)
    fr, mr = _project(field_ref, masses_ref, pos_ref, x0, dx_grid,
                      n_grid, periodic, dim)
    common = (mc > 0) & (mr > 0)
    if not bool(common.any()):
        return float("nan")
    diff = (fc[common] - fr[common]).flatten()
    return float(diff.norm() / diff.numel() ** 0.5)
