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

Shock diagnostics (1D reference cases such as Sedov / Sod): the common-grid
L2 error overweights a *misaligned* shock (a discontinuity of height Delta
shifted by delta costs O(sqrt(delta)) in L2), so two further metrics are
provided:

* `grid_error_p(..., p=1)` -- the L1 (area) error, which costs O(delta) for a
  shifted discontinuity and is the standard shock-capture metric;
* `aligned_1d_l2_error(...)` -- the L2 error after optimally translating the
  reference profile against the coarse one (sub-cell, linearly interpolated,
  periodic wrap), which isolates the shock-*shape* error from the
  shock-*position* error.

Pure torch, so it is unit-testable on CPU without warp or the float64 env.
"""

from __future__ import annotations

import itertools

import torch

__all__ = ["grid_l2_error", "grid_error_p", "aligned_1d_l2_error",
           "particle_error_norms"]


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
    return grid_error_p(field_coarse, masses_coarse, pos_coarse,
                        field_ref, masses_ref, pos_ref,
                        L, dim, periodic, n_grid, p=2)


def grid_error_p(field_coarse: torch.Tensor, masses_coarse: torch.Tensor,
                 pos_coarse: torch.Tensor,
                 field_ref: torch.Tensor, masses_ref: torch.Tensor,
                 pos_ref: torch.Tensor,
                 L: float, dim: int, periodic: bool,
                 n_grid: int, p: int = 2) -> float:
    """Generalized p-norm of the common-grid CIC difference
    (`(mean |diff|^p)^(1/p)` over the cells both sets fill). `p=1` is the L1
    (area) error -- the standard shock metric: a discontinuity of height
    Delta shifted by delta costs O(delta) in L1 vs O(sqrt(delta)) in L2.
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
    if p == 1:
        return float(diff.abs().mean())
    return float((diff.abs() ** p).mean() ** (1.0 / p))


def _shift1d_periodic(profile: torch.Tensor, delta_cells: float) -> torch.Tensor:
    """Periodically translate a 1D grid profile by `delta_cells` (in grid
    cells, sub-cell via linear interpolation): (shifted)[i] = profile[i + d]."""
    n = profile.shape[0]
    idx = torch.arange(n, dtype=profile.dtype, device=profile.device) \
        + delta_cells
    i0 = torch.floor(idx).long() % n
    i1 = (i0 + 1) % n
    w = idx - idx.floor()
    return profile[i0] * (1.0 - w) + profile[i1] * w


def _densify_1d(cell_field: torch.Tensor, mass_cell: torch.Tensor) -> torch.Tensor:
    """Dense (gap-filled) version of a 1D CIC projection: the projected
    profile is a cell average defined only on the mass-filled cells, with
    zeros in the gaps (typically most cells -- a particle spreads over 2 of
    the ~4-8 cells between neighbours). Shifts (or any continuous
    comparison) must act on a profile, so the empty cells are filled by
    periodic linear interpolation between the filled cells."""
    n = cell_field.shape[0]
    filled = (mass_cell > 0).nonzero().flatten()
    if filled.shape[0] == n:
        return cell_field
    if filled.shape[0] < 2:
        raise ValueError("not enough filled cells to densify")
    vals = cell_field[filled]
    # three-periodic extension so every cell in [0, n) has a filled cell
    # on each side; searchsorted gives the first filled index at/after i
    ext_idx = torch.cat([filled - n, filled, filled + n])
    ext_val = torch.cat([vals, vals, vals])
    pos = torch.arange(n, dtype=torch.float64, device=cell_field.device)
    j1 = torch.searchsorted(ext_idx, pos, side="left").clamp(max=n * 3 - 1)
    j0 = j1 - 1
    x0, x1 = ext_idx[j0].to(torch.float64), ext_idx[j1].to(torch.float64)
    v0, v1 = ext_val[j0], ext_val[j1]
    w = torch.where(x1 > x0, (pos - x0) / (x1 - x0), 0.0)
    return (v0 * (1.0 - w) + v1 * w).to(cell_field.dtype)


def aligned_1d_l2_error(field_coarse: torch.Tensor,
                        masses_coarse: torch.Tensor, pos_coarse: torch.Tensor,
                        field_ref: torch.Tensor, masses_ref: torch.Tensor,
                        pos_ref: torch.Tensor,
                        L: float, periodic: bool, n_grid: int,
                        max_shift: float) -> tuple[float, float]:
    """L2 error between two 1D particle sets after optimally translating the
    *reference* profile against the coarse one on the common CIC grid.

    The shift delta is searched in grid cells (coarse 2-cell scan over
    [-max_shift, +max_shift], then a 0.1-cell scan and a parabolic
    refinement); the reference is sampled at x + delta with linear
    interpolation and periodic wrap. Returns `(aligned_l2, shift)`, where
    `shift = delta* dx_grid` is in physical units: shift > 0 means the
    reference features sat `shift` to the *right* of the coarse ones before
    alignment (the translation applied to the reference to align).

    `max_shift` bounds the search (physical units) -- a few times the coarse
    dx covers the resolution-dependent shock-position drift; if the optimum
    saturates the bound, widen it.

    The CIC projections are gappy (a particle fills 2 of the ~4-8 cells
    between neighbours; empty cells hold zero), so both profiles are first
    densified (periodic linear interpolation over the gaps) and the error is
    taken over all cells.
    """
    if pos_coarse.shape[1] != 1:
        raise ValueError("aligned_1d_l2_error is 1D only")
    x0 = -L / 2.0
    dx_grid = L / n_grid
    fc, mc = _project(field_coarse, masses_coarse, pos_coarse, x0, dx_grid,
                      n_grid, periodic, 1)
    fr, mr = _project(field_ref, masses_ref, pos_ref, x0, dx_grid,
                      n_grid, periodic, 1)
    if not bool((mc > 0).any()) or not bool((mr > 0).any()):
        return float("nan"), float("nan")
    fc_d = _densify_1d(fc, mc).flatten()
    fr_d = _densify_1d(fr, mr).flatten()

    def err2(delta_cells: float) -> float:
        shifted = _shift1d_periodic(fr_d, delta_cells)
        return float(((fc_d - shifted) ** 2).mean())

    max_cells = max_shift / dx_grid
    # stage 1: 2-cell scan
    coarse_deltas = torch.arange(-max_cells, max_cells + 1e-9, 2.0,
                                 dtype=torch.float64)
    e = err2(coarse_deltas[0].item())
    best_d, best_e = coarse_deltas[0].item(), e
    for d in coarse_deltas[1:]:
        e = err2(d.item())
        if e < best_e:
            best_d, best_e = d.item(), e
    # stage 2: 0.1-cell scan around the stage-1 minimum
    fine_deltas = torch.arange(best_d - 2.0, best_d + 2.0 + 1e-9, 0.1,
                               dtype=torch.float64)
    errs = torch.tensor([err2(d.item()) for d in fine_deltas])
    j = int(errs.argmin().item())
    # stage 3: parabolic refinement (interior minimum only)
    d_star = fine_deltas[j].item()
    if 0 < j < len(fine_deltas) - 1:
        y0, y1, y2 = float(errs[j - 1]), float(errs[j]), float(errs[j + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-30:
            d_star = fine_deltas[j].item() + 0.1 * (y0 - y2) / (2.0 * denom)
            d_star = min(max(d_star, fine_deltas[j - 1].item()),
                         fine_deltas[j + 1].item())
    return err2(d_star) ** 0.5, d_star * dx_grid


def particle_error_norms(field: torch.Tensor, exact: torch.Tensor,
                         volumes: torch.Tensor) -> dict[str, float]:
    """Volume-weighted particle L1 / L2 error of `field` against an exact
    solution sampled at the particle positions (the `exact` metric of the
    reference cases).

    ``L1 = sum V|e| / sum V``, ``L2 = sqrt(sum V e^2 / sum V)`` with
    ``V = m / rho`` -- a quadrature of the continuous norm, so a case whose
    particle spacing varies across the domain (Sod: 4x denser on the left)
    is not weighted by particle count. Vector fields are reduced per
    particle by the Euclidean norm.
    """
    e = (field - exact).detach()
    if e.ndim > 1:
        e = e.flatten(start_dim=1).norm(dim=1)
    e = e.abs()
    w = volumes.detach().to(e.dtype)
    wsum = w.sum()
    return {
        "l1": float((w * e).sum() / wsum),
        "l2": float(((w * e * e).sum() / wsum).sqrt()),
    }
