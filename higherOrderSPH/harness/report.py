#!/usr/bin/env python3
"""Standardized reporting for the convergence harness.

Every phase of the higher-order plan writes its results through this
module so outputs are directly comparable (the parent plan's
cross-cutting rule: the harness is reused unmodified; a change here is a
versioning event that re-runs all prior phases).

* `Row` -- one measurement: one (case, field, probe, mode, region).
* `write_rows_csv` -- flat CSV, one row per measurement.
* `render_markdown` -- pivot-style tables grouped by a row attribute.
* `plot_error_curves` -- log-log error vs resolution, one line per
  (probe, field, mode, region), for convergence sweeps.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Row", "write_rows_csv", "render_markdown", "render_pivot",
           "plot_error_curves"]


@dataclass
class Row:
    # case description
    dim: int
    kernel: str
    mode: str                 # standard | crk | renorm
    periodic: bool
    jitter: float
    N: int
    dx: float
    h: float
    h_over_dx: float
    target_neighbors: int
    # probe description
    probe: str                # interpolate | gradient | laplacian
    field: str
    field_degree: int         # -1 for smooth fields
    region: str               # interior | boundary
    # result
    l1: float
    l2: float
    linf: float
    # provenance (tagged by the driver; "" in unit-level constructions)
    suite: str = ""           # patch | resolve-open | resolve-periodic | smoothing

    def key(self, attrs: tuple[str, ...]) -> tuple:
        return tuple(getattr(self, a) for a in attrs)


def write_rows_csv(rows: Iterable[Row], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    names = [f.name for f in fields(Row)]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for r in rows:
            writer.writerow([
                ("%.10g" % v) if isinstance(v, float) else str(v)
                for v in asdict(r).values()
            ])


def render_markdown(rows: Iterable[Row], group: tuple[str, ...],
                    value: str = "linf",
                    order: tuple[str, ...] | None = None) -> str:
    """Markdown table: one row per `group` combination, columns per
    `order` combination (or a single `value` column)."""
    rows = list(rows)
    if not rows:
        return "*(no rows)*\n"
    by_key: dict[tuple, Row] = {}
    for r in rows:
        by_key[r.key(group)] = r
    keys = sorted(by_key)
    header = list(group) + ([value] if order is None else list(order))
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for k in keys:
        r = by_key[k]
        cells = [str(getattr(r, a)) for a in group]
        if order is None:
            v = getattr(r, value)
            cells.append("n/a" if not np.isfinite(v) else f"{v:.3e}")
        else:
            for a in order:
                v = getattr(r, a)
                cells.append("n/a" if not np.isfinite(v) else f"{v:.3e}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_pivot(rows: Iterable[Row], row_keys: tuple[str, ...],
                 col_key: str, value: str = "linf",
                 col_order: tuple[str, ...] | None = None) -> str:
    """Markdown pivot table: rows = `row_keys` combinations, one column per
    `col_key` value, cell = `value`."""
    rows = list(rows)
    if not rows:
        return "*(no rows)*\n"
    cols = list(col_order) if col_order is not None else sorted(
        {getattr(r, col_key) for r in rows}, key=str)
    by: dict[tuple, dict] = {}
    for r in rows:
        by.setdefault(tuple(getattr(r, k) for k in row_keys), {})[
            getattr(r, col_key)] = getattr(r, value)
    header = list(row_keys) + cols
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for key in sorted(by, key=str):
        cells = [str(k) for k in key]
        for c in cols:
            v = by[key].get(c)
            cells.append("n/a" if v is None or not np.isfinite(v)
                         else f"{v:.3e}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_error_curves(rows: Iterable[Row], path: str | Path,
                      xattr: str = "dx",
                      group_by: tuple[str, ...] = ("mode", "probe", "field"),
                      regions: tuple[str, ...] = ("interior", "boundary"),
                      title: str = "") -> Path:
    """Log-log error (linf) vs resolution for one region per subplot;
    one line per `group_by` combination. Saturated points (flat within
    the metrics' ratio) are still plotted -- the eye sees the floor."""
    rows = [r for r in rows if r.region in regions and np.isfinite(r.linf)
            and r.linf > 0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    # Only subplot regions that actually occur (e.g. periodic-domain
    # sweeps have no boundary region): an empty subplot leaves a blank
    # panel plus a "no artists with labels" legend warning.
    present = [rg for rg in regions if any(r.region == rg for r in rows)]
    if not present:
        return path

    fig, axes = plt.subplots(1, len(present),
                             figsize=(6 * len(present), 4.5), sharey=False)
    if len(present) == 1:
        axes = [axes]
    for ax, region in zip(axes, present):
        curves: dict[tuple, list[Row]] = {}
        for r in rows:
            if r.region != region:
                continue
            curves.setdefault(r.key(group_by), []).append(r)
        for k, rs in curves.items():
            rs = sorted(rs, key=lambda r: getattr(r, xattr))
            xs = np.array([getattr(r, xattr) for r in rs])
            ys = np.array([r.linf for r in rs])
            label = " / ".join(str(v) for v in k)
            ax.loglog(xs, ys, "o-", ms=3, lw=1, label=label)
        ax.set_xlabel(xattr)
        ax.set_ylabel(r"error $L_\infty$")
        ax.set_title(f"{region} region")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
