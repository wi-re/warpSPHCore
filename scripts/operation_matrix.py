#!/usr/bin/env python3
"""Console diagnostic matrix for sphWarpCore operations.

Runs every operation (Density, Interpolate, Gradient, Divergence, Curl,
Laplacian) against every relevant scheme variant (GradientScheme /
LaplacianScheme), both neighbor-traversal modes (precomputed adjacency list
vs. grid dispatch via ``adjacency=None``), and all three correction paths
(none / CRK / renormalization) on a deterministic linear test field with a
known closed-form derivative. Every combination is executed in isolation so
one failing cell never aborts the run, and the result is a pass/fail/error
matrix printed to the console.

This is a manual diagnostic, not a pytest gate: some cells are expected to
fail today (curl, CPU renormalization) and that is the point -- the matrix
makes the current failure surface visible at a glance instead of hiding it
behind ``xfail`` markers scattered across the pytest suite.

Usage:
    python scripts/operation_matrix.py
    python scripts/operation_matrix.py --device cpu --threshold 0.3
    python scripts/operation_matrix.py --device both --verbose
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sphWarpCore import (
    OperationProperties,
    ParticleState,
    radiusSearchCompactHashMap,
    warpOperation,
)
from sphWarpCore.crk import computeCRKFactors
from sphWarpCore.enumTypes import (
    GradientScheme,
    KernelFunctions,
    LaplacianScheme,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from sphWarpCore.renorm import computeRenormalizationMatrices
from sphWarpCore.state import CRKState, RenormalizationState
from sphWarpCore.util import generateNeighborTestData

KERNEL = KernelFunctions.Wendland2
SUPPORT_MODE = SupportScheme.Gather
OP_MODE = OperationDirection.AllToAll

TRAVERSALS = ["adjacency", "grid"]
CORRECTIONS = ["base", "crk", "renorm"]
COLUMNS = [(t, c) for t in TRAVERSALS for c in CORRECTIONS]

GRADIENT_SCHEMES = [
    GradientScheme.Naive,
    GradientScheme.Symmetric,
    GradientScheme.Difference,
    GradientScheme.Summation,
]
LAPLACIAN_SCHEMES = [
    LaplacianScheme.Naive,
    LaplacianScheme.Brookshaw,
    LaplacianScheme.Dot,
    LaplacianScheme.Default,
]


# --------------------------------------------------------------------------
# Console rendering
# --------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


def magenta(text: str) -> str:
    return _c("35", text)


def dim(text: str) -> str:
    return _c("2", text)


# --------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------

@dataclass
class Cell:
    status: str  # "OK" | "HIGH" | "ERR" | "NAN" | "NA"
    mae: Optional[float]
    note: str


CELL_WIDTH = 10


def fmt_cell(cell: Optional[Cell]) -> str:
    if cell is None or cell.status == "NA":
        return dim(f"{'--':^{CELL_WIDTH}}")
    if cell.status == "ERR":
        return red(f"{'ERR':^{CELL_WIDTH}}")
    if cell.status == "NAN":
        return magenta(f"{'nonfinite':^{CELL_WIDTH}}")
    if cell.status == "HIGH":
        label = f"H {cell.mae:.3f}"
        return yellow(f"{label:^{CELL_WIDTH}}")
    label = f"ok {cell.mae:.3f}" if cell.mae is not None else "ok"
    return green(f"{label:^{CELL_WIDTH}}")


# --------------------------------------------------------------------------
# Deterministic scenario setup
# --------------------------------------------------------------------------

class Case:
    def __init__(self, device: torch.device, nx: int, target_neighbors: int, jitter: float = 0.0, seed: int = 0):
        dim = 2
        positions, supports, _, domain, dx = generateNeighborTestData(
            nx, target_neighbors, dim, True, device
        )
        if jitter > 0.0:
            # Matches the notebooks' make_jittered_particles: a perfect lattice
            # makes several corrections (CRK in particular) numerically close
            # to a no-op, since kernel sums are already exact on a regular
            # grid. Jittering is what actually exercises the correction path.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(positions.shape, generator=generator).to(device=device, dtype=positions.dtype)
            positions = (positions + jitter * dx * noise).contiguous()
        masses = torch.full((positions.shape[0],), dx ** dim, device=device, dtype=positions.dtype)
        kinds = torch.zeros(positions.shape[0], device=device, dtype=torch.int32)

        particles = ParticleState(
            positions=positions.contiguous(),
            supports=supports.contiguous(),
            masses=masses.contiguous(),
            densities=None,
            kinds=kinds,
        )

        adjacency = radiusSearchCompactHashMap(particles, domain, mode=SUPPORT_MODE)

        densities = warpOperation(
            particles,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Density,
                supportMode=SUPPORT_MODE,
                operationMode=OP_MODE,
            ),
            domain,
            adjacency=adjacency,
        )
        particles.densities = densities

        self.device = device
        self.dim = dim
        self.dx = dx
        self.domain = domain
        self.particles = particles
        self.adjacency = adjacency
        self.jitter = jitter

    def interior_mask(self, band_cells: float = 3.0) -> torch.Tensor:
        x = self.particles.positions
        band = band_cells * self.dx
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for axis in range(x.shape[1]):
            mask = mask & (x[:, axis] > self.domain.min[axis] + band)
            mask = mask & (x[:, axis] < self.domain.max[axis] - band)
        return mask


def linear_scalar_field(case: Case, ax=5.0, by=3.0, c=0.0):
    x = case.particles.positions
    return ax * x[:, 0] + by * x[:, 1] + c


def linear_vector_field(case: Case, a=2.0, b=-1.0, c=4.0, d=3.0):
    x = case.particles.positions
    vx = a * x[:, 0] + b * x[:, 1]
    vy = c * x[:, 0] + d * x[:, 1]
    return torch.stack((vx, vy), dim=1)


def linear_matrix_field(case: Case):
    x = case.particles.positions
    m = torch.empty((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)
    m[:, 0, 0] = 1.2 * x[:, 0]
    m[:, 0, 1] = -0.3 * x[:, 1]
    m[:, 1, 0] = 0.5 * x[:, 1]
    m[:, 1, 1] = -0.9 * x[:, 0]
    return m


def mean_abs_error(actual: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor) -> float:
    diff = torch.abs(actual - expected)
    if diff.ndim > 1:
        diff = torch.sum(diff, dim=tuple(range(1, diff.ndim))) / float(math.prod(diff.shape[1:]))
    return torch.mean(diff[mask]).item()


# --------------------------------------------------------------------------
# Correction-state construction (CRK / renormalization require an explicit
# adjacency list -- computeCRKFactors / computeRenormalizationMatrices both
# raise NotImplementedError for adjacency=None or a raw CompactHashMap. That
# is a real capability gap (see warpier_core.md), not a bug in this script:
# the factors are computed once via the adjacency list and then reused for
# both traversal columns of the operator call itself, since the correction
# terms are per-particle and independent of how neighbors were enumerated.
# --------------------------------------------------------------------------

@dataclass
class CRKBundle:
    state: CRKState
    volumes: torch.Tensor  # apparent area/volume -- the CRK correction is only
    # correct if this accompanies the A/B/gradA/gradB state as queryVolumes/
    # referenceVolumes (useVolume=True); see warp_gradient.ipynb's
    # apparent_area usage. Passing crkState without it (as this script did
    # originally) silently falls back to the uncorrected mj/rhoj volume and
    # under-corrects.


def build_crk_state(case: Case) -> CRKBundle:
    apparent_area, _, crk = computeCRKFactors(
        queryParticles=case.particles,
        domain=case.domain,
        kernel=KERNEL,
        operationMode=OP_MODE,
        adjacency=case.adjacency,
    )
    return CRKBundle(state=crk, volumes=apparent_area)


def build_renorm_state(case: Case):
    result = computeRenormalizationMatrices(
        queryParticles=case.particles,
        operationProperties=OperationProperties(
            kernel=KERNEL,
            operation=WarpOperation.Gradient,
            supportMode=SUPPORT_MODE,
            operationMode=OP_MODE,
            gradientMode=GradientScheme.Difference,
        ),
        domain=case.domain,
        adjacency=case.adjacency,
        returnEigVals=False,
    )
    if isinstance(result, RenormalizationState):
        return result
    if isinstance(result, tuple):
        return result[-1]
    raise AssertionError(f"Unexpected renormalization return type: {type(result)}")


# --------------------------------------------------------------------------
# Cell evaluation
# --------------------------------------------------------------------------

def run_op(
    case: Case,
    operation: WarpOperation,
    traversal: str,
    query_values=None,
    reference_values=None,
    gradient_mode=GradientScheme.Difference,
    laplacian_mode=LaplacianScheme.Default,
    crk_state=None,
    crk_volumes=None,
    renorm_state=None,
    divergenceDotMode=False,
):
    adjacency = case.adjacency if traversal == "adjacency" else None
    return warpOperation(
        case.particles,
        OperationProperties(
            kernel=KERNEL,
            operation=operation,
            supportMode=SUPPORT_MODE,
            operationMode=OP_MODE,
            gradientMode=gradient_mode,
            laplacianMode=laplacian_mode,
            divergenceDotMode=divergenceDotMode,
        ),
        case.domain,
        adjacency=adjacency,
        queryValues=query_values,
        referenceValues=reference_values,
        crkState=crk_state,
        queryVolumes=crk_volumes,
        referenceVolumes=crk_volumes,
        renormalizationState=renorm_state,
    )


def evaluate(
    case: Case,
    threshold: float,
    band_cells: float,
    traversal: str,
    correction: str,
    crk_state,
    renorm_state,
    **run_kwargs,
) -> Cell:
    kwargs = dict(run_kwargs)
    if correction == "crk":
        kwargs["crk_state"] = crk_state.state
        kwargs["crk_volumes"] = crk_state.volumes
    elif correction == "renorm":
        kwargs["renorm_state"] = renorm_state

    expected = kwargs.pop("expected", None)

    try:
        out = run_op(case, traversal=traversal, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a diagnostic sweep
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return Cell("ERR", None, msg[:160])

    if not torch.isfinite(out).all():
        return Cell("NAN", None, "output contains NaN/Inf")

    if expected is None:
        # No analytic reference (e.g. density): finiteness is the check.
        return Cell("OK", None, "finite")

    mask = case.interior_mask(band_cells)
    mae = mean_abs_error(out.view(out.shape[0], -1) if out.ndim > 1 else out.view(-1, 1),
                          expected.view(expected.shape[0], -1) if expected.ndim > 1 else expected.view(-1, 1),
                          mask)
    status = "OK" if mae < threshold else "HIGH"
    note = ""
    if status == "HIGH" and out.std().item() < 1e-8:
        # A near-constant/zero output on a linear (non-constant analytic)
        # field is not "high error", it is a sign that the kernel never
        # actually launched. Warp has been observed to do this silently for
        # every call after an earlier compile failure in the same module
        # within the same process (see the curl rows in this matrix and
        # warpier_core.md) -- the kernel-side exception is only raised once.
        note = "output is (near-)constant -- suspect a silent no-op after an earlier compile failure in this process, not a numerical error"
    return Cell(status, mae, note)


# --------------------------------------------------------------------------
# Matrix construction
# --------------------------------------------------------------------------

def build_rows(case: Case, threshold: float, band_cells: float):
    """Yields (row_label, applicable_corrections, cell_fn) tuples.

    cell_fn(traversal, correction, crk_state, renorm_state) -> Cell
    """
    rows = []

    # Density: no correction paths apply.
    def density_cell(traversal, correction, crk_state, renorm_state):
        if correction != "base":
            return Cell("NA", None, "density has no correction path")
        return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                         operation=WarpOperation.Density, expected=None)
    rows.append(("Density", density_cell))

    # Interpolate: base + CRK only (no renormalization support in the API).
    fields = {
        "scalar": linear_scalar_field(case),
        "vector": linear_vector_field(case),
        "matrix": linear_matrix_field(case),
    }
    for kind, field in fields.items():
        def interp_cell(traversal, correction, crk_state, renorm_state, field=field, kind=kind):
            if correction == "renorm":
                return Cell("NA", None, "interpolate has no renormalization path")
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Interpolate,
                             query_values=field, reference_values=field, expected=field)
        rows.append((f"Interpolate[{kind}]", interp_cell))

    # Gradient: scalar linear field, analytic gradient is constant.
    scalar_field = linear_scalar_field(case, ax=5.0, by=-2.0, c=0.7)
    grad_expected = torch.zeros((scalar_field.shape[0], case.dim), device=case.device, dtype=scalar_field.dtype)
    grad_expected[:, 0] = 5.0
    grad_expected[:, 1] = -2.0
    for scheme in GRADIENT_SCHEMES:
        def grad_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Gradient,
                             query_values=scalar_field, reference_values=scalar_field,
                             gradient_mode=scheme, expected=grad_expected)
        rows.append((f"Gradient[{scheme.name}]", grad_cell))

    # Divergence: vector field, analytic divergence is a constant scalar.
    # (dotMode does not affect rank-1 fields, see warpier_core.md.)
    vec_field = linear_vector_field(case, a=2.0, b=-1.0, c=4.0, d=3.0)
    div_expected = torch.full((vec_field.shape[0],), 2.0 + 3.0, device=case.device, dtype=vec_field.dtype)
    for scheme in GRADIENT_SCHEMES:
        def div_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Divergence,
                             query_values=vec_field, reference_values=vec_field,
                             gradient_mode=scheme, expected=div_expected, divergenceDotMode=True)
        rows.append((f"Divergence[{scheme.name}]", div_cell))

    # Divergence layout check: matrix field, verifies the fix for the div/dot
    # -mode bug documented in warpier_core.md. `OperationProperties.divergenceDotMode`
    # now exposes the flag that used to be hardcoded to False: with the field
    # laid out row-major as-is, `divergenceDotMode=True` should match the true
    # tensor divergence directly; with the field pre-transposed (the old
    # workaround from warp_divergence.ipynb), `divergenceDotMode=False`
    # (the old hardcoded default) should match it instead. Both are expected
    # to pass now -- this row exists to catch a regression in either
    # convention, not to demonstrate a bug.
    mat_field = linear_matrix_field(case)
    mat_div_expected = torch.zeros((mat_field.shape[0], 2), device=case.device, dtype=mat_field.dtype)
    mat_div_expected[:, 0] = 0.9  # d(1.2x)/dx + d(-0.3y)/dy
    mat_div_expected[:, 1] = 0.0  # d(0.5y)/dx + d(-0.9x)/dy
    # Use a tight threshold here regardless of the global one: the point of
    # this row is to distinguish "matches the analytic tensor divergence" from
    # "silently computes divergence(M^T) instead", and lattice discretization
    # error on a linear field is normally two-plus orders of magnitude below
    # the difference between the two conventions.
    mat_threshold = min(threshold, 0.05)
    mat_div_variants = (
        ("as-is/dotMode=True", mat_field, True),
        ("mT/dotMode=False", mat_field.mT.contiguous(), False),
    )
    for label, field, dot_mode in mat_div_variants:
        def mat_div_cell(traversal, correction, crk_state, renorm_state, field=field, dot_mode=dot_mode):
            if correction != "base":
                return Cell("NA", None, "layout check isolates the base path only")
            return evaluate(case, mat_threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Divergence,
                             query_values=field, reference_values=field,
                             gradient_mode=GradientScheme.Difference, expected=mat_div_expected,
                             divergenceDotMode=dot_mode)
        rows.append((f"Divergence-Matrix[{label}]", mat_div_cell))

    # Curl: vector field, analytic scalar curl is a constant.
    curl_expected = torch.full((vec_field.shape[0],), 4.0 - (-1.0), device=case.device, dtype=vec_field.dtype)
    for scheme in GRADIENT_SCHEMES:
        def curl_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Curl,
                             query_values=vec_field, reference_values=vec_field,
                             gradient_mode=scheme, expected=curl_expected)
        rows.append((f"Curl[{scheme.name}]", curl_cell))

    # Laplacian: scalar linear field, analytic laplacian is zero everywhere.
    lap_field = linear_scalar_field(case, ax=2.5, by=-3.2, c=1.0)
    lap_expected = torch.zeros(lap_field.shape[0], device=case.device, dtype=lap_field.dtype)
    for gscheme in GRADIENT_SCHEMES:
        for lscheme in LAPLACIAN_SCHEMES:
            def lap_cell(traversal, correction, crk_state, renorm_state, gscheme=gscheme, lscheme=lscheme):
                return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                                 operation=WarpOperation.Laplacian,
                                 query_values=lap_field, reference_values=lap_field,
                                 gradient_mode=gscheme, laplacian_mode=lscheme, expected=lap_expected)
            rows.append((f"Laplacian[{gscheme.name}/{lscheme.name}]", lap_cell))

    return rows


def run_matrix(device: torch.device, nx: int, target_neighbors: int, threshold: float, band_cells: float,
                jitter: float = 0.0, seed: int = 0):
    case = Case(device, nx, target_neighbors, jitter=jitter, seed=seed)

    try:
        crk_state = build_crk_state(case)
        crk_err = None
    except Exception as exc:  # noqa: BLE001
        crk_state = None
        crk_err = str(exc).strip().splitlines()[0][:160]

    try:
        renorm_state = build_renorm_state(case)
        renorm_err = None
    except Exception as exc:  # noqa: BLE001
        renorm_state = None
        renorm_err = str(exc).strip().splitlines()[0][:160]

    rows = build_rows(case, threshold, band_cells)

    results = {}
    for label, cell_fn in rows:
        results[label] = {}
        for traversal, correction in COLUMNS:
            if correction == "crk" and crk_state is None:
                results[label][(traversal, correction)] = Cell("ERR", None, f"CRK setup failed: {crk_err}")
                continue
            if correction == "renorm" and renorm_state is None:
                results[label][(traversal, correction)] = Cell("ERR", None, f"renorm setup failed: {renorm_err}")
                continue
            results[label][(traversal, correction)] = cell_fn(traversal, correction, crk_state, renorm_state)

    return case, results


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

GROUP_SEP = " | "


def _join_cells(cell_strs: list[str]) -> str:
    """Joins per-column cell strings, inserting GROUP_SEP between traversal groups
    (every len(CORRECTIONS) columns) so adjacency/grid blocks are visually distinct."""
    parts = []
    for i, s in enumerate(cell_strs):
        if i > 0 and i % len(CORRECTIONS) == 0:
            parts.append(GROUP_SEP)
        parts.append(s)
    return "".join(parts)


def print_matrix(device: torch.device, case: Case, results: dict, verbose: bool) -> dict:
    row_label_width = max(len(label) for label in results) + 2
    group_width = CELL_WIDTH * len(CORRECTIONS)
    group_header = " " * row_label_width + GROUP_SEP.join(f"{t:^{group_width}}" for t in TRAVERSALS)
    sub_header = " " * row_label_width + _join_cells([f"{c:^{CELL_WIDTH}}" for _, c in COLUMNS])
    rule = "-" * len(sub_header)

    print()
    title = f"=== Operation matrix -- device={device} nx={case.particles.positions.shape[0]} particles, dx={case.dx:.4f}"
    if case.jitter > 0.0:
        title += f", jitter={case.jitter:g}*dx"
    title += " ==="
    print(title)
    print(dim("Interior boundary band excluded from MAE to avoid the periodic-wrap"))
    print(dim("discontinuity of a linear field at the domain edge (see warpier_core.md)."))
    print(dim("Columns: traversal mode (adjacency-list vs. grid dispatch) / correction path (base, CRK, renormalization)."))
    print()
    print(group_header)
    print(sub_header)
    print(rule)

    flagged = []  # ERR / NAN / any HIGH with an explicit diagnostic note -- always worth surfacing
    plain_high = []  # HIGH with no special note beyond "MAE over threshold" -- only shown with --verbose
    counts = {"OK": 0, "HIGH": 0, "ERR": 0, "NAN": 0, "NA": 0}
    for label, cols in results.items():
        line = f"{label:<{row_label_width}}" + _join_cells([fmt_cell(cols[col]) for col in COLUMNS])
        for col in COLUMNS:
            cell = cols[col]
            counts[cell.status] += 1
            if cell.status in ("ERR", "NAN") or (cell.status == "HIGH" and cell.note):
                flagged.append((label, col, cell))
            elif cell.status == "HIGH":
                plain_high.append((label, col, cell))
        print(line)

    print(rule)
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Summary: {summary}")

    if flagged:
        print()
        print("Notes:")
        for label, col, cell in flagged:
            print(f"  {label} [{col[0]}/{col[1]}] {cell.status}: {cell.note}")

    if verbose and plain_high:
        print()
        print("High-error cells (MAE over threshold, no other diagnostic):")
        for label, col, cell in plain_high:
            print(f"  {label} [{col[0]}/{col[1]}] MAE={cell.mae:.4f}")

    return counts


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
    parser.add_argument("--nx", type=int, default=32, help="particles per axis (default: 32)")
    parser.add_argument("--target-neighbors", type=int, default=55)
    parser.add_argument("--threshold", type=float, default=0.4, help="MAE pass/fail threshold (default: 0.4)")
    parser.add_argument("--boundary-band", type=float, default=3.0, help="interior mask band in dx cells (default: 3.0)")
    parser.add_argument("--jitter", type=float, default=0.0,
                         help="position jitter scale as a fraction of dx, applied to the lattice before building "
                              "adjacency/CRK/renorm state (default: 0.0, a perfect lattice). A perfect lattice makes "
                              "CRK/renorm corrections numerically close to a no-op since kernel sums are already "
                              "exact on a regular grid -- try e.g. 0.15-0.3 to actually exercise them, matching the "
                              "notebooks' jittered examples.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --jitter (default: 0)")
    parser.add_argument("--verbose", action="store_true", help="also print notes for HIGH-error cells")
    parser.add_argument("--ci", action="store_true",
                         help="turn this diagnostic into a gate: exit with a non-zero status if any cell is "
                              "HIGH, ERR, or NAN (or if a device fatally fails to build), across all devices run.")
    args = parser.parse_args()

    wp.init()

    if args.device == "both":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
    elif args.device == "auto":
        devices = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available.", file=sys.stderr)
            sys.exit(1)
        devices = [torch.device(args.device)]

    had_failure = False
    for device in devices:
        try:
            case, results = run_matrix(device, args.nx, args.target_neighbors, args.threshold, args.boundary_band,
                                        jitter=args.jitter, seed=args.seed)
        except Exception:
            print(f"Fatal error building matrix for device={device}:", file=sys.stderr)
            traceback.print_exc()
            had_failure = True
            continue
        counts = print_matrix(device, case, results, args.verbose)
        if counts["HIGH"] or counts["ERR"] or counts["NAN"]:
            had_failure = True

    if args.ci and had_failure:
        print(file=sys.stderr)
        print("operation_matrix.py --ci: at least one cell was HIGH/ERR/NAN, or a device failed to build.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
