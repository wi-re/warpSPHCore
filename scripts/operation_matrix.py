#!/usr/bin/env python3
"""Console diagnostic matrix for warpSPHCore operations.

Runs every operation (Density, Interpolate, Gradient, Divergence, Curl,
Laplacian) against every relevant scheme variant (GradientScheme /
LaplacianScheme), both neighbor-traversal modes (precomputed adjacency list
vs. grid dispatch via ``adjacency=None``), and all three correction paths
(none / CRK / renormalization) on a deterministic linear test field with a
known closed-form derivative. Every combination is executed in isolation so
one failing cell never aborts the run, and the result is a pass/fail/error
matrix printed to the console.

This is a manual diagnostic, not a pytest gate: some cells are expected to
be N/A (e.g. LaplacianScheme.Dot on a scalar field -- see warpier_core.md)
and that is the point -- the matrix makes the current failure surface
visible at a glance instead of hiding it behind ``xfail`` markers scattered
across the pytest suite.

Precision and dimension are both configurable (``--precision``, ``--dim``),
covering the two axes that have each hidden a real bug so far (a raw-float
type-promotion bug only breaks compilation under float64; the Dot-scheme
out-of-bounds read only triggers when dim>1) -- see warpier_core.md's
Gradcheck Script Rollout Plan, Stage 6. float16 is deliberately not offered:
half precision is numerically nonsensical for this kind of SPH kernel sum,
a separate problem from what this smoke test checks.

Usage:
    python scripts/operation_matrix.py
    python scripts/operation_matrix.py --device cpu --threshold 0.3
    python scripts/operation_matrix.py --device both --verbose
    python scripts/operation_matrix.py --precision float64 --dim 3
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

# --------------------------------------------------------------------------
# warpSPHCore is imported lazily, inside _configure() (called from main()
# after --precision is parsed), NOT at module load time. Warp bakes scalar_t
# into every @wp.kernel/@wp.func the first time any warpSPHCore submodule is
# imported in this process (src/warpSPHCore/type_config.py reads
# warpSPHCore_PRECISION once, at that import); it cannot be changed
# afterwards. All the warpSPHCore-derived names below are populated by
# _configure() and referenced as globals by the functions/classes further
# down -- that's safe in Python because free-variable lookup for a global
# happens when the function is *called*, not when it's *defined*, and
# nothing here is called before main() runs _configure() first.
# --------------------------------------------------------------------------
OperationProperties = None
ParticleState = None
DomainDescription = None
radiusSearchCompactHashMap = None
warpOperation = None
computeCRKFactors = None
GradientScheme = None
KernelFunctions = None
LaplacianScheme = None
OperationDirection = None
SupportScheme = None
WarpOperation = None
computeRenormalizationMatrices = None
CRKState = None
RenormalizationState = None
generateNeighborTestData = None
n_h_to_nH = None

KERNEL = None
SUPPORT_MODE = None
OP_MODE = None
GRADIENT_SCHEMES = None
LAPLACIAN_SCHEMES = None

TRAVERSALS = ["adjacency", "grid"]
CORRECTIONS = ["base", "crk", "renorm"]
COLUMNS = [(t, c) for t in TRAVERSALS for c in CORRECTIONS]

PRECISION_TO_TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}


def _configure(precision: str) -> None:
    """Sets warpSPHCore_PRECISION and imports every warpSPHCore symbol this
    script needs. Must be called exactly once, before any Case/build_rows/
    evaluate/etc. call, and before any other warpSPHCore import in this
    process."""
    global OperationProperties, ParticleState, DomainDescription, radiusSearchCompactHashMap, warpOperation
    global computeCRKFactors
    global GradientScheme, KernelFunctions, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
    global computeRenormalizationMatrices, CRKState, RenormalizationState, generateNeighborTestData, n_h_to_nH
    global KERNEL, SUPPORT_MODE, OP_MODE, GRADIENT_SCHEMES, LAPLACIAN_SCHEMES

    os.environ["warpSPHCore_PRECISION"] = precision

    import warpSPHCore
    import warpSPHCore.crk
    import warpSPHCore.enumTypes
    import warpSPHCore.dataTypes
    import warpSPHCore.renorm
    import warpSPHCore.util

    OperationProperties = warpSPHCore.OperationProperties
    ParticleState = warpSPHCore.ParticleState
    DomainDescription = warpSPHCore.dataTypes.DomainDescription
    radiusSearchCompactHashMap = warpSPHCore.radiusSearchCompactHashMap
    warpOperation = warpSPHCore.warpOperation
    computeCRKFactors = warpSPHCore.crk.computeCRKFactors
    GradientScheme = warpSPHCore.enumTypes.GradientScheme
    KernelFunctions = warpSPHCore.enumTypes.KernelFunctions
    LaplacianScheme = warpSPHCore.enumTypes.LaplacianScheme
    OperationDirection = warpSPHCore.enumTypes.OperationDirection
    SupportScheme = warpSPHCore.enumTypes.SupportScheme
    WarpOperation = warpSPHCore.enumTypes.WarpOperation
    computeRenormalizationMatrices = warpSPHCore.renorm.computeRenormalizationMatrices
    CRKState = warpSPHCore.dataTypes.CRKState
    RenormalizationState = warpSPHCore.dataTypes.RenormalizationState
    generateNeighborTestData = warpSPHCore.util.generateNeighborTestData
    n_h_to_nH = warpSPHCore.n_h_to_nH

    KERNEL = KernelFunctions.Wendland2
    SUPPORT_MODE = SupportScheme.Gather
    OP_MODE = OperationDirection.AllToAll
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
    def __init__(self, device: torch.device, nx: int, n_h: float, dim: int = 2,
                 dtype: torch.dtype = torch.float32, jitter: float = 0.0, seed: int = 0):
        # n_h (particles per smoothing length, per axis) rather than a flat
        # target-neighbor-count -- a fixed neighbor count is not comparable
        # across dimensions (the same count implies a wildly different
        # support radius per dim), so it silently produced an oversized
        # support -- and badly wrong operator output -- for dim=1 before
        # n_h_to_nH existed. See n_h_to_nH's docstring (mathutil/wp_math.py).
        target_neighbors = int(round(n_h_to_nH(n_h, dim)))
        positions, supports, _, domain, dx = generateNeighborTestData(
            nx, target_neighbors, dim, True, device
        )
        # generateNeighborTestData hardcodes float32 internally (see
        # src/warpSPHCore/utils/wp_util.py) regardless of warpSPHCore_PRECISION
        # -- cast everything to the requested precision here rather than
        # touching that shared utility, which pytest fixtures and demo_util.py
        # also depend on at float32.
        positions = positions.to(dtype=dtype)
        supports = supports.to(dtype=dtype)
        domain = DomainDescription(
            min=domain.min.to(dtype=dtype),
            max=domain.max.to(dtype=dtype),
            periodic=domain.periodic,
            dim=domain.dim,
        )
        if jitter > 0.0:
            # Matches the notebooks' make_jittered_particles: a perfect lattice
            # makes several corrections (CRK in particular) numerically close
            # to a no-op, since kernel sums are already exact on a regular
            # grid. Jittering is what actually exercises the correction path.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(positions.shape, generator=generator).to(device=device, dtype=positions.dtype)
            positions = (positions + jitter * dx * noise).contiguous()
        masses = torch.full((positions.shape[0],), dx ** dim, device=device, dtype=dtype)
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
        self.dtype = dtype
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


# --------------------------------------------------------------------------
# Deterministic, dimension-generic linear test fields.
#
# Every field below is linear in the particle position x, so every operator's
# analytic derivative is a closed-form constant regardless of dim (1, 2, or
# 3) -- that's what lets the same test scaffolding run at any dimension
# without per-dim-hardcoded reference numbers. _coeff_vector/_coeff_matrix
# are deterministic (no RNG) and intentionally asymmetric (nonzero trace AND
# nonzero antisymmetric part) so both divergence and curl are meaningfully
# exercised, not accidentally zero.
# --------------------------------------------------------------------------

def _coeff_vector(dim_: int, device, dtype) -> torch.Tensor:
    """dim-length, non-degenerate (no equal/zero components) coefficient
    vector for a linear scalar field: f(x) = coeffs . x + const."""
    return torch.arange(1, dim_ + 1, device=device, dtype=dtype) * 1.3 - 0.4


def _coeff_matrix(dim_: int, device, dtype) -> torch.Tensor:
    """dim x dim, asymmetric coefficient matrix for a linear vector field:
    v(x) = A @ x. Nonzero trace (exercises divergence) and nonzero
    antisymmetric part (exercises curl)."""
    idx = torch.arange(dim_, device=device, dtype=dtype)
    return 0.6 * idx.unsqueeze(0) - 0.9 * idx.unsqueeze(1) + 2.0 * torch.eye(dim_, device=device, dtype=dtype)


def linear_scalar_field(case: Case) -> torch.Tensor:
    coeffs = _coeff_vector(case.dim, case.device, case.particles.positions.dtype)
    return case.particles.positions @ coeffs + 0.7


def linear_vector_field(case: Case) -> torch.Tensor:
    """v(x) = A @ x, per particle. divergence(v) = trace(A); curl(v) comes
    from A's antisymmetric part (see build_rows' Curl section)."""
    A = _coeff_matrix(case.dim, case.device, case.particles.positions.dtype)
    return case.particles.positions @ A.mT


def linear_matrix_field(case: Case) -> torch.Tensor:
    """M(x)_pq = A[p,q] * (coeffs . x), per particle -- a matrix field whose
    every entry is linear in x, sharing A/coeffs with the vector/scalar
    fields above. dM[p,q]/dx_d = A[p,q]*coeffs[d], so:
      * divergenceDotMode=True  (contracts M's 2nd index): (div M)_i = (A @ coeffs)[i]
      * divergenceDotMode=False (contracts M's 1st index): (div M)_i = (A.mT @ coeffs)[i]
    """
    A = _coeff_matrix(case.dim, case.device, case.particles.positions.dtype)
    coeffs = _coeff_vector(case.dim, case.device, case.particles.positions.dtype)
    scalar = case.particles.positions @ coeffs
    return A.unsqueeze(0) * scalar.view(-1, 1, 1)


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
    state: "CRKState"
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
    operation,
    traversal: str,
    query_values=None,
    reference_values=None,
    gradient_mode=None,
    laplacian_mode=None,
    crk_state=None,
    crk_volumes=None,
    renorm_state=None,
    divergenceDotMode=False,
):
    if gradient_mode is None:
        gradient_mode = GradientScheme.Difference
    if laplacian_mode is None:
        laplacian_mode = LaplacianScheme.Default
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
        return Cell("ERR", None, msg[:1024])

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
    dim_ = case.dim
    device = case.device
    dtype = case.particles.positions.dtype
    A = _coeff_matrix(dim_, device, dtype)
    coeffs = _coeff_vector(dim_, device, dtype)

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

    # Gradient: scalar linear field, analytic gradient is the constant coeffs vector.
    scalar_field = linear_scalar_field(case)
    grad_expected = coeffs.unsqueeze(0).expand(scalar_field.shape[0], dim_).contiguous()
    for scheme in GRADIENT_SCHEMES:
        def grad_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Gradient,
                             query_values=scalar_field, reference_values=scalar_field,
                             gradient_mode=scheme, expected=grad_expected)
        rows.append((f"Gradient[{scheme.name}]", grad_cell))

    # Divergence: vector field v(x) = A @ x, analytic divergence = trace(A), a
    # constant scalar for any dim. (dotMode does not affect rank-1 fields --
    # see warpier_core.md.)
    vec_field = linear_vector_field(case)
    div_expected = torch.full((vec_field.shape[0],), A.trace().item(), device=device, dtype=dtype)
    for scheme in GRADIENT_SCHEMES:
        def div_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Divergence,
                             query_values=vec_field, reference_values=vec_field,
                             gradient_mode=scheme, expected=div_expected, divergenceDotMode=True)
        rows.append((f"Divergence[{scheme.name}]", div_cell))

    # Divergence layout check: matrix field, verifies the fix for the div/dot
    # -mode bug documented in warpier_core.md. `OperationProperties.divergenceDotMode`
    # exposes the flag that used to be hardcoded to False: with the field
    # laid out as-is, `divergenceDotMode=True` should match the true tensor
    # divergence directly; with the field pre-transposed (the old workaround
    # from warp_divergence.ipynb), `divergenceDotMode=False` (the old
    # hardcoded default) should match it instead -- both reduce to the same
    # (A @ coeffs) expected value for linear_matrix_field's construction (see
    # its docstring). Both are expected to pass now -- this row exists to
    # catch a regression in either convention, not to demonstrate a bug.
    mat_field = linear_matrix_field(case)
    mat_div_expected = (A @ coeffs).unsqueeze(0).expand(mat_field.shape[0], dim_).contiguous()
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

    # Curl: same vector field v(x) = A @ x. Output shape/formula genuinely
    # depends on dim (matches computeSPHCurl_warpBackend's own branching in
    # wp_curl.py), so the expected value is built per-dim here:
    #   1D: curl is defined to be identically zero.
    #   2D: scalar curl = A[1,0] - A[0,1], output shape (n,1).
    #   3D: vector curl = (A[2,1]-A[1,2], A[0,2]-A[2,0], A[1,0]-A[0,1]).
    if dim_ == 1:
        curl_expected = torch.zeros((vec_field.shape[0],), device=device, dtype=dtype)
    elif dim_ == 2:
        curl_val = (A[1, 0] - A[0, 1]).item()
        curl_expected = torch.full((vec_field.shape[0], 1), curl_val, device=device, dtype=dtype)
    else:
        curl_vec = torch.stack([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])
        curl_expected = curl_vec.unsqueeze(0).expand(vec_field.shape[0], 3).contiguous()
    for scheme in GRADIENT_SCHEMES:
        def curl_cell(traversal, correction, crk_state, renorm_state, scheme=scheme):
            return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                             operation=WarpOperation.Curl,
                             query_values=vec_field, reference_values=vec_field,
                             gradient_mode=scheme, expected=curl_expected)
        rows.append((f"Curl[{scheme.name}]", curl_cell))

    # Laplacian: scalar linear field, analytic laplacian is zero everywhere.
    lap_field = linear_scalar_field(case)
    lap_expected = torch.zeros(lap_field.shape[0], device=device, dtype=dtype)
    for gscheme in GRADIENT_SCHEMES:
        for lscheme in LAPLACIAN_SCHEMES:
            # LaplacianScheme.Dot's computeLaplacianDot2 assumes the field's
            # flattened size is a multiple of the spatial dimension -- true
            # for vector fields (see the Laplacian-Vector rows below), never
            # true for this row's scalar field (flattened size 1) once
            # dim>1. wp_laplacian.py now raises for exactly this combination
            # (a real out-of-bounds-read bug found via gradcheck, see
            # warpier_core.md) rather than silently computing garbage, so
            # this is a structural NA here, not something to route through
            # evaluate() and flag ERR. For dim==1 the guard never fires
            # (flatInputShape==1==dim), so this still exercises Dot for real.
            if lscheme == LaplacianScheme.Dot and dim_ > 1:
                def lap_cell(traversal, correction, crk_state, renorm_state):
                    return Cell("NA", None, "Dot scheme doesn't support scalar fields in >1D -- see Laplacian-Vector rows")
                rows.append((f"Laplacian[{gscheme.name}/{lscheme.name}]", lap_cell))
                continue

            def lap_cell(traversal, correction, crk_state, renorm_state, gscheme=gscheme, lscheme=lscheme):
                return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                                 operation=WarpOperation.Laplacian,
                                 query_values=lap_field, reference_values=lap_field,
                                 gradient_mode=gscheme, laplacian_mode=lscheme, expected=lap_expected)
            rows.append((f"Laplacian[{gscheme.name}/{lscheme.name}]", lap_cell))

    # Laplacian-Vector: same vector field v(x) = A @ x as Divergence/Curl.
    # Its flattened size is exactly dim, so LaplacianScheme.Dot's
    # block/k-indexing (see computeLaplacianDot2) stays in-bounds for every
    # dim -- this is what actually exercises the Dot scheme's real math path,
    # since the scalar-field Laplacian rows above now correctly refuse it.
    # Laplacian of a linear field is zero component-wise regardless of dim.
    lap_vec_expected = torch.zeros((vec_field.shape[0], dim_), device=device, dtype=dtype)
    for gscheme in GRADIENT_SCHEMES:
        for lscheme in LAPLACIAN_SCHEMES:
            def lap_vec_cell(traversal, correction, crk_state, renorm_state, gscheme=gscheme, lscheme=lscheme):
                return evaluate(case, threshold, band_cells, traversal, correction, crk_state, renorm_state,
                                 operation=WarpOperation.Laplacian,
                                 query_values=vec_field, reference_values=vec_field,
                                 gradient_mode=gscheme, laplacian_mode=lscheme, expected=lap_vec_expected)
            rows.append((f"Laplacian-Vector[{gscheme.name}/{lscheme.name}]", lap_vec_cell))

    return rows


def run_matrix(device: torch.device, nx: int, n_h: float, threshold: float, band_cells: float,
                dim: int = 2, dtype: torch.dtype = torch.float32, jitter: float = 0.0, seed: int = 0):
    case = Case(device, nx, n_h, dim=dim, dtype=dtype, jitter=jitter, seed=seed)

    try:
        crk_state = build_crk_state(case)
        crk_err = None
    except Exception as exc:  # noqa: BLE001
        crk_state = None
        crk_err = str(exc).strip().splitlines()[0][:1024]

    try:
        renorm_state = build_renorm_state(case)
        renorm_err = None
    except Exception as exc:  # noqa: BLE001
        renorm_state = None
        renorm_err = str(exc).strip().splitlines()[0][:1024]

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
    title = (f"=== Operation matrix -- device={device} dim={case.dim} dtype={case.dtype} "
              f"nx={case.particles.positions.shape[0]} particles, dx={case.dx:.4f}")
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
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32",
                         help="scalar precision baked into every compiled kernel for this run (default: float32). "
                              "Warp bakes this in at first warpSPHCore import and cannot change it afterwards, so "
                              "this script sets warpSPHCore_PRECISION and imports warpSPHCore only after parsing "
                              "this flag -- one process/invocation tests exactly one precision. float16 is "
                              "deliberately not offered: half precision is numerically nonsensical for this kind "
                              "of SPH kernel sum, a separate problem from what this smoke test checks.")
    parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=2,
                         help="spatial dimension of the test domain (default: 2). Every analytic reference in this "
                              "script is dimension-generic (see linear_scalar_field/linear_vector_field/"
                              "linear_matrix_field), so 1D/3D exercise real, distinct code paths (e.g. Curl's "
                              "output shape and LaplacianScheme.Dot's indexing both depend on dim directly) rather "
                              "than being degenerate slices of the 2D case.")
    parser.add_argument("--nx", type=int, default=32, help="particles per axis (default: 32). Total particle count "
                         "is nx**dim, so consider a smaller value for --dim 3.")
    parser.add_argument("--n-h", type=float, default=4.0,
                         help="particles per smoothing length, per axis (default: 4.0). Converted to the actual "
                              "target neighbor count via n_h_to_nH (see mathutil/wp_math.py), which accounts for "
                              "--dim -- a flat, dimension-agnostic neighbor count is not comparable across "
                              "dimensions (the same count implies a wildly different support radius per dim), so "
                              "n_h is what actually stays meaningful when sweeping --dim.")
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

    _configure(args.precision)
    dtype = PRECISION_TO_TORCH_DTYPE[args.precision]

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
            case, results = run_matrix(device, args.nx, args.n_h, args.threshold, args.boundary_band,
                                        dim=args.dim, dtype=dtype, jitter=args.jitter, seed=args.seed)
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
