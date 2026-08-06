"""Grid-dispatch coverage (adjacency=None -> compact-hash-grid traversal).

Mirrors test_operations_core.py's base-path cases but forces the grid
traversal instead of a precomputed AdjacencyList. Per warpier_core.md, every
operator's unified backend now resolves adjacency=None into its own
compact-hash grid internally (via extractStateInfo) rather than dispatching
to a separate operations_grid backend -- this file only adds the missing
test cases, no product code changes are required to unlock it.
"""

import pytest
import torch

from sphWarpCore.enumTypes import GradientScheme, LaplacianScheme, WarpOperation

from conftest import (
    interior_mask,
    linear_scalar_field,
    linear_vector_field,
    matrix_field,
    mean_abs_error,
    op,
)


def test_density_is_positive_and_finite(particle_case):
    rho = op(particle_case, WarpOperation.Density, traversal="grid")
    assert torch.isfinite(rho).all()
    assert torch.all(rho > 0)


@pytest.mark.parametrize("field_kind", ["scalar", "vector", "matrix"])
def test_interpolate_preserves_shape_and_finiteness(particle_case, field_kind):
    if field_kind == "scalar":
        field = linear_scalar_field(particle_case)
    elif field_kind == "vector":
        field = linear_vector_field(particle_case)
    else:
        field = matrix_field(particle_case)

    out = op(
        particle_case,
        WarpOperation.Interpolate,
        query_values=field,
        reference_values=field,
        traversal="grid",
    )
    assert out.shape == field.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("field_kind", ["scalar", "vector", "matrix"])
def test_gradient_output_shape_by_field_rank(particle_case, field_kind):
    if field_kind == "scalar":
        field = linear_scalar_field(particle_case)
        expected_shape = (field.shape[0], particle_case["dim"])
    elif field_kind == "vector":
        field = linear_vector_field(particle_case)
        expected_shape = (field.shape[0], field.shape[1], particle_case["dim"])
    else:
        field = matrix_field(particle_case)
        expected_shape = (
            field.shape[0],
            field.shape[1],
            field.shape[2],
            particle_case["dim"],
        )

    out = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=field,
        reference_values=field,
        gradient_mode=GradientScheme.Difference,
        traversal="grid",
    )
    assert out.shape == expected_shape
    assert torch.isfinite(out).all()


def test_divergence_linear_vector_matches_trace_constant(particle_case):
    vec = linear_vector_field(particle_case, a=2.0, b=-1.0, c=4.0, d=3.0)
    expected = torch.full((vec.shape[0],), 2.0 + 3.0, device=vec.device, dtype=vec.dtype)

    div = op(
        particle_case,
        WarpOperation.Divergence,
        query_values=vec,
        reference_values=vec,
        gradient_mode=GradientScheme.Difference,
        traversal="grid",
    )

    div = div.view(-1)
    mask = interior_mask(particle_case)
    mae = mean_abs_error(div, expected, mask)
    assert mae < 0.40


def test_curl_linear_vector_matches_analytic_constant(particle_case):
    # For F=(a*x+b*y, c*x+d*y), scalar curl in 2D is dFy/dx - dFx/dy = c - b.
    vec = linear_vector_field(particle_case, a=2.0, b=-1.0, c=4.0, d=3.0)
    expected = torch.full((vec.shape[0],), 4.0 - (-1.0), device=vec.device, dtype=vec.dtype)

    curl = op(
        particle_case,
        WarpOperation.Curl,
        query_values=vec,
        reference_values=vec,
        gradient_mode=GradientScheme.Difference,
        traversal="grid",
    )

    curl = curl.view(-1)
    mask = interior_mask(particle_case)
    mae = mean_abs_error(curl, expected, mask)
    assert mae < 0.45


def test_laplacian_linear_scalar_is_near_zero(particle_case):
    f = linear_scalar_field(particle_case, ax=2.5, by=-3.2, c=1.0)
    lap = op(
        particle_case,
        WarpOperation.Laplacian,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        laplacian_mode=LaplacianScheme.Brookshaw,
        traversal="grid",
    ).view(-1)

    mask = interior_mask(particle_case)
    zero = torch.zeros_like(lap)
    mae = mean_abs_error(lap, zero, mask)
    assert mae < 0.40


def test_grid_and_adjacency_agree_on_gradient(particle_case):
    """Both traversal modes visit the same neighborhood, so results should match closely."""
    f = linear_scalar_field(particle_case, ax=5.0, by=-2.0, c=0.7)

    grad_adjacency = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        traversal="adjacency",
    )
    grad_grid = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        traversal="grid",
    )

    mask = interior_mask(particle_case)
    mae = mean_abs_error(grad_grid, grad_adjacency, mask)
    assert mae < 0.05
