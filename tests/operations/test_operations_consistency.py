import torch

from sphWarpCore.enumTypes import GradientScheme, WarpOperation

from conftest import interior_mask, linear_vector_field, mean_abs_error, op


def test_gradient_vector_trace_matches_divergence(particle_case):
    vec = linear_vector_field(particle_case, a=1.5, b=0.4, c=-0.7, d=2.2)

    grad = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=vec,
        reference_values=vec,
        gradient_mode=GradientScheme.Difference,
    )
    div = op(
        particle_case,
        WarpOperation.Divergence,
        query_values=vec,
        reference_values=vec,
        gradient_mode=GradientScheme.Difference,
    ).view(-1)

    trace = grad[:, 0, 0] + grad[:, 1, 1]
    mask = interior_mask(particle_case)

    mae = mean_abs_error(trace, div, mask)
    assert mae < 0.20


def test_interpolate_matrix_field_self_consistency(particle_case):
    x = particle_case["particles"].positions
    mat = torch.empty((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)
    mat[:, 0, 0] = x[:, 0] + 0.2 * x[:, 1]
    mat[:, 0, 1] = 0.4 * x[:, 0] - 0.3 * x[:, 1]
    mat[:, 1, 0] = -0.7 * x[:, 0] + 0.1 * x[:, 1]
    mat[:, 1, 1] = 1.1 * x[:, 0] + 0.6 * x[:, 1]

    interp = op(
        particle_case,
        WarpOperation.Interpolate,
        query_values=mat,
        reference_values=mat,
    )

    assert interp.shape == mat.shape
    assert torch.isfinite(interp).all()

    mask = interior_mask(particle_case)
    mae = mean_abs_error(interp, mat, mask)
    assert mae < 0.35
