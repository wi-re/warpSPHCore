import pytest
import torch

from sphWarpCore.enumTypes import GradientScheme, WarpOperation

from conftest import (
    crk_state,
    interior_mask,
    linear_scalar_field,
    mean_abs_error,
    op,
    renorm_state,
)


@pytest.mark.parametrize("mode", ["base", "crk", "renorm"])
def test_linear_scalar_gradient_against_analytic_with_corrections(particle_case, kernel, mode):
    f = linear_scalar_field(particle_case, ax=5.0, by=-2.0, c=0.7)
    expected = torch.zeros((f.shape[0], 2), dtype=f.dtype, device=f.device)
    expected[:, 0] = 5.0
    expected[:, 1] = -2.0

    kwargs = {}
    if mode == "crk":
        kwargs["crk_state"] = crk_state(particle_case, kernel)
    elif mode == "renorm":
        kwargs["renorm_state"] = renorm_state(particle_case, kernel)

    grad = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        **kwargs,
    )

    mask = interior_mask(particle_case)
    mae = mean_abs_error(grad, expected, mask)

    if mode == "base":
        assert mae < 0.35
    else:
        # Corrected variants (CRK, renormalization) are expected to be at least as good as base.
        assert mae < 0.25


def test_crk_gradient_not_worse_than_baseline_on_linear_field(particle_case, kernel):
    f = linear_scalar_field(particle_case, ax=4.0, by=1.5, c=-0.2)
    expected = torch.zeros((f.shape[0], 2), dtype=f.dtype, device=f.device)
    expected[:, 0] = 4.0
    expected[:, 1] = 1.5
    mask = interior_mask(particle_case)

    base = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
    )
    corrected = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        crk_state=crk_state(particle_case, kernel),
    )

    base_err = mean_abs_error(base, expected, mask)
    crk_err = mean_abs_error(corrected, expected, mask)

    assert crk_err <= base_err + 0.05


def test_renorm_state_path_executes_and_returns_finite_output(particle_case, kernel):
    f = linear_scalar_field(particle_case, ax=3.0, by=2.0)
    grad = op(
        particle_case,
        WarpOperation.Gradient,
        query_values=f,
        reference_values=f,
        gradient_mode=GradientScheme.Difference,
        renorm_state=renorm_state(particle_case, kernel),
    )
    assert torch.isfinite(grad).all()
