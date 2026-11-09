"""Backend-independent tensor periodic arithmetic and derivative contract."""
import pytest
import torch
from warpSPHCore.math.periodic import minimum_image_displacement, periodic_difference
from _support.periodic import cases, exact


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_pair_difference_matches_exact_rational_geometry(dtype, dim):
    x, y, length = cases(dtype, dim)
    flags = torch.tensor([True, False, True][:dim])
    actual = periodic_difference(x, y, flags, length)
    expected = exact(x, y, flags, length)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2*torch.finfo(dtype).eps)
    # Even a tolerance of epsilon is too weak for the ulp-sized local cases.
    assert torch.equal(actual[:4], (x-y)[:4])
    assert torch.equal(actual[:, ~flags], (x-y)[:, ~flags])
    assert torch.equal(actual[:2].sum(0), torch.zeros(dim, dtype=dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_displacement_wrap_keeps_local_bits_and_half_box_convention(dtype):
    eps = torch.finfo(dtype).eps
    delta = torch.tensor([-eps/4, eps/4, -.5, .5, -3.5, 3.5], dtype=dtype)
    out = minimum_image_displacement(delta, torch.tensor(True), torch.tensor(1., dtype=dtype))
    assert torch.equal(out[:2], delta[:2])
    assert torch.equal(out[2:], torch.full((4,), -.5, dtype=dtype))


def test_periodic_difference_derivatives_away_from_image_transitions():
    x = torch.tensor([[.25], [4.25]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[.26], [-.125]], dtype=torch.float64, requires_grad=True)
    flags, length = torch.tensor([True]), torch.tensor([1.], dtype=torch.float64)
    torch.autograd.gradcheck(lambda a,b: periodic_difference(a,b,flags,length), (x,y))
    _, tangent = torch.func.jvp(lambda a,b: periodic_difference(a,b,flags,length),
                               (x,y), (torch.ones_like(x),torch.zeros_like(y)))
    assert torch.equal(tangent, torch.ones_like(x))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_position_wrapping_keeps_in_domain_bits(dtype):
    from warpSPHCore.math.periodic import periodic_positions
    a = torch.tensor(.25, dtype=dtype)
    b = torch.nextafter(a, torch.full_like(a, float('inf')))
    x = torch.tensor([[a], [b], [2.25], [-2.25]], dtype=dtype)
    out = periodic_positions(x, torch.tensor([True]), x.new_tensor([-.5]), x.new_tensor([.5]))
    assert torch.equal(out[:2], x[:2])
    assert torch.equal(out[2:], x.new_tensor([[.25], [-.25]]))
