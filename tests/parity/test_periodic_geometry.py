"""Independent periodic geometry oracles, including FP32 cancellation targets."""

import pytest
import torch
import warp as wp

import warpSPHCore as sph
from warpSPHCore.math.wp_distance import moduloDistanceComponent
from warpSPHCore.type_config import scalar_t
from warpSPHCore.native.api import sphOperation as native

DTYPE = sph.get_torch_precision()


@wp.kernel
def distances(x: wp.array(dtype=scalar_t), y: wp.array(dtype=scalar_t),
              out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = moduloDistanceComponent(x[i], y[i], True, scalar_t(0), scalar_t(1))


def test_wrapping_preserves_small_separations_and_derivatives(device):
    wp.init()
    tiny = 2.**(-26 if DTYPE == torch.float32 else -55)
    x = torch.tensor([tiny, -tiny, .125 + 5*2.**-25, .75, 2.25, -3.75, .5, -.5],
                     dtype=DTYPE, device=device)
    y = torch.tensor([0, 0, .125, 0, 0, 0, 0, 0], dtype=DTYPE, device=device)
    out = torch.zeros_like(x)
    expected = torch.tensor([tiny, -tiny, 5*2.**-25, -.25, .25, .25, -.5, -.5],
                            dtype=DTYPE, device=device)
    wx = wp.from_torch(x, requires_grad=True)
    wy = wp.from_torch(y, requires_grad=True)
    wo = wp.from_torch(out, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(distances, len(x), inputs=[wx, wy, wo], device=device)
    wp.synchronize_device(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # Do not differentiate at the half-box discontinuity.
    seed = torch.ones_like(x); seed[-2:] = 0
    tape.backward(grads={wo: wp.from_torch(seed)})
    torch.testing.assert_close(wp.to_torch(wx.grad), seed, rtol=0, atol=0)
    torch.testing.assert_close(wp.to_torch(wy.grad), -seed, rtol=0, atol=0)


@pytest.mark.parametrize('origin', [0., .125, -.875, 2.125])
def test_periodic_pressure_gradient_matches_pair_oracle_and_balances(device, origin):
    # The separation is above Warp's gradient-normalization floor but below a
    # few half-box ulps. Legacy add/wrap/subtract changed it by 20 percent.
    delta = 5 * 2.**-25
    x = torch.tensor([[origin + delta], [origin]], device=device, dtype=DTYPE)
    h = torch.full((2,), .01, device=device, dtype=DTYPE)
    one = torch.ones_like(h)
    particles = sph.ParticleState(x, h, one, torch.zeros(2, device=device, dtype=torch.int32), one)
    domain = sph.DomainDescription(x.new_zeros(1), x.new_ones(1),
                                  torch.ones(1, device=device, dtype=torch.bool), 1)
    adjacency = sph.AdjacencyList(None, torch.tensor([1, 0], device=device, dtype=sph.get_torch_index_dtype()),
        torch.ones(2, device=device, dtype=torch.int32),
        torch.arange(2, device=device, dtype=sph.get_torch_offset_dtype()), 2, 2, numEdges=2)
    props = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
        gradientMode=sph.GradientScheme.Symmetric, supportMode=sph.SupportScheme.KernelMeanSymmetric,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    # Evaluate the exact stored input values in FP64, independently of Warp.
    dx = float(x[0, 0].double() - x[1, 0].double())
    hs = float(h[0]); q = abs(dx)/hs
    expected = x.new_tensor([[-30 * (1-q)**2 * dx/hs**3],
                             [30 * (1-q)**2 * dx/hs**3]])
    for implementation in (sph.warpOperation, native):
        actual = implementation(particles, props, domain, one, one, adjacency=adjacency)
        tol = 3e-6 if DTYPE == torch.float32 else 3e-12
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
        torch.testing.assert_close(actual.sum(0), x.new_zeros(1), rtol=0, atol=tol)


def test_half_box_predecessor_keeps_its_side(device):
    half = torch.tensor(.5, device=device, dtype=DTYPE)
    delta = torch.nextafter(half, torch.zeros_like(half))
    x = torch.stack((delta, torch.zeros_like(delta)))[:, None]
    h = torch.full((2,), .75, device=device, dtype=DTYPE)
    one = torch.ones_like(h)
    particles = sph.ParticleState(x, h, one, torch.zeros(2, device=device, dtype=torch.int32), one)
    domain = sph.DomainDescription(x.new_zeros(1), x.new_ones(1),
                                  torch.ones(1, device=device, dtype=torch.bool), 1)
    adjacency = sph.AdjacencyList(None, torch.tensor([1, 0], device=device, dtype=sph.get_torch_index_dtype()),
        torch.ones(2, device=device, dtype=torch.int32),
        torch.arange(2, device=device, dtype=sph.get_torch_offset_dtype()), 2, 2, numEdges=2)
    props = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
        gradientMode=sph.GradientScheme.Symmetric, supportMode=sph.SupportScheme.KernelMeanSymmetric,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    value = -30*(1-float(delta)/.75)**2*float(delta)/.75**3
    expected = x.new_tensor([[value], [-value]])
    for implementation in (sph.warpOperation, native):
        actual = implementation(particles, props, domain, one, one, adjacency=adjacency)
        tol = 3e-6 if DTYPE == torch.float32 else 3e-12
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
