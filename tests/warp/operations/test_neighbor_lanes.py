"""Multi-lane ("tiled") operator kernels (autograd/lanes.py) against the
thread-per-particle kernels they replace: same forward values and same
gradients, up to float summation order, on both traversal paths."""

import pytest
import torch

from warpSPHCore import setNeighborLanes
from warpSPHCore.enumTypes import GradientScheme, WarpOperation

from conftest import linear_scalar_field, linear_vector_field, op


@pytest.fixture
def cuda_case(particle_case):
    if particle_case["particles"].positions.device.type != "cuda":
        pytest.skip("tiled kernels are CUDA-only")
    yield particle_case
    setNeighborLanes(None)


def _both(fn):
    setNeighborLanes(1)
    ref = fn()
    setNeighborLanes(32)
    tiled = fn()
    setNeighborLanes(None)
    return ref, tiled


def _close(a, b, rtol=2e-5):
    scale = a.abs().max().clamp_min(1e-12)
    assert a.shape == b.shape
    assert float((a - b).abs().max() / scale) < rtol


CASES = {
    "interpolate_scalar": lambda c, f, v, t: op(c, WarpOperation.Interpolate, reference_values=f, traversal=t),
    "interpolate_vector": lambda c, f, v, t: op(c, WarpOperation.Interpolate, reference_values=v, traversal=t),
    "gradient": lambda c, f, v, t: op(c, WarpOperation.Gradient, query_values=f, reference_values=f,
                                      gradient_mode=GradientScheme.Difference, traversal=t),
    "divergence": lambda c, f, v, t: op(c, WarpOperation.Divergence, query_values=v, reference_values=v,
                                        gradient_mode=GradientScheme.Difference, traversal=t),
}


@pytest.mark.parametrize("traversal", ["adjacency", "grid"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiled_forward_matches(cuda_case, name, traversal):
    f = linear_scalar_field(cuda_case, ax=5.0, by=-2.0, c=0.7)
    v = linear_vector_field(cuda_case)
    ref, tiled = _both(lambda: CASES[name](cuda_case, f, v, traversal))
    _close(ref, tiled)


def test_tiled_covariance_matches(cuda_case):
    from warpSPHCore.renorm import computeRenormalizationMatrices
    p, d, a = cuda_case["particles"], cuda_case["domain"], cuda_case["adjacency"]
    from warpSPHCore import OperationProperties
    from warpSPHCore.enumTypes import KernelFunctions, SupportScheme
    props = OperationProperties(kernel=KernelFunctions.Wendland2, supportMode=SupportScheme.Gather)
    ref, tiled = _both(lambda: computeRenormalizationMatrices(p, props, d, adjacency=a))
    for x, y in zip(ref if isinstance(ref, tuple) else [ref], tiled if isinstance(tiled, tuple) else [tiled]):
        if hasattr(x, "renormalizationMatrices"):
            x, y = x.renormalizationMatrices, y.renormalizationMatrices
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            _close(x, y, rtol=1e-4)


@pytest.mark.parametrize("name", ["interpolate_scalar", "gradient", "divergence"])
def test_tiled_gradients_match(cuda_case, name):
    """Reverse mode through the tiled kernel (the tape replays it with its
    block_dim; the adjoint of the lane reduction broadcasts the cotangent)."""
    f0 = linear_scalar_field(cuda_case, ax=5.0, by=-2.0, c=0.7)
    v0 = linear_vector_field(cuda_case)

    def grads():
        f = f0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = CASES[name](cuda_case, f, v, "adjacency")
        w = torch.linspace(-1.0, 1.0, out.numel(), device=out.device, dtype=out.dtype).view_as(out)
        (out * w).sum().backward()
        g = f.grad if f.grad is not None else v.grad
        return g.detach().clone()

    ref, tiled = _both(grads)
    _close(ref, tiled)
