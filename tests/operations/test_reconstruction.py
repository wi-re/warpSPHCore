"""Core MLS-TENO / MLS-WENO interface reconstruction
(`warpSPHCore.polyfit.Reconstructor`): smooth-region order, boundedness at a
discontinuity, central-stencil selection in smooth data, gradients through the
nonlinear weights, and the exact smoothness Gram matrix. Periodic 2-D jittered
lattice; the pure-torch reference is `higherOrderSPH/harness/reconstruct.py`.
"""

import math

import pytest
import torch

from warpSPHCore import (DomainDescription, OperationProperties, ParticleState,
                         radiusSearchCompactHashMap, warpOperation)
from warpSPHCore.enumTypes import (KernelFunctions, OperationDirection,
                                   SupportScheme, WarpOperation)
from warpSPHCore.polyfit import Reconstructor, monomialExponents, smoothnessGram

KERNEL = KernelFunctions.Wendland2


def _periodic_case(device, n, jitter=0.3, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    ax = torch.arange(n, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2) / n
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5) * jitter / n
    pts = pts % 1.0
    N = pts.shape[0]
    dx = 1.0 / n
    domain = DomainDescription(torch.zeros(2, dtype=dtype, device=device),
                               torch.ones(2, dtype=dtype, device=device),
                               torch.ones(2, dtype=torch.bool, device=device), 2)
    P = ParticleState(positions=pts.to(dtype).to(device),
                      supports=torch.full((N,), 3.0 * dx, dtype=dtype, device=device),
                      masses=torch.full((N,), dx * dx, dtype=dtype, device=device),
                      densities=None, kinds=torch.zeros(N, dtype=torch.int32, device=device))
    adj = radiusSearchCompactHashMap(P, domain, mode=SupportScheme.SuperSymmetric)
    P.densities = warpOperation(P, OperationProperties(
        kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll), domain, adjacency=adj)
    # first-neighbour pairs with minimum-image vectors
    x = P.positions
    d = x[None, :, :] - x[:, None, :]
    d = d - torch.round(d)
    r = torch.linalg.norm(d, dim=-1)
    i, j = torch.nonzero((r < 1.6 * dx) & (r > 0), as_tuple=True)
    return P, domain, i, j, d[i, j]


def _smooth(x):
    return torch.sin(2 * math.pi * x[:, 0]) * torch.cos(2 * math.pi * x[:, 1])


def _exact_at(P, i, j, d, rec):
    h = rec.interfaceSupports
    oi = d * (h[i] / (h[i] + h[j])).unsqueeze(-1)
    pt = P.positions[i] + oi
    return _smooth(pt)


def test_gram_properties():
    G = smoothnessGram(torch.tensor(monomialExponents(2, 3), dtype=torch.int32))
    assert G[0].abs().max() == 0                               # the constant has no variation
    assert torch.allclose(G, G.T)
    assert torch.linalg.eigvalsh(G).min() > -1e-12             # positive semi-definite


@pytest.mark.parametrize("make, floor", [
    (lambda P, dom: Reconstructor.teno(P, dom, KERNEL, "O4"), 3.2),
    (lambda P, dom: Reconstructor.weno(P, dom, KERNEL, degree=3), 2.6),
])
def test_smooth_region_order(device, make, floor):
    errs = []
    for n in (24, 48):
        P, dom, i, j, d = _periodic_case(device, n)
        rec = make(P, dom)
        fl, fr = rec.interfaceStates(_smooth(P.positions), i, j, d)
        ex = _exact_at(P, i, j, d, rec)
        errs.append(max((fl - ex).abs().max(), (fr - ex).abs().max()).item())
    assert math.log(errs[0] / errs[1]) / math.log(2) > floor


def test_teno_keeps_the_central_stencil_in_smooth_data(device):
    P, dom, i, j, d = _periodic_case(device, 32)
    rec = Reconstructor.teno(P, dom, KERNEL, "O4")
    rec.interfaceStates(_smooth(P.positions), i, j, d)
    assert rec.last_central.float().mean() > 0.97
    assert torch.allclose(rec.last_omega.sum(1), torch.ones_like(rec.last_omega[:, 0]), atol=1e-5)


@pytest.mark.parametrize("make", [
    lambda P, dom: Reconstructor.teno(P, dom, KERNEL, "O4"),
    lambda P, dom: Reconstructor.weno(P, dom, KERNEL, degree=2),
])
def test_step_is_not_overshot(device, make):
    P, dom, i, j, d = _periodic_case(device, 40)
    f = (torch.sin(2 * math.pi * P.positions[:, 0] + 0.3) > 0).to(P.positions.dtype)
    fl, fr = make(P, dom).interfaceStates(f, i, j, d)
    for s in (fl, fr):
        assert torch.isfinite(s).all()
        assert s.max() < 1 + 1e-3 and s.min() > -1e-3
    # and a jump is actually resolved: pairs straddling it keep (almost) the full jump
    straddle = f[i] != f[j]
    assert (fl - fr).abs()[straddle].mean() > 0.95


def test_gradient_flows_through_the_nonlinear_weights(device):
    P, dom, i, j, d = _periodic_case(device, 24)
    rec = Reconstructor.teno(P, dom, KERNEL, "O4")
    f = _smooth(P.positions).clone().requires_grad_(True)
    fl, fr = rec.interfaceStates(f, i, j, d)
    (fl.sum() + fr.sum()).backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().max() > 0
