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


# --- pair lists and bundles of fields ------------------------------------------

def test_pairs_match_a_brute_force_pair_list(device):
    from warpSPHCore.polyfit import interfacePairs
    P, dom, *_ = _periodic_case(device, 20)
    P.supports = P.supports * (1.0 + 0.2 * torch.rand_like(P.supports))     # unequal supports
    i, j, d = interfacePairs(P, dom)
    x = P.positions
    dd = x[None, :, :] - x[:, None, :]
    dd = dd - torch.round(dd)
    r = torch.linalg.norm(dd, dim=-1)
    thr = torch.maximum(P.supports[:, None], P.supports[None, :])
    bi, bj = torch.nonzero((r <= thr) & (torch.arange(len(x), device=device)[:, None]
                                         < torch.arange(len(x), device=device)[None, :]),
                           as_tuple=True)
    key = lambda a, b: set(zip(a.tolist(), b.tolist()))
    # the neighbour search and the brute force may differ only on pairs at the support edge
    sym = key(i, j) ^ key(bi, bj)
    assert all(abs(r[a, b] - thr[a, b]) < 1e-5 for a, b in sym)
    assert (i < j).all()
    assert torch.allclose(d, dd[i, j])
    fi, fj, fd = interfacePairs(P, dom, halfList=False)
    assert len(fi) == 2 * len(i) and (fi != fj).all()


def _bundle(x):
    base = _smooth(x)
    step = (torch.sin(2 * math.pi * x[:, 0] + 0.3) > 0).to(x.dtype)
    return torch.stack([base, step, base + 0.5 * step], dim=1)


@pytest.mark.parametrize("make", [
    lambda P, dom: Reconstructor.teno(P, dom, KERNEL, "O4"),
    lambda P, dom: Reconstructor.weno(P, dom, KERNEL, degree=2),
])
def test_field_bundle_equals_componentwise(device, make):
    P, dom, *_ = _periodic_case(device, 28)
    rec = make(P, dom)
    i, j, d = rec.pairs()
    F = _bundle(P.positions)
    fl, fr = rec.interfaceStates(F, i, j, d)
    assert fl.shape == (len(i), 3) and fr.shape == fl.shape
    for k in range(3):
        sl, sr = rec.interfaceStates(F[:, k].contiguous(), i, j, d)
        # float32 round-off in the coefficients is amplified by the nonlinear weights of
        # the discontinuous components for the few pairs near a selection threshold
        for a, b in ((fl[:, k], sl), (fr[:, k], sr)):
            err = (a - b).abs()
            assert err.max() < (1e-5 if k == 0 else 1e-2) and (err > 1e-4).float().mean() < 0.02
    # the components are weighted independently: the smooth one keeps its central stencil
    rec.interfaceStates(F, i, j, d)
    assert rec.last_omega.shape[-1] == 3
    if rec.kind == "teno":
        assert rec.last_central[:, 0].float().mean() > rec.last_central[:, 1].float().mean()


def test_polyfit_bundle_interface_states(device):
    from warpSPHCore.polyfit import PolyFit
    P, dom, *_ = _periodic_case(device, 24)
    fit = PolyFit.build(P, dom, KERNEL, 2)
    i, j, d = fit.pairs()
    F = _bundle(P.positions)
    fl, fr = fit.interfaceStates(F, i, j, d)
    for k in range(3):
        sl, sr = fit.interfaceStates(F[:, k].contiguous(), i, j, d)
        assert torch.allclose(fl[:, k], sl, atol=1e-5) and torch.allclose(fr[:, k], sr, atol=1e-5)


def test_pairs_feed_the_smooth_reconstruction(device):
    P, dom, *_ = _periodic_case(device, 32)
    rec = Reconstructor.teno(P, dom, KERNEL, "O4")
    i, j, d = rec.pairs()
    fl, fr = rec.interfaceStates(_smooth(P.positions), i, j, d)
    ex = _exact_at(P, i, j, d, rec)
    assert (fl - ex).abs().max() < 1e-3 and (fr - ex).abs().max() < 1e-3


def test_bundle_gradient_and_field_tangent(device):
    import torch.autograd.forward_ad as fwAD
    P, dom, *_ = _periodic_case(device, 24)
    rec = Reconstructor.teno(P, dom, KERNEL, "O4")
    i, j, d = rec.pairs()
    F = _bundle(P.positions)
    Fg = F.clone().requires_grad_(True)
    fl, fr = rec.interfaceStates(Fg, i, j, d)
    (fl.sum() + fr.sum()).backward()
    assert torch.isfinite(Fg.grad).all() and Fg.grad.abs().max() > 0
    # smooth data keep the central stencil (the plain linear fit), so the field tangent
    # of a smooth bundle is that fit applied to the tangent
    S = torch.stack([_smooth(P.positions), torch.cos(2 * math.pi * P.positions[:, 1])], dim=1)
    v = torch.stack([torch.cos(2 * math.pi * P.positions[:, 0]), _smooth(P.positions)], dim=1)
    with fwAD.dual_level():
        fl, fr = rec.interfaceStates(fwAD.make_dual(S, v), i, j, d)
        tl = fwAD.unpack_dual(fl).tangent
    assert rec.last_central.all()
    from warpSPHCore.polyfit import PolyFit
    h = rec.interfaceSupports
    oi, _ = PolyFit.interfaceOffsets(d, h[i], h[j])        # Eq. 25 uses the particle supports
    ref = rec.fits[0].evaluate(v, i, oi)
    assert torch.allclose(tl, ref, atol=2e-4)
