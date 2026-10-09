"""Core order-p polynomial fit (MLS/RKPM, LABFM): `warpSPHCore.polyfit.PolyFit`.

Checks the warp moment / right-hand-side kernels end to end: exact
reproduction of polynomials through degree p (value, gradient, Hessian,
Laplacian) on an open jittered lattice including the wall band, both
traversal modes (compact-hash adjacency and adjacency=None), the LABFM form
(`constant=False`), rank-deficiency flagging, 1-D / 3-D, and agreement with
the pure-torch reference `higherOrderSPH/harness/rkpm.py` (skipped when the
harness is not importable).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from warpSPHCore import (DomainDescription, OperationProperties, ParticleState,
                         radiusSearchCompactHashMap, warpOperation)
from warpSPHCore.enumTypes import (KernelFunctions, OperationDirection,
                                   SupportScheme, WarpOperation)
from warpSPHCore.polyfit import PolyFit, monomialExponents
from warpSPHCore.util import volumeToSupport

KERNEL = KernelFunctions.Wendland2


def _tol(dtype):
    return 2e-3 if dtype == torch.float32 else 1e-8


def _case(device, dim=2, n=22, nbrs=40, jitter=0.3, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    axes = [torch.arange(n, dtype=torch.float64)] * dim
    pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    dx = 1.0 / n
    pts = (pts + 0.5) * dx
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5) * jitter * dx
    h = float(volumeToSupport(dx ** dim, nbrs, dim))
    N = pts.shape[0]
    positions = pts.to(dtype).to(device)
    domain = DomainDescription(
        torch.full((dim,), -0.5, dtype=dtype, device=device),
        torch.full((dim,), 1.5, dtype=dtype, device=device),
        torch.zeros(dim, dtype=torch.bool, device=device), dim)
    particles = ParticleState(
        positions=positions,
        supports=torch.full((N,), h, dtype=dtype, device=device),
        masses=torch.full((N,), dx ** dim, dtype=dtype, device=device),
        densities=None, kinds=torch.zeros(N, dtype=torch.int32, device=device))
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.SuperSymmetric)
    particles.densities = warpOperation(
        particles,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather,
                            operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency)
    return dict(particles=particles, domain=domain, adjacency=adjacency, dx=dx, h=h,
                dim=dim, positions=positions)


@pytest.fixture(scope="module")
def case2d(device):
    return _case(device)


def _poly2(x, deg):
    X, Y = x[:, 0], x[:, 1]
    f = X ** deg + 0.7 * X * Y ** max(deg - 1, 0) + 0.3 * Y ** deg
    gx = deg * X ** (deg - 1) + 0.7 * Y ** max(deg - 1, 0)
    gy = (0.7 * X * (deg - 1) * Y ** max(deg - 2, 0) if deg > 1 else 0 * X) \
        + 0.3 * deg * Y ** (deg - 1)
    out = dict(f=f, g=torch.stack([gx, gy], -1))
    if deg >= 2:
        fxx = deg * (deg - 1) * X ** (deg - 2)
        fxy = 0.7 * (deg - 1) * Y ** max(deg - 2, 0)
        fyy = (0.7 * X * (deg - 1) * (deg - 2) * Y ** max(deg - 3, 0)
               if deg > 2 else 0 * X) + 0.3 * deg * (deg - 1) * Y ** (deg - 2)
        out.update(fxx=fxx, fxy=fxy, fyy=fyy)
    return out


def test_monomial_exponents():
    for dim, order, count in [(1, 3, 4), (2, 2, 6), (2, 3, 10), (3, 2, 10)]:
        e = monomialExponents(dim, order)
        assert len(e) == count == math.comb(dim + order, order)
        assert e[0] == (0,) * dim and len(set(e)) == len(e)


@pytest.mark.parametrize("traversal", ["adjacency", "grid"])
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_polynomial_reproduction(case2d, p, constant, traversal):
    c = case2d
    adjacency = c["adjacency"] if traversal == "adjacency" else None
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, p, constant=constant,
                       adjacency=adjacency)
    assert not pf.deficient.any(), "40-neighbour stencils must be full rank through p = 3"
    tol = _tol(c["positions"].dtype)
    pos = c["positions"]
    # float32 at p = 3: wall-band rows reach cond ~1e3-1e4 (the Hessian of a
    # cubic carries 1/h^2 on top), beyond what single precision resolves.
    lowp = c["positions"].dtype == torch.float32
    rows = (pf.cond < 100) if (lowp and p >= 3) else torch.ones_like(pf.cond, dtype=torch.bool)
    for deg in range(p + 1):
        d = _poly2(pos, deg) if deg > 0 else dict(f=torch.ones_like(pos[:, 0]),
                                                  g=torch.zeros_like(pos))
        scale = max(1.0, float(d["f"].abs().max()))
        assert (pf.values(d["f"]) - d["f"]).abs()[rows].max() < tol * scale
        assert (pf.gradient(d["f"]) - d["g"]).abs()[rows].max() < tol * scale * 10
        if p >= 2 and deg >= 2:
            # float32: second derivatives carry 1/h^2 ~ 60x amplification of the
            # round-off, which on the ill-conditioned wall-band rows (cond up to
            # ~1e3 at p = 3) reaches O(0.1); check those rows in float64 only.
            m = (pf.cond < 100) if lowp else torch.ones_like(pf.cond, dtype=torch.bool)
            H = pf.hessian(d["f"])
            assert (H[:, 0, 0] - d["fxx"]).abs()[m].max() < tol * scale * 100
            assert (H[:, 0, 1] - d["fxy"]).abs()[m].max() < tol * scale * 100
            assert (H[:, 1, 1] - d["fyy"]).abs()[m].max() < tol * scale * 100
            assert (pf.laplacian(d["f"]) - d["fxx"] - d["fyy"]).abs()[m].max() < tol * scale * 200


def test_not_exact_above_order(case2d):
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, adjacency=c["adjacency"])
    d = _poly2(c["positions"], 3)
    assert (pf.gradient(d["f"]) - d["g"]).abs().max() > 1e-4


def test_hessian_laplacian_none_for_order_one(case2d):
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 1, adjacency=c["adjacency"])
    assert pf.hessian(c["positions"][:, 0]) is None and pf.laplacian(c["positions"][:, 0]) is None


def test_labfm_values_are_exact(case2d):
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, constant=False,
                       adjacency=c["adjacency"])
    f = torch.sin(3 * c["positions"][:, 0])
    assert torch.equal(pf.values(f), f) and pf.n_basis == 5


def test_traversals_agree(case2d):
    c = case2d
    a = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, adjacency=c["adjacency"])
    b = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, adjacency=None)
    f = torch.sin(3 * c["positions"][:, 0]) * torch.cos(2 * c["positions"][:, 1])
    assert torch.equal(a.num_nbrs, b.num_nbrs)
    assert (a.gradient(f) - b.gradient(f)).abs().max() < 1e-3
    assert (a.laplacian(f) - b.laplacian(f)).abs().max() < 1e-1


def test_vector_field_layout(case2d):
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 1, adjacency=c["adjacency"])
    x = c["positions"]
    F = torch.stack([x[:, 0], 2.0 * x[:, 1]], -1)
    G = pf.gradient(F)
    assert G.shape == (x.shape[0], 2, 2)
    assert (G - torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=x.dtype, device=x.device)).abs().max() < 1e-3


def test_rank_deficiency_flagged(device):
    c = _case(device, n=22, nbrs=6)       # ~6 neighbours: a cubic (10 basis fns) cannot be fit
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 3, adjacency=c["adjacency"])
    assert pf.deficient.all()


@pytest.mark.parametrize("dim, p, n, nbrs", [(1, 3, 200, 10), (3, 2, 9, 60)])
def test_other_dimensions(device, dim, p, n, nbrs):
    c = _case(device, dim=dim, n=n, nbrs=nbrs)
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, p, adjacency=c["adjacency"])
    ok = ~pf.deficient
    assert ok.sum() > 0.5 * ok.numel()
    x = c["positions"][:, 0]
    f = 1.0 + 2.0 * x + x ** 2
    tol = 5e-3 if x.dtype == torch.float32 else 1e-8
    assert (pf.values(f) - f).abs()[ok].max() < tol
    assert (pf.gradient(f)[:, 0] - (2.0 + 2.0 * x)).abs()[ok].max() < 10 * tol


def test_matches_pure_torch_reference(device):
    harness = Path(__file__).resolve().parents[2] / "higherOrderSPH" / "harness"
    if not (harness / "rkpm.py").exists():
        pytest.skip("harness reference not present")
    sys.path.insert(0, str(harness))
    try:
        from rkpm import RKPMOperator, build_system
    except Exception as e:                    # pragma: no cover
        pytest.skip(f"harness reference not importable: {e}")
    c = _case(device, dtype=torch.float32)
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, adjacency=c["adjacency"])
    p = c["particles"]
    vol = (p.masses / p.densities).to(torch.float64)
    ref = RKPMOperator(build_system(c["positions"].to(torch.float64), vol, c["h"], 2,
                                    kernel="wendland2"))
    f = (torch.sin(3 * c["positions"][:, 0]) * torch.cos(2 * c["positions"][:, 1]))
    ok = ~pf.deficient & ~ref.s.deficient
    # convention: the core kernel normalisation cancels, the reference uses the same shape
    assert (pf.gradient(f).double() - ref.gradient(f.double())).abs()[ok].max() < 5e-3
    assert (pf.laplacian(f).double() - ref.laplacian(f.double())).abs()[ok].max() < 5e-1


def test_backward_does_not_modify_the_incoming_gradient(case2d):
    """The AD bridge used to zero the caller's grad_output in place (the Warp
    tape zeroed the aliased seed), so reusing one gradient tensor for a second
    backward silently returned zeros."""
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 1, adjacency=c["adjacency"])
    f = torch.sin(3 * c["positions"][:, 0]).requires_grad_(True)
    out = pf.coefficients(f)
    g = torch.randn_like(out)
    g0 = g.clone()
    a = torch.autograd.grad(out, f, g, retain_graph=True)[0].clone()
    b = torch.autograd.grad(out, f, g, retain_graph=True)[0].clone()
    assert torch.equal(g, g0)
    # the pre-fix failure was an all-zero second result; float32 atomic scatter
    # order on CUDA still differs from run to run in the last bits
    scale = a.abs().max()
    assert scale > 0 and (a - b).abs().max() <= 1e-4 * scale


def _pairs(c, radius_factor=1.6):
    pos = c["positions"]
    d = pos[None, :, :] - pos[:, None, :]
    r = torch.linalg.norm(d, dim=-1)
    i, j = torch.nonzero((r < radius_factor * c["dx"]) & (r > 0), as_tuple=True)
    return i, j, (pos[j] - pos[i])


@pytest.mark.parametrize("constant", [True, False])
def test_interface_states_exact_with_variable_support(device, constant):
    """Polynomials of degree <= p are reproduced exactly at the Eq. 25 interface
    points, with a smoothly varying support (so h_i != h_j and the interface
    is not the arithmetic midpoint)."""
    c = _case(device)
    P = c["particles"]
    x = c["positions"]
    P.supports = P.supports * (1.0 + 0.25 * torch.sin(3 * x[:, 0]) * torch.cos(2 * x[:, 1]))
    adjacency = radiusSearchCompactHashMap(P, c["domain"], mode=SupportScheme.SuperSymmetric)
    p = 2
    pf = PolyFit.build(P, c["domain"], KERNEL, p, constant=constant, adjacency=adjacency)
    i, j, d = _pairs(c)
    f = _poly2(x, p)["f"]
    fl, fr = pf.interfaceStates(f, i, j, d)
    oi, _ = PolyFit.interfaceOffsets(d, P.supports[i], P.supports[j])
    rij = x[i] + oi
    X, Y = rij[:, 0], rij[:, 1]
    expect = X ** p + 0.7 * X * Y ** (p - 1) + 0.3 * Y ** p
    ok = ~pf.deficient[i] & ~pf.deficient[j] & (pf.cond[i] < 100) & (pf.cond[j] < 100)
    tol = 5e-3 if x.dtype == torch.float32 else 1e-8
    assert ok.sum() > 100
    assert (fl - expect).abs()[ok].max() < tol * 5
    assert (fr - expect).abs()[ok].max() < tol * 5
    # the interface differs from the arithmetic midpoint when supports differ
    mid = x[i] + 0.5 * d
    assert (rij - mid).abs().max() > 1e-4


def test_interface_offsets_equal_supports_is_the_midpoint():
    d = torch.randn(7, 2)
    h = torch.full((7,), 0.3)
    oi, oj = PolyFit.interfaceOffsets(d, h, h)
    assert torch.allclose(oi, 0.5 * d) and torch.allclose(oj, -0.5 * d)


def test_arbitrary_derivative(case2d):
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 3, adjacency=c["adjacency"])
    X, Y = c["positions"][:, 0], c["positions"][:, 1]
    f = X ** 3 + 0.5 * X * X * Y + Y ** 3                  # f_xxx = 6, f_xxy = 1, f_yyy = 6, f_xy = x
    m = pf.cond < 100
    tol = 0.5 if X.dtype == torch.float32 else 1e-5
    assert (pf.derivative(f, (3, 0)) - 6.0).abs()[m].max() < tol
    assert (pf.derivative(f, (2, 1)) - 1.0).abs()[m].max() < tol
    assert (pf.derivative(f, (0, 3)) - 6.0).abs()[m].max() < tol
    assert (pf.derivative(f, (1, 1)) - X).abs()[m].max() < tol
    with pytest.raises(ValueError):
        pf.derivative(f, (4, 0))


@pytest.mark.parametrize("constant", [True, False])
def test_forward_mode_field_tangent(case2d, constant):
    """JVP wrt the field: the fit is linear in f, so the output tangent equals
    the operator applied to the tangent (checked for value, gradient and
    Laplacian), and the primal is unchanged."""
    import torch.autograd.forward_ad as fwAD
    c = case2d
    pf = PolyFit.build(c["particles"], c["domain"], KERNEL, 2, constant=constant,
                       adjacency=c["adjacency"])
    x = c["positions"]
    f = torch.sin(3 * x[:, 0]) * torch.cos(2 * x[:, 1])
    v = torch.cos(5 * x[:, 0] + x[:, 1])
    m = pf.cond < 100
    with fwAD.dual_level():
        fd = fwAD.make_dual(f, v)
        for name in ("gradient", "laplacian"):
            out = getattr(pf, name)(fd)
            primal, tangent = fwAD.unpack_dual(out)
            assert tangent is not None
            ref_p, ref_t = getattr(pf, name)(f), getattr(pf, name)(v)
            tol = 1e-3 if x.dtype == torch.float32 else 1e-9
            assert (primal - ref_p).abs()[m].max() < tol * max(1.0, float(ref_p.abs().max()))
            assert (tangent - ref_t).abs()[m].max() < tol * max(1.0, float(ref_t.abs().max()))
