"""CPU unit tests for the order-p MLS/RKPM reference operator
(higherOrderSPH/harness/rkpm.py): basis enumeration, exact reproduction of
polynomials through degree p (value / gradient / Hessian, interior and
boundary), the Taylor-factor convention of the Hessian, periodic minimum
image, rank-deficiency flagging, and the order gain from p to p+1.

Pure torch -- no warp, no GPU.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "higherOrderSPH"))

from harness.rkpm import (RKPMOperator, build_system,  # noqa: E402
                          monomial_exponents)


def _lattice(n, dim, jitter=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    axes = [torch.arange(n, dtype=torch.float64)] * dim
    pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    dx = 1.0 / n
    pts = pts * dx
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=torch.float64)
                 - 0.5) * jitter * dx
    return pts, dx


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim, order, count", [
    (1, 3, 4), (2, 1, 3), (2, 2, 6), (2, 3, 10), (3, 1, 4), (3, 2, 10),
    (3, 3, 20)])
def test_basis_size(dim, order, count):
    e = monomial_exponents(dim, order)
    assert len(e) == count == math.comb(dim + order, order)
    assert e[0] == (0,) * dim            # constant first
    assert len(set(e)) == len(e)


# ---------------------------------------------------------------------------
# exact reproduction
# ---------------------------------------------------------------------------

def _poly(x, deg):
    """A mixed degree-`deg` polynomial in 2-D with analytic derivatives."""
    X, Y = x[:, 0], x[:, 1]
    f = X ** deg + 0.7 * X * Y ** max(deg - 1, 0) + 0.3 * Y ** deg
    gx = deg * X ** (deg - 1) + 0.7 * Y ** max(deg - 1, 0)
    gy = (0.7 * X * (deg - 1) * Y ** max(deg - 2, 0) if deg > 1 else 0 * X) \
        + 0.3 * deg * Y ** (deg - 1)
    if deg >= 2:
        fxx = deg * (deg - 1) * X ** (deg - 2)
        fxy = 0.7 * (deg - 1) * Y ** max(deg - 2, 0)
        fyy = (0.7 * X * (deg - 1) * (deg - 2) * Y ** max(deg - 3, 0)
               if deg > 2 else 0 * X) + 0.3 * deg * (deg - 1) * Y ** (deg - 2)
    else:
        fxx = fxy = fyy = None
    return f, torch.stack([gx, gy], -1), fxx, fxy, fyy


@pytest.mark.parametrize("p", [1, 2, 3])
def test_polynomial_reproduction_through_order_p(p):
    pts, dx = _lattice(22, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    s = build_system(pts, V, 4.0 * dx, p)
    assert not s.deficient.any()
    op = RKPMOperator(s)
    for deg in range(p + 1):
        f, g, fxx, fxy, fyy = _poly(pts, deg)
        assert (op.values(f) - f).abs().max() < 1e-9      # incl. boundary
        assert (op.gradient(f) - g).abs().max() < 1e-8
        if p >= 2 and deg >= 2:
            H = op.hessian(f)
            assert (H[:, 0, 0] - fxx).abs().max() < 1e-6
            assert (H[:, 0, 1] - fxy).abs().max() < 1e-6
            assert (H[:, 1, 0] - fxy).abs().max() < 1e-6
            assert (H[:, 1, 1] - fyy).abs().max() < 1e-6
            lap = op.laplacian(f)
            assert (lap - (fxx + fyy)).abs().max() < 1e-6


def test_not_exact_one_degree_above_order():
    pts, dx = _lattice(22, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    op = RKPMOperator(build_system(pts, V, 4.0 * dx, 2))
    f, g, *_ = _poly(pts, 3)
    assert (op.gradient(f) - g).abs().max() > 1e-5


def test_hessian_laplacian_need_order_two():
    pts, dx = _lattice(12, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    op = RKPMOperator(build_system(pts, V, 4.0 * dx, 1))
    f = pts[:, 0]
    assert op.hessian(f) is None and op.laplacian(f) is None


def test_vector_field_layout():
    pts, dx = _lattice(14, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    op = RKPMOperator(build_system(pts, V, 4.0 * dx, 1))
    F = torch.stack([pts[:, 0], 2.0 * pts[:, 1]], -1)       # (N, 2)
    G = op.gradient(F)                                       # (N, D, dim)
    assert G.shape == (pts.shape[0], 2, 2)
    expect = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    assert (G - expect).abs().max() < 1e-9


@pytest.mark.parametrize("dim, p", [(1, 2), (3, 1), (3, 2)])
def test_other_dimensions(dim, p):
    n = {1: 120, 3: 9}[dim]
    pts, dx = _lattice(n, dim)
    V = torch.full((pts.shape[0],), dx ** dim, dtype=torch.float64)
    s = build_system(pts, V, 3.5 * dx, p)
    op = RKPMOperator(s)
    ok = ~s.deficient
    f = 1.0 + 2.0 * pts[:, 0] + (pts[:, 0] ** 2 if p >= 2 else 0)
    g0 = 2.0 + (2.0 * pts[:, 0] if p >= 2 else 0)
    assert (op.values(f) - f).abs()[ok].max() < 1e-8
    assert (op.gradient(f)[:, 0] - g0).abs()[ok].max() < 1e-7


# ---------------------------------------------------------------------------
# periodicity, conditioning, convergence
# ---------------------------------------------------------------------------

def test_periodic_minimum_image_matches_shifted_copy():
    """A periodic field sampled across the seam: with the box given, the
    operator on points near the wrap equals the operator on the same
    configuration translated by half a box (wrapped)."""
    pts, dx = _lattice(20, 2, jitter=0.2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    box = torch.tensor([1.0, 1.0], dtype=torch.float64)
    two_pi = 2 * math.pi
    f = lambda x: torch.sin(two_pi * x[:, 0]) * torch.cos(two_pi * x[:, 1])
    op = RKPMOperator(build_system(pts, V, 4.0 * dx, 2, box=box))
    shift = torch.tensor([0.5, 0.5], dtype=torch.float64)
    pts2 = (pts + shift) % 1.0
    op2 = RKPMOperator(build_system(pts2, V, 4.0 * dx, 2, box=box))
    # same particles, translated: operator outputs agree particle by particle
    f2 = torch.sin(two_pi * (pts2[:, 0] - 0.5)) * torch.cos(two_pi * (pts2[:, 1] - 0.5))
    assert (op.gradient(f(pts)) - op2.gradient(f2)).abs().max() < 1e-9


def test_deficient_flag_for_sparse_support():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=torch.float64)
    V = torch.ones(3, dtype=torch.float64)
    s = build_system(pts, V, 1.5, 2)
    assert s.deficient.all()            # <6 neighbours everywhere


def test_collinear_support_is_ill_conditioned():
    x = torch.linspace(0, 1, 40, dtype=torch.float64)
    pts = torch.stack([x, torch.zeros_like(x)], -1)
    V = torch.full((40,), 1.0 / 40, dtype=torch.float64)
    s = build_system(pts, V, 0.2, 2)
    assert s.deficient.all()


def test_order_gain_p_to_p_plus_one():
    """Smooth-field gradient error drops faster for the higher order."""
    def err(p, n):
        pts, dx = _lattice(n, 2)
        V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
        op = RKPMOperator(build_system(pts, V, 4.0 * dx, p))
        f = torch.sin(2.0 * pts[:, 0]) * torch.cos(1.5 * pts[:, 1])
        g = torch.stack([2.0 * torch.cos(2.0 * pts[:, 0]) * torch.cos(1.5 * pts[:, 1]),
                         -1.5 * torch.sin(2.0 * pts[:, 0]) * torch.sin(1.5 * pts[:, 1])], -1)
        m = ((pts > 0.3) & (pts < 0.7)).all(1)       # interior
        return (op.gradient(f) - g).abs()[m].max().item()
    rate = {p: math.log(err(p, 16) / err(p, 32)) / math.log(2.0) for p in (1, 2)}
    # theory: O(h^p) gradient; the jittered max-norm of p=1 can read a bit
    # above 1, so assert the ordering and the p=2 rate rather than a margin
    assert rate[1] > 0.7
    assert rate[2] > 1.7
    assert rate[2] > rate[1]


# ---------------------------------------------------------------------------
# LABFM variant (constant=False)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [2, 3, 4])
def test_labfm_exact_through_order_and_reproduces_values(p):
    pts, dx = _lattice(24, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    s = build_system(pts, V, 4.5 * dx, p, constant=False)
    assert s.n_basis == math.comb(2 + p, p) - 1 and not s.has_constant
    assert not s.deficient.any()
    op = RKPMOperator(s)
    for deg in range(1, p + 1):
        f, g, fxx, fxy, fyy = _poly(pts, deg)
        assert torch.equal(op.values(f), f)               # f_i is exact
        assert (op.gradient(f) - g).abs().max() < 1e-7
        if deg >= 2:
            lap = op.laplacian(f)
            assert (lap - (fxx + fyy)).abs().max() < 1e-5


def test_labfm_and_mls_derivatives_agree_on_polynomials_differ_off_them():
    pts, dx = _lattice(24, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    mls = RKPMOperator(build_system(pts, V, 4.5 * dx, 2))
    lab = RKPMOperator(build_system(pts, V, 4.5 * dx, 2, constant=False))
    f, g, *_ = _poly(pts, 2)
    assert (mls.gradient(f) - lab.gradient(f)).abs().max() < 1e-8
    f3 = pts[:, 0] ** 3
    assert (mls.gradient(f3) - lab.gradient(f3)).abs().max() > 1e-6


@pytest.mark.parametrize("constant", [True, False])
def test_interface_states_exact_for_polynomials_through_order(constant):
    """Left/right midpoint states reproduce a degree-p polynomial exactly
    (so the L/R jump vanishes for it), for both the MLS and LABFM fits."""
    p = 3
    pts, dx = _lattice(22, 2)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    s = build_system(pts, V, 4.5 * dx, p, constant=constant)
    op = RKPMOperator(s)
    f, *_ = _poly(pts, p)
    i, j, fl, fr = op.interface_states(f)
    assert (i != j).all() and i.shape == fl.shape == fr.shape
    mid = 0.5 * (pts[i] + pts[j])
    X, Y = mid[:, 0], mid[:, 1]
    expect = X ** p + 0.7 * X * Y ** (p - 1) + 0.3 * Y ** p
    ok = ~s.deficient[i] & ~s.deficient[j]
    assert (fl - expect).abs()[ok].max() < 1e-8
    assert (fr - expect).abs()[ok].max() < 1e-8
    assert (fl - fr).abs()[ok].max() < 1e-8


# ---------------------------------------------------------------------------
# harness integration smoke (subprocess, float64; warp CPU or CUDA)
# ---------------------------------------------------------------------------

def test_run_rkpm_smoke_exits_zero():
    """`run_rkpm.py --smoke`: exact degree-p reproduction for p = 1..3
    through the registered harness modes (interior + boundary band) and the
    rkpm1 == crk value identity."""
    import subprocess
    driver = REPO_ROOT / "higherOrderSPH" / "harness" / "run_rkpm.py"
    proc = subprocess.run([sys.executable, str(driver), "--smoke"],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"smoke exited {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")
