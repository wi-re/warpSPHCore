"""CPU unit tests for the Phase 6/7 reconstruction layer
(higherOrderSPH/harness/reconstruct.py): the exact smoothness Gram matrix,
smooth-region order of the TENO / WENO interface states, boundedness at a
discontinuity, and 1-D operation. Pure torch.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "higherOrderSPH"))

from harness.reconstruct import (smoothness_gram, teno_reconstructor,  # noqa: E402
                                 weno_reconstructor)
from harness.rkpm import monomial_exponents  # noqa: E402

BOX2 = torch.tensor([1.0, 1.0], dtype=torch.float64)


def _lat(n, jitter=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n),
                                     indexing="ij"), -1).reshape(-1, 2).double() / n
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5) * jitter / n
    return pts, 1.0 / n


def test_gram_matches_numerical_quadrature():
    exps = torch.tensor(monomial_exponents(2, 3))
    G = smoothness_gram(exps)
    # Monte-Carlo-free check: Gauss-Legendre on [-1, 1]^2 of the same form
    x, w = torch.tensor(__import__("numpy").polynomial.legendre.leggauss(8),
                        dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    W = (w[:, None] * w[None, :]).reshape(-1)
    X, Y = X.reshape(-1), Y.reshape(-1)
    E = exps.tolist()

    def D(e, a):                                  # D^a (X^e0 Y^e1)
        if e[0] < a[0] or e[1] < a[1]:
            return torch.zeros_like(X)
        c = (math.factorial(e[0]) // math.factorial(e[0] - a[0])
             * math.factorial(e[1]) // math.factorial(e[1] - a[1]))
        return c * X ** (e[0] - a[0]) * Y ** (e[1] - a[1])

    ref = torch.zeros_like(G)
    for a in [(i, j) for i in range(4) for j in range(4) if 1 <= i + j <= 3]:
        d = torch.stack([D(e, a) for e in E])                 # (n, Q)
        ref += torch.einsum("aq,bq,q->ab", d, d, W)
    assert torch.allclose(G, ref, atol=1e-12)
    assert G[0].abs().max() == 0                               # constant: no variation


@pytest.mark.parametrize("make, expected", [
    (lambda p, v: teno_reconstructor(p, v, BOX2, "O4"), 3.5),
    (lambda p, v: weno_reconstructor(p, v, BOX2, 3), 3.0),
])
def test_smooth_region_order(make, expected):
    errs = []
    for n in (24, 48):
        pts, dx = _lat(n)
        V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
        R = make(pts, V)
        f = lambda x: torch.sin(2 * math.pi * x[:, 0]) * torch.cos(2 * math.pi * x[:, 1])
        i, j, fl, fr, d = R.interface_states(f(pts))
        ex = f(pts[i] + d)
        errs.append(max((fl - ex).abs().max(), (fr - ex).abs().max()).item())
    assert math.log(errs[0] / errs[1]) / math.log(2) > expected


@pytest.mark.parametrize("make", [
    lambda p, v: teno_reconstructor(p, v, BOX2, "O4"),
    lambda p, v: weno_reconstructor(p, v, BOX2, 2),
])
def test_discontinuity_is_not_overshot(make):
    pts, dx = _lat(48)
    V = torch.full((pts.shape[0],), dx * dx, dtype=torch.float64)
    f = (torch.sin(2 * math.pi * pts[:, 0] + 0.3) > 0).double()
    i, j, fl, fr, _d = make(pts, V).interface_states(f)
    assert fl.max() < 1 + 1e-6 and fl.min() > -1e-6
    assert fr.max() < 1 + 1e-6 and fr.min() > -1e-6


def test_one_dimension_smooth_order_and_central_selected():
    """1-D (not in the papers): TENO O4 reaches ~4th order and keeps the
    central stencil in smooth data (a starved central stencil would silently
    fall back to the 3rd-order sectors)."""
    errs = []
    for n in (100, 200):
        x = (torch.arange(n, dtype=torch.float64)
             + 0.2 * torch.sin(7.3 * torch.arange(n).double())) / n
        V = torch.full((n,), 1.0 / n, dtype=torch.float64)
        pts = x[:, None]
        R = teno_reconstructor(pts, V, torch.tensor([1.0], dtype=torch.float64), "O4")
        f = lambda q: torch.sin(2 * math.pi * q[:, 0])
        i, j, fl, fr, d = R.interface_states(f(pts))
        ex = f(pts[i] + d)
        errs.append(max((fl - ex).abs().max(), (fr - ex).abs().max()).item())
        assert R.last_central.float().mean() > 0.99
    assert math.log(errs[0] / errs[1]) / math.log(2) > 3.5
