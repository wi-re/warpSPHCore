"""CI gate for the convergence harness (higherOrderSPH/harness/).

Two layers, mirroring tests/kernels/test_kernel_audit.py's pattern:

1. Direct-import unit tests for the pure modules (metrics, test_fields,
   conditioning) -- CPU-only, no warp, fast.
2. A session-scoped subprocess run of `run_baseline.py --smoke` in
   float64 (the precision the harness is written for; the session's own
   precision is left untouched). The smoke verdict must be ok and the
   process must exit 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "higherOrderSPH" / "harness"
DRIVER = HARNESS_DIR / "run_baseline.py"

sys.path.insert(0, str(REPO_ROOT / "higherOrderSPH"))

from harness.metrics import SATURATION_RATIO, observed_order  # noqa: E402
from harness import conditioning  # noqa: E402
from harness import test_fields  # noqa: E402


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p, scale", [(1.0, 2.0), (2.0, 7.0), (4.0, 0.5)])
def test_observed_order_exact(p, scale):
    x = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    y = scale * x ** p
    res = observed_order(x, y)
    assert not res.saturated
    assert res.slope == pytest.approx(p, abs=1e-9)
    assert res.r_squared == pytest.approx(1.0, abs=1e-12)


def test_observed_order_saturated():
    x = np.array([1.0, 0.5, 0.25])
    y = np.array([1e-15, 1.2e-15, 0.9e-15])  # flat within the ratio
    res = observed_order(x, y)
    assert res.saturated
    assert np.isnan(res.slope)


def test_observed_order_exact_zero_errors():
    # machine-precision reproduction: errors exactly zero, nothing to fit
    x = np.array([1.0, 0.5, 0.25])
    res = observed_order(x, np.array([0.0, 0.0, 0.0]))
    assert res.exact
    assert np.isnan(res.slope)
    res = observed_order(x, np.array([1e-15, 0.0, 0.0]))
    assert res.exact


def test_observed_order_rejects_bad_input():
    with pytest.raises(ValueError):
        observed_order([1.0], [1.0])
    with pytest.raises(ValueError):
        observed_order([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        observed_order([1.0, 2.0], [1.0, -1.0])


# ---------------------------------------------------------------------------
# test_fields: analytic derivatives vs central finite differences
# ---------------------------------------------------------------------------

def _fd_check(field, x, eps=1e-6):
    v = field.value(x)
    g = field.grad(x)
    assert v.shape[0] == x.shape[0]
    assert g.shape[0] == x.shape[0]
    dim = x.shape[1]
    v_dim = v.shape[1] if v.ndim > 1 else 1
    for d in range(dim):
        e = torch.zeros_like(x)
        e[:, d] = eps
        vp = field.value(x + e)
        vm = field.value(x - e)
        fd = (vp - vm) / (2 * eps)
        scale = torch.abs(v).max().clamp_min(1.0)
        assert torch.allclose(g[:, d] if v.ndim == 1 else g[:, :, d],
                              fd, rtol=1e-4,
                              atol=1e-6 * scale.item() * 10)
    if field.lap is not None:
        lap = field.lap(x)
        if torch.allclose(lap, torch.zeros_like(lap)):
            # zero Laplacian: the finite-difference trace is pure float64
            # cancellation noise (~1e-5 at eps=1e-6); just bound it
            trace = torch.zeros_like(lap)
            for d in range(dim):
                e = torch.zeros_like(x)
                e[:, d] = eps
                vp = field.value(x + 2 * e)
                vm = field.value(x - 2 * e)
                vc = field.value(x)
                trace = trace + (vp - 2 * vc + vm) / (4 * eps ** 2)
            assert trace.abs().max() < 1e-3
            return
        trace = torch.zeros_like(lap)
        for d in range(dim):
            e = torch.zeros_like(x)
            e[:, d] = eps
            vp = field.value(x + 2 * e)
            vm = field.value(x - 2 * e)
            vc = field.value(x)
            trace = trace + (vp - 2 * vc + vm) / (4 * eps ** 2)
        scale = torch.abs(lap).max().clamp_min(1.0)
        assert torch.allclose(lap, trace, rtol=1e-3,
                              atol=1e-5 * scale.item() * 10)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_monomial_fields_derivatives(dim):
    torch.manual_seed(0)
    x = torch.rand(64, dim, dtype=torch.float64) * 0.8 + 0.1
    for f in test_fields.monomial_fields(dim, degree_max=3):
        _fd_check(f, x)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_monomial_fields_no_duplicates(dim):
    """Regression: the exponent generator must emit each exponent vector
    exactly once (an earlier version iterated 3-D compositions and
    truncated to `dim`, duplicating every monomial by its dropped
    exponents)."""
    import math
    for degree_max in (2, 4):
        fs = test_fields.monomial_fields(dim, degree_max=degree_max)
        names = [f.name for f in fs]
        assert len(set(names)) == len(names), (
            f"duplicate field names in dim={dim}: "
            f"{[n for n in names if names.count(n) > 1]}")
        # monomials of total degree <= D in dim variables: C(dim + D, dim)
        expected = math.comb(dim + degree_max, dim)
        assert len(fs) == expected, (f"dim={dim} D={degree_max}: "
                                     f"{len(fs)} != {expected}")
        # each field's value must match the exponents in its name
        import re
        x = torch.rand(16, dim, dtype=torch.float64) * 0.9 + 0.05
        for f in fs:
            assert f.degree <= degree_max
            exps = {s: int(e) if e else 1
                    for s, e in re.findall(r"([xyz])(?:\^(\d+))?", f.name)}
            expected = torch.ones(16, dtype=torch.float64)
            for s, e in exps.items():
                expected = expected * x[:, "xyz".index(s)] ** e
            assert torch.allclose(f.value(x), expected, atol=1e-12), \
                f.name


def test_vector_field_derivatives():
    x = torch.rand(32, 2, dtype=torch.float64)
    f = test_fields.vector_field(2)
    _fd_check(f, x)
    # Laplacian of a linear field is exactly zero
    assert torch.allclose(f.lap(x), torch.zeros(32, 2, dtype=torch.float64))


def test_periodic_fields_are_periodic_and_differentiable():
    box = [1.0, 1.2]
    for f in test_fields.smooth_periodic_fields(2, box):
        x = torch.rand(32, 2, dtype=torch.float64) * torch.tensor(box)
        _fd_check(f, x)
        # periodicity: f(x + L e_d) == f(x)
        for d in range(2):
            xs = x.clone()
            xs[:, d] = xs[:, d] + box[d]
            assert torch.allclose(f.value(xs), f.value(x),
                                  atol=1e-12 * f.value(x).abs().max())


def test_open_fields_derivatives():
    x = torch.rand(32, 2, dtype=torch.float64)
    for f in test_fields.smooth_open_fields(2, [1.0, 1.0]):
        _fd_check(f, x)


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------

def test_condition_numbers_and_fallback():
    C = torch.eye(2).repeat(4, 1, 1)
    C[0] = torch.diag(torch.tensor([4.0, 1.0]))
    C[1] = torch.diag(torch.tensor([9.0, 3.0]))
    eig = torch.stack([
        torch.tensor([4.0, 1.0]),
        torch.tensor([9.0, 3.0]),
        torch.tensor([1.0, 1.0]),   # identity fallback
        torch.tensor([1.0, 1.0]),   # identity fallback
    ])
    cond = conditioning.condition_numbers(eig)
    assert cond[0].item() == pytest.approx(4.0)
    assert cond[1].item() == pytest.approx(3.0)
    fb = conditioning.fallback_mask(C)
    assert fb.tolist() == [False, False, True, True]
    stats = conditioning.cond_summary(cond, ~fb)
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(3.5)
    assert stats["max"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# value renormalization (Randles--Libersky) helper
# ---------------------------------------------------------------------------

def test_renorm_value_scalar_and_vector_broadcast():
    # Local import: operators pulls in warpSPHCore (kept out of the pure-CPU
    # section above) and uses a bare sibling import, so the harness dir itself
    # must be importable too. _renormValue is plain torch either way.
    sys.path.insert(0, str(HARNESS_DIR))
    from harness.operators import _renormValue
    raw = torch.tensor([[2.0, 4.0], [6.0, 8.0]])  # (N, D) vector field
    S = torch.tensor([2.0, 4.0])                   # (N,) 0th moment
    out = _renormValue(raw, S)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [1.5, 2.0]]))
    # scalar field: (N,) / (N,)
    assert torch.allclose(_renormValue(torch.tensor([3.0, 5.0]),
                                       torch.tensor([1.0, 2.0])),
                          torch.tensor([3.0, 2.5]))


# ---------------------------------------------------------------------------
# smoke (subprocess, float64)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_run():
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--smoke"],
        env=env, cwd=str(HARNESS_DIR),
        capture_output=True, text=True, timeout=1200,
    )
    # the driver writes its verdict next to itself (results/, git-ignored)
    written = HARNESS_DIR / "results" / "smoke_verdict.json"
    payload = json.loads(written.read_text()) if written.exists() else {}
    return proc, payload


def test_smoke_exits_zero(smoke_run):
    proc, payload = smoke_run
    assert proc.returncode == 0, (
        f"smoke exited {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


def test_smoke_verdict_ok(smoke_run):
    _proc, payload = smoke_run
    assert payload, "no smoke verdict written"
    assert payload.get("ok") is True, json.dumps(payload, indent=2)
    c = payload["checks"]
    assert c["crk_interp_linear_interior_linf"] < 1e-10
    assert c["renorm_grad_linear_interior_linf"] < 1e-10
    assert c["standard_grad_linear_interior_linf"] > 1e-3
    assert c["standard_interp_const_boundary_linf"] > \
        c["standard_interp_const_interior_linf"]
