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

from harness.metrics import (REF_ORDER_MIN, SATURATION_RATIO,  # noqa: E402
                             observed_order, reference_corrected_order)
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


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("ladder, ref", [([200, 400, 800], 1600),
                                         ([32, 48, 64], 96)])
def test_reference_corrected_order_recovers_true_order(p, ladder, ref):
    # e(h) = C (h^p - h_ref^p): the plain fit over-reads (a true first order
    # fits ~1.4 / ~2.0 on these ladders); the corrected fit recovers p.
    x = 1.0 / np.array(ladder, dtype=float)
    y = 3.0 * (x ** p - (1.0 / ref) ** p)
    plain = observed_order(x, y)
    res = reference_corrected_order(x, y, 1.0 / ref)
    assert plain.slope > p + 0.15
    assert not res.saturated
    assert res.slope == pytest.approx(p, abs=1e-4)
    assert res.r_squared == pytest.approx(1.0, abs=1e-8)


def test_reference_corrected_order_flags_bound():
    # An error that *decelerates* toward the reference cannot be produced by
    # the model: the fit runs to the lower search bound and is flagged.
    x = 1.0 / np.array([32.0, 48.0, 64.0])
    y = np.array([1.0e-1, 4.0e-2, 3.9e-2])
    res = reference_corrected_order(x, y, 1.0 / 96)
    assert res.saturated
    assert res.slope == pytest.approx(REF_ORDER_MIN, abs=1e-2)


def test_reference_corrected_order_rejects_bad_reference():
    with pytest.raises(ValueError):
        reference_corrected_order([0.1, 0.05], [1.0, 0.5], 0.05)


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


def test_analytic_hessian_under_no_grad():
    """The drivers run under torch.set_grad_enabled(False); the analytic
    Hessian must not depend on the caller's grad mode (it silently returned
    zeros there until 2026-10-09)."""
    from harness.test_fields import analytic_hessian
    x = torch.rand(20, 2, dtype=torch.float64)
    f = {fl.name: fl for fl in test_fields.monomial_fields(2, 2)}["x^2"]
    with torch.no_grad():
        H = analytic_hessian(f, x)
    assert torch.allclose(H[:, 0, 0], torch.full_like(x[:, 0], 2.0))


def test_analytic_hessian_monomial_and_smooth():
    from harness.test_fields import analytic_hessian, smooth_open_fields
    x = torch.rand(50, 2, dtype=torch.float64)
    f = {fl.name: fl for fl in test_fields.monomial_fields(2, 3)}["x^2y"]
    H = analytic_hessian(f, x)
    X, Y = x[:, 0], x[:, 1]
    expect = torch.stack([torch.stack([2 * Y, 2 * X], -1),
                          torch.stack([2 * X, torch.zeros_like(X)], -1)], 1)
    assert torch.allclose(H, expect, atol=1e-12)
    # a field with an identically-constant gradient component (d(x^2)/dy)
    # and a linear one (all-zero Hessian, no graph at all)
    fx2 = {fl.name: fl for fl in test_fields.monomial_fields(2, 2)}
    H2 = analytic_hessian(fx2["x^2"], x)
    assert torch.allclose(H2[:, 0, 0], torch.full_like(X, 2.0))
    assert H2[:, 1].abs().max() == 0 and H2[:, 0, 1].abs().max() == 0
    assert analytic_hessian(fx2["x"], x).abs().max() == 0
    # trace == the field's own analytic Laplacian, for every smooth field
    for fl in smooth_open_fields(2, np.array([1.0, 1.0])):
        if fl.is_vector:
            continue
        H = analytic_hessian(fl, x)
        tr = H.diagonal(dim1=1, dim2=2).sum(-1)
        assert torch.allclose(tr, fl.lap(x), rtol=1e-10, atol=1e-10), fl.name
        assert torch.allclose(H, H.transpose(1, 2), atol=1e-10), fl.name


def test_external_mode_dispatch():
    # same local-import pattern as test_renorm_value_scalar_and_vector_broadcast
    sys.path.insert(0, str(HARNESS_DIR))
    from harness import operators
    calls = []

    def fake(case, cache, values, probe):
        calls.append(probe)
        return values * 2

    operators.register_mode("fakeMLS", fake)
    try:
        v = torch.ones(3)
        out = operators.run_probe(None, None, v, "hessian", "fakeMLS")
        assert torch.equal(out, v * 2) and calls == ["hessian"]
        with pytest.raises(ValueError):
            operators.register_mode("crk", fake)      # no shadowing built-ins
        with pytest.raises(ValueError):
            operators.run_probe(None, None, v, "nope", "fakeMLS")
    finally:
        operators.EXTERNAL_MODES.pop("fakeMLS", None)


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

def test_matrix_condition_numbers_general_n():
    # symmetric: agrees with the eigenvalue path
    A = torch.diag(torch.tensor([4.0, 2.0, 0.5], dtype=torch.float64))[None]
    k = conditioning.matrix_condition_numbers(A)
    assert k.item() == pytest.approx(8.0)
    assert conditioning.condition_numbers(
        torch.tensor([[4.0, 2.0, 0.5]])).item() == pytest.approx(8.0)
    # non-symmetric 6x6 (p=2 moment-matrix size in 2D) vs numpy's cond
    g = torch.Generator().manual_seed(0)
    M = torch.randn(5, 6, 6, generator=g, dtype=torch.float64)
    k = conditioning.matrix_condition_numbers(M)
    ref = np.linalg.cond(M.numpy())
    assert np.allclose(k.numpy(), ref, rtol=1e-10)
    # singular -> inf
    S = torch.zeros(1, 3, 3, dtype=torch.float64)
    S[0, 0, 0] = 1.0
    assert torch.isinf(conditioning.matrix_condition_numbers(S)).all()
    # equilibration removes a pure basis-scaling spread
    D = torch.diag(torch.tensor([1.0, 1e-2, 1e-4], dtype=torch.float64))[None]
    assert conditioning.matrix_condition_numbers(D).item() == pytest.approx(1e4)
    assert conditioning.matrix_condition_numbers(
        D, equilibrate=True).item() == pytest.approx(1.0)


def test_frozen_leg_pure_pieces():
    # exact solutions + RK4 of the frozen-particle PDE leg (no warp needed:
    # run_frozen only imports warp/warpSPHCore inside its driver functions)
    sys.path.insert(0, str(HARNESS_DIR))
    from harness import run_frozen as rf
    box = (1.0, 2.0)
    kappa = rf.wavevector(box, (1, 2))
    assert torch.allclose(kappa, torch.tensor([2 * np.pi, 2 * np.pi],
                                              dtype=torch.float64))
    x = torch.rand(20, 2, dtype=torch.float64)
    a = torch.tensor([1.0, 0.5], dtype=torch.float64)
    # advection: periodic in time with the box period along a
    u = rf.exact_solution("advection", x, 0.3, kappa, a=a)
    assert torch.allclose(u, torch.sin((x - 0.3 * a) @ kappa))
    ud = rf.exact_solution("diffusion", x, 0.2, kappa, nu=0.1)
    assert torch.allclose(ud, np.exp(-0.1 * float(kappa @ kappa) * 0.2)
                          * torch.sin(x @ kappa))
    # RK4 on u' = -u: 4th-order accurate, lands exactly on t_end
    u1, n, div = rf.rk4(lambda u: -u, torch.ones(1, dtype=torch.float64),
                        1.0, 0.1)
    assert n == 10 and not div
    assert abs(float(u1) - np.exp(-1.0)) < 1e-6
    # blow-up detection: u' = +50 u explodes past BLOWUP
    _, _, div = rf.rk4(lambda u: 50.0 * u, torch.ones(1, dtype=torch.float64),
                       1.0, 0.01)
    assert div


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
# particle sets: build_case_from_positions round-trip (GPU)
# ---------------------------------------------------------------------------

def test_build_case_from_positions_roundtrip():
    # build_case_from_positions must reproduce exactly the probe case
    # (densities, adjacency, domain) that build_case produces for the same
    # positions/masses/support: it is the glue for loading saved
    # distributions (e.g. higherOrderSPH/harness/data/tgv2d_*.npz) as probe
    # cases. Runs in a float64 subprocess (the precision the harness is
    # written for; the session's own precision is left untouched).
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    code = (
        "import warp as wp; wp.init()\n"
        "import torch\n"
        "from harness.particle_sets import (build_case,\n"
        "                                   build_case_from_positions)\n"
        "from warpSPHCore.enumTypes import KernelFunctions\n"
        "a = build_case(n=64, dim=2, target_neighbors=4, jitter=0.5,\n"
        "               seed=11, periodic=True, device='cuda',\n"
        "               kernel=KernelFunctions.Wendland4)\n"
        "b = build_case_from_positions(a.positions, a.particles.masses,\n"
        "                              a.h, a.box, a.dx,\n"
        "                              a.target_neighbors, periodic=True,\n"
        "                              device='cuda',\n"
        "                              kernel=KernelFunctions.Wendland4)\n"
        "assert b.N == a.N and b.dim == a.dim\n"
        "assert abs(b.h_over_dx - a.h_over_dx) <= 1e-12 * a.h_over_dx\n"
        "assert abs(b.cell_vol - a.cell_vol) <= 1e-12 * a.cell_vol\n"
        "assert torch.allclose(b.positions, a.positions)\n"
        "assert torch.equal(b.particles.densities, a.particles.densities)\n"
        "print('roundtrip ok')\n"
    )
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    pp = str(HARNESS_DIR.parent)
    env["PYTHONPATH"] = (pp + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else pp)
    proc = subprocess.run([sys.executable, "-c", code],
                          env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        f"roundtrip exited {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}")


def test_build_case_from_positions_on_saved_tgv_data():
    # The saved TGV distributions (disordered, and with positions drifted
    # outside the box -- the code never re-wraps) must load through
    # build_case_from_positions and reproduce the anisotropy module's
    # kernel sum: for uniform masses the SPH density is the normalised
    # kernel sum, so the min/max ratios must agree.
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    data_dir = HARNESS_DIR / "data"
    if not (data_dir / "tgv2d_noshift_nx128.npz").exists():
        pytest.skip("saved TGV test data not present")
    code = (
        "import warp as wp; wp.init()\n"
        "import numpy as np\n"
        "import torch\n"
        "from pathlib import Path\n"
        "from harness.particle_sets import build_case_from_positions\n"
        "from warpSPHCore.enumTypes import KernelFunctions\n"
        "DATA_DIR = Path(__DATA_DIR__)\n"
        "for name in ('tgv2d_noshift_nx128.npz', 'tgv2d_fullshift_nx128.npz'):\n"
        "    d = np.load(DATA_DIR / name)\n"
        "    L = float(d['L'])\n"
        "    case = build_case_from_positions(\n"
        "        positions=torch.tensor(d['t2_positions'],\n"
        "                               dtype=torch.float64, device='cuda'),\n"
        "        masses=torch.tensor(d['t2_masses'],\n"
        "                            dtype=torch.float64, device='cuda'),\n"
        "        h=float(d['t2_h']), box=np.array([L, L]),\n"
        "        dx=float(d['dx']), target_neighbors=32,\n"
        "        periodic=True, device='cuda',\n"
        "        kernel=KernelFunctions.Wendland4)\n"
        "    rho = case.particles.densities.detach().cpu()\n"
        "    ratio = float(rho.min() / rho.max())\n"
        "    cs = float(d['t2_kernel_sum_min'] / d['t2_kernel_sum_max'])\n"
        "    assert abs(ratio - cs) < 1e-9, (name, ratio, cs)\n"
        "print('saved-data glue ok')\n"
    ).replace("__DATA_DIR__", repr(str(data_dir)))
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    pp = str(HARNESS_DIR.parent)
    env["PYTHONPATH"] = (pp + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else pp)
    proc = subprocess.run([sys.executable, "-c", code],
                          env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, (
        f"saved-data glue exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}")


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
    # CRK linear-gradient exactness on a jittered open set (regression guard
    # for the crk/kernel.py gradB-transpose bug, 2026-09-26)
    assert c["crk_grad_linear_interior_linf"] < 1e-10
    assert c["crk_grad_linear_boundary_linf"] < 1e-10
    assert c["standard_grad_linear_interior_linf"] > 1e-3
    assert c["standard_interp_const_boundary_linf"] > \
        c["standard_interp_const_interior_linf"]


# ---------------------------------------------------------------------------
# disorder probe smoke (subprocess, float64)
# ---------------------------------------------------------------------------

DISORDER_DRIVER = HARNESS_DIR / "run_disorder_probe.py"
DISORDER_VERDICT = HARNESS_DIR / "results" / "disorder_probe_verdict.json"


@pytest.fixture(scope="session")
def disorder_smoke_run():
    if not torch.cuda.is_available():
        pytest.skip("disorder probe smoke requires a CUDA device")
    data = HARNESS_DIR / "data" / "tgv2d_noshift_nx128.npz"
    if not data.exists():
        pytest.skip("saved TGV test data not present")
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    proc = subprocess.run(
        [sys.executable, str(DISORDER_DRIVER), "--smoke"],
        env=env, cwd=str(HARNESS_DIR),
        capture_output=True, text=True, timeout=1200,
    )
    payload = json.loads(DISORDER_VERDICT.read_text()) \
        if DISORDER_VERDICT.exists() else {}
    return proc, payload


def test_disorder_probe_smoke_exits_zero(disorder_smoke_run):
    proc, _payload = disorder_smoke_run
    assert proc.returncode == 0, (
        f"disorder smoke exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")


def test_disorder_probe_smoke_verdict_ok(disorder_smoke_run):
    # The headline invariants of TGV_NOTES.md section 7.5: CRK interpolate
    # is distribution-independent (glass vs strongly disordered within 2x),
    # while the standard operators are disorder-sensitive (>10x) -- a
    # regression in the operators or the saved data breaks one or the other.
    _proc, payload = disorder_smoke_run
    assert payload, "no disorder probe verdict written"
    assert payload.get("ok") is True, json.dumps(payload, indent=2)
    c = payload["checks"]
    assert c["crk_interp_flat_ratio"] < 2.0
    assert c["std_interp_disorder_ratio"] > 10.0
    assert c["std_grad_disorder_ratio"] > 10.0
    assert c["all_errors_finite_positive"] is True


# ---------------------------------------------------------------------------
# frozen-particle PDE leg smoke (subprocess, float64)
# ---------------------------------------------------------------------------

FROZEN_DRIVER = HARNESS_DIR / "run_frozen.py"
FROZEN_VERDICT = HARNESS_DIR / "results" / "frozen_verdict_smoke.json"


@pytest.fixture(scope="session")
def frozen_smoke_run():
    if not torch.cuda.is_available():
        pytest.skip("frozen-leg smoke requires a CUDA device")
    env = dict(os.environ, warpSPHCore_PRECISION="float64",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    proc = subprocess.run([sys.executable, str(FROZEN_DRIVER), "--smoke"],
                          env=env, cwd=str(HARNESS_DIR),
                          capture_output=True, text=True, timeout=900)
    payload = (json.loads(FROZEN_VERDICT.read_text())
               if FROZEN_VERDICT.exists() else None)
    return proc, payload


def test_frozen_smoke_ok(frozen_smoke_run):
    proc, payload = frozen_smoke_run
    assert proc.returncode == 0, (
        f"frozen smoke exited {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}"
        f"\nstderr:\n{proc.stderr[-3000:]}")
    assert payload and payload["ok"] is True, payload
    # every built-in mode is stable on the smoke problem
    assert payload["diverged"] == []
    assert payload["n_rows"] == 2 * 4 * 2     # equations x modes x ladder
