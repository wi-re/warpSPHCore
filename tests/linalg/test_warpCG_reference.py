"""The upstream comparison harness verifies independently and replays new contents."""

from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts/benchmarks'))
    from _warp_cg_reference import WarpCGReference
    return WarpCGReference


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_portable_independent_verification(reference, dtype):
    b = torch.ones(4, dtype=dtype)
    solver = reference(lambda v: v * 2, b, maxiter=8, rtol=1e-5)
    result = solver.observe(b)
    assert result.verified
    torch.testing.assert_close(result.x, b / 2)
    assert result.trueResidual == (b - result.x * 2).norm()
    bad = reference(lambda v: v * float('nan'), b, maxiter=8, rtol=1e-5).observe(b)
    assert not bad.verified


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('schedule', ['eager', 'internal', 'cached'])
def test_reuse_changed_rhs_coefficients_and_initial_guess(reference, dtype, schedule):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        b = torch.ones(16, device='cuda', dtype=dtype)
        coefficient = torch.full_like(b, 2)
        inverse = coefficient.reciprocal()
        action = lambda v: coefficient * v
        solver = reference(action, b, maxiter=16, rtol=1e-5, precond=inverse, schedule=schedule)
        solver.prepare(b)
        first = solver.observe(b)
        snapshot = first.x.clone()
        coefficient.fill_(4)
        inverse.fill_(.25)
        second = solver.observe(b * 3)
        assert second.verified
        torch.testing.assert_close(second.x, b * .75)
        zero = solver.observe(b * 0, x0=b)
        assert zero.verified
        torch.testing.assert_close(zero.x, b * 0)
        exact = solver.observe(b, x0=b / 4)
        assert exact.verified and exact.scheduled == 0
        torch.testing.assert_close(first.x, snapshot, rtol=0, atol=0)
    side.synchronize()
