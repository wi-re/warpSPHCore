"""The opt-in fused CG chain preserves verified stopping and failure contracts."""

import warnings

import pytest
import torch

from warpSPHCore.autograd.compileGlue import compileGlueEnabled, setCompileGlue
from warpSPHCore.linalg import Status, cg


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compiler gate')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compiled_cg_verified_solution_and_failure_cases(dtype):
    previous = compileGlueEnabled()
    try:
        setCompileGlue(True)
        b = torch.linspace(1, 2, 32, dtype=dtype, device='cuda')
        diagonal = torch.linspace(2, 3, 32, dtype=dtype, device='cuda')
        with warnings.catch_warnings(record=True) as caught:
            result = cg(lambda v: diagonal * v, b, rtol=1e-5 if dtype == torch.float32 else 1e-10, maxiter=64)
            assert result.statusInt() >= 0
            torch.testing.assert_close(result.x, b / diagonal, rtol=2e-5 if dtype == torch.float32 else 2e-10, atol=1e-12)
            assert result.trueResidual <= (1e-5 if dtype == torch.float32 else 1e-10) * b.norm()
            exact = cg(lambda v: v, b, rtol=0, maxiter=8)
            assert exact.statusInt() >= 0 and exact.trueResidual == 0
            rejected = cg(lambda v: -v, b, maxiter=8)
            assert rejected.statusInt() == Status.INDEFINITE
            assert rejected.x.isfinite().all()
            threshold = cg(lambda v: v, b, threshold=.1, maxiter=8)
            assert threshold.statusInt() == Status.THRESHOLD
            zero = cg(lambda v: v, torch.zeros_like(b), b, maxiter=8)
            assert zero.statusInt() >= 0 and zero.trueResidual == 0
            nonfinite = cg(lambda v: v * float('nan'), b, maxiter=8)
            assert nonfinite.statusInt() == Status.INDEFINITE
            assert nonfinite.x.isfinite().all()
        assert not any('compileGlue:' in str(w.message) for w in caught), [str(w.message) for w in caught]
    finally:
        setCompileGlue(previous)
