"""Generated kernel inputs follow SPH precision rather than Torch's default."""

import pytest
import torch

from warpSPHCore import get_torch_precision
from warpSPHCore.util import generateNeighborTestData


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_generated_lattice_precision(dim):
    expected = get_torch_precision()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64 if expected == torch.float32 else torch.float32)
        positions, supports, n, domain, dx = generateNeighborTestData(8, 20, dim, False, 'cpu')
        assert n == 8 ** dim and positions.shape == (n, dim)
        assert all(t.dtype == expected for t in (positions, supports, domain.min, domain.max))
        assert domain.periodic.dtype == torch.bool
        assert dx == .25
        assert positions.isfinite().all() and (supports > 0).all()
    finally:
        torch.set_default_dtype(original)
