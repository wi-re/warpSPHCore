import pytest
import torch
from verify.native_inputs import verify_csr, verify_finite


def test_explicit_csr_verifier():
    offsets = torch.tensor([0, 2, 2])
    counts = torch.tensor([2, 0, 1], dtype=torch.int32)
    indices = torch.tensor([0, 2, 1, -1])  # unused padding is allowed
    verify_csr(offsets, counts, indices, 3)
    for bad_offsets, bad_counts, bad_indices, message in [
        (offsets, torch.tensor([2, -1, 1]), indices, 'nonnegative'),
        (offsets + 1, counts, indices, 'exclusive scan'),
        (offsets, counts, indices[:2], 'capacity'),
        (offsets, counts, torch.tensor([0, 3, 1]), 'reference index'),
    ]:
        with pytest.raises(ValueError, match=message):
            verify_csr(bad_offsets, bad_counts, bad_indices, 3)


def test_explicit_finite_verifier():
    verify_finite(torch.ones(3))
    with pytest.raises(ValueError, match='finite'):
        verify_finite(torch.tensor([float('nan')]))
