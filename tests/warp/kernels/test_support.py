"""Check the public support conversion against the unit-ball volume."""

import math

import pytest
import torch
import warp as wp

from warpSPHCore.type_config import scalar_t
from warpSPHCore.util import volumeToSupport_warp


@wp.kernel
def support_radii(result: wp.array(dtype=scalar_t)):
    i = wp.tid()
    result[i] = volumeToSupport_warp(scalar_t(0.125), 55, i + 1)


def test_volume_to_support_matches_ball_volume(device):
    result = wp.empty(3, dtype=scalar_t, device=str(device))
    wp.launch(support_radii, 3, outputs=[result], device=str(device))
    actual = wp.to_torch(result)
    expected = actual.new_tensor([
        (55 * 0.125 * math.gamma(1 + dim / 2) / math.pi ** (dim / 2)) ** (1 / dim)
        for dim in (1, 2, 3)
    ])
    torch.testing.assert_close(actual, expected, rtol=4 * torch.finfo(actual.dtype).eps, atol=0)
