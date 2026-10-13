"""The warp validity check decides like the torch reference (`validCheck._verlet_validity_metrics`)."""

import pytest
import torch
import warp as wp

from warpSPHCore import ParticleState, SupportScheme
from warpSPHCore.radiusSearch.verlet.validCheck import _verlet_validity_metrics
from warpSPHCore.radiusSearch.verlet.wp_validity import verletValidityMetrics
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(scope="module", autouse=True)
def _init_warp_once():
    wp.init()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("supportCase", [0, 1, 2])
@pytest.mark.parametrize("sameSet", [True, False])
def test_matchesTheTorchReference(device, periodic, supportCase, sameSet):
    positions, supports, n, domain, dx = generateNeighborTestData(10, 20, 2, periodic, device)
    g = torch.Generator(device="cpu").manual_seed(5)
    scale = 1.4
    prior = positions.clone()
    priorSupports = supports * scale
    moved = positions + 0.3 * dx * (torch.rand(positions.shape, generator=g) - 0.5).to(device)
    if periodic:                                     # push a few particles across the periodic boundary
        moved[:3] += (domain.max - domain.min)
    newSupports = supports * (1.0 + 0.05 * torch.rand(n, generator=g).to(device))
    if sameSet:
        q = r = moved
        pq = pr = prior
        sq = sr = newSupports
        sa = sb = priorSupports
    else:
        q, r = moved, moved.flip(0).contiguous()
        pq, pr = prior, prior.flip(0).contiguous()
        sq, sr = newSupports, newSupports.flip(0).contiguous()
        sa, sb = priorSupports, priorSupports.flip(0).contiguous()
    args = (q, r, pq, pr, sq, sr, sa, sb, domain.periodic, domain.min, domain.max, scale, supportCase)
    ref = _verlet_validity_metrics(*args)
    got = verletValidityMetrics(*args)
    for name, a, b in zip(("flag", "maxDistance", "supportFactor", "minSupport", "buffer"), ref, got):
        assert torch.allclose(a.float().cpu(), b.float().cpu(), rtol=1e-5, atol=1e-7), name
    # slack: rebuild earlier, never later
    slack = verletValidityMetrics(*args, rebuildSlack=0.9)[0]
    assert bool(slack) or not bool(ref[0])
