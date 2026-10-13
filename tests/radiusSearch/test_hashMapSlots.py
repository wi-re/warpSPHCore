"""The fixed-shape hash map (`compactHash/slots.py`) finds the neighbours the eager build finds."""

import pytest
import torch
import warp as wp

from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch.compactHash import GraphedBuild, buildCompactHashMap, buildFixed, makeHashMapSlots
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(scope="module", autouse=True)
def _init_warp_once():
    wp.init()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _case(device, periodic, dim=2, nx=14):
    positions, supports, n, domain, dx = generateNeighborTestData(nx, 20, dim, periodic, device)
    g = torch.Generator().manual_seed(2)
    positions = positions + 0.2 * dx * (torch.rand(positions.shape, generator=g) - 0.5).to(device)
    supports = supports * (1.0 + 0.2 * torch.rand(n, generator=g).to(device))
    return positions, supports, n, domain


def _rows(adj, n):
    i, j = adj.livePairs()
    out = [[] for _ in range(n)]
    for a, b in zip(i.tolist(), j.tolist()):
        out[a].append(b)
    return out


@pytest.mark.parametrize("periodic", [False, True])
def test_sameGridSameCsrOrderAsTheEagerBuild(device, periodic):
    positions, supports, n, domain = _case(device, periodic)
    mode = SupportScheme.SuperSymmetric
    H = n + 1
    eager = buildCompactHashMap(positions, positions, supports, supports, domain.periodic, domain, mode, H)
    slots = makeHashMapSlots(n, domain, float(supports.max()), device=device, mode=mode, hashMapLength=H)
    slots.positions.copy_(positions)
    slots.supports.copy_(supports)
    fixed = buildFixed(slots)
    assert bool(slots.fits) and fixed.hCell == pytest.approx(eager.hCell, rel=1e-6)
    ref = radiusSearchOnCompactHashMap(eager, positions, positions, supports, supports, domain.periodic, domain, mode, H)
    got = radiusSearchOnCompactHashMap(fixed, positions, positions, supports, supports, domain.periodic, domain, mode, H)
    assert got.numEdges == ref.numEdges
    assert _rows(got, n) == _rows(ref, n)                     # same neighbours in the same order, row by row


def test_largerRadiusGivesTheSameNeighbourSets(device):
    positions, supports, n, domain = _case(device, False)
    mode = SupportScheme.SuperSymmetric
    H = n + 1
    eager = buildCompactHashMap(positions, positions, supports, supports, domain.periodic, domain, mode, H)
    ref = radiusSearchOnCompactHashMap(eager, positions, positions, supports, supports, domain.periodic, domain, mode, H)
    slots = makeHashMapSlots(n, domain, 1.6 * float(supports.max()), device=device, mode=mode, hashMapLength=H)
    slots.positions.copy_(positions)
    slots.supports.copy_(supports)
    got = radiusSearchOnCompactHashMap(buildFixed(slots), positions, positions, supports, supports, domain.periodic, domain, mode, H)
    assert [sorted(r) for r in _rows(got, n)] == [sorted(r) for r in _rows(ref, n)]


def test_fitsFlagFalseWhenASupportOutgrowsTheRadius(device):
    positions, supports, n, domain = _case(device, False)
    slots = makeHashMapSlots(n, domain, float(supports.max()), device=device)
    slots.positions.copy_(positions)
    slots.supports.copy_(supports * 1.5)
    buildFixed(slots)
    assert not bool(slots.fits)


def test_graphedReplayMatchesEagerAfterTheParticlesMove():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    positions, supports, n, domain = _case(device, True)
    mode = SupportScheme.SuperSymmetric
    H = n + 1
    slots = makeHashMapSlots(n, domain, 1.2 * float(supports.max()), device=device, mode=mode, hashMapLength=H)
    graphed = GraphedBuild(slots)
    for step in range(4):                                     # warm-up, capture, replays
        moved = positions + 0.01 * step * torch.sin(7.0 * positions.flip(-1) + step)
        slots.positions.copy_(moved)
        slots.supports.copy_(supports)
        got = radiusSearchOnCompactHashMap(graphed.replay(), moved, moved, supports, supports, domain.periodic, domain, mode, H)
        slots2 = makeHashMapSlots(n, domain, 1.2 * float(supports.max()), device=device, mode=mode, hashMapLength=H)
        slots2.positions.copy_(moved)
        slots2.supports.copy_(supports)
        ref = radiusSearchOnCompactHashMap(buildFixed(slots2), moved, moved, supports, supports, domain.periodic, domain, mode, H)
        assert _rows(got, n) == _rows(ref, n), step
    assert graphed.graph is not None
