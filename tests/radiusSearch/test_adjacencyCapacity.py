"""Capacity-padded adjacency (ADJACENCY_CAPACITY_PLAN.md): the flat pair arrays are `paddedI` / `paddedJ`,
longer than the real edge count, with -1 in both beyond `numEdges`; the CSR view never touches the padding and
`livePairs()` is the exact unpadded list."""

import pytest
import torch
import warp as wp

from warpSPHCore import (
    CapacityPolicy,
    ParticleState,
    SupportScheme,
    buildVerletList,
    radiusSearchCompactHashMap,
)
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(scope="module", autouse=True)
def _init_warp_once():
    wp.init()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _particles(device, nx=12):
    positions, supports, n, domain, dx = generateNeighborTestData(nx, 20, 2, False, device)
    positions = positions + 0.1 * dx * torch.sin(7.0 * positions.flip(-1) + 1.0)
    state = ParticleState(positions=positions, supports=supports, masses=torch.ones(n, device=device),
                          densities=torch.ones(n, device=device),
                          kinds=torch.zeros(n, dtype=torch.int32, device=device))
    return state, domain


def test_oldMembersAreGone(device):
    state, domain = _particles(device)
    adj = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.SuperSymmetric)
    with pytest.raises(AttributeError):
        adj.i
    with pytest.raises(AttributeError):
        adj.j


def test_paddingIsMinusOneAndCountsAgree(device):
    state, domain = _particles(device)
    adj = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.SuperSymmetric)
    n = adj.numEdges
    assert adj.paddedI.shape[0] > n and adj.paddedJ.shape[0] == adj.paddedI.shape[0]
    assert int(adj.numNeighbors.sum()) == n
    assert bool((adj.paddedI[n:] == -1).all()) and bool((adj.paddedJ[n:] == -1).all())
    assert bool((adj.paddedI[:n] >= 0).all()) and bool((adj.paddedJ[:n] >= 0).all())


def test_livePairsEqualTheUnpaddedList(device):
    state, domain = _particles(device)
    padded = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.SuperSymmetric)
    exact = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.SuperSymmetric, capacity=None)
    assert exact.paddedI.shape[0] == exact.numEdges == padded.numEdges
    for a, b in zip(padded.livePairs(), exact.livePairs()):
        assert torch.equal(a, b)
    assert torch.equal(padded.edgeOffsets, exact.edgeOffsets)
    assert torch.equal(padded.numNeighbors, exact.numNeighbors)


def test_csrViewNeverReachesThePadding(device):
    state, domain = _particles(device)
    adj = radiusSearchCompactHashMap(state, domain, mode=SupportScheme.SuperSymmetric)
    last = adj.edgeOffsets + adj.numNeighbors
    assert int(last.max()) == adj.numEdges


def test_capacityPolicy():
    p = CapacityPolicy(headroom=0.1, shrinkBelow=0.5)
    assert p.capacityFor(1000) == 1100
    assert p.capacityFor(0) >= 1
    assert p.capacityFor(900, priorCapacity=1100) == 1100       # fits: keep the shape
    assert p.capacityFor(1100, priorCapacity=1100) == 1100      # exactly full still fits
    assert p.capacityFor(1200, priorCapacity=1100) == 1320      # overflow: grow with headroom
    assert p.capacityFor(400, priorCapacity=1100) == 440        # below half: shrink


def test_verletRebuildKeepsTheCapacityWhileItFits(device):
    state, domain = _particles(device)
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric)
    moved = ParticleState(positions=state.positions + 10.0 * state.supports[0],
                          supports=state.supports, masses=state.masses, densities=state.densities, kinds=state.kinds)
    second = buildVerletList(moved, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                             priorNeighborhood=first)
    assert second is not first                                   # the check forced a rebuild
    if first.paddedI.shape[0] * 0.5 <= second.numEdges <= first.paddedI.shape[0]:
        assert second.paddedI.shape[0] == first.paddedI.shape[0]
