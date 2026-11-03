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


def _moved(state, shift):
    return ParticleState(positions=state.positions + shift, supports=state.supports, masses=state.masses,
                         densities=state.densities, kinds=state.kinds)


def test_inPlaceRebuildKeepsTheBuffers(device):
    state, domain = _particles(device)
    policy = CapacityPolicy(headroom=0.5, inPlace=True)          # generous headroom: the new count fits
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric, capacity=policy)
    ptrs = [t.data_ptr() for t in (first.paddedI, first.paddedJ, first.numNeighbors, first.edgeOffsets)]
    moved = _moved(state, 10.0 * state.supports[0])               # far beyond the Verlet margin: a rebuild
    second = buildVerletList(moved, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                             priorNeighborhood=first, capacity=policy)
    assert second is not first
    assert [t.data_ptr() for t in (second.paddedI, second.paddedJ, second.numNeighbors, second.edgeOffsets)] == ptrs
    # the contents are the rebuilt list: equal to a fresh, exactly sized build of the moved particles
    exact = buildVerletList(moved, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric, capacity=None)
    assert second.numEdges == exact.numEdges
    for a, b in zip(second.livePairs(), exact.livePairs()):
        assert torch.equal(a, b)
    assert torch.equal(second.numNeighbors, exact.numNeighbors) and torch.equal(second.edgeOffsets, exact.edgeOffsets)
    assert bool((second.paddedI[second.numEdges:] == -1).all())


def test_inPlaceRebuildWritesTheReferenceSnapshotInPlaceToo(device):
    state, domain = _particles(device)
    policy = CapacityPolicy(headroom=0.5, inPlace=True)
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric, capacity=policy)
    names = ('queryPositions', 'referencePositions', 'querySupports', 'referenceSupports')
    ptrs = [getattr(first, n).data_ptr() for n in names]
    moved = _moved(state, 10.0 * state.supports[0])
    second = buildVerletList(moved, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                             priorNeighborhood=first, capacity=policy)
    assert [getattr(second, n).data_ptr() for n in names] == ptrs        # same addresses ...
    assert torch.equal(second.queryPositions, moved.positions)           # ... holding the new reference snapshot
    assert torch.equal(second.referencePositions, moved.positions)
    assert torch.allclose(second.querySupports, moved.supports * 1.4)


def test_capturedValidityFollowsAnInPlaceRebuild():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from warpSPHCore import verletUsage
    device = torch.device("cuda")
    state, domain = _particles(device)
    policy = CapacityPolicy(headroom=0.5, inPlace=True)
    held = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric, capacity=policy)
    probe = state.positions.clone()                                      # static input of the captured check
    supports = state.supports.clone()
    def usage():
        return verletUsage(probe, supports, held, domain, 1.4)
    usage()                                                              # warm-up
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        usage()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = usage()
    probe.copy_(_moved(state, 0.3 * state.supports[0]).positions)
    graph.replay()
    before = float(captured)
    assert before > 0.0                                                  # the particles moved against the list's snapshot
    rebuilt = buildVerletList(_moved(state, 0.3 * state.supports[0]), domain, verletScale=1.4,
                              supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=held, capacity=policy,
                              forceRebuild=True)
    assert rebuilt.queryPositions.data_ptr() == held.queryPositions.data_ptr()
    graph.replay()                                                       # the old graph, the new snapshot
    assert float(captured) == pytest.approx(0.0, abs=1e-6)               # the list is now fresh for these particles


def test_inPlaceRebuildThatOutgrowsTheCapacityReallocates(device):
    state, domain = _particles(device)
    tight = CapacityPolicy(headroom=0.0, inPlace=True)
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric, capacity=tight)
    squeezed = ParticleState(positions=state.positions * 0.5, supports=state.supports, masses=state.masses,
                             densities=state.densities, kinds=state.kinds)       # denser: more neighbours
    second = buildVerletList(squeezed, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                             priorNeighborhood=first, capacity=tight)
    assert second.numEdges > first.paddedI.shape[0]
    assert second.paddedI.data_ptr() != first.paddedI.data_ptr()
    assert second.paddedI.shape[0] >= second.numEdges


def test_defaultPolicyNeverSharesBuffers(device):
    state, domain = _particles(device)
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric)
    second = buildVerletList(_moved(state, 10.0 * state.supports[0]), domain, verletScale=1.4,
                             supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=first)
    assert second.paddedI.data_ptr() != first.paddedI.data_ptr()


def test_rebuildSlackRebuildsEarly(device):
    state, domain = _particles(device)
    first = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric)
    # all particles move: the check sums the query and the reference maximum, so this uses 60 % of the buffer
    buffer = 0.4 * float(state.supports.min())
    drifted = _moved(state, torch.tensor([0.3 * buffer, 0.0], device=device))
    kept = buildVerletList(drifted, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                           priorNeighborhood=first)
    assert kept is first                                           # 60 % of the buffer: still valid
    early = buildVerletList(drifted, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric,
                            priorNeighborhood=first, rebuildSlack=0.5)
    assert early is not first                                      # but past half of it: rebuilt early
