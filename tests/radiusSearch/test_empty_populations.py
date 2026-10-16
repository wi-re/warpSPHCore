"""Vacuous pair sets still carry meaningful CSR/grid and rebuild contracts."""

import math
from dataclasses import replace

import pytest
import torch
import warp as wp

from warpSPHCore import CapacityPolicy, DomainDescription, ParticleState, SupportScheme, radiusSearchCompactHashMap
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap, buildFixed, makeHashMapSlots, GraphedBuild
from warpSPHCore.radiusSearch.compactHash.wrapper import radiusSearchCompactHashMap_
from warpSPHCore.radiusSearch.verlet.build import buildVerletList_, deferVerletChecks
from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_
from warpSPHCore.radiusSearch.verlet.validCheck import _verlet_validity_metrics
from warpSPHCore.radiusSearch.verlet.wp_validity import verletValidityMetrics
from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    wp.init()
    return torch.device(request.param)


def cloud(n, dim, device):
    dtype = get_torch_precision()
    x = torch.arange(n * dim, dtype=dtype, device=device).reshape(n, dim).sin() * .3
    h = torch.linspace(1.1, 1.4, n, dtype=dtype, device=device)
    return ParticleState(x, h, torch.ones_like(h), torch.zeros(n, dtype=torch.int32, device=device), torch.ones_like(h))


def domain_for(dim, device, periodic=False):
    dtype = get_torch_precision()
    return DomainDescription(torch.full((dim,), -2., dtype=dtype, device=device),
                             torch.full((dim,), 2., dtype=dtype, device=device),
                             torch.full((dim,), periodic, dtype=torch.bool, device=device), dim)


def assert_empty(adj, nq, nr, capacity):
    assert adj.numRows == nq and adj.numCols == nr and adj.numEdges == 0
    assert adj.numNeighbors.shape == adj.edgeOffsets.shape == (nq,)
    assert adj.numNeighbors.dtype == torch.int32 and adj.edgeOffsets.dtype == get_torch_offset_dtype()
    assert adj.paddedI.dtype == adj.paddedJ.dtype == get_torch_index_dtype()
    assert torch.count_nonzero(adj.numNeighbors) == torch.count_nonzero(adj.edgeOffsets) == 0
    assert len(adj.livePairs()[0]) == len(adj.livePairs()[1]) == 0
    assert bool((adj.paddedI == -1).all()) and bool((adj.paddedJ == -1).all())
    if capacity is None:
        assert adj.paddedI.numel() == adj.paddedJ.numel() == 0
    else:
        assert adj.paddedI.numel() == adj.paddedJ.numel() >= 1


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
def test_empty_search_csr_and_grid_contract(device, dim, sizes, periodic, mode):
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D qualification runs on CUDA')
    nq, nr = sizes
    q, r = cloud(nq, dim, device), cloud(nr, dim, device)
    domain = domain_for(dim, device, periodic)
    for collection in ('fused', 'split'):
        for capacity in (None, CapacityPolicy()):
            adj = radiusSearchCompactHashMap(q, domain, referenceParticles=r, mode=mode,
                                             collection=collection, capacity=capacity)
            assert_empty(adj, nq, nr, capacity)
            grid = adj.hashMap
            assert grid.sortedPositions.shape == (nr, dim) and grid.sortedSupports.shape == (nr,)
            assert grid.sortIndex.shape == (nr,) and grid.sortIndex.dtype == get_torch_index_dtype()
            assert math.isfinite(grid.hCell) and grid.hCell > 0
            assert bool((grid.numCells > 0).all())
            if nr == 0:
                assert grid.sortedCellTable.shape == (0, 3)
                assert torch.count_nonzero(grid.hashTable[:, 1]) == 0


@pytest.mark.parametrize('missing', ['min', 'max', 'both'])
def test_empty_reference_requires_explicit_bounds(device, missing):
    q, r, domain = cloud(3, 2, device), cloud(0, 2, device), domain_for(2, device)
    domain = replace(domain, **{key: None for key in ('min', 'max') if missing in (key, 'both')})
    with pytest.raises(ValueError, match='[Ee]mpty reference.*[Bb]ounds'):
        radiusSearchCompactHashMap(q, domain, referenceParticles=r)


def test_empty_query_can_use_bounds_inferred_from_reference(device):
    q, r = cloud(0, 2, device), cloud(5, 2, device)
    domain = replace(domain_for(2, device), min=None, max=None)
    assert_empty(radiusSearchCompactHashMap(q, domain, referenceParticles=r), 0, 5, CapacityPolicy())


@pytest.mark.parametrize('mode', list(SupportScheme))
def test_empty_rebuild_transitions_and_fixed_addresses(device, mode):
    q, r, domain = cloud(3, 2, device), cloud(5, 2, device), domain_for(2, device)
    empty_q, empty_r = cloud(0, 2, device), cloud(0, 2, device)
    policy = CapacityPolicy(inPlace=True, shrinkBelow=0)

    def search(query, reference, prior=None):
        return radiusSearchCompactHashMap_(query.positions, reference.positions, query.supports,
            reference.supports, domain.periodic, domain, mode, capacity=policy, reuse=prior)

    original = search(q, r)
    held_pairs = [original.paddedI.data_ptr(), original.paddedJ.data_ptr()]
    empty = search(q, empty_r, original)
    assert_empty(empty, 3, 0, policy)
    assert held_pairs == [empty.paddedI.data_ptr(), empty.paddedJ.data_ptr()]
    restored = search(q, r, empty)
    oracle = search(q, r)
    for name in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets'):
        torch.testing.assert_close(getattr(restored, name), getattr(oracle, name), rtol=0, atol=0)
    assert_empty(search(empty_q, r, restored), 0, 5, policy)

    prior = None
    for query, reference in ((q, r), (q, empty_r), (empty_q, r), (empty_q, empty_r), (q, r)):
        adj = buildVerletList_(query.positions, reference.positions, query.supports, reference.supports,
            domain, verletScale=1.3, mode=mode, priorNeighborhood=prior, capacity=policy)
        assert (adj.numRows, adj.numCols) == (len(query.positions), len(reference.positions))
        if not len(query.positions) or not len(reference.positions):
            assert_empty(adj, len(query.positions), len(reference.positions), policy)
            filtered = filterVerletList_(query.positions, reference.positions, query.supports,
                                         reference.supports, domain, adj, mode)
            assert_empty(filtered, adj.numRows, adj.numCols, CapacityPolicy())
        prior = adj


@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('support_case', [0, 1, 2])
def test_empty_verlet_validity_is_vacuous_and_finite(device, sizes, support_case):
    q, r, domain = cloud(sizes[0], 2, device), cloud(sizes[1], 2, device), domain_for(2, device)
    args = (q.positions + 100, r.positions - 100, q.positions, r.positions,
            q.supports * 10, r.supports * 10, q.supports * 1.3, r.supports * 1.3,
            domain.periodic, domain.min, domain.max, 1.3, support_case)
    for fn in (_verlet_validity_metrics, verletValidityMetrics):
        flag, *stats = fn(*args)
        assert flag.shape == () and flag.dtype == torch.bool and not bool(flag)
        for stat in stats:
            assert stat.shape == () and stat.device == q.positions.device and stat.dtype == get_torch_precision()
            assert torch.isfinite(stat) and stat == 0
    adj = buildVerletList_(q.positions, r.positions, q.supports, r.supports, domain, verletScale=1.3)
    with deferVerletChecks() as flags:
        held = buildVerletList_(q.positions, r.positions, q.supports, r.supports, domain,
                                verletScale=1.3, priorNeighborhood=adj)
    assert held is adj and len(flags) == 1 and not bool(flags[0])


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_zero_particle_fixed_grid(device, dim):
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D qualification runs on CUDA')
    domain = domain_for(dim, device, True)
    slots = makeHashMapSlots(0, domain, 1.4, device=device)
    grid = buildFixed(slots)
    assert bool(slots.fits) and grid.sortedPositions.shape == (0, dim)
    assert torch.count_nonzero(grid.hashTable[:, 1]) == 0
    if device.type == 'cuda':
        replay = GraphedBuild(slots)
        for _ in range(3):
            assert replay.replay() is grid
        assert replay.graph is not None


@pytest.mark.parametrize('sizes,prior_sizes', [((3, 0), (3, 5)), ((3, 5), (3, 0)),
                                             ((0, 5), (3, 5)), ((3, 5), (0, 5))])
def test_verlet_birth_and_death_invalidate_before_reduction(device, sizes, prior_sizes):
    q, r = cloud(sizes[0], 2, device), cloud(sizes[1], 2, device)
    pq, pr = cloud(prior_sizes[0], 2, device), cloud(prior_sizes[1], 2, device)
    domain = domain_for(2, device)
    for fn in (_verlet_validity_metrics, verletValidityMetrics):
        flag, *stats = fn(q.positions, r.positions, pq.positions, pr.positions,
            q.supports, r.supports, pq.supports * 1.3, pr.supports * 1.3,
            domain.periodic, domain.min, domain.max, 1.3, 0)
        assert bool(flag) and all(torch.isfinite(s) and s == 0 for s in stats)


@pytest.mark.parametrize('mode', list(SupportScheme))
def test_empty_search_without_support_arrays_preserves_precision(device, mode):
    q, r = cloud(0, 2, device), cloud(0, 2, device)
    adj = radiusSearchCompactHashMap(replace(q, supports=None), domain_for(2, device),
        referenceParticles=replace(r, supports=None), mode=mode)
    assert_empty(adj, 0, 0, CapacityPolicy())
    assert adj.hashMap.sortedSupports.dtype == get_torch_precision()
