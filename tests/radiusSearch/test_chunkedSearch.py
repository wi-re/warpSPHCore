"""Direct chunk construction, row identity, resident reuse and filtering parity."""
from dataclasses import replace
import importlib

import pytest
import torch
import warp as wp

from warpSPHCore import DomainDescription, ParticleState, SupportScheme
from warpSPHCore.radiusSearch.chunked import (buildChunkedVerletList_, buildChunkedVerletList,
    radiusSearchRowChunks_, radiusSearchRowChunks, filterChunkedVerletList_, filterChunkedVerletList,
    chunkedVerletStaleFlag_)
from warpSPHCore.radiusSearch.compactHash.wrapper import radiusSearchCompactHashMap_
from warpSPHCore.radiusSearch.rowChunks import RowChunkPolicy
from warpSPHCore.radiusSearch.verlet.build import buildVerletList_, deferVerletChecks
from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_
from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


def sample(device, dim=2, periodic=True, asymmetric=False):
    dtype = get_torch_precision()
    axis = torch.linspace(.04, .94, 4, dtype=dtype, device=device)
    x = torch.cartesian_prod(*([axis]*dim))
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(41)).to(device)
    x = x[permutation].contiguous()
    h = torch.linspace(.20, .27, len(x), device=device, dtype=dtype)
    if periodic:
        x[::5] += 3  # wrapped coordinates must match the existing flat traversal
    if asymmetric:
        q = x[::2].flip(0).contiguous()
        q[::5] += .08 if periodic else 10
        qh = h[::2].flip(0).contiguous() * .8
    else:
        q, qh = x, h
    domain = DomainDescription(torch.zeros(dim, device=device, dtype=dtype),
                               torch.ones(dim, device=device, dtype=dtype),
                               torch.full((dim,), periodic, device=device, dtype=torch.bool), dim)
    return q, x, qh, h, domain


def policy(**kwargs):
    values = dict(edgeBudget=64, bytesPerEdge=8, maxRows=3)
    values.update(kwargs)
    return RowChunkPolicy(**values)


def flat(args, mode, storeI=False):
    q, r, qh, rh, d = args
    return radiusSearchCompactHashMap_(q, r, qh, rh, d.periodic, d, mode,
                                      len(q)+1, capacity=None, queryOrder='input', storeI=storeI)


def equalRows(actual, expected):
    assert actual.numRows == expected.numRows and actual.numCols == expected.numCols
    assert actual.numEdges == expected.numEdges
    assert actual.numNeighbors.dtype == torch.int32
    assert actual.globalPrefix.dtype == torch.int64
    assert actual.globalPrefix.tolist() == expected.edgeOffsets.tolist() + [expected.numEdges]
    assert torch.equal(actual.numNeighbors, expected.numNeighbors)
    for chunk in actual:
        m, a = chunk.metadata, chunk.adjacency
        assert m.owner == actual.owner and m.rowCount == a.numRows
        assert a.edgeOffsets.dtype == get_torch_offset_dtype()
        assert a.paddedJ.dtype == get_torch_index_dtype()
        assert len(a.paddedJ) == m.capacity <= actual.plan.policy.capacityLimit
        assert a.numEdges == m.liveEdges
        assert (a.paddedI is not None) == actual.storeI
        for local in range(m.rowCount):
            i = m.rowStart + local
            count = int(expected.numNeighbors[i])
            base, old = int(a.edgeOffsets[local]), int(expected.edgeOffsets[i])
            assert base == old - m.edgeStart
            assert torch.equal(a.paddedJ[base:base+count], expected.paddedJ[old:old+count])
            if actual.storeI:
                assert bool((a.paddedI[base:base+count] == i).all())
        assert bool((a.paddedJ[m.liveEdges:] == -1).all())
        if actual.storeI:
            assert bool((a.paddedI[m.liveEdges:] == -1).all())


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('asymmetric', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
@pytest.mark.parametrize('storeI', [False, True])
def test_direct_collection_matches_every_flat_row(device, dim, periodic, asymmetric, mode, storeI):
    args = sample(device, dim, periodic, asymmetric)
    got = radiusSearchRowChunks_(*args, policy=policy(), mode=mode, storeI=storeI)
    expected = flat(args, mode, storeI)
    equalRows(got, expected)
    assert len(got.chunks) > 1
    assert [i for c in got for i in range(c.metadata.rowStart, c.metadata.rowStop)] == list(range(len(args[0])))


@pytest.mark.parametrize('nq,nr', [(0, 0), (0, 4), (4, 0)])
@pytest.mark.parametrize('storeI', [False, True])
def test_empty_populations_own_zero_rows_without_pair_arrays(device, nq, nr, storeI):
    q, r, qh, rh, d = sample(device)
    args = q[:nq], r[:nr], qh[:nq], rh[:nr], d
    got = radiusSearchRowChunks_(*args, policy=policy(), storeI=storeI)
    expected = flat(args, SupportScheme.Gather, storeI)
    equalRows(got, expected)
    assert all(c.metadata.capacity == 0 for c in got)
    assert got.memoryUsage()['pairCapacityBytes'] == 0


def test_logical_owners_count_disjoint_ranges_and_allocate_only_their_chunks(device, monkeypatch):
    module = importlib.import_module('warpSPHCore.radiusSearch.chunked')
    args = sample(device)
    expected = flat(args, SupportScheme.Gather)
    countRows = []
    original = module.wp.launch
    def launch(kernel, *a, **kw):
        if kernel == module.radiusSearchCountNeighborsCompactHashMap:
            countRows.append(kw['dim'])
        return original(kernel, *a, **kw)
    monkeypatch.setattr(module.wp, 'launch', launch)
    results = [radiusSearchRowChunks_(*args, policy=policy(maxRows=6), numOwners=23, owner=owner)
               for owner in range(23)]
    assert len({p.plan.signature for p in results}) == 1
    assert sum(countRows) == len(args[0])*23  # each logical replica repeats counting, never sums duplicates
    covered = []
    for got in results:
        equalRows(got, expected)
        covered.extend(i for c in got for i in range(c.metadata.rowStart, c.metadata.rowStop))
    assert sorted(covered) == list(range(len(args[0])))
    assert any(not p.chunks for p in results)


def test_allocation_is_bounded_and_flat_search_is_never_called(device, monkeypatch):
    module = importlib.import_module('warpSPHCore.radiusSearch.chunked')
    flatModule = importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
    monkeypatch.setattr(flatModule, 'radiusSearchOnCompactHashMap', lambda *a, **k: pytest.fail('monolithic search'))
    allocations = []
    original = torch.empty
    def allocate(*a, **kw):
        if a and isinstance(a[0], int) and kw.get('dtype') == get_torch_index_dtype():
            allocations.append(a[0])
        return original(*a, **kw)
    monkeypatch.setattr(torch, 'empty', allocate)
    got = radiusSearchRowChunks_(*sample(device, 3), policy=policy(), storeI=False)
    assert got.numEdges > got.plan.policy.capacityLimit
    assert allocations == [c.metadata.capacity for c in got]
    assert max(allocations) <= got.plan.policy.capacityLimit
    assert all(c.adjacency.paddedI is None for c in got)
    report = got.memoryUsage()
    assert report['pairCapacityBytes'] == sum(allocations)*torch.empty((), dtype=get_torch_index_dtype()).element_size()
    assert report['pairCapacityBytes'] <= report['pairBudgetBytes']
    assert report['searchScratchBytes'] > 0 and report['hostPrefixBytes'] > 0
    assert not hasattr(got, 'paddedJ') and not hasattr(got, 'paddedI')


def test_signature_disagreement_fails_before_pair_allocation(device, monkeypatch):
    module = importlib.import_module('warpSPHCore.radiusSearch.chunked')
    monkeypatch.setattr(module, '_allocateChunks', lambda *a, **k: pytest.fail('allocation before signature agreement'))
    with pytest.raises(ValueError, match='signature mismatch'):
        radiusSearchRowChunks_(*sample(device), policy=policy(), expectedSignature='different-generation')


def test_count_and_collect_use_frozen_inputs(device, monkeypatch):
    module = importlib.import_module('warpSPHCore.radiusSearch.chunked')
    args = sample(device)
    expected = flat(args, SupportScheme.Gather)
    original = module.buildRowChunkPlan
    def planned(*a, **kw):
        result = original(*a, **kw)
        # Mutating the caller's positions/supports between count and collection
        # cannot change the generation's private grid/query snapshots.
        args[0].add_(.4)
        args[2].mul_(2)
        return result
    monkeypatch.setattr(module, 'buildRowChunkPlan', planned)
    got = radiusSearchRowChunks_(*args, policy=policy())
    equalRows(got, expected)
    assert not torch.equal(got.queryPositions, args[0])
    assert not torch.equal(got.querySupports, args[2])


@pytest.mark.parametrize('storeI', [False, True])
def test_resident_reuse_forced_rebuild_and_snapshot_lifetime(device, storeI):
    args = sample(device)
    kwargs = dict(policy=policy(), verletScale=1.4, storeI=storeI)
    a = buildChunkedVerletList_(*args, **kwargs)
    assert a.plan.generation.number == 0
    assert a.queryPositions is a.referencePositions and a.querySupports is a.referenceSupports
    assert buildChunkedVerletList_(*args, priorNeighborhood=a, **kwargs) is a
    assert not bool(chunkedVerletStaleFlag_(*args, a))
    moved = (args[0]+.001, args[1]+.001, args[2], args[3], args[4])
    assert buildChunkedVerletList_(*moved, priorNeighborhood=a, **kwargs) is a
    snapshots = a.queryPositions.clone()
    b = buildChunkedVerletList_(*args, priorNeighborhood=a, forceRebuild=True, inPlace=True, **kwargs)
    assert b is not a and b.plan.generation.number == 1 and b.rebuildReason == 'forced rebuild'
    for x, y in zip(a, b):
        assert x.adjacency.paddedJ.data_ptr() == y.adjacency.paddedJ.data_ptr()
        assert x.adjacency.edgeOffsets.data_ptr() == y.adjacency.edgeOffsets.data_ptr()
        if storeI:
            assert x.adjacency.paddedI.data_ptr() == y.adjacency.paddedI.data_ptr()
    q, r, qh, rh, d = args
    equalRows(b, flat((q, r, qh*1.4, rh*1.4, d), SupportScheme.SuperSymmetric, storeI))
    assert torch.equal(a.queryPositions, snapshots)
    assert b.queryPositions.data_ptr() != a.queryPositions.data_ptr()


@pytest.mark.parametrize('change', ['motion', 'support', 'scale', 'mode', 'domain', 'policy', 'layout', 'ordering', 'population'])
def test_invalidation_and_capacity_changes_preserve_old_default_buffers(device, change):
    args = list(sample(device))
    kwargs = dict(policy=policy(edgeBudget=256), verletScale=1.4, mode=SupportScheme.SuperSymmetric)
    a = buildChunkedVerletList_(*args, **kwargs)
    saved = [c.adjacency.paddedJ.clone() for c in a]
    oldSignature = a.plan.signature
    if change == 'motion':
        args[0] = args[0]+.35
        args[1] = args[0]
    elif change == 'support':
        args[2] = args[2]*2
        args[3] = args[2]
    elif change == 'scale':
        kwargs['verletScale'] = 1.6
    elif change == 'mode':
        kwargs['mode'] = SupportScheme.Gather
    elif change == 'domain':
        args[4].max.add_(.1)
    elif change == 'policy':
        kwargs['policy'] = policy(edgeBudget=128, maxRows=1)
    elif change == 'layout':
        kwargs['storeI'] = True
    elif change == 'ordering':
        kwargs['orderingKey'] = 'new-particle-order'
    elif change == 'population':
        args[:4] = [t[:7] for t in args[:4]]
    b = buildChunkedVerletList_(*args, priorNeighborhood=a, **kwargs)
    assert b is not a and b.plan.generation.number == 1 and b.plan.signature != oldSignature
    assert all(torch.equal(c.adjacency.paddedJ, before) for c, before in zip(a, saved))
    q, r, qh, rh, d = args
    equalRows(b, flat((q, r, qh*kwargs['verletScale'], rh*kwargs['verletScale'], d), kwargs['mode'], kwargs.get('storeI', False)))


def test_periodic_image_crossing_retains_resident_generation(device):
    q, r, qh, rh, d = sample(device)
    a = buildChunkedVerletList_(q, r, qh, rh, d, policy=policy(), verletScale=1.4)
    shifted = q + (d.max-d.min)*3
    assert buildChunkedVerletList_(shifted, shifted, qh, rh, d, policy=policy(), verletScale=1.4, priorNeighborhood=a) is a


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('asymmetric', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
@pytest.mark.parametrize('storeI', [False, True])
def test_filter_matches_flat_verlet_order_with_new_chunk_boundaries(device, dim, asymmetric, mode, storeI):
    args = sample(device, dim, True, asymmetric)
    cap = policy(edgeBudget=512, maxRows=1)
    old = buildChunkedVerletList_(*args, policy=cap, verletScale=1.4, mode=mode, storeI=storeI)
    flatOld = buildVerletList_(*args, verletScale=1.4, mode=mode, storeI=storeI)
    got = filterChunkedVerletList_(*args, old, policy=policy(edgeBudget=512, maxRows=7), storeI=storeI)
    expected = filterVerletList_(*args, flatOld, mode, capacity=None, storeI=storeI)
    equalRows(got, expected)
    assert got.sourceSignature == old.plan.signature and got.kind == 'filtered'
    assert len(got.chunks) < len(old.chunks)
    assert all(c.adjacency.paddedJ.data_ptr() not in {s.adjacency.paddedJ.data_ptr() for s in old} for c in got if c.metadata.capacity)


def test_stale_filter_source_is_rejected_and_filter_can_change_i_layout(device):
    args = sample(device)
    a = buildChunkedVerletList_(*args, policy=policy(), verletScale=1.4)
    filtered = filterChunkedVerletList_(*args, a, storeI=True)
    assert filtered.storeI and all(c.adjacency.paddedI is not None for c in filtered)
    q, r, qh, rh, d = args
    with pytest.raises(ValueError, match='stale'):
        filterChunkedVerletList_(q+.4, r+.4, qh, rh, d, a)
    with pytest.raises(ValueError, match='resident chunked Verlet'):
        filterChunkedVerletList_(*args, filtered)


def test_oversized_row_and_unsupported_contexts_fail_explicitly(device):
    args = sample(device)
    with pytest.raises(OverflowError, match='Rows cannot be split'):
        radiusSearchRowChunks_(*args, policy=policy(edgeBudget=1))
    with pytest.raises(NotImplementedError, match='AD'):
        radiusSearchRowChunks_(args[0].clone().requires_grad_(), *args[1:], policy=policy())
    with deferVerletChecks(), pytest.raises(NotImplementedError, match='graph'):
        buildChunkedVerletList_(*args, policy=policy())
    with pytest.raises(ValueError, match='bytesPerEdge'):
        radiusSearchRowChunks_(*args, policy=policy(bytesPerEdge=1))
    with pytest.raises(ValueError, match='owner'):
        radiusSearchRowChunks_(*args, policy=policy(), numOwners=2, owner=2)


def test_particle_wrappers_keep_owned_rows_explicit(device):
    q, r, qh, rh, d = sample(device)
    state = ParticleState(positions=q, supports=qh, masses=torch.ones_like(qh),
                          kinds=torch.zeros(len(q), dtype=torch.int32, device=device))
    exact = radiusSearchRowChunks(state, d, policy=policy())
    equalRows(exact, flat((q, r, qh, rh, d), SupportScheme.Gather))
    old = buildChunkedVerletList(state, d, policy=policy(), verletScale=1.4,
                                supportMode=SupportScheme.SuperSymmetric)
    filtered = filterChunkedVerletList(state, d, old)
    flatOld = buildVerletList_(q, r, qh, rh, d, verletScale=1.4)
    equalRows(filtered, filterVerletList_(q, r, qh, rh, d, flatOld, SupportScheme.SuperSymmetric, capacity=None))


def test_nonempty_populations_with_only_empty_rows_and_filter(device):
    q, r, qh, rh, d = sample(device, periodic=False, asymmetric=True)
    q = q + 100
    args = q, r, qh, rh, d
    old = buildChunkedVerletList_(*args, policy=policy(), verletScale=1.4)
    assert old.numEdges == 0 and len(old.chunks) > 1
    filtered = filterChunkedVerletList_(*args, old, policy=policy(maxRows=9))
    assert filtered.numEdges == 0
    equalRows(filtered, flat(args, SupportScheme.SuperSymmetric))


def test_particle_wrapper_preserves_the_existing_gather_default(device):
    q, r, qh, rh, d = sample(device)
    state = ParticleState(q, qh, torch.ones_like(qh), torch.zeros(len(q), dtype=torch.int32, device=device))
    old = buildChunkedVerletList(state, d, policy=policy(), verletScale=1.4)
    assert old.mode == SupportScheme.Gather
    equalRows(old, flat((q, r, qh*1.4, rh*1.4, d), SupportScheme.Gather))


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', list(SupportScheme))
def test_filter_reads_fresh_particle_fields_within_the_validity_buffer(device, dim, mode):
    args = sample(device, dim, True, True)
    old = buildChunkedVerletList_(*args, policy=policy(edgeBudget=512), verletScale=1.4, mode=mode)
    flatOld = buildVerletList_(*args, verletScale=1.4, mode=mode)
    q, r, qh, rh, d = args
    moved = q+.001*torch.sin(q), r+.002*torch.cos(r), qh*.95, rh*.97, d
    assert not bool(chunkedVerletStaleFlag_(*moved, old))
    filtered = filterChunkedVerletList_(*moved, old, policy=policy(edgeBudget=512, maxRows=5))
    equalRows(filtered, filterVerletList_(*moved, flatOld, mode, capacity=None))


def test_root_exports_remain_explicitly_opt_in():
    from warpSPHCore import (ChunkedAdjacency, OwnedRowChunk, RowChunkPolicy,
                            radiusSearchRowChunks, buildChunkedVerletList, filterChunkedVerletList)
    assert callable(radiusSearchRowChunks) and callable(buildChunkedVerletList) and callable(filterChunkedVerletList)
    assert ChunkedAdjacency.__name__ == 'ChunkedAdjacency' and OwnedRowChunk.__name__ == 'OwnedRowChunk'
    assert RowChunkPolicy.__name__ == 'RowChunkPolicy'


def test_oversized_rebuild_does_not_mutate_resident_pairs(device):
    q, r, qh, rh, d = sample(device, 3)
    old = buildChunkedVerletList_(q, r, qh, rh, d, policy=policy(), verletScale=1.4)
    saved = [c.adjacency.paddedJ.clone() for c in old]
    with pytest.raises(OverflowError, match='Rows cannot be split'):
        buildChunkedVerletList_(q, r, qh*10, rh*10, d, policy=policy(), verletScale=1.4,
                               priorNeighborhood=old, forceRebuild=True, inPlace=True)
    assert all(torch.equal(c.adjacency.paddedJ, before) for c, before in zip(old, saved))


def test_filter_never_materializes_coo_and_can_split_source_chunks(device, monkeypatch):
    from warpSPHCore import AdjacencyList
    args = sample(device, 3)
    old = buildChunkedVerletList_(*args, policy=policy(edgeBudget=512, maxRows=9), verletScale=1.4)
    flatOld = buildVerletList_(*args, verletScale=1.4, storeI=False)
    expected = filterVerletList_(*args, flatOld, SupportScheme.SuperSymmetric, capacity=None, storeI=False)
    monkeypatch.setattr(AdjacencyList, 'cooRows', lambda *a, **k: pytest.fail('hidden COO allocation'))
    filtered = filterChunkedVerletList_(*args, old, policy=policy(maxRows=1))
    equalRows(filtered, expected)
    assert len(filtered.chunks) > len(old.chunks)


def test_nondefault_cuda_stream_orders_grid_counts_collection_and_filter(device):
    if device.type != 'cuda':
        pytest.skip('CUDA stream control')
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        args = sample(device, 3)
        old = buildChunkedVerletList_(*args, policy=policy(), verletScale=1.4)
        filtered = filterChunkedVerletList_(*args, old)
        flatOld = buildVerletList_(*args, verletScale=1.4)
        expected = filterVerletList_(*args, flatOld, SupportScheme.SuperSymmetric, capacity=None)
    torch.cuda.current_stream(device).wait_stream(stream)
    equalRows(filtered, expected)
