"""Whole-row planner contract, metadata only: never allocate global pair arrays."""
from dataclasses import FrozenInstanceError, replace
import itertools

import pytest
import torch

from warpSPHCore.radiusSearch.csr import WARP_ARRAY_MAX
from warpSPHCore.radiusSearch.rowChunks import (
    RowChunkGeneration, RowChunkPolicy, buildRowChunkPlan,
)


def plan(values, *, policy=None, owners=1, generation=None):
    return buildRowChunkPlan(torch.tensor(values, dtype=torch.int32),
        generation=generation or RowChunkGeneration(7, WARP_ARRAY_MAX, 'stored-verlet-config'),
        policy=policy or RowChunkPolicy(edgeBudget=8, headroomNumerator=0), numOwners=owners)


def assertCoverage(p, values):
    rows = []
    assert p.globalOffsets().dtype == torch.int64
    assert p.globalOffsets().tolist() == [0] + list(itertools.accumulate(values))
    for index, chunk in enumerate(p.chunks):
        rows.extend(range(chunk.rowStart, chunk.rowStop))
        assert chunk.rowCount > 0
        assert chunk.liveEdges == sum(values[chunk.rowStart:chunk.rowStop])
        assert chunk.liveEdges <= chunk.capacity <= p.policy.capacityLimit
        assert chunk.capacity % p.policy.alignment == 0
        assert 0 <= chunk.owner < p.numOwners
        local = p.localOffsets(index)
        assert local.dtype == torch.int64
        assert (local + chunk.edgeStart).tolist() == list(p.prefix[chunk.rowStart:chunk.rowStop])
        assert local.tolist() == [0] + list(itertools.accumulate(values[chunk.rowStart:chunk.rowStop - 1]))
    assert rows == list(range(len(values)))
    counted = [row for first, stop in p.countingRanges for row in range(first, stop)]
    assert counted == rows
    assert len(p.countingRanges) == p.numOwners


@pytest.mark.parametrize('values', [[], [0], [0]*9, [2, 0, 6, 0, 1], [8, 1, 7, 0], [1]*17])
@pytest.mark.parametrize('owners', [1, 2, 4, 23])
def test_coverage_empty_rows_and_empty_owners(values, owners):
    p = plan(values, owners=owners)
    assertCoverage(p, values)
    assert len({c.owner for c in p.chunks}) == min(owners, len(p.chunks))
    assert [c.owner for c in p.chunks] == sorted(c.owner for c in p.chunks)
    if not values:
        assert p.prefix == (0,) and p.chunks == ()
        assert p.countingRanges == ((0, 0),)*owners


def test_exact_budget_and_forced_row_cap():
    p = plan([2, 0, 6, 0, 1, 7])
    assert [(c.rowStart, c.rowStop, c.liveEdges) for c in p.chunks] == [(0, 4, 8), (4, 6, 8)]
    p = plan([0]*7, policy=RowChunkPolicy(edgeBudget=1, maxRows=2), owners=2)
    assert [c.rowCount for c in p.chunks] == [2, 2, 2, 1]
    assert all(c.capacity == 0 for c in p.chunks)


def test_padding_alignment_and_byte_budget_bound_capacity():
    policy = RowChunkPolicy(edgeBudget=100, byteBudget=128, bytesPerEdge=16,
                           headroomNumerator=1, headroomDenominator=4, alignment=4)
    p = plan([3, 3, 1, 0, 3], policy=policy)
    assertCoverage(p, [3, 3, 1, 0, 3])
    assert [c.liveEdges for c in p.chunks] == [6, 4]
    assert [c.capacity for c in p.chunks] == [8, 8]
    assert policy.capacityFor(1) == 4
    assert policy.capacityLimit == 8
    assert all(c.capacity * policy.bytesPerEdge <= policy.byteBudget for c in p.chunks)


@pytest.mark.parametrize('policy', [
    RowChunkPolicy(edgeBudget=8, headroomNumerator=0),
    RowChunkPolicy(edgeBudget=9),  # padded 9-edge row needs 10
    RowChunkPolicy(byteBudget=35, bytesPerEdge=4, headroomNumerator=0),
    RowChunkPolicy(edgeBudget=9, headroomNumerator=0, alignment=4),
])
def test_oversized_rows_fail_instead_of_splitting(policy):
    with pytest.raises(OverflowError, match='Row 1.*cannot be split'):
        plan([0, 9, 1], policy=policy)


def test_wide_prefix_and_local_work_extent_limits_without_large_allocation():
    values = [WARP_ARRAY_MAX//3]*4 + [0, 11]
    policy = RowChunkPolicy(edgeBudget=WARP_ARRAY_MAX, maxElementsPerEdge=3, headroomNumerator=0)
    p = plan(values, policy=policy, owners=4)
    assertCoverage(p, values)
    assert p.totalEdges > WARP_ARRAY_MAX
    assert p.chunks[-1].edgeStart > WARP_ARRAY_MAX
    assert all(c.capacity * 3 <= WARP_ARRAY_MAX for c in p.chunks)
    with pytest.raises(OverflowError, match='Row 0'):
        plan([WARP_ARRAY_MAX//3+1], policy=policy)


def test_counting_assignment_is_independent_of_compute_chunks():
    p = plan([8, 8, 1, 1, 1, 1], owners=3)
    assert p.countingRanges == ((0, 2), (2, 4), (4, 6))
    assert [(c.rowStart, c.rowStop, c.owner) for c in p.chunks] == [(0, 1, 0), (1, 2, 1), (2, 6, 2)]


def test_replica_determinism_input_snapshot_and_immutable_metadata():
    counts = torch.tensor([3, 0, 4, 8, 2], dtype=torch.int32)
    generation = RowChunkGeneration(4, 20, 'ordering/support/domain/grid/acceptance')
    policy = RowChunkPolicy(edgeBudget=10)
    a = buildRowChunkPlan(counts, generation=generation, policy=policy, numOwners=4)
    b = buildRowChunkPlan(counts.clone(), generation=generation, policy=policy, numOwners=4)
    assert a == b
    a.validateSignature(b.signature)
    counts.fill_(0)
    assert a.totalEdges == 17
    offsets = a.globalOffsets()
    offsets.fill_(0)
    assert a.totalEdges == 17
    for obj, field, value in [(a, 'numOwners', 3), (a.chunks[0], 'owner', 2),
                              (generation, 'number', 9), (policy, 'edgeBudget', 12)]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


def test_signatures_cover_generation_config_all_counts_and_partition_rules():
    baseline = plan([2, 1, 3])
    variants = [plan([1, 2, 3]), plan([2, 1, 3], owners=2),
                plan([2, 1, 3], policy=RowChunkPolicy(edgeBudget=9, headroomNumerator=0)),
                plan([2, 1, 3], generation=replace(baseline.generation, number=8)),
                plan([2, 1, 3], generation=replace(baseline.generation, configSignature='new-grid')),
                plan([2, 1, 3], generation=replace(baseline.generation, numReferences=42))]
    assert len({baseline.signature, *(p.signature for p in variants)}) == 7
    for other in variants:
        with pytest.raises(ValueError, match='signature mismatch'):
            baseline.validateSignature(other.signature)


@pytest.mark.parametrize('counts', [torch.tensor([1.]), torch.tensor([1], dtype=torch.int64),
                                  torch.ones((1, 1), dtype=torch.int32)])
def test_count_shape_and_dtype(counts):
    with pytest.raises(TypeError, match='torch.int32'):
        buildRowChunkPlan(counts, generation=RowChunkGeneration(0, 1, 'cfg'), policy=RowChunkPolicy(edgeBudget=4))


def test_device_counts_require_an_explicit_cpu_transfer():
    # Meta exercises the non-CPU guard without starting a GPU workload.
    counts = torch.empty(2, dtype=torch.int32, device='meta')
    with pytest.raises(ValueError, match='requires CPU counts explicitly'):
        buildRowChunkPlan(counts, generation=RowChunkGeneration(0, 3, 'cfg'),
                          policy=RowChunkPolicy(edgeBudget=4))


def test_byte_budget_below_one_edge_still_owns_zero_rows():
    policy = RowChunkPolicy(byteBudget=3, bytesPerEdge=4)
    p = plan([0, 0], policy=policy, owners=4)
    assertCoverage(p, [0, 0])
    assert len(p.chunks) == 1 and p.chunks[0].capacity == 0
    with pytest.raises(OverflowError, match='Row 1'):
        plan([0, 1], policy=policy)


@pytest.mark.parametrize('values, refs', [([-1], 2), ([3], 2), ([1], 0)])
def test_invalid_counts(values, refs):
    with pytest.raises(ValueError, match='stored row count'):
        plan(values, generation=RowChunkGeneration(0, refs, 'cfg'))


@pytest.mark.parametrize('kwargs', [{}, {'edgeBudget': 0}, {'byteBudget': -1}, {'edgeBudget': True},
    {'edgeBudget': 4.0}, {'edgeBudget': 4, 'headroomNumerator': -1},
    {'edgeBudget': 4, 'headroomDenominator': 0}, {'edgeBudget': 4, 'alignment': 0},
    {'edgeBudget': 4, 'bytesPerEdge': 0}, {'edgeBudget': 4, 'maxRows': 0},
    {'edgeBudget': 4, 'maxElementsPerEdge': 0}])
def test_invalid_policy(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RowChunkPolicy(**kwargs)


@pytest.mark.parametrize('owners', [0, -1, True, 1.5])
def test_invalid_owner_count(owners):
    with pytest.raises((TypeError, ValueError)):
        plan([1], owners=owners)


def test_generation_validation_and_noncontiguous_counts():
    for args in [(-1, 1, 'cfg'), (0, -1, 'cfg'), (0, 1, ''), (0, 1, 42), (True, 1, 'cfg')]:
        with pytest.raises((TypeError, ValueError)):
            RowChunkGeneration(*args)
    with pytest.raises(OverflowError):
        RowChunkGeneration(0, WARP_ARRAY_MAX+1, 'cfg')
    values = torch.tensor([1, 0, 2, 0, 3, 0], dtype=torch.int32)[::2]
    p = buildRowChunkPlan(values, generation=RowChunkGeneration(0, 3, 'cfg'), policy=RowChunkPolicy(edgeBudget=8))
    assertCoverage(p, [1, 2, 3])


def test_exhaustive_small_rows_have_maximal_chunks_and_stable_replicas():
    for values in itertools.product(range(4), repeat=5):
        p = plan(values, owners=3)
        assertCoverage(p, values)
        for left, right in zip(p.chunks, p.chunks[1:]):
            assert left.liveEdges + values[right.rowStart] > p.policy.capacityLimit
        assert p == plan(values, owners=3)


@pytest.mark.parametrize('values,policy,owners,signature', [
    ([2, 0, 6, 0, 1, 7], RowChunkPolicy(edgeBudget=8, headroomNumerator=0), 3,
     'eda01253ad90aa6d56f42d87ba44f07cbb2317b355884faa5222fa2a728fb14a'),
    ([WARP_ARRAY_MAX//3]*4 + [0, 11],
     RowChunkPolicy(edgeBudget=WARP_ARRAY_MAX, maxElementsPerEdge=3, headroomNumerator=0), 4,
     '62901fb44470bff11e59d0aeba56177f3b179c35e768fb9c154c2ea0b2d01286'),
    ([], RowChunkPolicy(byteBudget=3, bytesPerEdge=4), 4,
     'f2a12de529a5ffb788c59e834ab73908b87d3d7f568095c0e8ea3db77c7ed860'),
])
def test_optimized_planner_preserves_v1_signature_bytes(values, policy, owners, signature):
    # Frozen from the original per-row implementation, including its little-
    # endian prefix encoding. Mixed-version peers must agree byte for byte.
    assert plan(values, policy=policy, owners=owners).signature == signature


def test_exact_padding_inversion_matches_brute_force_greedy_partition():
    # Exercise non-aligned limits and rational headroom that floats cannot
    # distinguish from zero. The oracle evaluates public capacityFor directly.
    for budget, alignment, fraction, maxRows in itertools.product(
            (1, 7, 17), (1, 4, 9), ((0, 1), (1, 3), (2, 7), (1, 10**40)), (None, 2)):
        numerator, denominator = fraction
        policy = RowChunkPolicy(edgeBudget=budget, alignment=alignment,
            headroomNumerator=numerator, headroomDenominator=denominator, maxRows=maxRows)
        fitting = max(v for v in range(18) if policy.capacityFor(v) <= policy.capacityLimit)
        values = [fitting, 0, min(1, fitting), 0, fitting, 0]
        expected, start = [], 0
        while start < len(values):
            limit = len(values) if maxRows is None else min(len(values), start + maxRows)
            stop = max(end for end in range(start + 1, limit + 1)
                       if policy.capacityFor(sum(values[start:end])) <= policy.capacityLimit)
            expected.append((start, stop))
            start = stop
        actual = plan(values, policy=policy, owners=3)
        assert [(c.rowStart, c.rowStop) for c in actual.chunks] == expected
        assertCoverage(actual, values)
