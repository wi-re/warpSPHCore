"""Owned pair-field bounds, generation safety, and eager/prepared global row mapping."""
from dataclasses import replace
from typing import Any

import pytest
import torch
import warp as wp
from warp.types import vector

from warpSPHCore import DomainDescription, ParticleState, OperationProperties, KernelFunctions
from warpSPHCore.autograd import (SPHContext, OperatorSpec, OutputSpec, ExtraSpec, ExtraKind,
                                  RowRole, PreparedLauncher, launchOperator)
from warpSPHCore.radiusSearch import radiusSearchRowChunks, RowChunkPolicy, ChunkedPairField
from warpSPHCore.type_config import scalar_t, offset_t, get_torch_precision


_vec2 = vector(length=2, dtype=scalar_t)


@wp.kernel
def _stamp(query: Any, reference: Any, domain: Any, useAdjacency: wp.bool,
           adjacency: Any, grid: Any, correction: Any, properties: Any,
           work: wp.array(dtype=vector(length=2, dtype=scalar_t)), rowStart: wp.int32,
           out: wp.array(dtype=scalar_t), scratch: wp.array(dtype=vector(length=2, dtype=scalar_t))):
    i = wp.tid()
    for edge in range(adjacency.numNeighbors[i]):
        at = adjacency.neighborOffsets[i] + offset_t(edge)
        j = adjacency.neighborList[at]
        work[at] = _vec2(scalar_t(i + rowStart), scalar_t(j))
    out[i] = scalar_t(i + rowStart)
    if i == 0:
        scratch[0] = _vec2(scalar_t(99.))


SPEC = OperatorSpec(kernel=_stamp, rowChunked=True,
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=vector(length=2, dtype=scalar_t), shape=1, chunkScratch=True)),
    extras=(ExtraSpec('work', ExtraKind.CHUNK_TENSOR, RowRole.CHUNK),
            ExtraSpec('rowStart', ExtraKind.SCALAR, RowRole.ROW_START)))


def sample(device, n=9, policy=None):
    dtype = get_torch_precision()
    t = torch.arange(n, device=device, dtype=dtype)
    x = torch.stack((t * .07 + .1, t * .05 + .12), 1)
    h = torch.full_like(t, .6)
    p = ParticleState(x, h, torch.ones_like(t), torch.zeros(n, device=device, dtype=torch.int32), torch.ones_like(t))
    d = DomainDescription(t.new_zeros(2), t.new_ones(2), torch.zeros(2, device=device, dtype=torch.bool), 2)
    policy = policy or RowChunkPolicy(edgeBudget=30, maxRows=2, maxElementsPerEdge=2,
                                     bytesPerEdge=4+2*t.element_size())
    a = radiusSearchRowChunks(p, d, policy=policy)
    return p, d, a


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('n', [0, 9])
def test_owned_field_global_rows_and_padding(device, prepared, n):
    p, d, a = sample(device, n)
    field = ChunkedPairField.empty(a, trailingShape=(2,), dtype=p.positions.dtype)
    for t in field.buffers:
        t.fill_(float('nan'))
    ctx = SPHContext(p, OperationProperties(kernel=KernelFunctions.Wendland2), d, a)
    output, scratch = (PreparedLauncher(ctx).launch(SPEC, work=field, rowStart=0) if prepared
                       else launchOperator(SPEC, ctx, work=field, rowStart=0))
    torch.testing.assert_close(output, torch.arange(n, device=device, dtype=p.positions.dtype), rtol=0, atol=0)
    assert scratch.shape == (1, 2)
    for c, t in zip(a, field.buffers):
        m, adj = c.metadata, c.adjacency
        for local in range(m.rowCount):
            start, count = int(adj.edgeOffsets[local]), int(adj.numNeighbors[local])
            assert bool((t[start:start+count, 0] == local + m.rowStart).all())
            torch.testing.assert_close(t[start:start+count, 1], adj.paddedJ[start:start+count].to(t.dtype), rtol=0, atol=0)
        assert torch.isnan(t[m.liveEdges:]).all()
    usage = field.memoryUsage()
    assert usage['liveBytes'] == a.numEdges * 2 * p.positions.element_size()
    assert usage['capacityBytes'] == sum(t.untyped_storage().nbytes() for t in field.buffers)


def test_invalid_owned_storage_fails_before_output_allocation(device, monkeypatch):
    p, d, a = sample(device)
    field = ChunkedPairField.empty(a, trailingShape=(2,), dtype=p.positions.dtype)
    ctx = SPHContext(p, OperationProperties(kernel=KernelFunctions.Wendland2), d, a)
    module = __import__('warpSPHCore.autograd.row_dispatch', fromlist=['_allocate_output'])
    monkeypatch.setattr(module, '_allocate_output', lambda *a, **k: pytest.fail('allocated before validation'))
    for invalid in (replace(field, adjacency=replace(a)), replace(field, buffers=field.buffers[:-1]), replace(field, buffers=list(field.buffers)),
                    replace(field, buffers=(field.buffers[0], field.buffers[0], *field.buffers[2:])),
                    replace(field, buffers=(field.buffers[0][:-1], *field.buffers[1:]))):
        with pytest.raises(ValueError, match='generation|buffer|storage'):
            launchOperator(SPEC, ctx, work=invalid, rowStart=0)
    with pytest.raises(NotImplementedError, match='opted'):
        launchOperator(replace(SPEC, outputs=tuple(reversed(SPEC.outputs))), ctx, work=field, rowStart=0)
    with pytest.raises(ValueError, match='ROW_START'):
        launchOperator(SPEC, ctx, work=field, rowStart=1)
    with pytest.raises(NotImplementedError, match='AD'):
        launchOperator(SPEC, ctx, work=replace(field,
            buffers=(field.buffers[0].clone().requires_grad_(), *field.buffers[1:])), rowStart=0)


def test_field_budget_and_inplace_rebuild_lease(device, monkeypatch):
    p, d, a = sample(device)
    field = ChunkedPairField.empty(a, trailingShape=(2,), dtype=p.positions.dtype)
    tiny = replace(a, plan=replace(a.plan, policy=RowChunkPolicy(edgeBudget=30)))
    original = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *args, **kw: pytest.fail('allocated oversized pair field')
                        if args and isinstance(args[0], tuple) else original(*args, **kw))
    with pytest.raises(ValueError, match='scalar extents'):
        ChunkedPairField.empty(tiny, trailingShape=(2,), dtype=p.positions.dtype)
    monkeypatch.setattr(torch, 'empty', original)
    radiusSearchRowChunks(p, d, policy=a.plan.policy, reuse=a, inPlace=True)
    with pytest.raises(ValueError, match='overwritten'):
        field.validate(a)
