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


@pytest.mark.parametrize('site', ['neighbors', 'work'])
def test_bulk_allocation_failure_releases_partial_storage_and_records_receipt(device, monkeypatch, site):
    import importlib, json, weakref
    p, d, adjacency = sample(device)
    before = tuple(c.adjacency.paddedJ.clone() for c in adjacency)
    original = torch.empty; refs = []; calls = 0
    error = torch.OutOfMemoryError('injected third bulk buffer')
    def empty(*args, **kwargs):
        nonlocal calls
        bulk = (args and isinstance(args[0], tuple)) if site=='work' else (args and isinstance(args[0], int) and args[0]>1 and kwargs.get('dtype')==torch.int32)
        if bulk:
            calls += 1
            if calls == 3:
                raise error
        value = original(*args, **kwargs)
        if bulk:
            refs.append(weakref.ref(value))
        return value
    monkeypatch.setattr(torch, 'empty', empty)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        if site=='work':
            ChunkedPairField.empty(adjacency, trailingShape=(2,), dtype=p.positions.dtype)
        else:
            chunk = importlib.import_module('warpSPHCore.radiusSearch.chunked')
            chunk._allocateChunks(adjacency.plan, adjacency.numNeighbors, adjacency.globalPrefix,
                0, False, adjacency, False)
    assert caught.value is error
    receipt = json.loads(error.__notes__[-1].split(': ',1)[1])
    assert receipt['kind']==('pair-field' if site=='work' else 'neighbors')
    assert receipt['completedChunks']==2 and receipt['requestedBytes']>0
    assert all(ref() is None for ref in refs)
    adjacency.assertStorageValid()
    for chunk, saved in zip(adjacency, before):
        torch.testing.assert_close(chunk.adjacency.paddedJ, saved, rtol=0, atol=0)


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('block', [64, 128])
def test_explicit_scalar_block_reaches_eager_and_prepared_chunk_launches(device, monkeypatch, prepared, block):
    p, domain, adjacency = sample(device)
    work = ChunkedPairField.empty(adjacency, trailingShape=(2,), dtype=p.positions.dtype)
    context = SPHContext(p, OperationProperties(kernel=KernelFunctions.Wendland2), domain, adjacency)
    launch = wp.launch
    seen = []
    def observed(kernel, *args, **kwargs):
        if kernel.key.startswith('_stamp'):
            seen.append(kwargs.get('block_dim'))
        return launch(kernel, *args, **kwargs)
    monkeypatch.setattr(wp, 'launch', observed)
    spec = replace(SPEC, blockDim=block)
    values, _ = (PreparedLauncher(context).launch(spec, work=work, rowStart=0) if prepared else
                 launchOperator(spec, context, work=work, rowStart=0))
    assert seen and all(value==block for value in seen)
    torch.testing.assert_close(values,torch.arange(len(p.positions),device=device,dtype=p.positions.dtype),rtol=0,atol=0)


@pytest.mark.parametrize('block', [True, 0, -1, 1025, 64.0])
def test_invalid_scalar_block_size_is_rejected_when_declared(block):
    with pytest.raises(ValueError, match='blockDim'):
        replace(SPEC, blockDim=block)


@wp.kernel
def _blockSquare(query: Any, reference: Any, domain: Any, useAdjacency: wp.bool,
                 adjacency: Any, grid: Any, correction: Any, properties: Any,
                 values: wp.array(dtype=scalar_t), output: wp.array(dtype=scalar_t)):
    i=wp.tid()
    output[i]=values[i]*values[i]


@pytest.mark.parametrize('block', [64, 128])
def test_explicit_scalar_block_preserves_reverse_mode(device, block):
    from warpSPHCore import radiusSearchCompactHashMap
    p, domain, _ = sample(device)
    adjacency=radiusSearchCompactHashMap(p,domain)
    context=SPHContext(p,OperationProperties(kernel=KernelFunctions.Wendland2),domain,adjacency)
    values=torch.linspace(-1.,1.,len(p.positions),device=device,dtype=p.positions.dtype,requires_grad=True)
    spec=OperatorSpec(kernel=_blockSquare,blockDim=block,outputs=(OutputSpec(dtype=scalar_t),),
        extras=(ExtraSpec('values',ExtraKind.TENSOR),))
    output=launchOperator(spec,context,values=values)
    output.sum().backward()
    torch.testing.assert_close(output,values.square(),rtol=0,atol=0)
    torch.testing.assert_close(values.grad,2*values,rtol=0,atol=0)
