"""Native neighborhood lifecycle inside a Warp runtime, with no Warp launches."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.radiusSearch import (RowChunkPolicy, buildChunkedVerletList,
    rowChunkCheckpointState, restoreRowChunkCheckpointState)


def scene(device, dim=2):
    g=torch.Generator().manual_seed(293)
    x=torch.rand((47,dim),generator=g).to(device)*3-1
    h=.2+torch.rand(47,generator=g).to(device)*.4
    return SimpleNamespace(positions=x,supports=h), sph.DomainDescription(
        x.new_zeros(dim),x.new_ones(dim),torch.tensor([True]+[False]*(dim-1),device=device),dim)


def pairs(adj):
    if hasattr(adj,'chunks'):
        result=[]
        for chunk in adj:
            a,m=chunk.adjacency,chunk.metadata
            rows=torch.repeat_interleave(torch.arange(m.rowStart,m.rowStop,device=a.paddedJ.device),a.numNeighbors.long(),output_size=a.numEdges)
            result.append(rows*adj.numCols+a.paddedJ[:a.numEdges].long())
        return torch.cat(result).sort().values
    i,j=adj.livePairs()
    return (i.long()*adj.numCols+j.long()).sort().values


def oracle(q,d,mode):
    dx=q.positions[:,None]-q.positions[None,:]
    L=d.max-d.min
    dx=torch.where(d.periodic,torch.remainder(dx+L/2,L)-L/2,dx)
    h=q.supports[:,None] if mode==sph.SupportScheme.Gather else q.supports[None,:] if mode==sph.SupportScheme.Scatter else torch.maximum(q.supports[:,None],q.supports[None,:])
    i,j=torch.where(dx.square().sum(-1)<=h.square())
    return i*len(q.positions)+j


@pytest.mark.parametrize('mode',[sph.SupportScheme.Gather,sph.SupportScheme.Scatter,sph.SupportScheme.SuperSymmetric])
def test_native_flat_and_bounded_lifecycle_without_warp(device,monkeypatch,mode):
    wp.init()
    q,d=scene(device)
    # Compile before the guard; all measured lifecycle calls must stay native.
    from warpSPHCore.native._build import build
    build()
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('native search launched Warp'))
    storeI = mode==sph.SupportScheme.Gather
    flat=sph.buildVerletList(q,d,1.4,mode,backend='native',storeI=storeI)
    assert flat.searchBackend=='native' and flat.edgeOffsets.dtype==torch.int64
    moved=SimpleNamespace(positions=q.positions+.001,supports=q.supports)
    assert sph.buildVerletList(moved,d,1.4,mode,priorNeighborhood=flat,backend='native',storeI=storeI) is flat
    assert not bool(sph.verletStaleFlag(moved.positions,moved.supports,flat,d,1.4,mode))
    exact=sph.filterVerletList(moved,d,flat,mode,capacity=None)
    torch.testing.assert_close(pairs(exact),oracle(moved,d,mode),rtol=0,atol=0)
    with sph.deferVerletChecks() as flags:
        assert sph.buildVerletList(moved,d,1.4,mode,priorNeighborhood=flat,backend='native',storeI=storeI) is flat
    assert len(flags)==1
    chunks=buildChunkedVerletList(q,d,policy=RowChunkPolicy(edgeBudget=2048,maxRows=12,bytesPerEdge=16),
        supportMode=mode,verletScale=1.4,storeI=True,backend='native')
    assert chunks.searchBackend=='native' and len(chunks.chunks)==4
    filtered=sph.filterVerletList(moved,d,chunks,mode)
    torch.testing.assert_close(pairs(filtered),oracle(moved,d,mode),rtol=0,atol=0)
    state=rowChunkCheckpointState(chunks)
    assert state['metadata']['searchBackend']=='native'
    resumed=restoreRowChunkCheckpointState(state,device=device,policy=RowChunkPolicy(edgeBudget=2048,maxRows=24,bytesPerEdge=16))
    assert resumed.searchBackend=='native' and len(resumed.chunks)==2
    torch.testing.assert_close(pairs(resumed),pairs(chunks),rtol=0,atol=0)
    changed=sph.buildVerletList(q,d,1.4,mode,priorNeighborhood=resumed,backend='native',storeI=True,forceRebuild=True)
    assert changed.plan.generation.number>resumed.plan.generation.number


@pytest.mark.parametrize('dim',[2,3])
def test_fixed_shape_native_grid_matches_warp_keys_and_capture(device,monkeypatch,dim):
    wp.init()
    from warpSPHCore.radiusSearch.compactHash.slots import makeHashMapSlots,buildFixed,GraphedBuild
    from warpSPHCore.native._build import build
    build()
    q,d=scene(device,dim)
    # Large integer cell coordinates exercise the hash's specified uint32 wrap.
    d=replace(d,min=d.min-20,max=d.max+20)
    slots=makeHashMapSlots(47,d,.05,device=device,dtype=q.positions.dtype)
    slots.positions.copy_(q.positions);slots.supports.copy_(q.supports*.05)
    old=buildFixed(slots)
    fields=('sortIndex','sortedPositions','sortedSupports','hashTable','sortedCellTable')
    expected={k:getattr(old,k).clone() for k in fields}
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('native fixed grid launched Warp'))
    builder=GraphedBuild(slots,backend='native')
    for _ in range(3):
        actual=builder.replay()
        for k,v in expected.items():torch.testing.assert_close(getattr(actual,k),v,rtol=0,atol=0)
    if torch.device(device).type=='cuda':
        assert builder.graph is not None
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):builder.replay()
        graph.replay();torch.cuda.synchronize()
        for k,v in expected.items():torch.testing.assert_close(getattr(slots.hashMap,k),v,rtol=0,atol=0)


def test_backend_transition_and_version1_checkpoint(device):
    wp.init()
    q,d=scene(device)
    w=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,backend='warp')
    n=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,priorNeighborhood=w,backend='native')
    assert n is not w and n.searchBackend=='native'
    torch.testing.assert_close(pairs(n),pairs(w),rtol=0,atol=0)
    again=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,priorNeighborhood=n,backend='warp')
    assert again is not n and again.searchBackend=='warp'
    c=buildChunkedVerletList(q,d,policy=RowChunkPolicy(edgeBudget=2048,maxRows=12,bytesPerEdge=16),
        supportMode=sph.SupportScheme.SuperSymmetric,verletScale=1.4,backend='warp')
    state=rowChunkCheckpointState(c)
    state['metadata'].pop('searchBackend');state['metadata']['version']=1
    from warpSPHCore.radiusSearch.chunkedState import _digest
    state['checksum']=_digest(state['metadata'],state['tensors'])
    restored=restoreRowChunkCheckpointState(state,device=device)
    assert restored.searchBackend=='warp'
    torch.testing.assert_close(pairs(restored),pairs(c),rtol=0,atol=0)


def test_native_flat_rebuild_does_not_narrow_into_foreign_capacity(device):
    q,d=scene(device)
    policy=sph.CapacityPolicy(headroom=.1,shrinkBelow=0.,inPlace=True)
    original=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,
        backend='native',capacity=policy,storeI=True)
    foreign=replace(original,paddedI=original.paddedI.int(),paddedJ=original.paddedJ.int(),edgeOffsets=original.edgeOffsets.int())
    snapshot=foreign.paddedJ.clone()
    rebuilt=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,
        backend='native',capacity=policy,storeI=True,priorNeighborhood=foreign,forceRebuild=True)
    assert rebuilt.paddedJ.dtype==rebuilt.edgeOffsets.dtype==torch.int64
    assert rebuilt.paddedJ.data_ptr()!=foreign.paddedJ.data_ptr()
    torch.testing.assert_close(foreign.paddedJ,snapshot,rtol=0,atol=0)
    torch.testing.assert_close(pairs(rebuilt),pairs(original),rtol=0,atol=0)
    assert bool((rebuilt.paddedJ[rebuilt.numEdges:]==-1).all())
    assert bool((rebuilt.paddedI[rebuilt.numEdges:]==-1).all())



def test_native_inplace_rebuild_retains_graph_snapshot_addresses(device):
    q,d=scene(device)
    policy=sph.CapacityPolicy(headroom=1.,shrinkBelow=0.,inPlace=True)
    prior=sph.buildVerletList(q,d,1.4,sph.SupportScheme.SuperSymmetric,backend='native',capacity=policy)
    names=('paddedJ','edgeOffsets','numNeighbors','queryPositions','referencePositions','querySupports','referenceSupports')
    addresses={k:getattr(prior,k).data_ptr() for k in names}
    moved=SimpleNamespace(positions=q.positions+.001,supports=q.supports)
    current=sph.buildVerletList(moved,d,1.4,sph.SupportScheme.SuperSymmetric,backend='native',
        priorNeighborhood=prior,capacity=policy,forceRebuild=True)
    assert {k:getattr(current,k).data_ptr() for k in names}==addresses
    torch.testing.assert_close(current.queryPositions,moved.positions,rtol=0,atol=0)
    assert not bool(sph.verletStaleFlag(moved.positions,moved.supports,current,d,1.4))
    torch.testing.assert_close(pairs(sph.filterVerletList(moved,d,current,sph.SupportScheme.SuperSymmetric)),oracle(moved,d,sph.SupportScheme.SuperSymmetric),rtol=0,atol=0)
