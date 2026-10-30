"""Portable search-history recipes retain global row order without pair IO."""
from copy import deepcopy
from dataclasses import replace
import importlib

import pytest
import torch
import warp as wp

from warpSPHCore import DomainDescription,ParticleState,SupportScheme,get_torch_precision,get_torch_offset_dtype
from warpSPHCore.radiusSearch import (RowChunkPolicy,buildChunkedVerletList,rowChunkCheckpointState,
    restoreRowChunkCheckpointState,chunkedVerletStaleFlag_)


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


def population(device,n,dim,shift):
    t=torch.arange(n,dtype=get_torch_precision(),device=device)
    x=torch.stack([.45*torch.sin(t*(.5+d)+shift) for d in range(dim)],1)
    return ParticleState(x,.4+.05*torch.cos(t),torch.ones_like(t),torch.zeros(n,dtype=torch.int32,device=device),torch.ones_like(t))


def setup(device,dim=2,sizes=(13,None),storeI=False):
    q=population(device,sizes[0],dim,.2)
    r=q if sizes[1] is None else population(device,sizes[1],dim,.8)
    domain=DomainDescription(torch.full((dim,),-2.,device=device,dtype=q.positions.dtype),
        torch.full((dim,),2.,device=device,dtype=q.positions.dtype),torch.zeros(dim,device=device,dtype=torch.bool),dim)
    policy=RowChunkPolicy(edgeBudget=80,maxRows=3,bytesPerEdge=32,maxElementsPerEdge=3)
    adjacency=buildChunkedVerletList(q,domain,referenceParticles=r,policy=policy,
        supportMode=SupportScheme.SuperSymmetric,verletScale=1.2,storeI=storeI)
    return q,r,domain,adjacency


def row(adj,i):
    chunk=next(c for c in adj if c.metadata.rowStart<=i<c.metadata.rowStop)
    local=i-chunk.metadata.rowStart
    base=int(chunk.adjacency.edgeOffsets[local])
    n=int(chunk.adjacency.numNeighbors[local])
    return chunk.adjacency.paddedJ[base:base+n]


@pytest.mark.parametrize('dim,sizes,storeI',[(2,(13,None),False),(3,(9,14),True),
    (2,(0,0),False),(2,(0,5),True),(2,(3,0),False)])
def test_recipe_reconstructs_same_rows_under_new_budget(device,dim,sizes,storeI):
    q,r,domain,old=setup(device,dim,sizes,storeI)
    state=rowChunkCheckpointState(old)
    assert set(state['tensors']).isdisjoint({'paddedJ','paddedI','W','edgeOffsets'})
    new=restoreRowChunkCheckpointState(state,device=device,policy=replace(old.plan.policy,maxRows=2),storeI=not storeI)
    torch.testing.assert_close(new.numNeighbors,old.numNeighbors,rtol=0,atol=0)
    torch.testing.assert_close(new.globalPrefix,old.globalPrefix,rtol=0,atol=0)
    assert new.plan.generation.number==old.plan.generation.number
    assert (new.queryPositions is new.referencePositions)==(old.queryPositions is old.referencePositions)
    assert (new.querySupports is new.referenceSupports)==(old.querySupports is old.referenceSupports)
    for i in range(new.numRows):
        torch.testing.assert_close(row(new,i),row(old,i),rtol=0,atol=0)
    assert all(c.adjacency.edgeOffsets.dtype==get_torch_offset_dtype() for c in new)
    flag=chunkedVerletStaleFlag_(q.positions,r.positions,q.supports,r.supports,domain,new)
    assert not bool(flag)
    assert buildChunkedVerletList(q,domain,referenceParticles=r,policy=new.plan.policy,
        supportMode=SupportScheme.SuperSymmetric,verletScale=1.2,priorNeighborhood=new,storeI=not storeI) is new


@pytest.mark.parametrize('bad',['checksum','counts','version','snapshot','missing','budget','recount','geometry'])
def test_recipe_rejects_invalid_history_before_pair_allocation(device,monkeypatch,bad):
    _,_,_,old=setup(device)
    state=deepcopy(rowChunkCheckpointState(old))
    policy=old.plan.policy
    if bad=='checksum':state['checksum']='bad'
    elif bad=='counts':state['tensors']['counts'][0]+=1
    elif bad=='version':state['metadata']['version']=99
    elif bad=='snapshot':state['tensors']['queryPositions'][0,0]+=.01
    elif bad=='missing':state['tensors'].pop('counts')
    elif bad=='budget':policy=replace(policy,edgeBudget=1)
    elif bad in ('recount','geometry'):
        # Re-sign a structurally valid recipe to exercise recount/plan agreement,
        # beyond the checksum guard. Still reject before pair allocation.
        if bad=='recount':state['tensors']['counts'][0]+=1
        else:state['tensors']['queryPositions'][0,0]+=1
        recipeModule=importlib.import_module('warpSPHCore.radiusSearch.chunkedState')
        state['checksum']=recipeModule._digest(state['metadata'],state['tensors'])
    module=importlib.import_module('warpSPHCore.radiusSearch.chunked')
    def forbidden(*a,**kw):
        raise AssertionError('invalid recipe allocated pair buffers')
    monkeypatch.setattr(module,'_allocateChunks',forbidden)
    with pytest.raises((ValueError,OverflowError)):
        restoreRowChunkCheckpointState(state,device=device,policy=policy)


def test_export_is_independent_and_rejects_overwritten_storage(device):
    q,r,domain,old=setup(device)
    state=rowChunkCheckpointState(old)
    saved=state['tensors']['queryPositions'].clone()
    q.positions.add_(.01)
    torch.testing.assert_close(state['tensors']['queryPositions'],saved,rtol=0,atol=0)
    buildChunkedVerletList(q,domain,policy=old.plan.policy,verletScale=1.2,
        supportMode=SupportScheme.SuperSymmetric,priorNeighborhood=old,forceRebuild=True,inPlace=True)
    with pytest.raises(ValueError,match='overwritten'):
        rowChunkCheckpointState(old)
