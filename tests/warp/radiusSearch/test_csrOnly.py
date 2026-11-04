"""CSR-only allocation preserves neighbour identity/order/rebuild contracts."""
from dataclasses import replace
import importlib
import os
import pytest
import torch
import warp as wp
from warpSPHCore import (ParticleState, SupportScheme, CapacityPolicy, buildVerletList,
                         radiusSearchCompactHashMap, filterVerletList)
from warpSPHCore.util import generateNeighborTestData

@pytest.fixture(scope='module', autouse=True)
def init():
    wp.init()

def particles(dim=2):
    device=os.environ.get('WARPSPH_TEST_DEVICE','cpu')
    x,h,n,domain,dx=generateNeighborTestData(7,20,dim,True,device)
    x=x+.07*dx*torch.sin(3*x.flip(-1))
    return ParticleState(positions=x,supports=h,masses=torch.ones(n,device=device),
                         densities=torch.ones(n,device=device),kinds=torch.zeros(n,dtype=torch.int32,device=device)),domain

@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('collection',['fused','split'])
@pytest.mark.parametrize('order',['input','spatial'])
def test_exact_csr_and_explicit_coo(dim,collection,order):
    p,d=particles(dim)
    a=radiusSearchCompactHashMap(p,d,mode=SupportScheme.SuperSymmetric,collection=collection,queryOrder=order)
    b=radiusSearchCompactHashMap(p,d,mode=SupportScheme.SuperSymmetric,collection=collection,queryOrder=order,storeI=False)
    assert b.paddedI is None and a.numEdges==b.numEdges
    for name in ('paddedJ','numNeighbors','edgeOffsets'):
        assert torch.equal(getattr(a,name),getattr(b,name))
    assert torch.equal(a.paddedI,b.cooRows())
    assert all(torch.equal(x,y) for x,y in zip(a.livePairs(),b.livePairs()))
    assert b.paddedI is None  # requesting compatibility rows never retains them

@pytest.mark.parametrize('storeI',[False,True])
def test_rebuild_reuse_filter_and_layout_transition(storeI):
    p,d=particles()
    cap=CapacityPolicy(inPlace=True)
    a=buildVerletList(p,d,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric,capacity=cap,storeI=storeI)
    kept=buildVerletList(p,d,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric,capacity=cap,storeI=storeI,priorNeighborhood=a)
    assert kept is a
    moved=replace(p,positions=p.positions+.001)
    b=buildVerletList(moved,d,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric,capacity=cap,storeI=storeI,priorNeighborhood=a,forceRebuild=True)
    assert b.paddedJ.data_ptr()==a.paddedJ.data_ptr()
    assert (b.paddedI is not None)==storeI
    f=filterVerletList(moved,d,b,supportMode=SupportScheme.SuperSymmetric)
    assert (f.paddedI is not None)==storeI
    exact=radiusSearchCompactHashMap(moved,d,mode=SupportScheme.SuperSymmetric)
    fi,fj=f.livePairs(); ei,ej=exact.livePairs()
    assert torch.equal(torch.sort(fi.long()*len(p.positions)+fj.long()).values,
                       torch.sort(ei.long()*len(p.positions)+ej.long()).values)
    baseline=filterVerletList(moved,d,b,supportMode=SupportScheme.SuperSymmetric,storeI=True)
    for name in ('paddedJ','numNeighbors','edgeOffsets'):
        assert torch.equal(getattr(f,name),getattr(baseline,name))
    other=buildVerletList(moved,d,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric,capacity=cap,storeI=not storeI,priorNeighborhood=b)
    assert other is not b and (other.paddedI is not None)!=storeI
    assert other.paddedJ.data_ptr()!=b.paddedJ.data_ptr()


def test_no_edge_i_allocation(monkeypatch):
    m=importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
    original=m.allocateTorchWarp
    calls=[]
    def observed(n,*args,**kwargs):
        calls.append(n)
        return original(n,*args,**kwargs)
    monkeypatch.setattr(m,'allocateTorchWarp',observed)
    p,d=particles()
    a=radiusSearchCompactHashMap(p,d,storeI=False)
    assert calls.count(a.paddedJ.numel())==1
    assert a.paddedI is None


def test_empty_rows_and_no_padding():
    p,d=particles()
    # Unequal populations with no candidates: CSR rows still reconstruct safely.
    q=replace(p,positions=p.positions[:2],supports=torch.zeros_like(p.supports[:2]))
    ref=replace(p,positions=p.positions[2:3],supports=torch.zeros_like(p.supports[2:3]))
    a=radiusSearchCompactHashMap(q,d,referenceParticles=ref,capacity=None,storeI=False)
    assert a.numEdges==0 and a.paddedI is None and a.paddedJ.numel()==0
    assert a.cooRows().numel()==0
    assert a.numNeighbors.tolist()==[0,0]


@pytest.mark.parametrize('index_dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('offset_dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('counts',[[2,0,1], [0,0,0], []])
def test_explicit_native_coo_with_wide_csr_in_warp_process(device,index_dtype,offset_dtype,counts,monkeypatch):
    from warpSPHCore import AdjacencyList
    # The native consumer owns its backend even while the process default is Warp.
    # Reconstruction must not wrap the independently wide CSR in Warp arrays.
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('unexpected Warp launch'))
    c=torch.tensor(counts,device=device,dtype=torch.int32)
    offsets=torch.cat((torch.zeros(1,device=device,dtype=offset_dtype),c.to(offset_dtype).cumsum(0)))[:-1]
    live=sum(counts)
    j=torch.full((live+2,),-1,device=device,dtype=index_dtype)
    j[:live]=torch.arange(live,device=device,dtype=index_dtype)
    adj=AdjacencyList(None,j,c,offsets,len(counts),max(live,1),numEdges=live)
    expected=torch.tensor([i for i,n in enumerate(counts) for _ in range(n)],device=device,dtype=index_dtype)
    torch.testing.assert_close(adj.cooRows(padded=False,backend='native'),expected,rtol=0,atol=0)
    torch.testing.assert_close(adj.cooRows(backend='native'),torch.cat((expected,j.new_full((2,),-1))),rtol=0,atol=0)
    actual_i,actual_j=adj.livePairs(backend='native')
    torch.testing.assert_close(actual_i,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual_j,j[:live],rtol=0,atol=0)
    assert adj.paddedI is None
