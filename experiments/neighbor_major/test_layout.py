"""Exact search/order/padding and matched operator checks for the isolated layout."""

import sys
from pathlib import Path

import pytest
import torch
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parent))
import layout as experiment
from warpSPHCore import (ParticleState,OperationProperties,KernelFunctions,WarpOperation,
                         SupportScheme,GradientScheme,LaplacianScheme,setNeighborLanes)
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


def case(device,dim=2,periodic=True):
    p,h,n,domain,dx=generateNeighborTestData(6,15,dim,periodic,device)
    rng=torch.Generator().manual_seed(3)
    p+=(torch.rand(p.shape,generator=rng).to(device)-.5)*(.2*dx)
    h=h*(.5+.5*torch.rand(n,generator=rng).to(device))
    particles=ParticleState(positions=p,supports=h,masses=torch.full_like(h,dx**dim),
                            densities=torch.ones_like(h),kinds=torch.zeros(n,dtype=torch.int32,device=device))
    return particles,domain


def assert_rows(csr,ell):
    torch.testing.assert_close(csr.numNeighbors,ell.numNeighbors,rtol=0,atol=0)
    view=ell.paddedJ.view(-1,ell.numRows)
    for i in range(ell.numRows):
        size=int(csr.numNeighbors[i]);begin=int(csr.edgeOffsets[i])
        torch.testing.assert_close(view[:size,i],csr.paddedJ[begin:begin+size],rtol=0,atol=0)
        assert bool((view[size:,i]==-1).all())


@pytest.mark.parametrize('periodic',[False,True])
@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('mode',list(SupportScheme))
def test_direct_search_matches_production_order(device,periodic,dim,mode):
    particles,domain=case(device,dim,periodic)
    p,h=particles.positions,particles.supports
    queries=p.clone()
    if not periodic:queries[::7]+=10
    grid=experiment.buildCompactHashMap(queries,p,h,h,domain.periodic,domain,mode,len(p)+1)
    reference=radiusSearchOnCompactHashMap(grid,queries,p,h,h,domain.periodic,domain,mode,len(p)+1,capacity=None)
    csr=experiment.search(grid,queries,h,domain.periodic,'csr')
    ell=experiment.search(grid,queries,h,domain.periodic,'neighbor-major')
    torch.testing.assert_close(csr.paddedJ,reference.paddedJ,rtol=0,atol=0)
    assert_rows(csr,ell)


@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('lanes',[1,32])
@pytest.mark.parametrize('name',list(experiment.BACKENDS))
def test_primal_operators(device,dim,lanes,name):
    particles,domain=case(device,dim)
    p,h=particles.positions,particles.supports
    grid=experiment.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,len(p)+1)
    csr=experiment.search(grid,p,h,domain.periodic,'csr')
    ell=experiment.search(grid,p,h,domain.periodic,'neighbor-major')
    scalar=torch.sin(p[:,0]*3).contiguous()
    values=torch.stack([scalar.roll(i) for i in range(dim)],1).contiguous() if name in ('curl','divergence') else scalar
    props=OperationProperties(kernel=KernelFunctions.Wendland2,operation=getattr(WarpOperation,name.capitalize()),
                              gradientMode=GradientScheme.Difference,laplacianMode=LaplacianScheme.Brookshaw,
                              supportMode=SupportScheme.Gather)
    setNeighborLanes(lanes)
    try:
        expected=experiment.makeOperation(name,csr,particles,props,domain,values)()
        result=experiment.makeOperation(name,ell,particles,props,domain,values)()
        tolerance=3e-12 if p.dtype==torch.float64 else 3e-5
        torch.testing.assert_close(result,expected,rtol=tolerance,atol=tolerance*max(1.,float(expected.abs().max())))
    finally:setNeighborLanes(None)


def test_overflow_and_all_empty_rows(device):
    particles,domain=case(device,periodic=False)
    p,h=particles.positions,particles.supports
    grid=experiment.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,len(p)+1)
    with pytest.raises(OverflowError):experiment.search(grid,p,h,domain.periodic,'neighbor-major',capacity=0)
    ell=experiment.search(grid,p+10,h,domain.periodic,'neighbor-major')
    assert ell.maximum==0 and ell.paddedJ.numel()==0 and ell.numEdges==0
    assert bool((ell.numNeighbors==0).all())


def test_generated_primal_path_rejects_ad(device):
    particles,domain=case(device)
    p,h=particles.positions,particles.supports
    grid=experiment.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,len(p)+1)
    ell=experiment.search(grid,p,h,domain.periodic,'neighbor-major')
    props=OperationProperties(kernel=KernelFunctions.Wendland2)
    with pytest.raises(NotImplementedError):experiment.makeOperation('interpolate',ell,particles,props,domain,torch.ones_like(h,requires_grad=True))


def test_replay_with_changed_neighbor_ids():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from warpSPHCore.autograd.stream import torchWarpStream,torchWarpCapture
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        particles,domain=case('cuda')
        p,h=particles.positions,particles.supports
        grid=experiment.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,len(p)+1)
        ell=experiment.search(grid,p,h,domain.periodic,'neighbor-major')
        common,wrapped=experiment.commonInputs(grid,p,h,domain.periodic)
        def call():
            ell.paddedJ.fill_(-1)
            experiment.collectInto(grid,common,ell)
        call();stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torchWarpStream(p.device),torchWarpCapture(graph):call()
        grid.sortIndex.copy_(grid.sortIndex.flip(0).clone())
        graph.replay();captured=ell.paddedJ.clone()
        call();stream.synchronize()
        torch.testing.assert_close(captured,ell.paddedJ,rtol=0,atol=0)
        assert not bool(ell.overflow)
