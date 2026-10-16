"""Independent CSR row expansion and exact prototype/production collection parity."""

from pathlib import Path

import pytest
import torch
import warp as wp


@pytest.fixture
def prototype(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts/benchmarks'))
    import _split_collect
    return _split_collect


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


@pytest.mark.parametrize('lanes',[32,128])
def test_rows_empty_irregular_and_padding(prototype,device,lanes):
    counts=torch.tensor([0,1,0,33,257,2,0],dtype=torch.int32,device=device)
    offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1].int()))
    output=torch.full((int(counts.sum())+37,),-1,dtype=torch.int64,device=device)
    prototype.launchRows(wp.from_torch(counts),wp.from_torch(offsets),wp.from_torch(output),len(counts),lanes)
    wp.synchronize_device(str(device))
    expected=torch.repeat_interleave(torch.arange(len(counts),device=device),counts.long())
    torch.testing.assert_close(output[:len(expected)],expected,rtol=0,atol=0)
    assert bool((output[len(expected):]==-1).all())


@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('periodic',[False,True])
@pytest.mark.parametrize('modeName',['Gather','Scatter','MeanSymmetric','KernelMeanSymmetric','SuperSymmetric','PartialSymmetric'])
def test_split_search_exact_arrays(prototype,device,dim,periodic,modeName):
    from warpSPHCore import SupportScheme,CapacityPolicy
    from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
    from warpSPHCore.util import generateNeighborTestData
    positions,supports,n,domain,dx=generateNeighborTestData(5,15,dim,periodic,device)
    rng=torch.Generator().manual_seed(31)
    positions=positions+.1*dx*(torch.rand(positions.shape,generator=rng)-.5).to(device)
    supports=supports*(.7+.3*torch.rand(n,generator=rng).to(device))
    # Same-sized query/reference sets, with empty nonperiodic query rows.
    queries=positions.clone(); queries[::7]+=10
    mode=getattr(SupportScheme,modeName); H=n+1
    grid=buildCompactHashMap(queries,positions,supports,supports,domain.periodic,domain,mode,H)
    def search(**kwargs):
        kwargs.setdefault('collection','fused')
        kwargs.setdefault('queryOrder','input')
        return radiusSearchOnCompactHashMap(grid,queries,positions,supports,supports,domain.periodic,domain,mode,H,**kwargs)
    expected=search(); jOnly,_=prototype.loadJOnly()
    for lanes in (32,128):
        with prototype.splitSearch(jOnly,lanes): got=search()
        assert expected.numEdges==got.numEdges
        for name in ('numNeighbors','edgeOffsets','paddedI','paddedJ'):
            torch.testing.assert_close(getattr(got,name),getattr(expected,name),rtol=0,atol=0)
        addresses=[x.data_ptr() for x in (got.paddedI,got.paddedJ,got.numNeighbors,got.edgeOffsets)]
        with prototype.splitSearch(jOnly,lanes): reused=search(capacity=CapacityPolicy(inPlace=True),reuse=got)
        assert addresses==[x.data_ptr() for x in (reused.paddedI,reused.paddedJ,reused.numNeighbors,reused.edgeOffsets)]
        torch.testing.assert_close(reused.paddedI,expected.paddedI,rtol=0,atol=0)
        torch.testing.assert_close(reused.paddedJ,expected.paddedJ,rtol=0,atol=0)


def test_nondefault_stream_replay_changed_rows(prototype):
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from warpSPHCore.autograd.stream import torchWarpCapture,torchWarpStream,currentTorchWarpStream
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        counts=torch.tensor([1,0,3,2],dtype=torch.int32,device='cuda')
        offsets=torch.tensor([0,1,1,4],dtype=torch.int32,device='cuda')
        output=torch.full((16,),-1,dtype=torch.int64,device='cuda')
        def call():
            output.fill_(-1)
            prototype.launchRows(wp.from_torch(counts),wp.from_torch(offsets),wp.from_torch(output),4,32,currentTorchWarpStream(counts.device))
        call();stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torchWarpStream(counts.device),torchWarpCapture(graph): call()
        for values in ([0,4,0,1],[2,1,3,0],[0,0,0,0]):
            counts.copy_(torch.tensor(values,dtype=torch.int32,device='cuda'))
            offsets[0]=0; offsets[1:]=counts.cumsum(0)[:-1].int()
            graph.replay()
            result=output.clone()
            expected=torch.repeat_interleave(torch.arange(4,device='cuda'),counts.long(),output_size=sum(values))
            stream.synchronize()
            torch.testing.assert_close(result[:len(expected)],expected,rtol=0,atol=0)
            assert bool((result[len(expected):]==-1).all())
