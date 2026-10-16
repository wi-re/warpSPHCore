"""Run in fresh processes for each configured ID/offset width and precision."""
import pytest
from dataclasses import replace
import torch
import warp as wp

from warpSPHCore import DomainDescription, SupportScheme, CapacityPolicy
from warpSPHCore.type_config import (index_t, offset_t, get_torch_precision,
    get_torch_index_dtype, get_torch_offset_dtype, get_type_config, _resolve_integer_type)
from warpSPHCore.radiusSearch.csr import (buildCSROffsets, validateEdgeCount,
    validateEdgeCapacity, validateParticleCounts, WARP_ARRAY_MAX)
from warpSPHCore.radiusSearch.grid_util import laneSubRange
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.radiusSearch.small.wp_radius_small import warp_radius_search_small
from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


def test_configuration_roles():
    config=get_type_config()
    assert config['index_t']==index_t and config['offset_t']==offset_t
    assert config['neighbor_count_t']==wp.int32
    assert wp.dtype_to_torch(index_t)==get_torch_index_dtype()
    assert wp.dtype_to_torch(offset_t)==get_torch_offset_dtype()
    for invalid in ('float32','uint32','int16','banana'):
        with pytest.raises(ValueError):_resolve_integer_type(invalid,'offset_dtype')


@pytest.mark.parametrize('values',[[],[0],[0,0,0],[3,0,2],[2**31-3,2,0]])
def test_wide_scan_before_checked_narrowing(device,values):
    counts=torch.tensor(values,dtype=torch.int32,device=device)
    offsets,total=buildCSROffsets(counts)
    expected=[];running=0
    for value in values:expected.append(running);running+=value
    assert total==running and offsets.tolist()==expected
    assert offsets.dtype==get_torch_offset_dtype() and counts.dtype==torch.int32


def test_total_over_int32_without_large_allocation(device):
    counts=torch.tensor([2**31-1,1,0],dtype=torch.int32,device=device)
    if offset_t==wp.int32:
        with pytest.raises(OverflowError,match='CSR edge count'):buildCSROffsets(counts)
    else:
        offsets,total=buildCSROffsets(counts)
        assert total==2**31 and offsets.tolist()==[0,2**31-1,2**31]
        with pytest.raises(OverflowError,match='Warp flat-array'):validateEdgeCapacity(total,total)


def test_allocation_and_population_bounds_without_allocation():
    validateParticleCounts(0,2**31-1)
    with pytest.raises(OverflowError):validateParticleCounts(1,2**31)
    with pytest.raises(OverflowError):validateParticleCounts(-1,1)
    validateEdgeCount(2**31-1,torch.int32)
    with pytest.raises(OverflowError):validateEdgeCount(2**31,torch.int32)
    validateEdgeCount(2**31,torch.int64)
    with pytest.raises(OverflowError):validateEdgeCount(2**63,torch.int64)
    validateEdgeCapacity(0,0)
    with pytest.raises(ValueError):validateEdgeCapacity(2,1)
    with pytest.raises(OverflowError,match='Warp flat-array'):validateEdgeCapacity(2**31-1,2**31)


@wp.kernel
def slicedRows(begin: offset_t, count: wp.int32,
               starts: wp.array(dtype=offset_t), lengths: wp.array(dtype=wp.int32)):
    lane=wp.tid()
    first,size=laneSubRange(begin,count,lane,32)
    starts[lane]=first;lengths[lane]=size


@pytest.mark.parametrize('case',['end-of-int32','huge-row','wide-offset'])
def test_lane_addresses_do_not_narrow_or_overflow(device,case):
    if case=='wide-offset' and offset_t==wp.int32:pytest.skip('int64 address case')
    begin,count={'end-of-int32':(2**31-4,3),'huge-row':(0,2**31-1),'wide-offset':(2**31+17,57)}[case]
    starts=torch.empty(32,dtype=get_torch_offset_dtype(),device=device)
    lengths=torch.empty(32,dtype=torch.int32,device=device)
    stream=wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type=='cuda' else None
    wp.launch(slicedRows,32,inputs=[offset_t(begin),count,wp.from_torch(starts),wp.from_torch(lengths)],device=device.type,stream=stream)
    chunk=(count+31)//32
    expected=[min(lane*chunk,count) for lane in range(32)]
    assert starts.tolist()==[begin+start for start in expected]
    assert lengths.tolist()==[min(chunk,count-start) for start in expected]
    assert int(lengths.sum())==count


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('mode',list(SupportScheme))
def test_search_backends_and_verlet_use_configured_widths(device,empty,mode):
    dtype=get_torch_precision()
    reference=torch.tensor([[0.,0.],[.2,0.],[.4,0.],[0.,.2],[.2,.2],[.4,.2]],device=device,dtype=dtype)
    query=reference[[0,2,4]].clone()
    if empty:query+=3
    qh=torch.full((len(query),),.31,device=device,dtype=dtype)
    rh=torch.full((len(reference),),.27,device=device,dtype=dtype)
    domain=DomainDescription(min=torch.tensor([-1.,-1.],device=device,dtype=dtype),
        max=torch.tensor([5.,5.],device=device,dtype=dtype),periodic=torch.tensor([False,False],device=device),dim=2)
    grid=buildCompactHashMap(query,reference,qh,rh,domain.periodic,domain,mode,17)
    assert grid.sortIndex.dtype==get_torch_index_dtype()
    csr=radiusSearchOnCompactHashMap(grid,query,reference,qh,rh,domain.periodic,domain,mode,17,capacity=None)
    small=warp_radius_search_small(query,reference,qh,rh,domain.periodic,domain,mode)
    filtered=filterVerletList_(query,reference,qh,rh,domain,csr,mode,capacity=None)
    def rows(adj):
        return [sorted(adj.paddedJ[int(start):int(start)+int(count)].tolist()) for start,count in zip(adj.edgeOffsets,adj.numNeighbors)]
    expected=[]
    for i in range(len(query)):
        row=[]
        for j in range(len(reference)):
            threshold={SupportScheme.Gather:qh[i],SupportScheme.Scatter:rh[j],
                SupportScheme.MeanSymmetric:(qh[i]+rh[j])/2,
                SupportScheme.KernelMeanSymmetric:torch.maximum(qh[i],rh[j]),
                SupportScheme.PartialSymmetric:torch.maximum(qh[i],rh[j]),
                SupportScheme.SuperSymmetric:torch.maximum(qh[i],rh[j])}[mode]
            if bool((query[i]-reference[j]).norm()<=threshold):row.append(j)
        expected.append(row)
    for adj in (csr,small,filtered):
        assert adj.numRows==len(query) and adj.numCols==len(reference)
        assert adj.paddedI.dtype==adj.paddedJ.dtype==get_torch_index_dtype()
        assert adj.edgeOffsets.dtype==get_torch_offset_dtype() and adj.numNeighbors.dtype==torch.int32
        assert rows(adj)==expected
        assert adj.numEdges==sum(map(len,expected))


def test_inplace_reuse_with_incompatible_dtypes_allocates_configured_buffers(device):
    dtype=get_torch_precision()
    p=torch.tensor([[0.,0.],[.2,0.]],device=device,dtype=dtype)
    h=torch.full((2,),.3,device=device,dtype=dtype)
    domain=DomainDescription(min=torch.tensor([-1.,-1.],device=device,dtype=dtype),
        max=torch.tensor([1.,1.],device=device,dtype=dtype),periodic=torch.tensor([False,False],device=device),dim=2)
    grid=buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,5)
    def search(**kwargs):return radiusSearchOnCompactHashMap(grid,p,p,h,h,domain.periodic,domain,SupportScheme.Gather,5,**kwargs)
    original=search()
    otherID=torch.int64 if get_torch_index_dtype()==torch.int32 else torch.int32
    otherOffset=torch.int64 if get_torch_offset_dtype()==torch.int32 else torch.int32
    incompatible=replace(original,paddedI=original.paddedI.to(otherID),paddedJ=original.paddedJ.to(otherID),
                         edgeOffsets=original.edgeOffsets.to(otherOffset))
    before=incompatible.paddedJ.clone()
    got=search(capacity=CapacityPolicy(inPlace=True),reuse=incompatible)
    assert got.paddedJ.dtype==get_torch_index_dtype() and got.edgeOffsets.dtype==get_torch_offset_dtype()
    assert got.paddedJ.data_ptr()!=incompatible.paddedJ.data_ptr()
    torch.testing.assert_close(got.paddedJ,original.paddedJ,rtol=0,atol=0)
    torch.testing.assert_close(incompatible.paddedJ,before,rtol=0,atol=0)
