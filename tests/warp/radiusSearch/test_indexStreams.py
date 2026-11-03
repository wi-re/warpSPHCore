"""Small/filtered CSR producers follow asynchronous writes on Torch's stream."""
import pytest
import torch

from warpSPHCore import DomainDescription, SupportScheme
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.radiusSearch.small.wp_radius_small import warp_radius_search_small
from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA stream ownership')
@pytest.mark.parametrize('producer',['small','filtered'])
def test_asynchronous_positions_and_supports_on_nondefault_stream(producer):
    dtype=get_torch_precision()
    p=torch.tensor([[0.],[.2],[.7]],dtype=dtype,device='cuda')
    h=torch.full((3,),.9,dtype=dtype,device='cuda')
    domain=DomainDescription(min=torch.tensor([-1.],dtype=dtype,device='cuda'),
        max=torch.tensor([2.],dtype=dtype,device='cuda'),
        periodic=torch.tensor([False],device='cuda'),dim=1)
    grid=buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,5)
    prior=radiusSearchOnCompactHashMap(grid,p,p,h,h,domain.periodic,domain,SupportScheme.Gather,5,capacity=None)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Delay the writes after the prior superset is ready. The filtered
        # producer has no metadata read before its count launch to hide a race.
        torch.cuda._sleep(2_000_000)
        p[2]=.4
        h.fill_(.3)
        if producer=='small':
            adj=warp_radius_search_small(p,p,h,h,domain.periodic,domain,SupportScheme.Gather)
        else:
            adj=filterVerletList_(p,p,h,h,domain,prior,SupportScheme.Gather,capacity=None)
        counts,offsets,neighbors=adj.numNeighbors.clone(),adj.edgeOffsets.clone(),adj.paddedJ.clone()
    stream.synchronize()
    rows=[sorted(neighbors[int(start):int(start)+int(count)].tolist()) for start,count in zip(offsets,counts)]
    assert rows==[[0,1],[0,1,2],[1,2]]
