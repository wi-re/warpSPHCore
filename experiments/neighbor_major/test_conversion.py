"""Conversion checks, including poisoned padding and partial tiles."""
import sys
from pathlib import Path

import pytest
import torch
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parent))
import layout as experiment
import conversion


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('n',[1,35,67])
@pytest.mark.parametrize('method',['rows','tile'])
def test_ragged_conversion_overwrites_live_and_padding(device,n,method):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    if device=='cpu' and method=='tile':pytest.skip('block-cooperative transpose requires CUDA')
    wp.init()
    counts=torch.tensor([(i*17)%65 for i in range(n)],dtype=torch.int32,device=device)
    if n==1:counts[0]=0
    offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1].int()))
    edges=int(counts.sum()); maximum=int(counts.max())
    source=torch.arange(edges,dtype=torch.int64,device=device)*3+7
    csr=experiment.Neighbors(source,counts,offsets,n,100,edges,'csr',maximum,counts.new_zeros(1))
    target=conversion.allocate(csr,method)
    target.paddedJ.fill_(123456)
    conversion.fill(csr,target,method)
    wp.synchronize_device(device)
    expected=torch.full((maximum,n),-1,dtype=torch.int64,device=device)
    for i in range(n):expected[:int(counts[i]),i]=source[int(offsets[i]):int(offsets[i]+counts[i])]
    torch.testing.assert_close(target.paddedJ.view(maximum,n),expected,rtol=0,atol=0)


@pytest.mark.parametrize('method',['rows','tile'])
def test_conversion_replay_reads_changed_source(method):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from warpSPHCore.autograd.stream import torchWarpStream,torchWarpCapture
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        counts=torch.tensor([0,33,1,2,34],dtype=torch.int32,device='cuda')
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1].int()))
        edges=int(counts.sum());maximum=int(counts.max());n=len(counts)
        source=torch.arange(edges,dtype=torch.int64,device='cuda')
        csr=experiment.Neighbors(source,counts,offsets,n,100,edges,'csr',maximum,counts.new_zeros(1))
        target=conversion.convert(csr,method)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torchWarpStream(source.device),torchWarpCapture(graph):conversion.fill(csr,target,method)
        source.add_(101)
        graph.replay();observed=target.paddedJ.clone();stream.synchronize()
        expected=torch.full((maximum,n),-1,dtype=torch.int64,device='cuda')
        for i in range(n):expected[:int(counts[i]),i]=source[int(offsets[i]):int(offsets[i]+counts[i])]
        torch.testing.assert_close(observed.view(maximum,n),expected,rtol=0,atol=0)
