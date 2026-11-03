"""Seeded local distributed storage/failure stress; never a large allocation."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import argparse
import contextlib
import gc
import importlib
import json
import os
import random
import time
import weakref
from unittest.mock import patch

import torch
import torch.distributed as dist
import warp as wp

from _rowChunkWorker import sample,check_rows,equal
from warpSPHCore import SupportScheme,setNeighborLanes
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedContractError
from warpSPHCore.radiusSearch import buildChunkedVerletList,filterChunkedVerletList,RowChunkPolicy,ChunkedPairField
from warpSPHCore.radiusSearch.verlet.build import buildVerletList_
from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_


def cycle(comm,prior,index,rng):
    modes=list(SupportScheme)
    nq=rng.choice((0,1,2,5,11,19,29));nr=None if index%3==0 else rng.choice((0,1,7,17,31))
    # Guaranteed population/ownership transitions, beyond random draws.
    if index%12 in (0,1):nq,nr=29,None
    dim=2+index%2
    q,r,domain=sample(comm.device,nq,nr,dim)
    if index%4:
        q.positions[nq//3:,0]+=10+3*torch.arange(nq-nq//3,device=comm.device,dtype=q.positions.dtype)
        if r is not q:r.positions[len(r.positions)//2:]*=.1
    if index%5==0:domain.periodic[:]=True
    mode=modes[index%len(modes)]
    alignment=rng.choice((1,4,8));headroom=rng.choice((0,1,3))
    minimum=((len(r.positions)*(10+headroom)+9)//10+alignment-1)//alignment*alignment
    edgeBudget=max(1,minimum)*rng.choice((1,2,5))
    rows=rng.choice((1,2,3,7,80))
    if index%12==0:rows,edgeBudget=1,8192
    if index%12==1:rows,edgeBudget=80,8192
    policy=RowChunkPolicy(edgeBudget=edgeBudget,maxRows=rows,alignment=alignment,
        headroomNumerator=headroom,bytesPerEdge=8+dim*q.positions.element_size(),maxElementsPerEdge=dim)
    resident=buildChunkedVerletList(q,domain,referenceParticles=r,policy=policy,verletScale=1.5,
        supportMode=mode,storeI=index%2==0,communication=comm,priorNeighborhood=prior,forceRebuild=True)
    if prior is not None:prior.assertStorageValid()
    inplace=False
    if index%8==7:
        old=resident
        resident=buildChunkedVerletList(q,domain,referenceParticles=r,policy=policy,verletScale=1.5,
            supportMode=mode,storeI=index%2==0,communication=comm,priorNeighborhood=old,forceRebuild=True,inPlace=True)
        if old.chunks:
            try:old.assertStorageValid()
            except ValueError:pass
            else:raise AssertionError('in-place lease stayed valid')
        inplace=True
    flat=buildVerletList_(q.positions,r.positions,q.supports,r.supports,domain,
                         verletScale=1.5,mode=mode,storeI=index%2==0)
    q.positions.add_(.0001)
    targetRows=rng.choice((1,2,5,80));targetBudget=max(1,minimum)*rng.choice((1,3,8))
    if index%12==0:targetRows,targetBudget=80,8192
    if index%12==1:targetRows,targetBudget=1,8192
    target=replace(policy,edgeBudget=targetBudget,maxRows=targetRows)
    work=ChunkedPairField.empty(resident,trailingShape=(dim,),dtype=q.positions.dtype)
    for w in work.buffers:w.fill_(index*.125+comm.rank)
    before=[c.adjacency.paddedJ.clone() for c in resident]
    saved=[w.clone() for w in work.buffers]
    faults=0
    def rejected(function,text):
        try:function()
        except DistributedContractError as exc:assert text in str(exc),str(exc)
        else:raise AssertionError('injected fault was not propagated')
    if index%12==0:
        original=comm._checkedAgree
        fired=False
        def injected(tag,function):
            nonlocal fired
            if tag.startswith('filter:rows:') and not fired:
                fired=True
                def fail():
                    metadata=function()
                    if comm.rank==comm.size-1:raise ValueError('injected stress transfer contract')
                    return metadata
                return original(tag,fail)
            return original(tag,function)
        with patch.object(comm,'_checkedAgree',injected):
            rejected(lambda:filterChunkedVerletList(q,domain,resident,referenceParticles=r,policy=target),
                     'injected stress transfer')
        assert fired
        faults+=1
    actual=filterChunkedVerletList(q,domain,resident,referenceParticles=r,policy=target,storeI=index%2!=0)
    expected=filterVerletList_(q.positions,r.positions,q.supports,r.supports,domain,flat,mode,capacity=None,storeI=index%2!=0)
    check_rows(actual,expected)
    for c,j,w,want in zip(resident,before,work.buffers,saved):equal(c.adjacency.paddedJ,j);equal(w,want)
    work.validate(resident)
    # Exercise bounded packet reuse as ownership and row count change.
    a,b=actual.plan.ownerRanges[comm.rank]
    expectedA=torch.arange(nq*2,dtype=q.positions.dtype,device=comm.device).reshape(nq,2)
    expectedB=torch.arange(nq,dtype=q.positions.dtype,device=comm.device)
    A=expectedA.clone();B=expectedB.clone();A[:a]=-1;A[b:]=-1;B[:a]=-1;B[b:]=-1
    comm.exchangeParticleOutputs(actual,(A,B),tag='stress:outputs')
    equal(A,expectedA);equal(B,expectedB)
    bound=max((c.metadata.liveEdges*4+c.metadata.rowCount*torch.empty((),dtype=c.adjacency.edgeOffsets.dtype).element_size()
               for c in resident),default=0)
    assert actual.searchScratchBytes<=bound
    return resident,[weakref.ref(c.adjacency.paddedJ) for c in resident],dict(
        cycle=index,rows=nq,references=len(r.positions),dim=dim,mode=mode.name,
        sourceRanges=resident.plan.ownerRanges,targetRanges=actual.plan.ownerRanges,
        chunks=len(resident.chunks),edges=resident.numEdges,inPlace=inplace,faults=faults,
        scratchBytes=actual.searchScratchBytes,scratchBound=bound)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu');parser.add_argument('--cycles',type=int,default=48)
    parser.add_argument('--seed',type=int,default=991);parser.add_argument('--output',required=True)
    parser.add_argument('--die-transfer-rank',type=int)
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
    args=parser.parse_args()
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=45))
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks,particleWorkspaceBytes=4096)
    wp.init();setNeighborLanes(1)
    stream=torch.cuda.Stream(device=comm.device) if comm.device.type=='cuda' else None
    rng=random.Random(args.seed);history=[];references=[];prior=None;address=None;started=time.perf_counter()
    try:
        if args.die_transfer_rank is not None:
            assert args.die_transfer_rank in (0,comm.size-1)
            send=torch.arange(32,dtype=torch.int32,device=comm.device) if comm.rank==0 else None
            receive=torch.empty(32,dtype=torch.int32,device=comm.device) if comm.rank==comm.size-1 else None
            originalSend,originalReceive=dist.send,dist.recv
            def payload(function):
                def transfer(*a,**kw):
                    if comm.rank==args.die_transfer_rank:
                        print('injected fatal participant after transfer contract',flush=True)
                        os._exit(17)
                    return function(*a,**kw)
                return transfer
            with patch.object(dist,'send',payload(originalSend)),patch.object(dist,'recv',payload(originalReceive)):
                comm.transferOwned(send,receive,source=0,target=comm.size-1,count=32,dtype=torch.int32,
                                   tag='stress:fatal-transfer',signature='fatal-participant')
            comm.agree('stress:fatal-unexpected-return',None)
            raise AssertionError('fatal transfer participant unexpectedly survived')
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            for index in range(args.cycles):
                prior,refs,report=cycle(comm,prior,index,rng)
                references.extend((index,ref) for ref in refs)
                gc.collect()
                alive=sum(ref() is not None for generation,ref in references if generation<index)
                assert alive==0,('retained obsolete pair tensors',index,alive)
                if comm._particleWorkspace is not None:
                    if address is None:address=comm._particleWorkspace.data_ptr()
                    assert comm._particleWorkspace.data_ptr()==address
                if stream is not None:
                    stream.synchronize()
                    report['allocatedBytes']=torch.cuda.memory_allocated(comm.device)
                    report['reservedBytes']=torch.cuda.memory_reserved(comm.device)
                history.append(report)
                comm.agree('stress:cycle-passed',index)
                print(f'[stress rank {comm.rank}] cycle {index+1}/{args.cycles}: passed, rows={report["rows"]}, stalePairs={alive}',flush=True)
        if stream is not None:stream.synchronize()
        assert comm.particleWorkspaceAllocations==1
        assert any(r['sourceRanges']!=r['targetRanges'] for r in history)
        assert any(any(a==b for a,b in r['targetRanges']) for r in history)
        report=dict(status='passed',seed=args.seed,cycles=args.cycles,history=history,
                    wallSeconds=time.perf_counter()-started,obsoletePairTensorsAlive=0,**comm.summary())
        if comm.device.type=='cuda':
            import ctypes
            version=ctypes.c_int();ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            report['runtimeNccl']=version.value
            # Fixed tiny populations and one 4KiB cache should not grow with cycles.
            allocated=[h['allocatedBytes'] for h in history[8:]]
            assert max(allocated)-min(allocated)<2*1024*1024,allocated
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/f'stress-rank{comm.rank}.json').write_text(json.dumps(report,indent=2))
        print(f'[stress rank {comm.rank}] all stress checks passed',flush=True)
    finally:
        setNeighborLanes(None);dist.destroy_process_group()

if __name__=='__main__':main()
