"""Opt-in physical wide-edge access qualification; small CPU cases run in pytest.

Dense coincident particles make an exhaustive independent J oracle affordable:
each row must be the same full permutation of all reference IDs. No edge-sized
oracle or global pair tensor is constructed. Actual search/owned buffers, the
row operator adapter, prepared density, work mapping and filtering are exercised.
Launch directly or under torchrun; CUDA ranks intentionally share cuda:0.
"""
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import contextlib
import ctypes
import json
import os
import threading
import time

import torch
import torch.distributed as dist
import warp as wp
from warp.types import vector

from warpSPHCore import (DomainDescription, ParticleState, OperationProperties,
    KernelFunctions, SupportScheme, setNeighborLanes, get_torch_precision)
from warpSPHCore.autograd import (SPHContext, PreparedLauncher, OperatorSpec,
    OutputSpec, ExtraSpec, ExtraKind, RowRole, launchOperator)
from warpSPHCore.coreOperations.wp_density import _DENSITY_SPEC
from warpSPHCore.distributed import RowChunkProcessGroup
from warpSPHCore.radiusSearch import (RowChunkPolicy, buildChunkedVerletList,
    filterChunkedVerletList, ChunkedPairField)
from warpSPHCore.radiusSearch.verlet.build import buildVerletList_
from warpSPHCore.type_config import scalar_t, offset_t, get_torch_offset_dtype

INT_MAX=2**31-1
MODULUS=16777213  # every residue is represented exactly in float32
_vec=vector(length=3,dtype=scalar_t)


@wp.kernel
def stamp(query: Any, reference: Any, domain: Any, useAdjacency: wp.bool,
          adjacency: Any, grid: Any, correction: Any, properties: Any,
          work: wp.array(dtype=_vec), rowStart: wp.int32, width: wp.int32,
          output: wp.array(dtype=scalar_t)):
    i=wp.tid()
    for column in range(adjacency.numNeighbors[i]):
        at=adjacency.neighborOffsets[i]+offset_t(column)
        j=adjacency.neighborList[at]
        globalAt=wp.int64(i+rowStart)*wp.int64(width)+wp.int64(column)
        work[at]=_vec(scalar_t(i+rowStart),scalar_t(j),scalar_t(globalAt%wp.int64(16777213)))
    output[i]=scalar_t(i+rowStart)


STAMP=OperatorSpec(kernel=stamp,rowChunked=True,outputs=(OutputSpec(dtype=scalar_t),),
    extras=(ExtraSpec('work',ExtraKind.CHUNK_TENSOR,RowRole.CHUNK),
            ExtraSpec('rowStart',ExtraKind.SCALAR,RowRole.ROW_START),ExtraSpec('width',ExtraKind.SCALAR)))


@wp.kernel
def audit(j: wp.array(dtype=wp.int32), work: wp.array(dtype=_vec),
          pattern: wp.array(dtype=wp.int32), width: wp.int32, edgeStart: wp.int64,
          total: wp.int64, checkWork: wp.bool, errors: wp.array(dtype=wp.int32),
          probes: wp.array2d(dtype=wp.int64)):
    k=wp.tid()
    globalAt=edgeStart+wp.int64(k)
    column=wp.int32(globalAt%wp.int64(width))
    row=wp.int32(globalAt//wp.int64(width))
    value=j[k]
    if value!=pattern[column]:
        wp.atomic_add(errors,0,1)
    if checkWork:
        v=work[k]
        if v[0]!=scalar_t(row) or v[1]!=scalar_t(value) or v[2]!=scalar_t(globalAt%wp.int64(16777213)):
            wp.atomic_add(errors,1,1)
    probe=wp.int32(-1)
    if globalAt==wp.int64(2147483647):probe=0
    if globalAt==wp.int64(2147483648):probe=1
    if globalAt==total-wp.int64(1):probe=2
    if probe>=0:
        probes[probe,0]=globalAt
        probes[probe,1]=wp.int64(value)
        probes[probe,2]=wp.int64(row)


class Phases:
    def __init__(self,rank):self.rank=rank;self.records=[]
    @contextlib.contextmanager
    def __call__(self,name,device):
        start=time.perf_counter();stop=threading.Event()
        def heartbeat():
            while not stop.wait(10):
                print(f'[wide rank {self.rank}] {name}: running {time.perf_counter()-start:.1f}s',flush=True)
        thread=threading.Thread(target=heartbeat,daemon=True)
        print(f'[wide rank {self.rank}] {name}: start',flush=True);thread.start()
        try:
            yield
            if device.type=='cuda':torch.cuda.synchronize(device)
        finally:
            stop.set();thread.join()
            wall=time.perf_counter()-start
            self.records.append(dict(phase=name,wallSeconds=wall))
            print(f'[wide rank {self.rank}] {name}: end {wall:.3f}s',flush=True)


def run(args,device,comm):
    rank,size=(0,1) if comm is None else (comm.rank,comm.size)
    phases=Phases(rank);n=args.particles;total=n*n
    if args.require_wide and total<=INT_MAX:raise ValueError('--require-wide needs a real edge count above INT_MAX')
    policy=RowChunkPolicy(edgeBudget=args.edge_budget,headroomNumerator=0,bytesPerEdge=16,maxElementsPerEdge=3)
    # W(3 FP32) + J, optional overlapping filtered J and one source-chunk scratch.
    scalarBytes=torch.empty((),dtype=get_torch_precision()).element_size()
    policy=replace(policy,bytesPerEdge=4+3*scalarBytes)
    expectedBytes=total*(policy.bytesPerEdge+(4 if args.filter else 0))+args.edge_budget*8+size*512*2**20
    available=None
    if device.type=='cuda':
        available,capacity=torch.cuda.mem_get_info(device)
        if expectedBytes>args.memory_budget_gib*2**30 or available<expectedBytes+args.reserve_gib*2**30:
            raise MemoryError(f'Wide preflight needs {expectedBytes/2**30:.2f} GiB plus {args.reserve_gib} GiB reserve; free={available/2**30:.2f} GiB, budget={args.memory_budget_gib} GiB')
        torch.cuda.reset_peak_memory_stats(device)
    print(f'[wide rank {rank}] preflight particles={n} edges={total} pair/work estimate={expectedBytes/2**30:.3f}GiB free={available}',flush=True)
    if comm:comm.agree('wide:configuration',(n,args.edge_budget,args.filter,args.lanes,total))
    dtype=get_torch_precision()
    p=ParticleState(torch.full((n,2),.5,device=device,dtype=dtype),torch.ones(n,device=device,dtype=dtype),
        torch.ones(n,device=device,dtype=dtype),torch.zeros(n,device=device,dtype=torch.int32),torch.ones(n,device=device,dtype=dtype))
    domain=DomainDescription(torch.zeros(2,device=device,dtype=dtype),torch.ones(2,device=device,dtype=dtype),torch.zeros(2,device=device,dtype=torch.bool),2)
    with phases('count/scan/allocate/collect resident J',device):
        adj=buildChunkedVerletList(p,domain,policy=policy,verletScale=1.,supportMode=SupportScheme.SuperSymmetric,communication=comm)
    assert adj.numEdges==total and int(adj.globalPrefix[-1])==total and adj.globalPrefix.dtype==torch.int64
    assert bool((adj.numNeighbors==n).all())
    assert sum(c.metadata.liveEdges for c in adj)==sum(c.liveEdges for c in adj.plan.chunks if c.owner==rank)
    # A single flat row is O(N) and below the ABI bound, independent of chunk offsets.
    with phases('independent single-row flat oracle',device):
        flat=buildVerletList_(p.positions[:1],p.positions,p.supports[:1],p.supports,domain,
            verletScale=1.,mode=SupportScheme.SuperSymmetric,storeI=False)
        assert int(flat.numNeighbors[0])==n
        pattern=flat.paddedJ[:n].clone()
        torch.testing.assert_close(pattern.sort().values,torch.arange(n,device=device,dtype=torch.int32),rtol=0,atol=0)
    props=OperationProperties(kernel=KernelFunctions.Wendland2,supportMode=SupportScheme.SuperSymmetric)
    first=replace(p,positions=p.positions[:1],supports=p.supports[:1],masses=p.masses[:1],kinds=p.kinds[:1],densities=p.densities[:1])
    expected=launchOperator(_DENSITY_SPEC,SPHContext(first,props,domain,flat,p))[0].clone()
    ctx=SPHContext(p,props,domain,adj)
    with phases('production density eager and prepared',device):
        for prepared in (False,True):
            out=PreparedLauncher(ctx).launch(_DENSITY_SPEC) if prepared else launchOperator(_DENSITY_SPEC,ctx)
            torch.testing.assert_close(out,expected.expand(n),rtol=0,atol=0)
    with phases('allocate real owned work',device):
        field=ChunkedPairField.empty(adj,trailingShape=(3,),dtype=dtype)
        for t in field.buffers:t.fill_(float('nan'))
        field.validate(adj)
    with phases('stamp all owned work through prepared row dispatch',device):
        output=PreparedLauncher(ctx).launch(STAMP,work=field,rowStart=0,width=n)
        torch.testing.assert_close(output,torch.arange(n,device=device,dtype=dtype),rtol=0,atol=0)
    stream=wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type=='cuda' else None
    def check(adjacency,work,tag):
        probes=torch.full((3,3),-1,dtype=torch.int64,device=device)
        records=[];storages=[]
        with phases(tag,device):
            for index,c in enumerate(adjacency):
                a,m=c.adjacency,c.metadata
                assert a.edgeOffsets.dtype==get_torch_offset_dtype() and a.paddedI is None
                assert a.paddedJ.numel()==m.capacity and m.capacity<=policy.capacityLimit
                tensors=[a.paddedJ]+([] if work is None else [work.buffers[index]])
                for t in tensors:
                    assert t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()
                    storages.append((t.data_ptr(),t.data_ptr()+t.untyped_storage().nbytes()))
                errors=torch.zeros(2,dtype=torch.int32,device=device)
                buffer=work.buffers[index] if work is not None else torch.empty((1,3),dtype=dtype,device=device)
                wp.launch(audit,dim=m.liveEdges,inputs=[wp.from_torch(a.paddedJ),wp.from_torch(buffer,dtype=_vec),
                    wp.from_torch(pattern),n,wp.int64(m.edgeStart),wp.int64(total),work is not None,
                    wp.from_torch(errors),wp.from_torch(probes)],device=str(device),stream=stream)
                assert errors.tolist()==[0,0],(tag,index,errors.tolist())
                records.append(dict(rowStart=m.rowStart,rowStop=m.rowStop,edgeStart=m.edgeStart,liveEdges=m.liveEdges,
                    capacity=m.capacity,jPointer=a.paddedJ.data_ptr(),workPointer=None if work is None else buffer.data_ptr()))
                print(f'[wide rank {rank}] {tag}: chunk {index+1}/{len(adjacency.chunks)} global edges [{m.edgeStart},{m.edgeStart+m.liveEdges}) verified',flush=True)
        ordered=sorted(storages)
        assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),tag
        localProbes=probes.cpu().tolist()
        if comm:dist.all_reduce(probes,op=dist.ReduceOp.MAX,group=comm.group)
        allProbes=probes.cpu().tolist()
        wanted=[INT_MAX,INT_MAX+1,total-1]
        for i,at in enumerate(wanted):
            if at<total:
                assert allProbes[i]==[at,int(pattern[at%n]),at//n],(at,allProbes[i])
        return dict(chunks=records,localBoundaryProbes=localProbes,boundaryProbes=allProbes,
                    verifiedEdges=sum(c.metadata.liveEdges for c in adjacency),distinctStorageIntervals=len(storages))
    original=check(adj,field,'verify every J and work entry')
    filtered=None
    if args.filter:
        rows=max(1,min(1024,max(c.rowCount for c in adj.plan.chunks)//2))
        target=replace(policy,maxRows=rows)
        with phases('filter and redistribute changed row boundaries',device):
            new=filterChunkedVerletList(p,domain,adj,policy=target)
        assert new.numEdges==total and new.plan.signature!=adj.plan.signature
        field.validate(adj)
        filtered=check(new,None,'verify every redistributed J')
        retained=check(adj,field,'verify source J/work survived redistribution')
        with phases('production density on redistributed rows',device):
            actual=PreparedLauncher(replace(ctx,adjacency=new)).launch(_DENSITY_SPEC)
            torch.testing.assert_close(actual,expected.expand(n),rtol=0,atol=0)
        filtered.update(ownerRanges=new.plan.ownerRanges,sourceRetained=True,retainedAudit=retained)
    if comm:comm.agree('wide:passed',(total,adj.plan.signature))
    import warpSPHCore
    package=Path(warpSPHCore.__file__).resolve().parent
    result=dict(status='passed',rank=rank,ranks=size,device=str(device),particles=n,totalEdges=total,
        aboveIntMax=total>INT_MAX,edgesAboveIntMax=max(0,total-INT_MAX),localOffsetDtype=str(get_torch_offset_dtype()),
        globalPrefixDtype=str(adj.globalPrefix.dtype),lanes=args.lanes,sourceFiles={str(f.relative_to(package)):sha256(f.read_bytes()).hexdigest() for f in sorted(package.rglob('*.py'))},
        workerSha256=sha256(Path(__file__).read_bytes()).hexdigest(),preflight=dict(expectedGlobalBytes=expectedBytes,freeDeviceBytes=available),
        original=original,filtered=filtered,logicalJBytes=adj.memoryUsage()['pairCapacityBytes'],
        logicalWorkBytes=field.memoryUsage()['capacityBytes'],ownerRanges=adj.plan.ownerRanges,phases=phases.records,
        torch=torch.__version__,warp=wp.__version__,densityReference=float(expected))
    if device.type=='cuda':
        result.update(gpu=torch.cuda.get_device_name(device),gpuUuid=str(torch.cuda.get_device_properties(device).uuid),
            peakAllocatedBytes=torch.cuda.max_memory_allocated(device),peakReservedBytes=torch.cuda.max_memory_reserved(device),
            freeAfterBytes=torch.cuda.mem_get_info(device)[0])
    if comm:
        result['communication']=comm.summary()
        if comm.backend=='nccl':
            version=ctypes.c_int();assert ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))==0
            result['runtimeNccl']=version.value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles',type=int,default=17)
    parser.add_argument('--edge-budget',type=int,default=64)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--lanes',type=int,choices=(1,32),default=1)
    parser.add_argument('--filter',action='store_true')
    parser.add_argument('--require-wide',action='store_true')
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
    parser.add_argument('--memory-budget-gib',type=float,default=64.)
    parser.add_argument('--reserve-gib',type=float,default=8.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.particles<1 or args.memory_budget_gib<=0 or args.reserve_gib<0:parser.error('Positive particle/budget values and nonnegative reserve required')
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    setNeighborLanes(args.lanes);wp.init()
    distributed='RANK' in os.environ
    comm=None
    if distributed:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=600))
        comm=RowChunkProcessGroup(dist.group.WORLD,device,replicaChecks=args.replica_checks)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        result=run(args,device,comm)
        (args.output/f'wide-rank{result["rank"]}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'[wide rank {result["rank"]}] all wide-access checks passed',flush=True)
    finally:
        if distributed:dist.destroy_process_group()


if __name__=='__main__':main()
