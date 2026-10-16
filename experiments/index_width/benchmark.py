#!/usr/bin/env python3
"""Matched CSR int32/int64 IDs, collection, SPH kernels and Torch bridges."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['WARPSPHCORE_PROFILING']='0'
os.environ['WARPSPHCORE_AUDIT']='0'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import torch
import warp as wp
import kernels
import layout
from build_paths import productionQueryInputs
from bench_neighbor_search import graphMeasure, measure
from bench_krylov import timeMatvec
from warpSPHCore import (ParticleState,OperationProperties,KernelFunctions,WarpOperation,SupportScheme,
                         GradientScheme,LaplacianScheme,setNeighborLanes)
from warpSPHCore.util import generateNeighborTestData
from warpSPHCore.math import getNextPrime
from warpSPHCore.autograd.operator_spec import SPHContext
from warpSPHCore.mfm.fastLaunch import PreparedLauncher
from warpSPHCore.autograd.stream import currentTorchWarpStream


def torchCompatibility(device):
    x=torch.arange(8.,device=device); idx=torch.tensor([0,3,6],dtype=torch.int32,device=device)
    val=torch.ones(3,device=device)
    actions={
        'advanced_index':lambda i:x[i], 'index_select':lambda i:torch.index_select(x,0,i),
        'gather':lambda i:torch.gather(x,0,i), 'take':lambda i:torch.take(x,i),
        'index_add':lambda i:torch.zeros_like(x).index_add_(0,i,val),
        'scatter':lambda i:torch.zeros_like(x).scatter_(0,i,val),
        'scatter_add':lambda i:torch.zeros_like(x).scatter_add_(0,i,val),
        'scatter_reduce_sum':lambda i:torch.zeros_like(x).scatter_reduce_(0,i,val,reduce='sum'),
    }
    result={}
    for name,call in actions.items():
        reference=call(idx.long())
        try:
            actual=call(idx)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            result[name]={'int32':True,'matches_int64':True}
        except RuntimeError as exc:result[name]={'int32':False,'error':str(exc).split('\n')[0]}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns',nargs='+',type=int,default=[1024,65536,1048576,4194304])
    parser.add_argument('--neighbors',nargs='+',type=int,default=[55,220])
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--no-shuffled',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    wp.init();torch.set_num_threads(2)
    paths=sorted((ROOT/'src/warpSPHCore').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'experiments/neighbor_major'/p for p in ('layout.py','build_paths.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    modules={name:kernels.operator(name) for name in ('density','gradient','laplacian')}
    collectors={bits:kernels.collector(bits) for bits in (32,64)}
    report=dict(complete=False,records=[],torch=torch.__version__,warp=wp.__version__,gpu=torch.cuda.get_device_name(),
                free_bytes_at_start=torch.cuda.mem_get_info()[0],source_sha256=hashes,
                configuration=vars(args)|{'output':str(args.output)},
                torch_compatibility={d:torchCompatibility(d) for d in ('cpu','cuda')},
                generated_sources={name:info for name,(_,info) in modules.items()}|
                    {str(bits):info for bits,(_,info) in collectors.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for requested in args.neighbors:
        for n in args.ns:
            for shuffled in ([False,True] if n==max(args.ns) and not args.no_shuffled else [False]):
                side=round(n**.5)
                assert side**2==n and 0<n<2**31
                p,h,_,domain,dx=generateNeighborTestData(side,requested,2,True,'cuda')
                rng=torch.Generator(device='cuda').manual_seed(17)
                if shuffled:
                    p=(p+(torch.rand(p.shape,device='cuda',generator=rng)-.5)*(.4*dx)-domain.min).remainder(domain.max-domain.min)+domain.min
                    order=torch.randperm(n,device='cuda',generator=rng)
                    p,h=p[order].contiguous(),h[order].contiguous();del order
                particles=ParticleState(positions=p,supports=h,masses=torch.full_like(h,dx**2),densities=torch.ones_like(h),
                                        kinds=torch.zeros(n,dtype=torch.int32,device='cuda'))
                grid=layout.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,getNextPrime(n))
                # Reference search is used only for verified counts and J; its timing is excluded.
                csr=layout.search(grid,p,h,domain.periodic,'csr')
                edges=csr.numEdges
                assert edges<2**31
                common,wrapped=productionQueryInputs(grid,p,h,domain.periodic)
                expectedJ=csr.paddedJ.clone()
                ids={64:csr.paddedJ,32:csr.paddedJ.int()}
                rows={bits:torch.empty_like(ids[bits]) for bits in (32,64)}
                row=dict(n=n,requested_neighbors=requested,layout='jittered-shuffled' if shuffled else 'lattice',edges=edges,degree=edges/n,
                         pair_storage_bytes={str(b):2*edges*(b//8) for b in (32,64)},
                         j_storage_bytes={str(b):edges*(b//8) for b in (32,64)},
                         csr_metadata_bytes=8*n,collect={},operators=[],casts={})
                bitOrder=(32,64) if len(report['records'])%2 else (64,32)
                for bits in bitOrder:
                    kernel=collectors[bits][0]
                    inputs=common+[wp.from_torch(t) for t in (csr.numNeighbors,csr.edgeOffsets,grid.sortIndex,rows[bits],ids[bits])]
                    def collect():wp.launch(kernel,n,inputs=inputs,device='cuda',stream=currentTorchWarpStream(p.device))
                    row['collect'][str(bits)]=graphMeasure(collect,args.repeat,3,args.batch)
                    # Poisoned outputs ensure every live entry is overwritten on replay.
                    ids[bits].fill_(-1);rows[bits].fill_(-1);collect()
                torch.testing.assert_close(ids[32].long(),ids[64],rtol=0,atol=0)
                torch.testing.assert_close(ids[64],expectedJ,rtol=0,atol=0)
                torch.testing.assert_close(rows[32].long(),rows[64],rtol=0,atol=0)
                del expectedJ
                # Validate CSR row identities without constructing a full repeat_interleave temporary.
                for lo in range(0,n,65536):
                    hi=min(n,lo+65536)
                    begin=int(csr.edgeOffsets[lo]);end=int(csr.edgeOffsets[hi-1]+csr.numNeighbors[hi-1])
                    expected=torch.repeat_interleave(torch.arange(lo,hi,device='cuda'),csr.numNeighbors[lo:hi].long())
                    torch.testing.assert_close(rows[64][begin:end],expected,rtol=0,atol=0)
                row['checks']={'all_collected_i_j_entries_equal':True,'all_row_ids_correct':True}
                values=torch.randn(n,1,device='cuda',generator=rng)
                for name,(module,_) in modules.items():
                    props=OperationProperties(kernel=KernelFunctions.Wendland2,operation=getattr(WarpOperation,name.capitalize()),
                        gradientMode=GradientScheme.Difference,laplacianMode=LaplacianScheme.Brookshaw,supportMode=SupportScheme.Gather)
                    ctx=SPHContext(query=particles,reference=particles,properties=props,domain=domain,adjacency=csr)
                    prepared={b:PreparedLauncher(ctx) for b in (32,64)}
                    a=kernels.Adjacency32();a.neighborList=wp.from_torch(ids[32]);a.neighborOffsets=wp.from_torch(csr.edgeOffsets);a.numNeighbors=wp.from_torch(csr.numNeighbors)
                    prefix=list(prepared[32].prefix);prefix[4]=a;prepared[32].prefix=tuple(prefix)
                    extras={} if name=='density' else dict(numDims=2,flatInputShape=1,flatOutputShape=2 if name=='gradient' else 1,
                                                          queryValuesFlat=values,referenceValuesFlat=values)
                    spec=getattr(module,'_'+name.upper()+'_SPEC')
                    for lanes in (1,32):
                        setNeighborLanes(lanes)
                        outputs={b:prepared[b].launch(spec,**extras) for b in (32,64)}
                        torch.testing.assert_close(outputs[32],outputs[64],rtol=0,atol=0)
                        assert bool(torch.isfinite(outputs[32]).all())
                        # Also compare cloned CSR-only code to the unmodified production kernel.
                        original=__import__('importlib').import_module('warpSPHCore.coreOperations.wp_'+name)
                        reference=prepared[64].launch(getattr(original,'_'+name.upper()+'_SPEC'),**extras)
                        torch.testing.assert_close(outputs[64],reference,rtol=0,atol=0)
                        for bits in bitOrder:
                            call=lambda _:prepared[bits].launch(spec,**extras)
                            timing=timeMatvec(call,values,repeat=args.repeat,batch=args.batch)
                            ms=timing['median_graph_device_ms_per_call']
                            row['operators'].append(dict(operation=name,bits=bits,lanes=lanes,timing=timing,
                                nonself_interactions_per_second=(edges-n)/(ms*1e-3),exact_outputs=True,exact_production_reference=True))
                        del outputs,reference,call
                    del prepared,a,prefix,ctx
                setNeighborLanes(None)
                # Reuse already allocated int64 outputs: conversion bandwidth without allocation.
                row['casts']['j32_to64_preallocated']=graphMeasure(lambda:ids[64].copy_(ids[32]),args.repeat,3,args.batch)
                row['casts']['ij32_to64_preallocated']=graphMeasure(lambda:(ids[64].copy_(ids[32]),rows[64].copy_(rows[32])),args.repeat,3,args.batch)
                def allocateCast():ids[32].long()
                row['casts']['j32_to64_allocating_wall']=measure(allocateCast,args.repeat,3)
                row['torch_indexing']={}
                source=values[:,0].contiguous()
                for opname in ('advanced_index','index_select','gather'):
                    row['torch_indexing'][opname]={}
                    for bits in bitOrder:
                        index=ids[bits]
                        if opname=='advanced_index':action=lambda:source[index]
                        elif opname=='index_select':action=lambda:torch.index_select(source,0,index)
                        else:action=lambda:torch.gather(source,0,index)
                        row['torch_indexing'][opname][str(bits)]=graphMeasure(action,args.repeat,3,args.batch)
                del source,index,action
                row['checks']['eager_matches_capture']=True
                report['records'].append(row);save()
                print(f'N={n} k={edges/n:.2f} {row["layout"]}: collected IDs and 12 operator variants exactly matched',flush=True)
                del allocateCast,collect,inputs,kernel,csr,ids,rows,common,wrapped,particles,grid,p,h,values
                gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
    report['complete']=True;save()


if __name__=='__main__':
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):main()
    stream.synchronize()
