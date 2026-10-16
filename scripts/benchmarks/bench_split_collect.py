#!/usr/bin/env python3
"""Compare fused I/J collection with identical J search plus cooperative CSR row fill.

No particle reordering, compression, index-width changes or production policy
changes. Bare graph tests use preallocated buffers; public searches include
allocation, padding, counting and verification reads exactly as production does.
Use the production J-only specialization and split32 API when present. The
legacy prototype fallback removes one I store via a checked AST transform.
Split128 remains a benchmark-only row-fill policy.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _split_collect import loadJOnly, launchRows, splitSearch
from bench_neighbor_search import measure, graphMeasure, candidateVisits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024,65536,1048576,4194304])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[55,220])
    parser.add_argument('--layouts', nargs='+', choices=['lattice','jittered-shuffled'],default=['lattice','jittered-shuffled'])
    parser.add_argument('--dim',type=int,choices=[2,3],default=2)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(*args.ns,*args.neighbors,args.repeat,args.warmup,args.batch)<=0:
        parser.error('sizes and sample counts must be positive')
    if not torch.cuda.is_available(): parser.error('accessible CUDA required')
    from warpSPHCore import SupportScheme, CapacityPolicy
    from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors import radiusSearchCollectCompactHashMap
    from warpSPHCore.util import generateNeighborTestData
    from warpSPHCore.math import getNextPrime
    from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype

    torch.set_num_threads(2)
    wp.init()
    jOnly,generated=loadJOnly()
    productionSplit=hasattr(__import__('warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors',fromlist=['fillCSRRows']),'fillCSRRows')
    paths=sorted((ROOT/'src/warpSPHCore').rglob('*.py'))+[Path(__file__),Path(__file__).with_name('_split_collect.py'),Path(__file__).with_name('bench_neighbor_search.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(configuration=vars(args)|{'output':str(args.output)},records=[],complete=False,scope=__doc__,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,warp=wp.__version__,dtype=str(get_torch_precision()),
                index_dtype=str(get_torch_index_dtype()),offset_dtype=str(get_torch_offset_dtype()),
                source_sha256=hashes,generated_kernel=generated,production_split32=productionSplit,
                gpu_free_at_start_bytes=torch.cuda.mem_get_info()[0],seed=17)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    caseIndex=0
    for requested in args.neighbors:
        for n in args.ns:
            side=round(n**(1/args.dim))
            if side**args.dim!=n: parser.error('N must be an exact dim-th power')
            for layout in args.layouts:
                torch.cuda.reset_peak_memory_stats()
                positions,supports,_,domain,dx=generateNeighborTestData(side,requested,args.dim,True,'cuda')
                assert positions.shape[0]==n
                if layout=='jittered-shuffled':
                    rng=torch.Generator(device='cuda').manual_seed(17)
                    positions+=(torch.rand(positions.shape,device='cuda',dtype=positions.dtype,generator=rng)-.5)*(.4*dx)
                    positions=(positions-domain.min).remainder(domain.max-domain.min)+domain.min
                    order=torch.randperm(n,device='cuda',generator=rng)
                    positions,supports=positions[order].contiguous(),supports[order].contiguous()
                    del order
                H=getNextPrime(n); mode=SupportScheme.Gather; periodic=domain.periodic
                grid=buildCompactHashMap(positions,positions,supports,supports,periodic,domain,mode,H)
                adjacency=radiusSearchOnCompactHashMap(grid,positions,positions,supports,supports,periodic,domain,mode,H,collection='fused',queryOrder='input')
                edges=adjacency.numEdges
                counts=adjacency.numNeighbors.clone(); offsets=adjacency.edgeOffsets.clone()
                candidates=int(candidateVisits(positions,grid,periodic).sum())
                wrapped=(positions-domain.min).remainder(domain.max-domain.min)+domain.min
                inputs=[wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in
                        [wrapped,supports,grid.sortedPositions,grid.sortedSupports,grid.hashTable,grid.sortedCellTable,
                         grid.qMin,grid.hCell,grid.D,grid.cellOffsets,grid.numCells,grid.qMax,grid.qMin,periodic,wp.uint32(mode.value),
                         counts,offsets,grid.sortIndex,adjacency.paddedI,adjacency.paddedJ]]
                stream=wp.stream_from_torch(torch.cuda.current_stream())
                def fused(): wp.launch(radiusSearchCollectCompactHashMap,n,inputs=inputs,stream=stream)
                def jonly(): wp.launch(jOnly,n,inputs=inputs,stream=stream)
                def rows32():
                    if productionSplit:
                        from warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors import fillCSRRows
                        wp.launch(fillCSRRows,(n,32),inputs=[inputs[15],inputs[16],inputs[18]],block_dim=128,device=inputs[15].device,stream=stream)
                    else: launchRows(inputs[15],inputs[16],inputs[18],n,32,stream)
                def rows128(): launchRows(inputs[15],inputs[16],inputs[18],n,128,stream)
                def split32(): jonly(); rows32()
                def split128(): jonly(); rows128()
                # Check every output entry in bounded chunks without cloning the
                # entire multi-GB reference. J is checked against captured baseline
                # samples here and full exact-array equality on small test cases.
                sample=torch.linspace(0,edges-1,min(edges,65536),device='cuda').long()
                referenceJ=adjacency.paddedJ[sample].clone()
                bare={}
                phases=[('fused',fused),('j_only',jonly),('i_rows32',rows32),('i_rows128',rows128),('split32',split32),('split128',split128)]
                rotation=caseIndex%len(phases)
                for name,call in phases[rotation:]+phases[:rotation]:
                    bare[name]=graphMeasure(call,args.repeat,args.warmup,args.batch)
                    # Validate each measured phase before a later fused launch
                    # could overwrite evidence from the J-only specialization.
                    torch.testing.assert_close(adjacency.paddedJ[sample],referenceJ,rtol=0,atol=0)
                torch.testing.assert_close(adjacency.paddedJ[sample],referenceJ,rtol=0,atol=0)
                for begin in range(0,n,65536):
                    ids=torch.arange(begin,min(begin+65536,n),device='cuda',dtype=get_torch_index_dtype())
                    start=int(offsets[begin]); end=int(offsets[min(begin+65536,n)-1]+counts[min(begin+65536,n)-1])
                    expected=torch.repeat_interleave(ids,counts[begin:begin+65536].long(),output_size=end-start)
                    torch.testing.assert_close(adjacency.paddedI[start:end],expected,rtol=0,atol=0)
                    del ids,expected
                assert bool((adjacency.paddedI[edges:]==-1).all()) and bool((adjacency.paddedJ[edges:]==-1).all())
                row=dict(n=n,dim=args.dim,requested_neighbors=requested,layout=layout,degree=edges/n,edges=edges,
                         candidate_visits_per_grid_pass=candidates,search_candidate_distance_tests=2*candidates,
                         bare=bare,public={},
                         storage=dict(pair_capacity=adjacency.paddedI.numel(),
                                      pair_bytes=(adjacency.paddedI.numel()+adjacency.paddedJ.numel())*adjacency.paddedI.element_size(),
                                      csr_metadata_bytes=counts.numel()*counts.element_size()+offsets.numel()*offsets.element_size(),
                                      index_bytes=adjacency.paddedI.element_size(),additional_split_buffer_bytes=0),
                         checks=dict(all_i_entries=True,sampled_j_entries=sample.numel(),sampled_j_after_each_phase=True,padding=True))
                del adjacency,inputs,wrapped,sample,referenceJ,phases,call,fused,jonly,rows32,rows128,split32,split128
                gc.collect(); torch.cuda.current_stream().synchronize(); torch.cuda.empty_cache()
                def search(): radiusSearchOnCompactHashMap(grid,positions,positions,supports,supports,periodic,domain,mode,H,collection='fused',queryOrder='input')
                # Rotate order, keeping contexts outside timed calls.
                variants=['fused','split32','split128']
                rotation=caseIndex%len(variants)
                for name in variants[rotation:]+variants[:rotation]:
                    torch.cuda.current_stream().synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_bytes=torch.cuda.memory_allocated()
                    if name=='fused': row['public'][name]=measure(search,args.repeat,args.warmup)
                    elif name=='split32' and productionSplit:
                        def productionSearch():
                            radiusSearchOnCompactHashMap(grid,positions,positions,supports,supports,periodic,domain,mode,H,collection='split',queryOrder='input')
                        row['public'][name]=measure(productionSearch,args.repeat,args.warmup)
                    else:
                        with splitSearch(jOnly,int(name.removeprefix('split'))):
                            row['public'][name]=measure(search,args.repeat,args.warmup)
                    row['public'][name]['peak_additional_torch_allocated_bytes']=torch.cuda.max_memory_allocated()-baseline_bytes
                    row['public'][name]['useful_nonself_pairs_per_second']=(edges-n)*1000/row['public'][name]['median_wall_ms']
                    row['public'][name]['candidate_distance_tests_per_second']=2*candidates*1000/row['public'][name]['median_wall_ms']
                # In-place buffers: preserve shapes/addresses and overwritten tails.
                held=radiusSearchOnCompactHashMap(grid,positions,positions,supports,supports,periodic,domain,mode,H,collection='fused',queryOrder='input')
                addresses=[x.data_ptr() for x in (held.paddedI,held.paddedJ,held.numNeighbors,held.edgeOffsets)]
                for lanes in (32,128):
                    with splitSearch(jOnly,lanes):
                        got=radiusSearchOnCompactHashMap(grid,positions,positions,supports,supports,periodic,domain,mode,H,
                                                        capacity=CapacityPolicy(inPlace=True),reuse=held,collection='fused',queryOrder='input')
                    assert addresses==[x.data_ptr() for x in (got.paddedI,got.paddedJ,got.numNeighbors,got.edgeOffsets)]
                    torch.testing.assert_close(got.numNeighbors,counts,rtol=0,atol=0)
                    torch.testing.assert_close(got.edgeOffsets,offsets,rtol=0,atol=0)
                row['checks']['inplace_addresses_counts_offsets']=True
                row['torch_allocated_after_checks_bytes']=torch.cuda.memory_allocated()
                report['records'].append(row)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(f'{args.dim}D/{layout} N={n} degree={edges/n:.2f}: public ms '+str({k:round(v['median_wall_ms'],3) for k,v in row['public'].items()})+
                      '; bare graph ms '+str({k:round(v['graph']['median_device_interval_ms'],3) for k,v in bare.items()}),flush=True)
                del held,got,search,grid,positions,supports,counts,offsets
                gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
                caseIndex+=1
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    os.environ['WARPSPHCORE_PROFILING']='0'
    os.environ['WARPSPHCORE_AUDIT']='0'
    if torch.cuda.is_available():
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream): main()
        stream.synchronize()
    else: main()
