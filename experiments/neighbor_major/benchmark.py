#!/usr/bin/env python3
"""Direct neighbor-major search and seven primal operators versus compact CSR-J.

Only experiment files are used; production modules/specs are not patched. Both
layouts store int64 J only (no I), counts, and required metadata, isolating the
layout comparison from the independent I-array experiment. Positions stay in
original order. Search includes count, metadata read, allocation, initialization,
collect and checked overflow read. Grid building is timed separately.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['WARPSPHCORE_PROFILING']='0'
os.environ['WARPSPHCORE_AUDIT']='0'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))

import torch
import warp as wp
import layout as experiment
from bench_neighbor_search import measure,graphMeasure,candidateVisits
from bench_krylov import timeMatvec
from _work_metrics import WorkLedger
from warpSPHCore import (ParticleState,OperationProperties,KernelFunctions,WarpOperation,SupportScheme,
                         GradientScheme,LaplacianScheme,setNeighborLanes)
from warpSPHCore.math import getNextPrime
from warpSPHCore.util import generateNeighborTestData


def verifyRows(csr,ell):
    """Check every live/padded ELL entry in bounded row chunks against CSR."""
    counts,offsets=csr.numNeighbors,csr.edgeOffsets
    n=csr.numRows;slots=ell.paddedJ.view(-1,n);maximum=ell.maximum
    torch.testing.assert_close(ell.numNeighbors,counts,rtol=0,atol=0)
    k=torch.arange(maximum,device=counts.device).view(-1,1)
    for begin in range(0,n,8192):
        end=min(begin+8192,n)
        valid=k<counts[begin:end].view(1,-1)
        address=offsets[begin:end].long().view(1,-1)+k
        # Non-live slots may point at E; clamp only the unused gather addresses.
        expected=csr.paddedJ[address.clamp(max=max(0,csr.numEdges-1))] if csr.numEdges else torch.full_like(address,-1)
        expected=torch.where(valid,expected,-1)
        torch.testing.assert_close(slots[:,begin:end],expected,rtol=0,atol=0)
    return dict(all_live_j_entries=True,all_padding_entries=True,counts=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns',type=int,nargs='+',default=[1024,65536,1048576,4194304])
    parser.add_argument('--neighbors',type=int,nargs='+',default=[55,220])
    parser.add_argument('--layouts',nargs='+',choices=['lattice','jittered-shuffled','support-tail'],default=['lattice','jittered-shuffled','support-tail'])
    parser.add_argument('--operations',nargs='+',choices=list(experiment.BACKENDS),default=list(experiment.BACKENDS))
    parser.add_argument('--lanes',type=int,nargs='+',choices=[1,4,8,16,32],default=[1,32])
    parser.add_argument('--dim',type=int,choices=[2,3],default=2)
    parser.add_argument('--stress-max-n',type=int,default=1048576)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(*args.ns,*args.neighbors,args.repeat,args.warmup,args.batch)<=0:parser.error('positive sizes/sample counts required')
    if not torch.cuda.is_available():parser.error('accessible CUDA required')
    wp.init();torch.set_num_threads(2)
    sources=sorted((ROOT/'src/warpSPHCore').rglob('*.py'))+[Path(__file__),Path(__file__).with_name('layout.py')]
    sources += [ROOT/'scripts/benchmarks'/name for name in ('bench_neighbor_search.py','bench_krylov.py','_work_metrics.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(configuration=vars(args)|{'output':str(args.output)},records=[],complete=False,scope=__doc__,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,warp=wp.__version__,seed=17,
                gpu_free_at_start_bytes=torch.cuda.mem_get_info()[0],source_sha256=hashes,generated_sources={},
                precision=os.environ.get('warpSPHCore_PRECISION','default'),skipped_cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    caseIndex=0
    for requested in args.neighbors:
        for n in args.ns:
            side=round(n**(1/args.dim))
            if side**args.dim!=n:parser.error('N must be an exact dim-th power')
            for layout in args.layouts:
                if layout=='support-tail' and (requested!=min(args.neighbors) or n>args.stress_max_n):
                    report['skipped_cases'].append(dict(n=n,requested_neighbors=requested,layout=layout,reason='bounded padding stress subset'))
                    continue
                torch.cuda.reset_peak_memory_stats()
                free=torch.cuda.mem_get_info()[0]
                p,h,_,domain,dx=generateNeighborTestData(side,requested,args.dim,True,'cuda')
                assert len(p)==n
                rng=torch.Generator(device='cuda').manual_seed(17)
                if layout!='lattice':
                    p+=(torch.rand(p.shape,device='cuda',dtype=p.dtype,generator=rng)-.5)*(.4*dx)
                    p=(p-domain.min).remainder(domain.max-domain.min)+domain.min
                    order=torch.randperm(n,device='cuda',generator=rng)
                    p,h=p[order].contiguous(),h[order].contiguous()
                    del order
                if layout=='support-tail':h[::100]*=2
                originalPositions=p.clone()
                particles=ParticleState(positions=p,supports=h,masses=torch.full_like(h,dx**args.dim),
                                        densities=torch.ones_like(h),kinds=torch.zeros(n,dtype=torch.int32,device='cuda'))
                H=getNextPrime(n);mode=SupportScheme.Gather;periodic=domain.periodic
                grid=experiment.buildCompactHashMap(p,p,h,h,periodic,domain,mode,H)
                def build():experiment.buildCompactHashMap(p,p,h,h,periodic,domain,mode,H)
                buildTime=measure(build,args.repeat,args.warmup)
                lists={kind:experiment.search(grid,p,h,periodic,kind) for kind in ('csr','neighbor-major')}
                csr,ell=lists['csr'],lists['neighbor-major']
                checks=verifyRows(csr,ell)
                edges=csr.numEdges;maximum=ell.maximum;candidates=int(candidateVisits(p,grid,periodic).sum())
                row=dict(n=n,dim=args.dim,requested_neighbors=requested,layout=layout,edges=edges,degree=edges/n,
                         minimum=int(csr.numNeighbors.min()),maximum=maximum,candidate_visits_per_grid_pass=candidates,
                         padding_ratio=ell.paddedJ.numel()/edges if edges else 1.,wasted_slots=ell.paddedJ.numel()-edges,
                         storage_bytes={k:v.storageBytes for k,v in lists.items()},
                         j_bytes={k:v.paddedJ.numel()*8 for k,v in lists.items()},counts_dtype='int32',indices_dtype='int64',
                         grid_build=buildTime,search={},collect={},operators=[],checks=checks,free_before_case_bytes=free)
                # Bare collector reuses counts/storage. No transpose appears in
                # construction: each collector writes directly to its layout.
                common,wrapped=experiment.commonInputs(grid,p,h,periodic)
                kinds=['csr','neighbor-major']
                if caseIndex%2:kinds.reverse()
                for kind in kinds:
                    topology=lists[kind]
                    def collect():experiment.collectInto(grid,common,topology)
                    row['collect'][kind]=graphMeasure(collect,args.repeat,args.warmup,args.batch)
                    if bool(topology.overflow):raise OverflowError('bare collection overflow')
                row['checks'].update(verifyRows(csr,ell))
                del collect,topology,common,wrapped
                scalar=torch.randn(n,device='cuda',dtype=p.dtype,generator=rng)
                vector=torch.stack([scalar.roll(i) for i in range(args.dim)],dim=1).contiguous()
                for name in args.operations:
                    values=vector if name in ('curl','divergence') else scalar
                    props=OperationProperties(kernel=KernelFunctions.Wendland2,operation=getattr(WarpOperation,name.capitalize()),
                                              gradientMode=GradientScheme.Difference,laplacianMode=LaplacianScheme.Brookshaw,
                                              supportMode=mode)
                    modules={kind:experiment.operatorModule(name,kind)[0] for kind in kinds}
                    for lanes in args.lanes:
                        setNeighborLanes(lanes)
                        reference=None
                        order=kinds if (caseIndex+lanes)%2 else list(reversed(kinds))
                        for kind in order:
                            call=experiment.makeOperation(name,lists[kind],particles,props,domain,values)
                            before=time.perf_counter();result=call();torch.cuda.current_stream().synchronize()
                            setup=(time.perf_counter()-before)*1e3
                            if reference is None:reference=result.clone()
                            scale=max(1.,float(reference.abs().max()));difference=float((result-reference).abs().max())/scale
                            tolerance=3e-12 if p.dtype==torch.float64 else 3e-5
                            if not bool(torch.isfinite(result).all()) or difference>tolerance:raise RuntimeError(f'operator mismatch: {name}, {kind}, {lanes}, {difference}')
                            timing=timeMatvec(lambda _:call(),values,repeat=args.repeat,batch=args.batch)
                            ledger=WorkLedger().pair(name,evaluated_nonself_pairs=edges-n,candidate_entries=edges,
                                                    basis='exact gather neighbor count, no skin; same neighbor order; valid entries only, padding never evaluated')
                            op=dict(operation=name,storage=kind,lanes=lanes,timing=timing,
                                    relative_max_difference=difference,initial_setup_including_compile_ms=setup,
                                    device_work=ledger.report(wall_ms=timing['median_graph_device_ms_per_call'],timing_scope='per backend action, warmed graph batch'),
                                    wall_work=ledger.report(wall_ms=timing['median_eager_wall_ms'],timing_scope='matched primal state-backend call, synchronized wall'))
                            row['operators'].append(op)
                        del reference
                    del call,result,modules
                setNeighborLanes(None)
                torch.testing.assert_close(p,originalPositions,rtol=0,atol=0)
                row['checks']['particle_order_and_positions_unchanged']=True
                del lists,csr,ell,scalar,vector,values,originalPositions
                gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
                # Time allocating/checked search calls after releasing held lists.
                for kind in kinds:
                    def search():experiment.search(grid,p,h,periodic,kind)
                    row['search'][kind]=measure(search,args.repeat,args.warmup)
                    del search
                row['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
                report['records'].append(row)
                report['generated_sources']={f'collect/{k}':v[1] for k,v in experiment._COLLECT.items()}|{f'operator/{k}':v[1] for k,v in experiment._OPERATORS.items()}
                save()
                print(f'{args.dim}D/{layout} N={n} mean/max={edges/n:.2f}/{maximum}, padding={row["padding_ratio"]:.2f}x; '
                      f'search ms '+str({k:round(v['median_wall_ms'],3) for k,v in row['search'].items()})+
                      f'; {len(row["operators"])} operator variants passed',flush=True)
                del build,grid,particles,p,h
                gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
                caseIndex+=1
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    report['complete']=True;save()


if __name__=='__main__':
    if torch.cuda.is_available():
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):main()
        stream.synchronize()
    else:main()
