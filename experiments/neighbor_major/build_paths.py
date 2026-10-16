#!/usr/bin/env python3
"""Compare direct and CSR-conversion construction, including search + Laplacian.

Selects the largest completed geometry for each dimension/degree/layout from
primary inputs. Complete pipeline uses the best measured 1/32 lane choice per
storage from those independent operator samples. No dispatch policy is changed.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ['WARPSPHCORE_PROFILING']='0'
os.environ['WARPSPHCORE_AUDIT']='0'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import torch
import warp as wp
import layout as experiment
import conversion
from benchmark import verifyRows
from bench_neighbor_search import measure,graphMeasure
from warpSPHCore import (ParticleState,OperationProperties,KernelFunctions,WarpOperation,SupportScheme,
                         GradientScheme,LaplacianScheme,setNeighborLanes)
from warpSPHCore.util import generateNeighborTestData
from warpSPHCore.math import getNextPrime

ORIGINAL_COMMON=experiment.commonInputs


def productionQueryInputs(grid,queryPositions,querySupports,periodic):
    """Match production search.py's vstack/mT query wrapping and strides."""
    flags=periodic.cpu().tolist()
    wrapped=torch.vstack([component if not flag else
        torch.remainder(component-grid.qMin[d],grid.qMax[d]-grid.qMin[d])+grid.qMin[d]
        for d,(component,flag) in enumerate(zip(queryPositions.mT,flags))]).mT
    values=[wrapped,querySupports,grid.sortedPositions,grid.sortedSupports,grid.hashTable,grid.sortedCellTable,
            grid.qMin,grid.hCell,grid.D,grid.cellOffsets,grid.numCells,grid.qMax,grid.qMin,periodic,wp.uint32(grid.mode_uint)]
    return [wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in values],wrapped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--repeat',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.repeat,args.warmup,args.batch)<=0:parser.error('positive sample counts required')
    if not torch.cuda.is_available():parser.error('CUDA required')
    inputs=[json.loads(p.read_text()) for p in args.inputs]
    if not all(p['complete'] for p in inputs):parser.error('primary sweeps must be complete')
    wp.init();torch.set_num_threads(2)
    selected=[]
    for report in inputs:
        groups={}
        for row in report['records']:
            key=(row['requested_neighbors'],row['layout'])
            if key not in groups or row['n']>groups[key]['n']:groups[key]=row
        selected.extend(groups.values())
    paths=sorted((ROOT/'src/warpSPHCore').rglob('*.py'))+[Path(__file__),Path(__file__).with_name('layout.py'),Path(__file__).with_name('conversion.py'),
           Path(__file__).with_name('benchmark.py'),ROOT/'scripts/benchmarks/bench_neighbor_search.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(records=[],complete=False,scope=__doc__,source_sha256=hashes,
                query_preparation='production vstack/mT column-major; row-major primary prototype also measured as a same-run control',
                primary_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,warp=wp.__version__,
                configuration=vars(args)|{'inputs':[str(p) for p in args.inputs],'output':str(args.output)},
                gpu_free_at_start_bytes=torch.cuda.mem_get_info()[0])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(selected):
        torch.cuda.reset_peak_memory_stats()
        n,dim,requested,geometry=row['n'],row['dim'],row['requested_neighbors'],row['layout']
        p,h,_,domain,dx=generateNeighborTestData(round(n**(1/dim)),requested,dim,True,'cuda')
        rng=torch.Generator(device='cuda').manual_seed(17)
        if geometry!='lattice':
            p+=(torch.rand(p.shape,device='cuda',dtype=p.dtype,generator=rng)-.5)*(.4*dx)
            p=(p-domain.min).remainder(domain.max-domain.min)+domain.min
            order=torch.randperm(n,device='cuda',generator=rng)
            p,h=p[order].contiguous(),h[order].contiguous();del order
        if geometry=='support-tail':h[::100]*=2
        scalar=torch.randn(n,device='cuda',dtype=p.dtype,generator=rng)
        particles=ParticleState(positions=p,supports=h,masses=torch.full_like(h,dx**dim),densities=torch.ones_like(h),
                                kinds=torch.zeros(n,dtype=torch.int32,device='cuda'))
        grid=experiment.buildCompactHashMap(p,p,h,h,domain.periodic,domain,SupportScheme.Gather,getNextPrime(n))
        csr=experiment.search(grid,p,h,domain.periodic,'csr')
        if csr.numEdges!=row['edges'] or csr.maximum!=row['maximum']:raise RuntimeError('geometry differs from primary sweep')
        record={k:row[k] for k in ('n','dim','layout','requested_neighbors','degree','maximum','padding_ratio')}
        record.update(conversion={},search={},pipeline={},checks={},pipeline_lanes={})
        record['neighbor_storage_bytes']=row['storage_bytes']
        for method in ('rows','tile'):
            target=conversion.convert(csr,method)
            record['temporary_csr_plus_neighbor_major_bytes']=csr.storageBytes+target.storageBytes-csr.numNeighbors.numel()*4
            verifyRows(csr,target)
            record['checks'][method+'_all_entries']=True
            def fill():conversion.fill(csr,target,method)
            record['conversion'][method]=graphMeasure(fill,args.repeat,args.warmup,args.batch)
            del target,fill
        del csr
        gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
        record['query_layout_control']={}
        for label,commonFn in [('row-major',ORIGINAL_COMMON),('production-column-major',productionQueryInputs)]:
            experiment.commonInputs=commonFn
            record['query_layout_control'][label]={}
            for storage in ('csr','neighbor-major'):
                def control():experiment.search(grid,p,h,domain.periodic,storage)
                record['query_layout_control'][label][storage]=measure(control,args.repeat,args.warmup)
                del control
        experiment.commonInputs=productionQueryInputs
        props=OperationProperties(kernel=KernelFunctions.Wendland2,operation=WarpOperation.Laplacian,
                                  gradientMode=GradientScheme.Difference,laplacianMode=LaplacianScheme.Brookshaw,supportMode=SupportScheme.Gather)
        # Builders produce a fully owned result; no CSR transposition is hidden
        # in the direct path. Conversion paths include their temporary CSR.
        def construct(path):
            if path=='direct':return experiment.search(grid,p,h,domain.periodic,'neighbor-major')
            source=experiment.search(grid,p,h,domain.periodic,'csr')
            return source if path=='csr' else conversion.convert(source,path)
        pathsOrder=['csr','direct','rows','tile']
        if index%2:pathsOrder.reverse()
        reference=None
        for path in pathsOrder:
            storage='csr' if path=='csr' else 'neighbor-major'
            best=min([o for o in row['operators'] if o['operation']=='laplacian' and o['storage']==storage],
                     key=lambda o:o['timing']['median_graph_device_ms_per_call'])['lanes']
            record['pipeline_lanes'][path]=best
            setNeighborLanes(best)
            built=construct(path)
            output=experiment.makeOperation('laplacian',built,particles,props,domain,scalar)()
            if reference is None:reference=output.clone()
            tol=3e-5 if p.dtype==torch.float32 else 3e-12
            torch.testing.assert_close(output,reference,rtol=tol,atol=tol*max(1.,float(reference.abs().max())))
            del built,output
            gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
            def search():construct(path)
            record['search'][path]=measure(search,args.repeat,args.warmup)
            def pipeline():
                topology=construct(path)
                experiment.makeOperation('laplacian',topology,particles,props,domain,scalar)()
            record['pipeline'][path]=measure(pipeline,args.repeat,args.warmup)
            del search,pipeline
        setNeighborLanes(None)
        record['checks']['pipeline_outputs']=True
        record['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['records'].append(record)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'{dim}D/{geometry} N={n} degree={row["degree"]:.2f}: build ms '+str({k:round(v['median_wall_ms'],3) for k,v in record['search'].items()})+
              '; search + Lap ms '+str({k:round(v['median_wall_ms'],3) for k,v in record['pipeline'].items()}),flush=True)
        del construct,grid,p,h,particles,scalar,reference
        gc.collect();torch.cuda.current_stream().synchronize();torch.cuda.empty_cache()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    if torch.cuda.is_available():
        stream=torch.cuda.Stream()
        experiment.commonInputs=productionQueryInputs
        try:
            with torch.cuda.stream(stream):main()
            stream.synchronize()
        finally:experiment.commonInputs=ORIGINAL_COMMON
    else:main()
