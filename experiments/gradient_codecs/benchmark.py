#!/usr/bin/env python3
"""Compare original/sorted Warp, native Torch, and raw/compressed native primals.

Every operator uses fixed exact CSR. Sorting, search, CRK-factor construction,
encoding, and qualification occur outside the operator timing interval.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library',type=Path,required=True)
parser.add_argument('--native-artifact',type=Path,required=True)
parser.add_argument('--planner-library',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--precision',choices=('float32','float64'),required=True)
parser.add_argument('--n',type=int,required=True)
parser.add_argument('--dimension',type=int,required=True)
parser.add_argument('--nh',type=float,required=True)
parser.add_argument('--screen',action='store_true')
parser.add_argument('--selection',type=Path)
parser.add_argument('--repeat',type=int,default=7)
parser.add_argument('--target-ms',type=float,default=10)
parser.add_argument('--profile',help='work:corrected:path, e.g. 1:1:band; profiles one resident call')
args=parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp',warpSPHCore_PRECISION=args.precision,
    warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64',WARPSPHCORE_NEIGHBOR_LANES='32')
sys.path.insert(0,str(ROOT/'src'))
import torch
import warp as wp
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch,sphOperation as native
from operator_runtime import Library,encode,request_for,packing_pitch,output_shape,NAMES,WORKLOADS,RAW64,RAW32,DELTA,BAND,NIBBLE
from common import fixture,morton_order,reorder,properties,crk_factors,activity,error,sample,WarpBatches
wp.init()
for module in ('wp_gradient','wp_divergence','wp_covariance'):
    wp.set_module_options({'enable_backward':False},module=f'warpSPHCore.coreOperations.native.{module}')
for module in ('crk_volume','crk_moments'):
    wp.set_module_options({'enable_backward':False},module=f'warpSPHCore.crk.{module}')
torch.ops.load_library(str(args.native_artifact))
build._extension=torch.ops.warpsph_native
torch.set_num_threads(16);torch.set_num_interop_threads(1)
library=Library(args.library)
selection=json.loads(args.selection.read_text())['selection'] if args.selection else {}
n,d=args.n,args.dimension
state,scalar,vector,domain=fixture(n,d,getattr(torch,args.precision),args.nh)
order=morton_order(state,args.planner_library,d)
inverse=torch.empty_like(order);inverse[order]=torch.arange(n,device='cuda')
sample_ids=torch.randperm(n,generator=torch.Generator().manual_seed(261101))[:min(n,16384)].to('cuda')
report=dict(complete=False,configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
    environment=dict(torch=torch.__version__,warp=wp.__version__,gpu=torch.cuda.get_device_name(),warp_lanes=32),
    particles=n,dimension=d,nh=args.nh,precision=args.precision,records=[],qualification=[],planning=[],selection={},activity=[],
    sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [args.library,args.native_artifact,args.planner_library,*Path(__file__).parent.glob('*.py'),
         Path(__file__).parent/'kernels.cu',ROOT/'experiments/csr_small_gaps/kernels.cu',ROOT/'experiments/csr_compression/kernels.cu']})


def save():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


def check_activity():
    snapshot=activity();report['activity'].append(snapshot)
    if snapshot['other_processes']:
        save();raise RuntimeError(f'Other GPU jobs: {snapshot["other_processes"]}')


def qualification(actual,expected,work,corrected,path,order_name):
    metrics=error(actual,expected)
    tolerance=4e-4 if args.precision=='float32' else 3e-8
    qualified=metrics['finite'] and metrics['relative_l2']<tolerance
    # Warp's periodic (dx+L/2)-L/2 loses significant bits for tiny FP32 1D
    # supports. Retain its measurements, but never label these outputs equivalent.
    known_geometry_case=args.precision=='float32' and d==1
    if not qualified and not known_geometry_case:
        raise AssertionError(f'{path}: unexpected numerical disagreement {metrics}')
    report['qualification'].append(dict(work=WORKLOADS[work],corrected=corrected,path=path,
        order=order_name,qualified=qualified,known_fp32_1d_geometry=known_geometry_case and not qualified,**metrics))


def measure_group(functions,work,corrected,order_name,configurations=None):
    for function in functions.values():function()
    pilot=sample(next(iter(functions.values())),1)
    iterations=min(20,max(1,round(args.target_ms/pilot['event_ms'])))
    samples={name:[] for name in functions}
    for repeat in range(args.repeat):
        names=list(functions);random.Random(261101+repeat).shuffle(names)
        for name in names:samples[name].append(sample(functions[name],iterations))
    for name,data in samples.items():
        report['records'].append(dict(work=WORKLOADS[work],corrected=corrected,order=order_name,path=name,
            event_ms=statistics.median(x['event_ms'] for x in data),wall_ms=statistics.median(x['wall_ms'] for x in data),
            samples=data,iterations=iterations,configuration=configurations.get(name) if configurations else None))


check_activity();save()
print('Preparing original exact CSR and CRK factors',flush=True)
adjacency=radiusSearch(state,domain)
report['edges']=adjacency.numEdges;report['mean_neighbors']=adjacency.numEdges/n
original_warp_plan=WarpBatches(state,adjacency,domain) if adjacency.numEdges>2**31-1 else None
report['warp_batches_original']=len(original_warp_plan.batches) if original_warp_plan else 1
volumes,correction,report['crk_setup']=crk_factors(state,domain,adjacency,batch_plan=original_warp_plan)


def original_phase():
    indices32=adjacency.paddedJ.int()
    adj32=sph.AdjacencyList(None,indices32,adjacency.numNeighbors,adjacency.edgeOffsets,n,n,numEdges=adjacency.numEdges)
    expected_samples={}
    if not args.screen and not args.profile:
        for work,corrected in itertools.product(range(4),(False,True)):
            props=properties(work);values=scalar if work==0 else vector
            kwargs=dict(queryVolumes=volumes,referenceVolumes=volumes,crkState=correction) if corrected else {}
            warp_call=(original_warp_plan.operation(props,values,volumes if corrected else None,correction if corrected else None)
                       if original_warp_plan else lambda:sph.warpOperation(state,props,domain,values,adjacency=adjacency,**kwargs))
            expected=warp_call();expected_samples[work,corrected]=expected[sample_ids].clone()
            functions={'warp_unsorted':warp_call}
            if not corrected:
                native_call=lambda:native(state,props,domain,values,adjacency=adj32)
                qualification(native_call(),expected,work,corrected,'native_torch_unsorted','original')
                functions['native_torch_unsorted']=native_call
            measure_group(functions,work,corrected,'original')
            save()
    return expected_samples,adjacency.numNeighbors.clone()


original_samples,original_counts=original_phase()
del adjacency,original_warp_plan
torch.cuda.empty_cache()
print('Preparing refined Morton exact CSR',flush=True)
state=reorder(state,order);scalar=scalar[order];vector=vector[order];volumes=volumes[order]
correction=sph.CRKState(**{name:getattr(correction,name)[order].contiguous() for name in ('A','B','gradA','gradB')})
adjacency=radiusSearch(state,domain)
assert torch.equal(adjacency.numNeighbors,original_counts[order]),'Sorting changed neighbor counts'
report['sorting_membership']='Identical per-particle counts; small qualification separately checks full edge membership.'
indices32=adjacency.paddedJ.int()
adj32=sph.AdjacencyList(None,indices32,adjacency.numNeighbors,adjacency.edgeOffsets,n,n,numEdges=adjacency.numEdges)
report['csr32_bytes']=indices32.numel()*4+adjacency.numNeighbors.numel()*4+adjacency.edgeOffsets.numel()*8
report['csr64_bytes']=adjacency.paddedJ.numel()*8+adjacency.numNeighbors.numel()*4+adjacency.edgeOffsets.numel()*8
sorted_warp_plan=WarpBatches(state,adjacency,domain) if adjacency.numEdges>2**31-1 else None
report['warp_batches_sorted']=len(sorted_warp_plan.batches) if sorted_warp_plan else 1
check_activity()
for work,corrected in itertools.product(range(4),(False,True)):
    key=f'{args.precision}:{d}:{args.nh:g}:{work}:{int(corrected)}'
    props=properties(work);values=scalar if work==0 else vector
    kwargs=dict(queryVolumes=volumes,referenceVolumes=volumes,crkState=correction) if corrected else {}
    warp_call=(sorted_warp_plan.operation(props,values,volumes if corrected else None,correction if corrected else None)
               if sorted_warp_plan else lambda:sph.warpOperation(state,props,domain,values,adjacency=adjacency,**kwargs))
    expected=warp_call()
    if (work,corrected) in original_samples:
        qualification(expected[inverse[sample_ids]],original_samples[work,corrected],work,corrected,'warp_sorted_permutation','sorted')
    choices={encoding:(list(itertools.product((8,16,32),(1,2),(1,2))) if args.screen else [selection[key][NAMES[encoding]]])
             for encoding in NAMES}
    plans={}
    for encoding in (DELTA,BAND,NIBBLE):
        for chunk in sorted({configuration[0] for configuration in choices[encoding]}):
            torch.cuda.synchronize();begin=time.perf_counter()
            plan=encode(library,adjacency.paddedJ,adjacency.numNeighbors,adjacency.edgeOffsets,chunk,encoding)
            torch.cuda.synchronize()
            plans[encoding,chunk]=plan
            report['planning'].append(dict(work=WORKLOADS[work],corrected=corrected,format=NAMES[encoding],chunk=chunk,
                wall_ms=(time.perf_counter()-begin)*1000,storage=plan.storage()))
    packed=values.new_empty((n,packing_pitch(work,d)))
    output=values.new_empty(output_shape(work,n,d))
    common={name:getattr(state,name) for name in ('positions','supports','masses','densities')}
    common.update(values=values,lower=domain.min,upper=domain.max,periodic=domain.periodic,packed=packed,output=output)
    functions,configurations,owners={}, {}, []
    for encoding,configs in choices.items():
        for chunk,items,batch in configs:
            name=f'{NAMES[encoding]}_{chunk}_{items}_{batch}' if args.screen else NAMES[encoding]
            topology=plans[encoding,chunk].buffers if encoding>=DELTA else dict(counts=adjacency.numNeighbors,
                row_offsets=adjacency.edgeOffsets,indices=adjacency.paddedJ if encoding==RAW64 else indices32)
            resident=request_for(common,topology,n,d,int(args.precision=='float64'),volumes,correction)
            owners.append(resident)
            def resident_call(request=resident,encoding=encoding,chunk=chunk,items=items,batch=batch):
                library.run(request,encoding,work,corrected,chunk,items,batch)
                return output
            qualification(resident_call(),expected,work,corrected,name,'sorted')
            if args.screen:
                functions[name]=resident_call
            else:
                def public(topology=topology,encoding=encoding,chunk=chunk,items=items,batch=batch):
                    local_packed=torch.empty_like(packed);local_output=torch.empty_like(output)
                    request=request_for(dict(common,packed=local_packed,output=local_output),topology,n,d,
                                        int(args.precision=='float64'),volumes,correction)
                    library.run(request,encoding,work,corrected,chunk,items,batch)
                    return local_output
                functions[name]=public
            configurations[name]=[chunk,items,batch]
    if not args.screen:
        functions['warp_sorted']=warp_call
        if not corrected:
            native_call=lambda:native(state,props,domain,values,adjacency=adj32)
            qualification(native_call(),expected,work,corrected,'native_torch_sorted','sorted')
            functions['native_torch_sorted']=native_call
    if args.profile:
        requested_work,requested_corrected,path=args.profile.split(':')
        if int(requested_work)==work and bool(int(requested_corrected))==corrected:
            torch.cuda.nvtx.range_push(f'{WORKLOADS[work]}_crk{int(corrected)}_{path}')
            torch.cuda.cudart().cudaProfilerStart();functions[path]();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.nvtx.range_pop()
    else:
        print(f'Timing {WORKLOADS[work]}, CRK={corrected}',flush=True)
        measure_group(functions,work,corrected,'sorted',configurations)
    if args.screen:
        records=[r for r in report['records'] if r['work']==WORKLOADS[work] and r['corrected']==corrected]
        report['selection'][key]={NAMES[encoding]:min((r for r in records if r['path'].startswith(NAMES[encoding]+'_')),
            key=lambda r:r['event_ms'])['configuration'] for encoding in NAMES}
    save()
    functions.clear();plans.clear();owners.clear()
    if not args.screen:del public
    del plan,resident,topology,common,packed,output,expected
    torch.cuda.empty_cache()
check_activity()
report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
report['complete']=True
report['uncontended_process_snapshots']=all(not x['other_processes'] for x in report['activity'])
save()
print('Complete',args.output,flush=True)
