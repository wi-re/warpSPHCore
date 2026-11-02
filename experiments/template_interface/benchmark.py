"""Paired CPU/GPU timings for the template, direct-loop and existing operators."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import torch
from template_runtime import Library,Plan,SLOTS,field_count,transpose
from fixtures import load_core,fixture,to_device


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core',type=Path,required=True)
parser.add_argument('--native',type=Path,required=True);parser.add_argument('--planner',type=Path,required=True)
parser.add_argument('--cpu',type=Path,required=True);parser.add_argument('--gpu',type=Path,required=True)
parser.add_argument('--precision',choices=('float32','float64'),required=True)
parser.add_argument('--n',type=int,required=True);parser.add_argument('--dimension',type=int,required=True)
parser.add_argument('--nh',type=float,default=4);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--target-ms',type=float,default=8)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
torch.set_num_threads(24);torch.set_num_interop_threads(1)
sph=load_core(args.core,args.native,args.precision)
from warpSPHCore.native.api import sphOperation as native
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
import warp as wp
wp.init()
for module in ('density','interpolate','gradient','covariance'):
    wp.set_module_options({'enable_backward':False},module=f'warpSPHCore.coreOperations.native.wp_{module}')
wp.set_module_options({'enable_backward':False},module='warpSPHCore.crk.crk_moments')
library=Library(args.cpu,args.gpu)
report=dict(complete=False,n=args.n,dimension=args.dimension,precision=args.precision,nh=args.nh,
            threads=torch.get_num_threads(),records=[],qualification=[],activity=[],planning={})


def save():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


def activity():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip().splitlines()
    others=[line for line in lines if line.split(',')[0].strip()!=str(os.getpid())]
    item=dict(processes=lines,other_processes=others,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True).strip())
    report['activity'].append(item)
    if others:
        save();raise RuntimeError(f'Concurrent GPU workload: {others}')


def qualify(value,expected,label):
    delta=value.reshape(-1).double()-expected.reshape(-1).double()
    error=float(delta.norm()/expected.double().norm().clamp_min(1e-30))
    finite=bool(torch.isfinite(value).all())
    tolerance=5e-4 if args.precision=='float32' else 3e-9
    qualified=finite and error<tolerance
    report['qualification'].append(dict(label=label,error=error,finite=finite,qualified=qualified))
    # Warp's existing FP32 1D periodic geometry loses tiny separations. Retain
    # timings but explicitly flag those comparisons rather than relaxing all tests.
    if not qualified and not ('warp' in label and args.dimension==1 and args.precision=='float32'):
        save();raise AssertionError((label,error))


def sample(function,device,iterations,batched=False):
    if device=='cuda':
        torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();begin.record()
        if batched:result=function(iterations)
        else:
            for _ in range(iterations):result=function()
        end.record();end.synchronize()
        return dict(event_ms=begin.elapsed_time(end)/iterations,wall_ms=(time.perf_counter()-start)*1000/iterations)
    start=time.perf_counter()
    if batched:result=function(iterations)
    else:
        for _ in range(iterations):result=function()
    return dict(wall_ms=(time.perf_counter()-start)*1000/iterations)


def paired(calls,device,context,batched=False):
    active={}
    for name,function in calls.items():
        if batched:function(1)
        else:function()
        measurement=sample(function,device,16 if batched else 1,batched)
        elapsed=measurement.get('event_ms',measurement['wall_ms'])
        active[name]=max(1,min(1000 if batched else 100,int(args.target_ms/max(elapsed,.005))))
    batches={name:[] for name in calls}
    generator=random.Random(7751+len(report['records']))
    for repeat in range(args.repeats):
        names=list(calls);generator.shuffle(names)
        for name in names:batches[name].append(sample(calls[name],device,active[name],batched))
    for name,values in batches.items():
        record=dict(context,device=device,path=name,samples=values,iterations=active[name])
        for key in values[0]:record[key]=statistics.median(item[key] for item in values)
        report['records'].append(record)
    save()


activity()
state,scalar,vector,domain,adjacency,csr,tiles,data,planning=fixture(sph,args.core,args.planner,args.n,args.dimension,getattr(torch,args.precision),args.nh)
report['planning'].update(planning)
start=time.perf_counter();reverse=transpose(csr,args.n,args.n);torch.cuda.synchronize()
report['planning'].update(transpose_ms=(time.perf_counter()-start)*1000,transpose_bytes=reverse.bytes())


def properties(work):
    operation={0:sph.WarpOperation.Density,1:sph.WarpOperation.Interpolate,
        2:sph.WarpOperation.Gradient,3:sph.WarpOperation.Gradient,4:sph.WarpOperation.Covariance,5:sph.WarpOperation.Interpolate}[work]
    return sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=operation,
        gradientMode=sph.GradientScheme.Difference,supportMode=sph.SupportScheme.Gather,
        operationMode=sph.OperationDirection.TrueAllToToAll)


for device in ('cuda','cpu'):
    activity()
    local=to_device(data,device);local_csr=csr.to(device);local_reverse=reverse.to(device);local_tiles=tiles.to(device)
    local64=Plan(dict(local_csr.buffers,indices=local_csr.buffers['indices'].long()))
    local_scalar=scalar.to(device);local_vector=vector.to(device)
    local_state=replace(state,**{name:getattr(state,name).to(device) for name in ('positions','supports','masses','densities','kinds')})
    local_domain=sph.DomainDescription(domain.min.to(device),domain.max.to(device),domain.periodic.to(device),args.dimension)
    local_adjacency=sph.AdjacencyList(None,local_csr.buffers['indices'],local_csr.buffers['counts'],local_csr.buffers['offsets'],args.n,args.n,numEdges=planning['edges'])
    warp_adjacency=sph.AdjacencyList(None,local64.buffers['indices'],local64.buffers['counts'],local64.buffers['offsets'],args.n,args.n,numEdges=planning['edges'])
    works=(1,2,5) if args.smoke else (0,1,2,3,4,5,6)
    for work in works:
        fields=local_vector if work==3 else local_scalar
        current=dict(local,qvalue=fields if work in (2,3) else None,
                     rvalue=fields if field_count(work,args.dimension) else None)
        expected=library.forward(current,local_csr,work).values
        def call(plan=local_csr,direct=False):return library.forward(current,plan,work,direct=direct).values
        calls={'template32':call,'direct32':lambda:call(direct=True),'template64':lambda:call(local64)}
        if device=='cuda':calls['cell_shared']=lambda:call(local_tiles)
        if work!=6:
            props=properties(work)
            public_fields=fields if work==3 else fields[:,0]
            ones=torch.ones_like(public_fields)
            def public(backend):
                function=native if backend=='native' else sph.warpOperation
                selected_adjacency=local_adjacency if backend=='native' else warp_adjacency
                kwargs=dict(queryValues=public_fields,referenceValues=public_fields,adjacency=selected_adjacency)
                if work==0:kwargs.pop('queryValues');kwargs.pop('referenceValues')
                if work==5:
                    numerator=function(local_state,props,local_domain,**kwargs)
                    denominator=function(local_state,props,local_domain,queryValues=ones,referenceValues=ones,adjacency=selected_adjacency)
                    return torch.where(denominator!=0,numerator/torch.where(denominator!=0,denominator,torch.ones_like(denominator)),torch.zeros_like(numerator))
                return function(local_state,props,local_domain,**kwargs)
            calls['native_torch']=lambda:public('native')
            if device=='cuda':calls['warp']=lambda:public('warp')
        elif device=='cuda':
            moment_props=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Interpolate,supportMode=sph.SupportScheme.Gather)
            moment_volumes=local_state.masses/local_state.densities
            def warp_moments():
                raw=_computeCRKMoments_stateBackend(local_state,moment_props,local_domain,moment_volumes,moment_volumes,adjacency=warp_adjacency)
                return torch.cat([value.reshape(args.n,-1) for value in raw[:6]],1)
            calls['warp']=warp_moments
        for name,function in calls.items():qualify(function(),expected,f'{device}:{work}:{name}')
        paired(calls,device,dict(phase='forward',work=work))
        # A resident pair isolates template orchestration from Python allocation
        # and packing. Both paths use exactly the same interaction and schedule.
        buffers=library.forward(current,local_csr,work)
        paired({'template32':lambda iterations:library.forward(current,local_csr,work,buffers=buffers,pack=False,iterations=iterations),
                'direct32':lambda iterations:library.forward(current,local_csr,work,buffers=buffers,pack=False,direct=True,iterations=iterations)},
               device,dict(phase='resident_forward',work=work),batched=True)
        if work not in (1,2,5):continue
        saved=library.forward(current,local_csr,work,save=True)
        cot=torch.cos(torch.arange(saved.values.numel(),device=device,dtype=fields.dtype)).reshape_as(saved.values)
        tangents=tuple(torch.full_like(current[name],.01) if current.get(name) is not None else None for name in SLOTS)
        for requested in ('all','fields'):
            if requested=='fields':
                gradient_buffers=(None,)*5+(torch.empty_like(current['qvalue']) if current['qvalue'] is not None else None,
                                            torch.empty_like(current['rvalue']))
                tangent_buffers=(None,)*5+(tangents[5],tangents[6])
            else:
                gradient_buffers=tuple(torch.empty_like(current[name]) if current.get(name) is not None else None for name in SLOTS)
                tangent_buffers=tangents
            owned=lambda:library.derivative(current,local_csr,local_reverse,work,cotangent=cot,cache=saved.cache,gradients=gradient_buffers)
            atomic=lambda:library.derivative(current,local_csr,local_reverse,work,cotangent=cot,cache=saved.cache,gradients=gradient_buffers,atomic=True)
            # Verification already checks each implementation against dense AD.
            # Here retain one paired agreement check on the actual large graph.
            first=tuple(value.clone() if value is not None else None for value in owned())
            second=atomic()
            for slot,(a,b) in enumerate(zip(first,second)):
                if a is not None:qualify(b,a,f'{device}:{work}:vjp:{requested}:{slot}')
            paired({'owned':owned,'atomic':atomic},device,dict(phase='vjp',work=work,requested=requested))
            paired({'template':lambda:library.derivative(current,local_csr,None,work,tangents=tangent_buffers)},device,
                   dict(phase='jvp',work=work,requested=requested))
    if device=='cuda':torch.cuda.synchronize()
activity();report['complete']=True;save()
print(f'Complete: N={args.n}, D={args.dimension}, {args.precision}, {len(report["records"])} records',flush=True)
