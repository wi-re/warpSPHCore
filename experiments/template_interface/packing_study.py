"""Matched follow-up for the operation-specific compact interpolation record."""
import argparse
from dataclasses import replace
import json
import os
import subprocess
from pathlib import Path
import random
import statistics
import time
import torch
from template_runtime import Library,Plan,SLOTS,transpose
from fixtures import load_core,fixture,to_device


def sample(function,device,iterations,batched):
    if device=='cuda':
        torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();begin.record()
        if batched:function(iterations)
        else:
            for _ in range(iterations):function(1)
        end.record();end.synchronize()
        return dict(event_ms=begin.elapsed_time(end)/iterations,wall_ms=(time.perf_counter()-start)*1000/iterations)
    start=time.perf_counter()
    if batched:function(iterations)
    else:
        for _ in range(iterations):function(1)
    return dict(wall_ms=(time.perf_counter()-start)*1000/iterations)


def paired(calls,device,resident,repeats=7):
    samples={key:[] for key in calls};iterations={}
    for key,function in calls.items():
        measurement=sample(function,device,16 if resident else 1,resident)
        milliseconds=measurement.get('event_ms',measurement['wall_ms'])
        # Longer CPU intervals reduce thread wake-up and scheduling noise.
        target=150 if device=='cpu' else 15
        iterations[key]=max(1,min(1000,int(target/max(milliseconds,.005))))
    rng=random.Random(661)
    for repeat in range(repeats):
        keys=list(calls);rng.shuffle(keys)
        for key in keys:samples[key].append(sample(calls[key],device,iterations[key],resident))
    return {key:dict(samples=values,iterations=iterations[key],**{
        metric:statistics.median(value[metric] for value in values) for metric in values[0]})
        for key,values in samples.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('core','baseline','old-cpu','old-gpu','cpu','gpu','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--precision',required=True);parser.add_argument('--dimension',type=int,required=True)
    parser.add_argument('--n',type=int,default=1048576);args=parser.parse_args()
    torch.set_num_threads(24);torch.set_num_interop_threads(1)
    sph=load_core(args.core,args.baseline/'native-torch-final.so',args.precision)
    from warpSPHCore.native.api import sphOperation as native
    old=Library(args.old_cpu,args.old_gpu);new=Library(args.cpu,args.gpu)
    state,scalar,vector,domain,adjacency,csr,tiles,data,planning=fixture(sph,args.core,
        args.baseline/'exploration-v2.so',args.n,args.dimension,getattr(torch,args.precision),4.)
    report=dict(n=args.n,dimension=args.dimension,precision=args.precision,planning=planning,records=[],activity=[],complete=False)
    def activity():
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip().splitlines()
        others=[line for line in lines if line.split(',')[0].strip()!=str(os.getpid())]
        report['activity'].append(dict(processes=lines,other_processes=others))
        if others:raise RuntimeError(f'Concurrent GPU workload: {others}')
    activity()
    for device in ('cuda','cpu'):
        activity()
        current=dict(to_device(data,device),qvalue=None,rvalue=scalar.to(device))
        plan=csr.to(device);shared=tiles.to(device)
        local_state=replace(state,**{name:getattr(state,name).to(device) for name in ('positions','supports','masses','densities','kinds')})
        local_domain=sph.DomainDescription(domain.min.to(device),domain.max.to(device),domain.periodic.to(device),args.dimension)
        graph=sph.AdjacencyList(None,plan.buffers['indices'],plan.buffers['counts'],plan.buffers['offsets'],args.n,args.n,numEdges=planning['edges'])
        properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Interpolate,
            supportMode=sph.SupportScheme.Gather,operationMode=sph.OperationDirection.TrueAllToToAll)
        # V2 expects a larger allocation. A flat buffer with its original pitch
        # keeps the old binary independent of the new Python allocation policy.
        old_buffers=old.allocate(current,1)
        if device=='cuda':
            pitch=((args.dimension+2+3)//4)*4
            old_buffers.packed=current['qx'].new_empty((args.n,pitch))
        old.forward(current,plan,1,buffers=old_buffers)
        new_buffers=new.forward(current,plan,1)
        error=float((new_buffers.values-old_buffers.values).double().norm()/old_buffers.values.double().norm())
        assert error<(3e-6 if args.precision=='float32' else 1e-12),error
        for resident in (False,True):
            calls={
                'volume_and_field':lambda count:old.forward(current,plan,1,buffers=old_buffers,pack=not resident,iterations=count),
                'weighted_field':lambda count:new.forward(current,plan,1,buffers=new_buffers,pack=not resident,iterations=count),
                'weighted_direct':lambda count:new.forward(current,plan,1,buffers=new_buffers,pack=not resident,direct=True,iterations=count),
            }
            if not resident:
                # Both versions allocate their own outputs/reference buffers.
                def old_allocating(count):
                    result=old.allocate(current,1)
                    if device=='cuda':result.packed=current['qx'].new_empty((args.n,((args.dimension+5)//4)*4))
                    return old.forward(current,plan,1,buffers=result)
                calls['volume_and_field']=old_allocating
                calls['weighted_field']=lambda count:new.forward(current,plan,1)
                calls['weighted_direct']=lambda count:new.forward(current,plan,1,direct=True)
                calls['native_torch']=lambda count:native(local_state,properties,local_domain,
                    queryValues=current['rvalue'][:,0],referenceValues=current['rvalue'][:,0],adjacency=graph)
                if device=='cuda':calls['weighted_shared']=lambda count:new.forward(current,shared,1)
            result=paired(calls,device,resident)
            report['records'].append(dict(device=device,resident=resident,error=error,results=result))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        if device=='cuda' and args.dimension==3:
            # Nonlinear finalization now pulls its cotangent back once per row,
            # rather than repeating divisions inside every directed pair.
            reverse=transpose(plan,args.n,args.n)
            saved=new.forward(current,plan,5,save=True)
            cotangent=torch.cos(torch.arange(args.n,device=device,dtype=current['qx'].dtype))[:,None]
            for requested in ('all','fields'):
                gradients=tuple(torch.empty_like(current[name]) if current.get(name) is not None
                    and (requested=='all' or name in ('qvalue','rvalue')) else None for name in SLOTS)
                calls={}
                for label,library in [('per_edge',old),('per_query',new)]:
                    for atomic in (False,True):
                        def call(count,library=library,atomic=atomic):
                            return library.derivative(current,plan,reverse,5,cotangent=cotangent,
                                cache=saved.cache,gradients=gradients,atomic=atomic,iterations=count)
                        calls[label+('_atomic' if atomic else '_owned')]=call
                first=tuple(value.clone() if value is not None else None for value in calls['per_edge_owned'](1))
                second=calls['per_query_owned'](1)
                error=max(float((a-b).double().norm()/a.double().norm().clamp_min(1e-30))
                    for a,b in zip(first,second) if a is not None)
                assert error<(4e-5 if args.precision=='float32' else 1e-11),error
                report['records'].append(dict(device=device,phase='finalization_vjp',requested=requested,
                    error=error,results=paired(calls,device,False)))
                args.output.write_text(json.dumps(report,indent=2)+'\n')
    activity()
    report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)
