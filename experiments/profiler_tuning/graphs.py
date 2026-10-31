#!/usr/bin/env python3
"""Test CUDA Graph replay for a fixed interpolation topology and changing fields.

Planning and capture are outside measurements. Each replay still prepares the
weighted field and computes interpolation. The caller owns graph input/output
lifetimes; this is not an automatic cache in the library.
"""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',type=Path,required=True)
parser.add_argument('--library',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='native',warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/benchmarks'),str(ROOT/'experiments/cluster_pairs')]
import torch
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch,sphOperation
from runtime import Library,build_cells
import bench_backends as bench

torch.set_num_threads(16)
torch.ops.load_library(str(args.artifact))
build._extension=torch.ops.warpsph_native
library=Library(args.library)
properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,
    operation=sph.WarpOperation.Interpolate,operationMode=sph.OperationDirection.TrueAllToToAll)
report=dict(complete=False,pid=os.getpid(),artifact=str(args.artifact),artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),cases=[],activity=[])

def sample(function):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start=time.perf_counter()
    begin.record()
    for _ in range(20): result=function()
    end.record()
    end.synchronize()
    return dict(event_ms=begin.elapsed_time(end)/20,wall_ms=(time.perf_counter()-start)*1000/20)

for dimension in [1,2,3]:
    for n in [32768,1000000]:
        report['activity'].append(bench.activity())
        generator=torch.Generator().manual_seed(9067+n*3+dimension)
        x=torch.rand((n,dimension),generator=generator,dtype=torch.float64).cuda().float()
        side=round(n**(1/dimension))
        spacing=1/side if side**dimension==n else n**(-1/dimension)
        h=x.new_full((n,),4*spacing)
        state=sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
        values=torch.sin(2*math.pi*x)+.1*x
        domain=sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
        cells=build_cells(library,state,values,float(h[0]),8)
        adjacency=radiusSearch(cells.state,domain)
        function=lambda:sphOperation(cells.state,properties,domain,cells.values,adjacency=adjacency)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5): result=function()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured=function()
        graph.replay()
        torch.testing.assert_close(captured,function(),rtol=3e-4,atol=3e-4)
        # A changed field must be observed on replay, including its packing pass.
        cells.values.add_(.125)
        graph.replay()
        torch.testing.assert_close(captured,function(),rtol=3e-4,atol=3e-4)
        functions={'ordinary':function,'graph':graph.replay}
        samples={name:[] for name in functions}
        for repeat in range(11):
            names=list(functions)
            random.Random(1711+repeat).shuffle(names)
            for name in names:samples[name].append(sample(functions[name]))
        results={name:dict(samples=data,**{key:statistics.median(v[key] for v in data) for key in ['event_ms','wall_ms']}) for name,data in samples.items()}
        report['cases'].append(dict(dimension=dimension,n=n,n_h=4,precision='float32',results=results,changed_field_parity=True))
        report['activity'].append(bench.activity())
        bench.save(args.output,report)
        print(dimension,n,{name:round(r['wall_ms'],4) for name,r in results.items()},flush=True)
        del graph,captured,adjacency,cells,state
        torch.cuda.empty_cache()
report['complete']=True
bench.save(args.output,report)
