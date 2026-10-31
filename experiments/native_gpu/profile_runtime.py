#!/usr/bin/env python3
"""Capture actual Torch/CUDA runtime stages for one native or Warp public call."""
import argparse,json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--precision',choices=['float32','float64'],default='float32')
parser.add_argument('--operation',choices=['density','interpolate','gradient','divergence','curl','laplacian','covariance','search'],default='search')
parser.add_argument('--backend',choices=['native','warp'],default='native')
parser.add_argument('--n',type=int,default=1000000)
parser.add_argument('--artifact',type=Path)
parser.add_argument('--index-bits',type=int,choices=[32,64],default=64)
parser.add_argument('--offset-bits',type=int,choices=[32,64],default=64)
parser.add_argument('--dimension',type=int,choices=[1,2,3],default=2)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp',warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0]=[str(root/'src'),str(root/'scripts/benchmarks')]
import torch,warp as wp,warpSPHCore as sph
import bench_backends as b
from warpSPHCore.native.api import radiusSearch,sphOperation
b.torch,b.sph=torch,sph
if args.artifact:
    import warpSPHCore.native._build as build
    torch.ops.load_library(str(args.artifact.resolve()))
    build._extension=torch.ops.warpsph_native
torch.set_num_threads(16);wp.init()
q,domain,values=b.fixture(args.n,args.dimension,'cuda',32,getattr(torch,args.precision))
if args.operation=='search':
    if args.backend=='native': call=lambda:radiusSearch(q,domain)
    else: call=lambda:sph.radiusSearchCompactHashMap(q,domain,sph.SupportScheme.Gather,
        sph.HashMapLengthMode.Fixed,args.n,capacity=None,collection='fused',queryOrder='input',storeI=False)
else:
    source=radiusSearch(q,domain)
    adjacency=sph.AdjacencyList(None,source.paddedJ.to(getattr(torch,f'int{args.index_bits}')),
        source.numNeighbors,source.edgeOffsets.to(getattr(torch,f'int{args.offset_bits}')),
        args.n,args.n,numEdges=source.numEdges)
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=getattr(sph.WarpOperation,args.operation.capitalize()),
        gradientMode=sph.GradientScheme.Difference,laplacianMode=sph.LaplacianScheme.Brookshaw,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    operation=sphOperation if args.backend=='native' else sph.warpOperation
    call=lambda:operation(q,properties,domain,values,adjacency=adjacency)
for _ in range(3): result=call()
torch.cuda.synchronize()
args.output.parent.mkdir(parents=True,exist_ok=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
    with torch.profiler.record_function(f'{args.backend}.{args.operation}'):
        result=call()
        torch.cuda.synchronize()
profile.export_chrome_trace(str(args.output))
trace=json.loads(args.output.read_text())
kernels=[event for event in trace['traceEvents'] if event.get('cat')=='kernel']
print('GPU kernels:')
for event in kernels: print(f"{event['dur']/1000:.4f} ms {event['name'][:150]}")
print('Sum GPU kernel duration:',sum(event['dur'] for event in kernels)/1000,'ms')
print(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=14))
