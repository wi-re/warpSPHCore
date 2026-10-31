#!/usr/bin/env python3
"""Dense-reference correctness and stream checks, outside timed operation calls."""
import argparse
import ctypes as C
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/cluster_pairs'))
import torch
from runtime import Library,ClusterCase,build_cells,build_clusters
from api import CsrCase
spec=importlib.util.spec_from_file_location('cluster_verifier',ROOT/'experiments/cluster_pairs/verify.py')
reference=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=reference
spec.loader.exec_module(reference)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--quick',action='store_true')
args=parser.parse_args()
torch.set_num_threads(8)
library=Library(args.library)
library.library.tuned_csr.argtypes=[C.POINTER(CsrCase)]+[C.c_int]*5
library.library.tuned_csr.restype=C.c_int
library.library.tuned_cluster.argtypes=[C.POINTER(ClusterCase)]+[C.c_int]*3
library.library.tuned_cluster.restype=C.c_int
results=[]

def check_case(state,values,support,refinement):
    cells=build_cells(library,state,values,support,refinement)
    expected=reference.dense(cells.state,cells.values)
    dtype=cells.values.dtype
    n,dimension=cells.values.shape
    tolerance=3e-5 if dtype==torch.float32 else 3e-10
    # This independent small dense search constructs input CSR for the verifier.
    # Its host size reads are not part of a running operation or benchmark.
    x=cells.state.positions.double()
    dx=x[:,None,:]-x[None,:,:]
    dx-=torch.floor(dx+.5)
    mask=dx.norm(dim=-1)<=cells.state.supports.double()[:,None]
    counts=mask.sum(1).int()
    offsets=torch.cat([counts.new_zeros(1).long(),counts.long().cumsum(0)])
    indices=mask.nonzero()[:,1]
    packed=values.new_empty((n,((2*dimension+3)//4)*4))
    output=torch.empty_like(values)
    lower,upper=values.new_zeros(dimension),values.new_ones(dimension)
    periodic=torch.ones(dimension,device='cuda',dtype=torch.bool)
    for bits in [32,64]:
        ids=indices.to(getattr(torch,f'int{bits}'))
        descriptor=CsrCase(dimension,int(dtype==torch.float64),n,bits)
        descriptor.stream=library.stream
        tensors={name:getattr(cells.state,name) for name in ['positions','supports','masses','densities']}
        tensors.update(values=cells.values,offsets=offsets,indices=ids,counts=counts,lower=lower,upper=upper,
                       periodic=periodic,packed=packed,output=output)
        for name,tensor in tensors.items():setattr(descriptor,name,tensor.data_ptr())
        batches,widths,blocks=([1,2],[4,16],[128]) if args.quick else ([1,2,4,8],[2,4,8,16,32],[128,256])
        for batch,width,block in itertools.product(batches,widths,blocks):
            library.check(library.library.tuned_csr(C.byref(descriptor),batch,width,block,1,1))
            torch.testing.assert_close(output.double(),expected,rtol=tolerance,atol=tolerance)
            results.append(dict(kind='csr',dimension=dimension,precision=str(dtype),support=support,
                                bits=bits,batch=batch,width=width,block=block))
    for i,j,group,exact in itertools.product([4,8],[4,8],[1,8],[False,True]):
        if args.quick and (i,j,group,exact)!=(4,8,8,True):continue
        plan=build_clusters(library,cells,i,j,group,exact,compression=False)
        cluster_packed=values.new_empty((2*dimension,n))
        descriptor=plan.request(cluster_packed,output)
        for variant in range(6):
            library.check(library.library.tuned_cluster(C.byref(descriptor),variant,1,1))
            torch.testing.assert_close(output.double(),expected,rtol=tolerance,atol=tolerance)
            results.append(dict(kind='cluster',dimension=dimension,precision=str(dtype),support=support,
                                tile=[i,j],group=group,exact=exact,variant=variant))

for dtype,dimension,support,refinement in itertools.product([torch.float32,torch.float64],[1,2,3],[.140001,.310001],[1,8]):
    if args.quick and (dimension!=3 or support>.2 or refinement!=8):continue
    state,values=reference.fixture(257,dimension,dtype,support)
    check_case(state,values,support,refinement)
# Retain the targeted near-periodic-cutoff regression from the cluster study.
for dtype in [torch.float32,torch.float64]:
    unit=2.**-24
    x=torch.tensor([[1-3*unit],[51*unit]],device='cuda',dtype=dtype)
    h=x.new_full((2,),54.5*unit)
    state=reference.State(x,h,h.new_ones(2),torch.zeros(2,device='cuda',dtype=torch.int32),h.new_ones(2))
    check_case(state,x.new_tensor([[0.],[1.]]),float(h[0]),8)
stream=torch.cuda.Stream()
with torch.cuda.stream(stream):
    state,values=reference.fixture(257,3,torch.float32,.140001)
    check_case(state,values,.140001,8)
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
args.output.write_text(json.dumps(dict(complete=True,cases=len(results),results=results,nondefault_stream=True),indent=2)+'\n')
print(f'Passed {len(results)} numerical configurations and nondefault stream',flush=True)
