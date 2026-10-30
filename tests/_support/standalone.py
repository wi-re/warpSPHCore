"""Independent oracles for the framework-free library, using a test-only C bridge.

Set WARPSPHCORE_STANDALONE_LIBRARY to libnative_standalone_test.so from CMake.
Torch owns fixture memory here; neither the tested library nor bridge links it.
"""
import ctypes as C
import os
import pytest
import torch
import warpSPHCore as sph
if sph.backend!='native':pytest.skip('native oracle process only',allow_module_level=True)
path=os.environ.get('WARPSPHCORE_STANDALONE_LIBRARY')
if not path:pytest.skip('set WARPSPHCORE_STANDALONE_LIBRARY to qualify standalone CMake build',allow_module_level=True)

N=17
class Request(C.Structure):
    _fields_=[*( (name,C.c_int) for name in ('task','device','precision','dimension','rank','operation','kernel','support','scheme','lap','direction','dot','offset_bits','index_bits','validation','threads')),
        ('queries',C.c_int64),('references',C.c_int64),('inputs',C.c_void_p*N),('sizes',C.c_int64*N),
        ('output',C.c_void_p),('output_size',C.c_int64),('cotangent',C.c_void_p),('cotangent_size',C.c_int64),
        ('gradients',C.c_void_p*N),('gradient_sizes',C.c_int64*N),('mask',C.c_uint64),('accumulate',C.c_int),('stream',C.c_void_p),('result',C.c_int64)]
lib=C.CDLL(path)
lib.standalone_run.argtypes=[C.POINTER(Request)];lib.standalone_run.restype=C.c_int
lib.standalone_error.restype=C.c_char_p
lib.standalone_supports.argtypes=[C.c_int];lib.standalone_supports.restype=C.c_int


def run(r):
    if lib.standalone_run(C.byref(r)):raise RuntimeError(lib.standalone_error().decode())
    return r.result


def request(q,r,d,qf,rf,prop,a=None):
    req=Request();req.device=(2 if torch.version.hip else 1) if q.positions.is_cuda else 0;req.precision=int(q.positions.dtype==torch.float64)
    req.dimension=d.dim;req.rank=qf.dim()-1;req.queries=len(q.positions);req.references=len(r.positions)
    req.operation=prop.operation.value;req.kernel=prop.kernel.value;req.support=prop.supportMode.value
    req.scheme=prop.gradientMode.value;req.lap=prop.laplacianMode.value;req.direction=prop.operationMode.value;req.dot=prop.divergenceDotMode
    req.offset_bits=req.index_bits=64;req.threads=4
    if req.device:req.stream=torch.cuda.current_stream().cuda_stream
    empty=q.positions.new_empty(0)
    inputs=[q.positions,r.positions,q.supports,r.supports,r.masses,q.kinds,r.kinds,d.min,d.max,d.periodic,
            *(a if a is not None else [empty,empty,empty]),qf,rf,r.densities,q.densities]
    for s,t in enumerate(inputs):
        if t is not None:req.inputs[s]=t.data_ptr();req.sizes[s]=t.numel()
    if a:
        req.offset_bits=32 if a[0].dtype==torch.int32 else 64
        req.index_bits=32 if a[2].dtype==torch.int32 else 64
    return req


def output(req,t,task=0):
    req.task=task;req.output=t.data_ptr();req.output_size=t.numel();run(req);return t


def search(q,r,d,qf,rf,prop):
    req=request(q,r,d,qf,rf,prop)
    counts=torch.empty(len(q.positions),device=q.positions.device,dtype=torch.int32)
    output(req,counts,2)
    req.inputs[11]=counts.data_ptr();req.sizes[11]=counts.numel()
    offsets=torch.empty(len(q.positions),device=q.positions.device,dtype=torch.int64)
    req.task=3;req.output=offsets.data_ptr();req.output_size=offsets.numel();total=run(req)
    req.inputs[10]=offsets.data_ptr();req.sizes[10]=offsets.numel()
    ids=torch.full((total,),-1,device=q.positions.device,dtype=torch.int64)
    output(req,ids,4)
    return offsets,counts,ids


def vjp(req,values,slots,cot,*,accumulate=False,buffers=None):
    req.task=1;req.cotangent=cot.data_ptr();req.cotangent_size=cot.numel();req.mask=sum(1<<s for s in slots);req.accumulate=accumulate
    buffers=[torch.empty_like(x) for x in values] if buffers is None else buffers
    for s,b in zip(slots,buffers):req.gradients[s]=b.data_ptr();req.gradient_sizes[s]=b.numel()
    run(req);return buffers


def valid(op,rank,dim):
    if op==sph.WarpOperation.Divergence and rank==0:pytest.skip('vector field required')
    if op==sph.WarpOperation.Curl and rank==0 and dim>1:pytest.skip('vector field required')
