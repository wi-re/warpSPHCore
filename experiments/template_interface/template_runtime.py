"""Opt-in borrowed-buffer prototype, with an ordinary Torch backward bridge.

Planning performs sorting and transpose construction. Execution only loads the
finished plan; it does not inspect CSR contents or copy tensor data to the CPU.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable


class Problem(C.Structure):
    _fields_=[(name,C.c_int) for name in
              ('dimension','precision','work','queries','references','wide','direction')]+[
              (name,C.c_void_p) for name in
              ('stream','qx','rx','h','mass','density','qvalue','rvalue','volume',
               'qkind','rkind','lower','upper','periodic')]


class Topology(C.Structure):
    _fields_=[(name,C.c_int) for name in ('mode','tasks','wide')]+[
        (name,C.c_void_p) for name in ('offsets','counts','indices','begins','ends','query_begin','query_end')]


class Outputs(C.Structure):
    _fields_=[(name,C.c_void_p) for name in ('values','cache','packed','counts','winners')]


SLOTS=('qx','rx','h','mass','density','qvalue','rvalue')
GRADIENTS=('gx','grx','gh','gm','grho','gqvalue','grvalue')
TANGENTS=('tx','trx','th','tm','trho','tqvalue','trvalue')


class Derivatives(C.Structure):
    _fields_=[(name,C.c_void_p) for name in ('cotangent',*TANGENTS,*GRADIENTS,'final_bars')]


def pointer(value):
    return value.data_ptr() if value is not None else None


def field_count(work,dimension):
    return dimension if work in (3,8) else 1 if work in (1,2,5,7) else 0


def output_count(work,dimension):
    if work==2:return dimension
    if work in (3,4):return dimension*dimension
    if work==6:return 1+2*dimension+2*dimension**2+dimension**3
    if work==8:return dimension
    return 1


@dataclass
class Plan:
    buffers: dict
    mode: int=0
    tasks: int=0

    def descriptor(self):
        result=Topology(self.mode,self.tasks,int(
            'indices' in self.buffers and self.buffers['indices'].dtype==torch.int64))
        for name,value in self.buffers.items():setattr(result,name,pointer(value))
        return result

    def to(self,device):
        return Plan({name:value.to(device) for name,value in self.buffers.items()},self.mode,self.tasks)

    def bytes(self):
        return sum(value.numel()*value.element_size() for value in self.buffers.values())


def transpose(plan,queries,references):
    """Build once per topology. No original-edge map is needed for recomputation."""
    ids=plan.buffers['indices']
    counts=plan.buffers['counts']
    rows=torch.repeat_interleave(torch.arange(queries,device=ids.device,dtype=torch.int32),
                                 counts.long(),output_size=ids.numel())
    order=torch.argsort(ids,stable=True)
    columns=torch.bincount(ids.long(),minlength=references).to(torch.int32)
    offsets=torch.cat([columns.new_zeros(1,dtype=torch.int64),columns.long().cumsum(0)])[:-1]
    return Plan(dict(indices=rows[order].contiguous(),counts=columns,offsets=offsets))


@dataclass
class Buffers:
    values: torch.Tensor | None
    cache: torch.Tensor | None
    packed: torch.Tensor | None
    counts: torch.Tensor | None
    winners: torch.Tensor | None

    def descriptor(self):
        return Outputs(*(pointer(getattr(self,name)) for name,_ in Outputs._fields_))


class Library:
    def __init__(self,cpu,gpu=None):
        self.cpu=C.CDLL(str(cpu));self.gpu=C.CDLL(str(gpu)) if gpu else None
        for library,backend in [(self.cpu,'cpu'),(self.gpu,'gpu')]:
            if library is None:continue
            forward=getattr(library,'template_forward_'+backend)
            forward.argtypes=[C.POINTER(Problem),C.POINTER(Topology),C.POINTER(Outputs)]+[C.c_int]*(4 if backend=='gpu' else 2)
            forward.restype=C.c_int
            derivative=getattr(library,'template_derivative_'+backend)
            derivative.argtypes=[C.POINTER(Problem),C.POINTER(Topology),C.POINTER(Topology),C.POINTER(Outputs),C.POINTER(Derivatives)]+[C.c_int]*(3 if backend=='gpu' else 2)
            derivative.restype=C.c_int

    def problem(self,data,work):
        qx=data['qx'];dimension=qx.shape[1]
        result=Problem(dimension,int(qx.dtype==torch.float64),work,len(qx),len(data['rx']),0,data.get('direction',0))
        if qx.is_cuda:
            stream=torch.cuda.current_stream(qx.device)
            result.stream=stream.cuda_stream
            # Borrowed inputs may have been allocated on a different stream.
            for value in data.values():
                if isinstance(value,torch.Tensor) and value.is_cuda:value.record_stream(stream)
        for name,_ in Problem._fields_[8:]:setattr(result,name,pointer(data.get(name)))
        return result

    def allocate(self,data,work,save=False,auxiliary=False):
        qx=data['qx'];n,d=qx.shape;fields=field_count(work,d)
        pitch=((d+fields+int(work not in (1,8))+3)//4)*4
        return Buffers(qx.new_empty((n,output_count(work,d))),
            qx.new_empty((n,fields+1)) if save and work==5 else None,
            qx.new_empty((len(data['rx']),pitch)) if qx.is_cuda else None,
            torch.empty(n,dtype=torch.int32,device=qx.device) if auxiliary and work in (4,6) else None,
            torch.empty(n,dtype=torch.int32,device=qx.device) if auxiliary and work==7 else None)

    def forward(self,data,plan,work,*,direct=False,width=16,save=False,auxiliary=False,buffers=None,pack=True,iterations=1):
        result=buffers if buffers is not None else self.allocate(data,work,save,auxiliary)
        request=self.problem(data,work);topology=plan.descriptor();output=result.descriptor()
        if data['qx'].is_cuda:
            stream=torch.cuda.current_stream(data['qx'].device)
            for value in (*plan.buffers.values(),*(getattr(result,name) for name,_ in Outputs._fields_)):
                if value is not None:value.record_stream(stream)
            status=self.gpu.template_forward_gpu(C.byref(request),C.byref(topology),C.byref(output),width,direct,pack,iterations)
        else:status=self.cpu.template_forward_cpu(C.byref(request),C.byref(topology),C.byref(output),direct,iterations)
        if status:raise RuntimeError(f'Native template forward error {status}')
        return result

    def derivative(self,data,plan,reverse,work,*,cotangent=None,tangents=None,cache=None,
                   atomic=False,width=16,gradients=None,output=None,iterations=1):
        if work not in (0,1,2,3,5,8):raise NotImplementedError('AD is qualified only for density/interpolation/gradient/normalized interpolation')
        if data.get('volume') is not None:raise NotImplementedError('Apparent-volume AD is not implemented in this initial prototype')
        if plan.mode!=0:raise NotImplementedError('Initial AD execution uses explicit CSR and its transpose')
        qx=data['qx'];request=self.problem(data,work)
        forward=plan.descriptor();transposed=reverse.descriptor() if reverse is not None else Topology()
        derivative=Derivatives();derivative.cotangent=pointer(cotangent)
        mode=1 if tangents is not None else 3 if atomic else 2
        if tangents is not None:
            for name,value in zip(TANGENTS,tangents):setattr(derivative,name,pointer(value))
            result=output if output is not None else qx.new_empty((len(qx),output_count(work,qx.shape[1])))
            outputs=Outputs(pointer(result),pointer(cache),None,None,None)
        else:
            if work==5 and cache is None:raise ValueError('Normalized interpolation VJP requires saved numerator/denominator')
            if not atomic and reverse is None:raise ValueError('Owned VJP requires a transposed plan')
            result=gradients if gradients is not None else tuple(
                torch.empty_like(data[name]) if data.get(name) is not None else None for name in SLOTS)
            for name,value in zip(GRADIENTS,result):setattr(derivative,name,pointer(value))
            outputs=Outputs(None,pointer(cache),None,None,None)
        final_bars=qx.new_empty((len(qx),2)) if mode!=1 and work==5 else None
        derivative.final_bars=pointer(final_bars)
        if qx.is_cuda:
            stream=torch.cuda.current_stream(qx.device)
            for value in (*plan.buffers.values(),*(reverse.buffers.values() if reverse is not None else ())):
                value.record_stream(stream)
            for value in (cotangent,cache,*(tangents or ()),*(result if isinstance(result,tuple) else (result,))):
                if value is not None:value.record_stream(stream)
            status=self.gpu.template_derivative_gpu(C.byref(request),C.byref(forward),C.byref(transposed),C.byref(outputs),C.byref(derivative),mode,width,iterations)
        else:status=self.cpu.template_derivative_cpu(C.byref(request),C.byref(forward),C.byref(transposed),C.byref(outputs),C.byref(derivative),mode,iterations)
        if status:raise RuntimeError(f'Native template derivative error {status}')
        return result


class NativeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,library,plan,reverse,work,metadata,*inputs):
        data=dict(metadata,**dict(zip(SLOTS,inputs)))
        result=library.forward(data,plan,work,save=work==5)
        ctx.library,ctx.plan,ctx.reverse,ctx.work=library,plan,reverse,work
        # Tensor metadata (bounds/masks) and both topology views are saved too:
        # PyTorch then detects in-place changes instead of using a stale plan.
        ctx.metadata_keys=tuple(name for name,value in metadata.items()
            if isinstance(value,torch.Tensor) or value is None)
        ctx.scalar_metadata={name:value for name,value in metadata.items() if name not in ctx.metadata_keys}
        ctx.topology_keys=(tuple(plan.buffers),tuple(reverse.buffers))
        ctx.topology_metadata=(plan.mode,plan.tasks,reverse.mode,reverse.tasks)
        saved=(*inputs,result.cache,*(metadata[name] for name in ctx.metadata_keys),*plan.buffers.values(),*reverse.buffers.values())
        ctx.save_for_backward(*saved)
        return result.values

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        saved=ctx.saved_tensors
        inputs=saved[:7];cache=saved[7]
        metadata=dict(ctx.scalar_metadata)
        metadata.update(zip(ctx.metadata_keys,saved[8:8+len(ctx.metadata_keys)]))
        data=dict(metadata,**dict(zip(SLOTS,inputs)))
        start=8+len(ctx.metadata_keys)
        forward_keys,reverse_keys=ctx.topology_keys
        forward=Plan(dict(zip(forward_keys,saved[start:start+len(forward_keys)])),
                     *ctx.topology_metadata[:2])
        start+=len(forward_keys)
        reverse=Plan(dict(zip(reverse_keys,saved[start:start+len(reverse_keys)])),
                     *ctx.topology_metadata[2:])
        requested=ctx.needs_input_grad[5:]
        gradients=tuple(torch.empty_like(value) if needed and value is not None else None
                        for needed,value in zip(requested,inputs))
        result=ctx.library.derivative(data,forward,reverse,ctx.work,cotangent=cotangent.contiguous(),
                                     cache=cache,gradients=gradients)
        return (None,)*5+result


def differentiable(library,data,plan,reverse,work):
    """First-order Torch bridge; explicit JVP is Library.derivative(tangents=...)."""
    if work not in (0,1,2,3,5,8):raise NotImplementedError('No prototype backward for this operation')
    if data.get('volume') is not None:raise NotImplementedError('Apparent-volume AD is not qualified')
    if any(data[name].requires_grad for name in ('lower','upper')):raise NotImplementedError('Domain-bound gradients are not implemented')
    metadata={name:value for name,value in data.items() if name not in SLOTS}
    return NativeFunction.apply(library,plan,reverse,work,metadata,*(data.get(name) for name in SLOTS))
