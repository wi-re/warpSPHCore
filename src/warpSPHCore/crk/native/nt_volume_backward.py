"""First-order CRK volume products with deterministic CSR/CSC ownership."""
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._differential import REFERENCE


def used_slots(flat):
    return {0,1,2,3,7,8}


def primal(config,flat):
    from ...native._differential import _preparation_primal
    return _preparation_primal(config,flat)


def product(config,flat,tangents,example):
    from ...native._build import build
    if not any(t is not None for t in tangents):return torch.zeros_like(example)
    return build().corrected_jvp(flat,tangents,example,*config[:8])[:,0]


class _Volume(torch.autograd.Function):
    @staticmethod
    def forward(ctx,config,topology,*inputs):
        ctx.config,ctx.topology,ctx.token=config,topology,topology.token()
        ctx.used=used_slots(inputs)
        ctx.mask=sum(1<<slot for slot in ctx.used if ctx.needs_input_grad[slot+2])
        ctx.set_materialize_grads(False)
        output,cache=primal(config,list(inputs[:26]))
        ctx.save_for_backward(*inputs,cache)
        ctx.save_for_forward(*inputs,output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent,*counts):
        from ...native._build import build
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native CRK volume backwards')
        inputs,cache=ctx.saved_tensors[:-1],ctx.saved_tensors[-1]
        if cotangent is None:return (None,)*(len(inputs)+2)
        if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in (*inputs,cotangent)):
            raise NotImplementedError('CRK volume reverse products are first-order; forward-over-backward is unsupported')
        transpose=[]
        if any(ctx.mask&(1<<slot) for slot in REFERENCE):
            csc=ctx.topology.transpose();transpose=[csc.edgeOffsets,csc.numNeighbors,csc.paddedJ]
        gradients=build().corrected_vjp(list(inputs[:26]),cotangent.reshape(-1,1).contiguous(),transpose,ctx.mask,cache,*ctx.config[:8])
        result=[]
        for slot,value in enumerate(inputs):
            gradient=gradients[slot] if slot in ctx.used else None
            if slot in (7,8) and gradient is not None:gradient=gradient.sum(0)
            if gradient is None and ctx.needs_input_grad[slot+2]:gradient=torch.zeros_like(value)
            result.append(gradient)
        return (None,None,*result)

    @staticmethod
    def jvp(ctx,_config,_topology,*tangents):
        from ...native._differential import _FirstOrderTangent
        inputs=ctx.saved_tensors[:-1]
        directions=[t.contiguous() if slot in ctx.used and t is not None else None for slot,t in enumerate(tangents[:26])]
        with torch.no_grad():result=product(ctx.config,list(inputs[:26]),directions,ctx.saved_tensors[-1])
        dependencies=[x for x in (*inputs,*tangents) if x is not None and x.requires_grad]
        if torch.is_grad_enabled() and dependencies:result=_FirstOrderTangent.apply(result,*dependencies)
        return result


def apply(inputs,options,topology,ignored=()):
    from ...native._build import build
    # Retain state-aware zero gradients for unread mass/density inputs, including
    # when they are the only trainable tensors. They never enter pair arithmetic.
    flat=[x.contiguous() if x is not None else None for x in (*inputs,*([None]*9))]
    flat[4]=None
    extras=[x.contiguous() if x is not None else None for x in ignored]
    config=(8,*options,1,2,False,False,False)
    differentiating=any(x is not None and ((torch.is_grad_enabled() and x.requires_grad) or forward_ad.unpack_dual(x).tangent is not None) for x in (*flat,*extras))
    if not differentiating:return build().crk_volume_forward(inputs,*options)
    return _Volume.apply(config,topology,*flat,*extras)
