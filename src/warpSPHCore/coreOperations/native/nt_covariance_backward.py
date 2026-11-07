"""First-order covariance products with deterministic CSR/CSC ownership."""
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology
from ...native._differential import REFERENCE


def used_slots(flat):
    used={0,1,2,3,7,8}
    used.update((18,) if flat[18] is not None else (4,15))
    used.update(slot for slot in (19,20,21,22,24) if flat[slot] is not None)
    return used


def primal(config,flat):
    from ...native._build import build
    from ...native._differential import _core_primal
    if not config[8] and all(x is None for x in flat[17:]):
        empty=flat[0].new_empty(0)
        return build().covariance_forward([x if x is not None else empty for x in flat[:17]],*config[1:6],False)
    return _core_primal(config,flat)


def product(config,flat,tangents):
    from ...native._build import build
    example=primal((*config[:8],False),flat)
    if not any(t is not None for t in tangents):return torch.zeros_like(example)
    return build().corrected_jvp(flat,tangents,example,*config[:8])


class _Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,config,topology,*inputs):
        ctx.config,ctx.topology,ctx.token=config,topology,topology.token()
        ctx.used=used_slots(inputs)
        ctx.mask=sum(1<<slot for slot in ctx.used if ctx.needs_input_grad[slot+2])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        output=primal(config,list(inputs[:26]))
        matrix=output[0] if config[8] else output
        ctx.save_for_forward(*inputs,matrix)
        if config[8]:ctx.mark_non_differentiable(output[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent,*counts):
        from ...native._build import build
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native covariance backwards')
        inputs=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(inputs)+2)
        if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in (*inputs,cotangent)):
            raise NotImplementedError('Covariance reverse products are first-order; forward-over-backward is unsupported')
        transpose=[]
        if any(ctx.mask&(1<<slot) for slot in REFERENCE):
            csc=ctx.topology.transpose();transpose=[csc.edgeOffsets,csc.numNeighbors,csc.paddedJ]
        gradients=build().corrected_vjp(list(inputs[:26]),materialize_cotangent(cotangent),transpose,ctx.mask,None,*ctx.config[:8])
        result=[]
        for slot,value in enumerate(inputs):
            gradient=gradients[slot] if slot in ctx.used else None
            if slot in (7,8) and gradient is not None:gradient=gradient.sum(0)
            if gradient is None and ctx.needs_input_grad[slot+2]:gradient=torch.zeros_like(value)
            result.append(gradient)
        return (None,None,*result)

    @staticmethod
    def jvp(ctx,_config,_topology,*tangents):
        from .nt_classic_jvp import product as classic_product
        inputs=ctx.saved_tensors[:-1]
        result=classic_product(ctx.config,inputs,tangents)
        if result is not None:
            return (result,None) if ctx.config[8] else result
        from ...native._differential import _FirstOrderTangent
        inputs=ctx.saved_tensors[:-1]
        directions=[t.contiguous() if slot in ctx.used and t is not None else None for slot,t in enumerate(tangents[:26])]
        with torch.no_grad():result=product(ctx.config,list(inputs[:26]),directions)
        dependencies=[x for x in (*inputs,*tangents) if x is not None and x.requires_grad]
        if torch.is_grad_enabled() and dependencies:result=_FirstOrderTangent.apply(result,*dependencies)
        return (result,None) if ctx.config[8] else result


def apply(inputs,properties,corrections=None,query_mass=None,ignored=(),topology=None,counts=False):
    flat=[x.contiguous() if x is not None else None for x in (*inputs,*(corrections or [None]*9))]
    extras=[x.contiguous() if x is not None else None for x in (query_mass,*ignored)]
    config=(7,properties.supportMode.value,properties.operationMode.value,properties.kernel.value,
            properties.gradientMode.value,properties.laplacianMode.value,False,False,counts)
    differentiating=any(x is not None and ((torch.is_grad_enabled() and x.requires_grad) or forward_ad.unpack_dual(x).tangent is not None) for x in (*flat,*extras))
    if not differentiating:return primal(config,flat)
    if topology is None:topology=CSRTopology(AdjacencyList(None,flat[12],flat[11],flat[10],len(flat[0]),len(flat[1])))
    return _Covariance.apply(config,topology,*flat,*extras)
