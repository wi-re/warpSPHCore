"""Explicit density JVP and its first reverse, matching the Warp API."""
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable



class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.save_for_backward(*values);ctx.set_materialize_grads(False)
        return build().density_product(list(values[:17]),list(values[17:22]),None,[],*options)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native density JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().density_product(list(values[:17]),list(values[17:22]),cotangent.contiguous(),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.options)
        mapping={0:0,1:1,2:2,3:3,4:4,17:5,18:6,19:7,20:8,21:9}
        result=[]
        for slot,value in enumerate(values):
            # Domain bounds are frozen by the explicit reference launcher.
            if value is None or not ctx.needs_input_grad[slot+2] or slot in (7,8):bar=None
            elif slot in mapping:bar=bars[mapping[slot]]
            else:bar=torch.zeros_like(value) if value is not None and ctx.needs_input_grad[slot+2] else None
            result.append(bar)
        return None,None,*result


def compute(query,reference,domain,properties,adjacency,qt,rt):
    from .nt_corrected import _prepare
    from ..._models import AdjacencyList
    from ...native._topology import CSRTopology
    if type(adjacency).__name__ in ('CompactHashMap','SearchIndex'):adjacency=None
    reference=query if reference is None else reference
    inputs,_,_=_prepare(query,reference,domain,properties,adjacency,None,None,None,None,None,None,None,allow_forward=True)
    inputs[7]=forward_ad.unpack_dual(inputs[7]).primal.detach()
    inputs[8]=forward_ad.unpack_dual(inputs[8]).primal.detach()
    directions=[getattr(state,attr,None) if state is not None else None for state,attr in
                ((qt,'positions'),(rt,'positions'),(qt,'supports'),(rt,'supports'),(rt,'masses'))]
    values=[*inputs,*directions,query.masses]
    if any(v is not None and forward_ad.unpack_dual(v).tangent is not None for v in values):
        raise NotImplementedError('The explicit reference JVP has no classic dual JVP registration')
    values=[v.contiguous() if v is not None else None for v in values]
    topology=CSRTopology(AdjacencyList(None,inputs[12],inputs[11],inputs[10],len(inputs[0]),len(inputs[1])))
    options=(properties.supportMode.value,properties.operationMode.value,properties.kernel.value)
    return _Product.apply(options,topology,*values)
