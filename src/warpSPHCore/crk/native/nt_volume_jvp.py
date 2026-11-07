"""Explicit given-volume JVP and its first reverse, matching the Warp API."""
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from .nt_preparation import prepare


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.save_for_backward(*values);ctx.set_materialize_grads(False)
        return build().crk_volume_product(list(values[:17]),list(values[18:22]),values[17],None,[],options[1],options[2])[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native volume JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().crk_volume_product(list(values[:17]),list(values[18:22]),values[17],materialize_cotangent(cotangent),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],ctx.options[1],ctx.options[2])
        mapping={0:0,1:1,2:2,3:3,17:8,18:4,19:5,20:6,21:7}
        result=[]
        for slot,value in enumerate(values):
            # Domain bounds are frozen by the explicit reference launcher.
            if value is None or not ctx.needs_input_grad[slot+2] or slot in (7,8):bar=None
            elif slot in mapping:bar=bars[mapping[slot]]
            else:bar=torch.zeros_like(value) if value is not None and ctx.needs_input_grad[slot+2] else None
            result.append(bar)
        return None,None,*result


def compute(query,reference,domain,properties,adjacency,volumes,qt,rt):
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):raise NotImplementedError('Row-chunk operators do not support AD')
    if type(adjacency).__name__ in ('CompactHashMap','SearchIndex'):adjacency=None
    reference=query if reference is None else reference
    inputs,options,topology=prepare(query,reference,domain,properties,adjacency,volumes)
    inputs[7]=forward_ad.unpack_dual(inputs[7]).primal.detach()
    inputs[8]=forward_ad.unpack_dual(inputs[8]).primal.detach()
    directions=[getattr(state,attr,None) if state is not None else None for state,attr in
                ((qt,'positions'),(rt,'positions'),(qt,'supports'),(rt,'supports'))]
    values=[*inputs,volumes,*directions,query.masses,reference.masses,query.densities,reference.densities]
    if any(v is not None and forward_ad.unpack_dual(v).tangent is not None for v in values):
        raise NotImplementedError('The explicit reference JVP has no classic dual JVP registration')
    values=[v.contiguous() if v is not None else None for v in values]
    return _Product.apply(options,topology,*values)
