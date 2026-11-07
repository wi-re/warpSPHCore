"""Full explicit covariance products and their owned CSR/CSC first reverse."""
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.set_materialize_grads(False);ctx.save_for_backward(*values)
        return build().covariance_product(list(values[:26]),list(values[26:52]),None,[],*options)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native covariance JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().covariance_product(list(values[:26]),list(values[26:52]),materialize_cotangent(cotangent),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.options)
        return None,None,*(None if value is None or not ctx.needs_input_grad[k+2] or k in (7,8) else
            bars[k] if k<52 else torch.zeros_like(value) for k,value in enumerate(values))


def apply(config,flat,tangents,query_mass=None,ignored=()):
    flat=list(flat);flat[7]=flat[7].detach();flat[8]=flat[8].detach();tangents=list(tangents)
    topology=CSRTopology(AdjacencyList(None,flat[12],flat[11],flat[10],len(flat[0]),len(flat[1])))
    inputs=[v.contiguous() if v is not None else None for v in (*flat,*tangents,query_mass,*ignored)]
    return _Product.apply(config[1:5],topology,*inputs)
