"""Full interpolation product and owned reverse, including coefficient roles."""
import math
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology
from .nt_interpolate_backward import field_blocks

_SLOTS=(0,1,2,3,4,15,14,18,19,20)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.set_materialize_grads(False);ctx.save_for_backward(*values)
        return build().interpolate_product(list(values[:26]),list(values[26:52]),None,[],*options)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native interpolation JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().interpolate_product(list(values[:26]),list(values[26:52]),materialize_cotangent(cotangent),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.options)
        mapping={slot:k for k,slot in enumerate(_SLOTS)};mapping.update({slot+26:k+10 for k,slot in enumerate(_SLOTS)})
        volume=values[18] is not None;crk=values[19] is not None;result=[]
        ignored=({4,15,30,41} if volume else {18,44})|({19,20,45,46} if not crk else set())
        for slot,value in enumerate(values):
            if value is None or not ctx.needs_input_grad[slot+2]:bar=None
            elif slot in mapping and slot not in ignored:bar=bars[mapping[slot]]
            elif slot in (7,8):bar=None
            else:bar=torch.zeros_like(value)
            result.append(bar)
        return None,None,*result


def apply(config,flat,tangents,query_mass=None,ignored=()):
    flat=list(flat);flat[7]=flat[7].detach();flat[8]=flat[8].detach()
    # The explicit reference launcher never receives query densities.
    flat[16]=None;tangents=list(tangents);tangents[16]=None
    topology=CSRTopology(AdjacencyList(None,flat[12],flat[11],flat[10],len(flat[0]),len(flat[1])))
    result=[]
    for values,start,width in field_blocks(flat):
        directions=list(tangents)
        if directions[14] is not None:
            field=directions[14];directions[14]=field.reshape(len(field),math.prod(field.shape[1:]))[:,start:start+width].contiguous()
        inputs=[v.contiguous() if v is not None else None for v in (*values,*directions,query_mass,*ignored)]
        result.append(_Product.apply(config[1:4],topology,*inputs))
    return torch.cat(result,1).reshape(flat[0].shape[0],*flat[14].shape[1:])
