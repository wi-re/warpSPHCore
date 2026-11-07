"""Full explicit laplacian products and their owned CSR/CSC first reverse."""
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology
from .nt_laplacian_backward import field_blocks,field_layout,pack_field,finish


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.set_materialize_grads(False);ctx.save_for_backward(*values)
        output=build().laplacian_product(list(values[:26]),list(values[26:52]),None,[],*options[:6],(list(values[52:54]) if values[52] is not None else []))[0]
        ctx.save_for_forward(*values,output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native laplacian JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().laplacian_product(list(values[:26]),list(values[26:52]),materialize_cotangent(cotangent),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.options[:6],(list(values[52:54]) if values[52] is not None else []))
        result=[]
        for k,value in enumerate(values):
            if value is None or not ctx.needs_input_grad[k+2]:bar=None
            elif k<52 and k%26 in (7,8):bar=bars[k].sum(0) if ctx.options[5] else None
            elif k<52:bar=bars[k]
            else:bar=torch.zeros_like(value)
            result.append(bar)
        return None,None,*result

    @staticmethod
    def jvp(ctx,_options,_topology,*tangents):
        if not ctx.options[6]:raise NotImplementedError('Full geometry explicit products have no second forward registration')
        values=ctx.saved_tensors[:-1];example=ctx.saved_tensors[-1]
        from .nt_laplacian_backward import used_slots
        flat=list(values[:26]);directions=list(tangents[:26])
        gates=[flat[13],flat[14]]
        for slot in (13,14):
            flat[slot]=values[26+slot] if values[26+slot] is not None else torch.zeros_like(flat[slot])
            directions[slot]=tangents[26+slot]
        # Omega is a constant in the explicit geometry API, but the ordinary
        # field operator's forward rule includes it. Fold that quotient term
        # into the supplied field direction before invoking the native product.
        for field,omega in ((13,25),(14,23)):
            if tangents[omega] is not None and values[omega] is not None:
                shape=(len(flat[field]),*([1]*(flat[field].ndim-1)))
                term=flat[field]*(tangents[omega]/values[omega]).reshape(shape)
                directions[field]=-term if directions[field] is None else directions[field]-term
            directions[omega]=None
        if any(directions[k] is not None for k in (7,8)) and directions[0] is None:
            # A zero query-position carrier records the relative-displacement
            # pullback, which is also reduced into the box-direction adjoints.
            directions[0]=torch.zeros_like(flat[0])
        used=used_slots(flat,ctx.options[3])
        directions=[v if k in used else None for k,v in enumerate(directions)]
        if not any(v is not None for v in directions):return torch.zeros_like(example)
        args=[v.contiguous() if v is not None else None for v in (*flat,*directions,*gates,*values[54:])]
        return _Product.apply((*ctx.options[:6],False),ctx.topology,*args)



def apply(config,flat,tangents,query_mass=None,ignored=(),bounds_ad=False,field_only=False):
    flat=list(flat)
    if not bounds_ad:flat[7]=flat[7].detach();flat[8]=flat[8].detach()
    tangents=list(tangents)
    topology=CSRTopology(AdjacencyList(None,flat[12],flat[11],flat[10],len(flat[0]),len(flat[1])))
    result=[]
    for values,start,width in field_blocks(config,flat):
        directions=list(tangents)
        for slot in (13,14):
            if directions[slot] is not None:
                directions[slot]=pack_field(directions[slot],start,width,flat[0].shape[1])
        inputs=[v.contiguous() if v is not None else None for v in (*values,*directions,None,None,query_mass,*ignored)]
        result.append((_Product.apply((*config[1:6],bounds_ad,field_only),topology,*inputs),width))
    return finish(config,flat,result)
