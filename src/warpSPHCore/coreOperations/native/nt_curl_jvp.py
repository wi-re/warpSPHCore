"""Full explicit curl products and their owned CSR/CSC first reverse."""
import torch
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology
from .nt_curl_backward import field_blocks,field_layout,pack_field,finish


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,*values):
        from ...native._build import build
        ctx.options,ctx.topology,ctx.token=options,topology,topology.token()
        ctx.set_materialize_grads(False);ctx.save_for_backward(*values)
        return build().curl_product(list(values[:26]),list(values[26:52]),None,[],*options)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors
        if cotangent is None:return (None,)*(len(values)+2)
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native curl JVP reverse')
        csc=ctx.topology.transpose()
        bars=build().curl_product(list(values[:26]),list(values[26:52]),cotangent.contiguous(),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.options)
        return None,None,*(None if value is None or not ctx.needs_input_grad[k+2] or k in (7,8) else
            bars[k] if k<52 else torch.zeros_like(value) for k,value in enumerate(values))


def apply(config,flat,tangents,query_mass=None):
    flat=list(flat);flat[7]=flat[7].detach();flat[8]=flat[8].detach();tangents=list(tangents)
    topology=CSRTopology(AdjacencyList(None,flat[12],flat[11],flat[10],len(flat[0]),len(flat[1])))
    result=[];_,components=field_layout(flat)
    for values,start,width in field_blocks(flat):
        directions=list(tangents)
        for slot in (13,14):
            if directions[slot] is not None:
                directions[slot]=pack_field(directions[slot],start,width,flat[0].shape[1],components)
        if flat[0].shape[1]==3 and components:
            # Preserve the established 3D extension as differences of scalar
            # gradient products, including their separate reverse reductions.
            # Combining kernel cotangents before those reductions changes
            # singular cancellation even when the smooth equations agree.
            from .nt_gradient_jvp import apply as gradient_product
            gradients=[]
            for c in range(3):
                scalar=list(values);tangent=list(directions)
                for slot in (13,14):
                    scalar[slot]=values[slot].reshape(len(values[slot]),3,width)[:,c,:].contiguous()
                    if directions[slot] is not None:tangent[slot]=directions[slot].reshape(len(directions[slot]),3,width)[:,c,:].contiguous()
                if scalar[25] is not None and config[4]!=1:
                    omega=scalar[25][:,None]
                    scalar[13]=scalar[13]/omega
                    if tangent[13] is not None:tangent[13]=tangent[13]/omega
                gradients.append(gradient_product(config,scalar,tangent,query_mass))
            x,y,z=gradients
            output=torch.stack((z[...,1]-y[...,2],x[...,2]-z[...,0],y[...,0]-x[...,1]),1)
            result.append((output,width))
        else:
            inputs=[v.contiguous() if v is not None else None for v in (*values,*directions,query_mass)]
            result.append((_Product.apply(config[1:5],topology,*inputs),width))
    return finish(flat,result)
