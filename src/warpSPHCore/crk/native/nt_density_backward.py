"""Owned scalar-value reverse for the CRK consistency ratio."""
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import materialize_cotangent
from ...native._differential import REFERENCE, _FirstOrderTangent

USED = {0, 1, 3, 4, 7, 8, 18, 19, 20}


class _Density(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *values):
        from ...native._build import build
        ctx.config, ctx.topology, ctx.token = config, topology, topology.token()
        ctx.mask = sum(1 << k for k in USED if ctx.needs_input_grad[k+2])
        ctx.set_materialize_grads(False)
        cache = build().preparation_accumulators(list(values[:26]), 10, *config[1:4])
        numerator, denominator = cache[:,0], cache[:,1]
        result = torch.where(denominator != 0, numerator/torch.where(denominator != 0, denominator, 1.), 0.)
        ctx.save_for_backward(*values, cache)
        ctx.save_for_forward(*values, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        from ...native._build import build
        values, cache = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        if cotangent is None: return (None,)*(len(values)+2)
        if ctx.topology.token() != ctx.token: raise RuntimeError('CSR topology changed before native CRK density backwards')
        if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in (*values, cotangent)):
            raise NotImplementedError('CRK density ordinary reverse is first-order; no forward-over-backwards registration')
        transpose = []
        if any(ctx.mask & (1 << k) for k in REFERENCE):
            csc = ctx.topology.transpose(); transpose = [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ]
        bars = build().corrected_vjp(list(values[:26]), materialize_cotangent(cotangent).reshape(-1,1),
                                    transpose, ctx.mask, cache, *ctx.config[:8])
        return None, None, *(None if x is None or not ctx.needs_input_grad[k+2] else
            (bars[k].sum(0) if k in (7,8) else bars[k]) if k in USED else torch.zeros_like(x)
            for k,x in enumerate(values))

    @staticmethod
    def jvp(ctx, _config, _topology, *tangents):
        from ...native._build import build
        values, example = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        directions = [t.contiguous() if k in USED and t is not None else None for k,t in enumerate(tangents[:26])]
        with torch.no_grad():
            product = (build().corrected_jvp(list(values[:26]), directions, example, *ctx.config[:8]).reshape_as(example)
                       if any(t is not None for t in directions) else torch.zeros_like(example))
        dependencies = [x for x in (*values,*tangents) if x is not None and x.requires_grad]
        # The reference density launch has no JVPSpec. Preserve this existing
        # native first-order extension independently of the moment explicit API.
        return _FirstOrderTangent.apply(product,*dependencies) if torch.is_grad_enabled() and dependencies else product


def apply(inputs, options, topology, volumes, A, B, ignored=()):
    from ...native._build import build
    values = [*inputs,None,volumes,A,B,*([None]*5),*ignored]
    values = [x.contiguous() if x is not None else None for x in values]
    active = any(x is not None and ((torch.is_grad_enabled() and x.requires_grad)
                 or forward_ad.unpack_dual(x).tangent is not None) for x in values)
    if not active: return build().crk_density_forward(inputs,volumes,A,B,*options)
    return _Density.apply((10,*options,1,2,False,False,False),topology,*values)
