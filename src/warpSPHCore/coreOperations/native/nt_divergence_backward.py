"""First-order divergence products with deterministic CSR/CSC ownership.

Packed fields are processed in bounded blocks; geometry and coefficient
reductions retain one owner per physical row, with no floating atomics.
"""
import math
import torch
from ...native._inputs import materialize_cotangent
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology
from ...native._differential import REFERENCE


def used_slots(flat, scheme, consistent=False):
    used={0,1,2,3,7,8,14}
    if scheme!=1:
        used.add(13)
        if flat[25] is not None:used.add(25)
    if scheme==2:used.update((4,15,16))
    else:used.update((18,) if flat[18] is not None else (4,15))
    if consistent and scheme!=2:
        used.add(16)
        if flat[18] is not None:used.add(15)
        else:used.discard(15)
    used.update(slot for slot in range(19,25) if flat[slot] is not None)
    return used


def field_layout(flat,dot):
    q,r=flat[13],flat[14];dimension=flat[0].shape[1]
    if q.shape[1:]!=r.shape[1:]:raise ValueError('Divergence query/reference field axes must match')
    axes=tuple(r.shape[1:])
    if not axes:
        if dimension!=1:raise ValueError('Divergence requires a spatial contraction axis')
        return (),1
    axis=0 if dot else len(axes)-1
    if axes[axis]!=dimension:raise ValueError('Divergence contraction axis must match spatial dimension')
    shape=axes[:axis]+axes[axis+1:]
    width=math.prod(shape)
    if width==0:raise ValueError('Divergence requires nonempty field axes')
    return shape,width


def pack_field(value,start,width,dimension,dot):
    if value.dim()==1:return value.reshape(value.shape[0],1)
    moved=value.movedim(1 if dot else -1,1).reshape(value.shape[0],dimension,math.prod(value.shape[1:])//dimension)
    block=moved[:,:,start:start+width]
    if not dot:block=block.transpose(1,2)
    return block.reshape(value.shape[0],dimension).contiguous() if width==1 else block.contiguous()


def field_blocks(flat,dot):
    _,components=field_layout(flat,dot);dimension=flat[0].shape[1]
    start=0
    while start<components:
        width=dimension if components-start>=dimension else 1
        values=list(flat)
        for slot in (13,14):values[slot]=pack_field(flat[slot],start,width,dimension,dot)
        yield values,start,width
        start+=width


def primal(config,flat):
    from ...native._build import build
    from ...native._differential import _core_primal
    if not config[7] and all(value is None for value in flat[17:]):
        empty=flat[0].new_empty(0)
        borrowed=[value if value is not None else empty for value in flat[:17]]
        return build().divergence_forward(borrowed,*config[1:5],2,config[6])
    return _core_primal(config,flat)


def product(config,flat,tangents):
    from ...native._build import build
    outputs=[]
    for values,start,width in field_blocks(flat,config[6]):
        directions=list(tangents)
        for slot in (13,14):
            if directions[slot] is not None:
                directions[slot]=pack_field(directions[slot],start,width,flat[0].shape[1],config[6])
        example=primal(config,values)
        outputs.append(build().corrected_jvp(values,directions,example,*config[:8]).reshape(flat[0].shape[0],width))
    return torch.cat(outputs,1).reshape(flat[0].shape[0],*field_layout(flat,config[6])[0])


class _Divergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *inputs):
        ctx.config, ctx.topology, ctx.token = config, topology, topology.token()
        ctx.used = used_slots(inputs,config[4],config[7])
        ctx.mask = sum(1 << slot for slot in ctx.used if ctx.needs_input_grad[slot + 2])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        output = primal(config, list(inputs[:26]))
        ctx.save_for_forward(*inputs, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        from ...native._build import build
        if ctx.topology.token() != ctx.token:
            raise RuntimeError('CSR topology changed before native divergence backwards')
        inputs = ctx.saved_tensors
        if cotangent is None:
            return (None,) * (len(inputs) + 2)
        if any(value is not None and forward_ad.unpack_dual(value).tangent is not None
               for value in (*inputs, cotangent)):
            raise NotImplementedError('Divergence reverse products are first-order; forward-over-backward is unsupported')
        transpose = []
        if any(ctx.mask & (1 << slot) for slot in REFERENCE):
            csc = ctx.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        gradients = build().corrected_vjp(list(inputs[:26]), materialize_cotangent(cotangent),
                                        transpose, ctx.mask, None, *ctx.config[:8])
        result = []
        for slot, value in enumerate(inputs):
            gradient = gradients[slot] if slot in ctx.used else None
            if slot in (7, 8) and gradient is not None:
                gradient = gradient.sum(0)
            if gradient is None and ctx.needs_input_grad[slot + 2]:
                gradient = torch.zeros_like(value)
            result.append(gradient)
        return (None, None, *result)

    @staticmethod
    def jvp(ctx, _config, _topology, *tangents):
        from .nt_classic_jvp import product as classic_product
        inputs=ctx.saved_tensors[:-1]
        result=classic_product(ctx.config,inputs,tangents)
        if result is not None:
            return result
        from ...native._build import build
        from ...native._differential import _FirstOrderTangent
        inputs, example = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        directions = [t.contiguous() if slot in ctx.used and t is not None else None
                      for slot, t in enumerate(tangents[:26])]
        with torch.no_grad():
            result = build().corrected_jvp(list(inputs[:26]), directions, example, *ctx.config[:8])
        dependencies = [value for value in (*inputs, *tangents)
                        if value is not None and value.requires_grad]
        if torch.is_grad_enabled() and dependencies:
            result = _FirstOrderTangent.apply(result, *dependencies)
        return result


def apply(inputs, properties, corrections=None, query_mass=None, ignored=(), topology=None, consistent=False):
    flat = [value.contiguous() if value is not None else None
            for value in (*inputs, *(corrections or [None] * 9))]
    extras = [value.contiguous() if value is not None else None for value in (query_mass, *ignored)]
    config = (3, properties.supportMode.value, properties.operationMode.value,
              properties.kernel.value, properties.gradientMode.value, 2, properties.divergenceDotMode, consistent, False)
    differentiating = any(value is not None and (
        (torch.is_grad_enabled() and value.requires_grad) or forward_ad.unpack_dual(value).tangent is not None)
        for value in (*flat, *extras))
    if not differentiating:
        outputs = [primal(config, values).reshape(flat[0].shape[0],width) for values, _, width in field_blocks(flat,config[6])]
        return torch.cat(outputs, 1).reshape(flat[0].shape[0], *field_layout(flat,config[6])[0])
    if topology is None:
        topology = CSRTopology(AdjacencyList(None, flat[12], flat[11], flat[10], len(flat[0]), len(flat[1])))
    outputs = [_Divergence.apply(config, topology, *values, *extras).reshape(flat[0].shape[0],width) for values, _, width in field_blocks(flat,config[6])]
    return torch.cat(outputs, 1).reshape(flat[0].shape[0], *field_layout(flat,config[6])[0])
