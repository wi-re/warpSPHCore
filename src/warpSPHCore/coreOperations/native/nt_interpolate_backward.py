"""First-order interpolation products with deterministic CSR/CSC ownership.

Packed fields are processed in bounded blocks; geometry and coefficient
reductions retain one owner per physical row, with no floating atomics.
"""
import math
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology


def used_slots(flat):
    used = {0, 1, 2, 3, 7, 8, 14}
    used.update((18,) if flat[18] is not None else (4, 15))
    if flat[19] is not None:
        used.update((19, 20))
    return used


def field_blocks(flat):
    """Flatten arbitrary field axes without expanding per-edge storage."""
    field = flat[14]
    width = math.prod(field.shape[1:])
    packed = field.reshape(field.shape[0], width)
    dimension = flat[0].shape[1]
    if width == 0:
        raise ValueError('Interpolation requires nonempty field axes')
    start = 0
    while start < width:
        block = next(size for size in (dimension ** 2, dimension, 1) if size <= width - start)
        values = list(flat)
        values[14] = packed[:, start:start + block].contiguous()
        yield values, start, block
        start += block


def primal(config, flat):
    from ...native._build import build
    from ...native._differential import _core_primal
    if all(value is None for value in flat[17:]):
        empty=flat[0].new_empty(0)
        borrowed=[value if value is not None else empty for value in flat[:17]]
        return build().interpolate_forward(borrowed,*config[1:5],2,False)
    return _core_primal(config,flat)


def product(config, flat, tangents):
    from ...native._build import build
    outputs = []
    for values, start, width in field_blocks(flat):
        directions = list(tangents)
        if directions[14] is not None:
            t = directions[14]
            directions[14] = t.reshape(t.shape[0], math.prod(t.shape[1:]))[:, start:start + width].contiguous()
        example = primal(config, values)
        outputs.append(build().corrected_jvp(values, directions, example, *config[:8]))
    return torch.cat(outputs, 1).reshape(flat[0].shape[0], *flat[14].shape[1:])


class _Interpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *inputs):
        ctx.config, ctx.topology, ctx.token = config, topology, topology.token()
        ctx.used = used_slots(inputs)
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
            raise RuntimeError('CSR topology changed before native interpolation backwards')
        inputs = ctx.saved_tensors
        if cotangent is None:
            return (None,) * (len(inputs) + 2)
        if any(value is not None and forward_ad.unpack_dual(value).tangent is not None
               for value in (*inputs, cotangent)):
            raise NotImplementedError('Interpolation reverse products are first-order; forward-over-backward is unsupported')
        transpose = []
        if any(ctx.mask & (1 << slot) for slot in (1, 3, 4, 14, 15, 18)):
            csc = ctx.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        gradients = build().corrected_vjp(list(inputs[:26]), cotangent.contiguous(),
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


def apply(inputs, properties, corrections=None, query_mass=None, ignored=(), topology=None):
    flat = [value.contiguous() if value is not None else None
            for value in (*inputs, *(corrections or [None] * 9))]
    extras = [value.contiguous() if value is not None else None for value in (query_mass, *ignored)]
    config = (1, properties.supportMode.value, properties.operationMode.value,
              properties.kernel.value, properties.gradientMode.value, 2, False, False, False)
    differentiating = any(value is not None and (
        (torch.is_grad_enabled() and value.requires_grad) or forward_ad.unpack_dual(value).tangent is not None)
        for value in (*flat, *extras))
    if not differentiating:
        outputs = [primal(config, values) for values, _, _ in field_blocks(flat)]
        return torch.cat(outputs, 1).reshape(flat[0].shape[0], *flat[14].shape[1:])
    if topology is None:
        topology = CSRTopology(AdjacencyList(None, flat[12], flat[11], flat[10], len(flat[0]), len(flat[1])))
    outputs = [_Interpolate.apply(config, topology, *values, *extras) for values, _, _ in field_blocks(flat)]
    return torch.cat(outputs, 1).reshape(flat[0].shape[0], *flat[14].shape[1:])
