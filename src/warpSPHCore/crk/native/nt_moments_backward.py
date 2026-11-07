"""Six raw moment outputs with owned CSR/CSC reverse and discrete counts."""
import math
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import materialize_cotangent
from ...native._differential import REFERENCE, _FirstOrderTangent


USED = {0, 1, 2, 3, 7, 8, 18}


class _Moments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *inputs):
        from ...native._build import build
        ctx.config, ctx.topology, ctx.token = config, topology, topology.token()
        ctx.mask = sum(1 << k for k in USED if ctx.needs_input_grad[k + 2])
        ctx.set_materialize_grads(False)
        outputs = tuple(build().crk_moments_forward(list(inputs[:17]), inputs[18], *config[1:4]))
        ctx.shapes = [tuple(x.shape) for x in outputs[:6]]
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs, outputs[0])
        ctx.mark_non_differentiable(outputs[-1])
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        from ...native._build import build
        values = ctx.saved_tensors
        if not any(x is not None for x in cotangents[:6]):
            return (None,) * (len(values) + 2)
        if ctx.topology.token() != ctx.token:
            raise RuntimeError('CSR topology changed before native CRK moment backwards')
        if any(x is not None and forward_ad.unpack_dual(x).tangent is not None
               for x in (*values, *cotangents)):
            raise NotImplementedError('CRK moment ordinary reverse is first-order; no forward-over-backward registration')
        cotangent = torch.cat([
            (materialize_cotangent(bar) if bar is not None else values[0].new_zeros(shape)).reshape(
                shape[0], math.prod(shape[1:]))
            for bar, shape in zip(cotangents[:6], ctx.shapes)
        ], 1)
        transpose = []
        if any(ctx.mask & (1 << k) for k in REFERENCE):
            csc = ctx.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        bars = build().corrected_vjp(list(values[:26]), cotangent, transpose, ctx.mask, None, *ctx.config[:8])
        result = []
        for k, value in enumerate(values):
            if value is None or not ctx.needs_input_grad[k + 2]:
                bar = None
            elif k in USED:
                bar = bars[k].sum(0) if k in (7, 8) else bars[k]
            else:
                bar = torch.zeros_like(value)
            result.append(bar)
        return None, None, *result

    @staticmethod
    def jvp(ctx, _config, _topology, *tangents):
        from ...native._build import build
        values, example = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        directions = [t.contiguous() if k in USED and t is not None else None
                      for k, t in enumerate(tangents[:26])]
        width = sum(math.prod(shape[1:]) for shape in ctx.shapes)
        with torch.no_grad():
            product = (build().corrected_jvp(list(values[:26]), directions, example, *ctx.config[:8])
                       if any(t is not None for t in directions) else example.new_zeros((len(example), width)))
        dependencies = [x for x in (*values, *tangents) if x is not None and x.requires_grad]
        # The reference ordinary moment launch has no JVPSpec. Keep this
        # existing native first-order extension separate from its explicit API.
        if torch.is_grad_enabled() and dependencies:
            product = _FirstOrderTangent.apply(product, *dependencies)
        outputs, start = [], 0
        for shape in ctx.shapes:
            size = math.prod(shape[1:])
            outputs.append(product[:, start:start + size].reshape(shape).contiguous())
            start += size
        return *outputs, None


def apply(inputs, options, topology, volumes, ignored=()):
    from ...native._build import build
    flat = [*inputs, None, volumes, *([None] * 7)]
    mass = flat[4]
    flat[4] = None
    values = [x.contiguous() if x is not None else None for x in (*flat, mass, *ignored)]
    differentiating = any(x is not None and ((torch.is_grad_enabled() and x.requires_grad)
                         or forward_ad.unpack_dual(x).tangent is not None) for x in values)
    if not differentiating:
        return tuple(build().crk_moments_forward(inputs, volumes, *options))
    return _Moments.apply((9, *options, 1, 2, False, False, True), topology, *values)
