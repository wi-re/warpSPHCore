"""Density first-order reverse/JVP bridge over native CSR/CSC pair products.

Floating reductions have one owner per query/reference row. The transposed
integer topology is sorted and cached by CSRTopology; no floating atomics or
per-edge Jacobians are required. Warp's unsupported higher/functional products
remain outside the reference contract.
"""
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ..._models import AdjacencyList
from ...native._topology import CSRTopology

_USED = (0, 1, 2, 3, 4, 7, 8)
_REFERENCE = (1, 3, 4)


def _flat(inputs):
    # Density ignores field values, particle densities and all corrections.
    return [*(value if slot < 13 else None for slot, value in enumerate(inputs[:17])), *([None] * 9)]


class _Density(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *inputs):
        from ...native._build import build
        ctx.config, ctx.topology = config, topology
        ctx.token = topology.token()
        ctx.set_materialize_grads(False)
        ctx.mask = sum(1 << slot for slot in _USED if ctx.needs_input_grad[slot + 2])
        ctx.save_for_backward(*inputs)
        output = build().density_forward(list(inputs[:17]), *config[1:4])
        ctx.save_for_forward(*inputs, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        from ...native._build import build
        if ctx.topology.token() != ctx.token:
            raise RuntimeError('CSR topology changed before native density backwards')
        inputs = ctx.saved_tensors
        if cotangent is None:
            return (None,) * (len(inputs) + 2)
        if any(value is not None and forward_ad.unpack_dual(value).tangent is not None
               for value in (*inputs, cotangent)):
            raise NotImplementedError('Density reverse products are first-order; forward-over-backward is unsupported')
        transpose = []
        if any(ctx.mask & (1 << slot) for slot in _REFERENCE):
            csc = ctx.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        gradients = build().corrected_vjp(_flat(inputs), cotangent.contiguous(),
                                        transpose, ctx.mask, None, *ctx.config)
        result = []
        for slot, value in enumerate(inputs):
            gradient = gradients[slot] if slot in _USED else None
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
        directions = [t.contiguous() if slot in _USED and t is not None else None
                      for slot, t in enumerate(tangents[:17])] + [None] * 9
        with torch.no_grad():
            result = build().corrected_jvp(_flat(inputs), directions, example, *ctx.config)
        dependencies = [value for value in (*inputs, *tangents)
                        if value is not None and value.requires_grad]
        if torch.is_grad_enabled() and dependencies:
            result = _FirstOrderTangent.apply(result, *dependencies)
        return result


def apply(inputs, properties, query_mass=None):
    from ...native._build import build
    packed = [value.contiguous() if value is not None else None for value in (*inputs, query_mass)]
    used = (*_USED, 15, 16, 17)
    differentiating = any(packed[slot] is not None and (
        (torch.is_grad_enabled() and packed[slot].requires_grad) or
        forward_ad.unpack_dual(packed[slot]).tangent is not None) for slot in used)
    if not differentiating:
        return build().density_forward(packed[:17], properties.supportMode.value,
                                      properties.operationMode.value, properties.kernel.value)
    topology = CSRTopology(AdjacencyList(None, packed[12], packed[11], packed[10],
                                        len(packed[0]), len(packed[1])))
    config = (6, properties.supportMode.value, properties.operationMode.value,
              properties.kernel.value, 1, 2, False, False)
    return _Density.apply(config, topology, *packed)
