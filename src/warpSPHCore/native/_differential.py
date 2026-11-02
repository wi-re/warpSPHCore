"""First-order corrected core products; native pairs own all numerical derivatives."""
import math
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ._autograd import REVERSE_KERNELS
from ._topology import CSRTopology
from .._models import AdjacencyList

GEOMETRY = (0, 1, 2, 3, 7, 8)
REFERENCE = (1, 3, 4, 14, 15, 18, 23)


def _check_geometry(config, flat, tangents=None):
    moving = any((tangents[slot] is not None if tangents is not None else
                  torch.is_grad_enabled() and flat[slot] is not None and flat[slot].requires_grad)
                 for slot in GEOMETRY)
    if moving and config[3] not in REVERSE_KERNELS:
        raise NotImplementedError('Corrected geometry AD for cusp/singular kernels is not qualified')


def _core_primal(config, flat):
    from ._build import build
    operation, support, direction, kernel, scheme, lap, dot, consistent, counts = config
    names = {1: 'interpolate', 2: 'gradient', 3: 'divergence', 4: 'curl', 5: 'laplacian', 7: 'covariance'}
    function = getattr(build(), names[operation] + '_corrected_forward')
    args = (flat[:17], flat[17:], support, direction, kernel, scheme)
    if operation in (1, 2): return function(*args)
    args += (lap, dot)
    if operation == 3: return function(*args, consistent)
    if operation == 7:
        result = function(*args, counts)
        return tuple(result) if counts else result[0]
    return function(*args)


def _preparation_primal(config, flat):
    from ._build import build
    kind, support, direction, kernel = config[:4]
    if kind == 9:
        return tuple(build().crk_moments_forward(flat[:17], flat[18], support, direction, kernel)), None
    cache = build().preparation_accumulators(flat, kind, support, direction, kernel)
    denominator = cache[:, 0] if kind == 8 else cache[:, 1]
    numerator = torch.ones_like(denominator) if kind == 8 else cache[:, 0]
    result = torch.where(denominator != 0, numerator / torch.where(
        denominator != 0, denominator, torch.ones_like(denominator)), torch.zeros_like(denominator))
    return result, cache


class _FirstOrderTangent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, product, *dependencies):
        return product

    @staticmethod
    def backward(ctx, *cotangents):
        raise NotImplementedError('Native corrected/preparation JVPs are first-order; reverse-over-JVP is unsupported')


class _Corrected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, topology, *flat):
        ctx.config, ctx.topology = config, topology
        ctx.set_materialize_grads(False)
        ctx.mask = sum(1 << slot for slot in range(len(flat)) if ctx.needs_input_grad[slot + 2])
        ctx.topology_token = topology.token()
        if config[0] >= 8: result, cache = _preparation_primal(config, list(flat))
        else: result, cache = _core_primal(config, list(flat)), None
        ctx.save_for_backward(*flat, cache)
        outputs = result if isinstance(result, tuple) else (result,)
        ctx.shapes = [tuple(value.shape) for value in outputs if value.is_floating_point()]
        ctx.save_for_forward(*flat, outputs[0])
        if isinstance(result, tuple): ctx.mark_non_differentiable(result[-1])
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent, *remaining):
        from ._build import build
        if ctx.topology.token() != ctx.topology_token:
            raise RuntimeError('CSR topology changed before native backwards')
        flat, cache = list(ctx.saved_tensors[:-1]), ctx.saved_tensors[-1]
        if any(isinstance(value, torch.Tensor) and forward_ad.unpack_dual(value).tangent is not None
               for value in (*flat, cotangent, *remaining)):
            raise NotImplementedError('Native corrected/preparation VJPs are first-order; forward-over-backwards is unsupported')
        if ctx.config[0] == 9:
            # Only row accumulators, never per-edge Jacobians, enter this
            # gather. Unused moment outputs have exactly zero cotangent.
            bars = (cotangent, *remaining[:-1])
            cotangent = torch.cat([(value if value is not None else flat[0].new_zeros(shape)).reshape(
                shape[0], 1 if len(shape) == 1 else math.prod(shape[1:]))
                for value, shape in zip(bars, ctx.shapes)], 1)
        elif ctx.config[0] >= 8:
            cotangent = cotangent.reshape(flat[0].shape[0], 1)
        transpose = []
        if any(ctx.mask & (1 << slot) for slot in REFERENCE):
            csc = ctx.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        if ctx.config[0] <= 5 and (ctx.mask & ~((1 << 13) | (1 << 14))) == 0:
            # Reuse the tuned field-only ownership policy when no coefficient
            # or geometry derivatives are requested.
            gradients = [None] * len(flat)
            config = ctx.config
            gradients[13], gradients[14] = build().field_pullback(flat[:17], flat[17:],
                cotangent.contiguous(), transpose, bool(ctx.mask & (1 << 13)),
                bool(ctx.mask & (1 << 14)), config[0], config[1], config[2], config[3],
                config[4], config[5], config[6], config[7])
        else:
            gradients = build().corrected_vjp(flat, cotangent.contiguous(),
                transpose, ctx.mask, cache, *ctx.config[:8])
        for slot in (7, 8):
            if gradients[slot] is not None: gradients[slot] = gradients[slot].sum(0)
        return (None, None, *gradients)

    @staticmethod
    def jvp(ctx, _config, _topology, *tangents):
        from ._build import build
        flat, example = list(ctx.saved_tensors[:-1]), ctx.saved_tensors[-1]
        _check_geometry(ctx.config, flat, tangents)
        direction = [value.contiguous() if value is not None else None for value in tangents]
        with torch.no_grad():
            if ctx.config[0] <= 5 and all(value is None for slot, value in enumerate(direction) if slot not in (13, 14)):
                result = build().field_product(flat[:17], flat[17:], direction[13], direction[14], *ctx.config[:8])
            else:
                result = build().corrected_jvp(flat, direction, example, *ctx.config[:8])
        dependencies = [value for value in (*flat, *tangents)
                        if isinstance(value, torch.Tensor) and value.requires_grad]
        if torch.is_grad_enabled() and dependencies:
            # A downstream Torch factor solve can otherwise build a partial
            # higher-order graph through its algebra while missing the native
            # Hessian. Keep the first-order product usable and reject that
            # reverse request explicitly when it reaches this boundary.
            result = _FirstOrderTangent.apply(result, *dependencies)
        if ctx.config[0] == 9:
            values, offset = [], 0
            for shape in ctx.shapes:
                width = 1 if len(shape) == 1 else math.prod(shape[1:])
                values.append(result[:, offset:offset + width].reshape(shape).contiguous())
                offset += width
            return (*values, None)
        if ctx.config[0] >= 8: result = result.reshape(ctx.shapes[0])
        return (result, None) if ctx.config[8] else result


def apply_preparation(kind, inputs, options, topology, volumes=None, A=None, B=None):
    corrections = [None, volumes, A, B, None, None, None, None, None]
    flat = [*inputs, *corrections]
    # Apparent volumes and moments have no mass dependency. Do not retain an
    # unrelated trainable mass tensor or manufacture an unused gradient path.
    if kind in (8, 9): flat[4] = None
    config = (kind, *options, 1, 2, False, False, kind == 9)
    _check_geometry(config, flat)
    differentiating = any(isinstance(value, torch.Tensor) and (
        (torch.is_grad_enabled() and value.requires_grad) or forward_ad.unpack_dual(value).tangent is not None)
        for value in flat)
    if differentiating: return _Corrected.apply(config, topology, *flat)
    from ._build import build
    if kind == 8: return build().crk_volume_forward(inputs, *options)
    if kind == 9: return tuple(build().crk_moments_forward(inputs, volumes, *options))
    return build().crk_density_forward(inputs, volumes, A, B, *options)


def apply_corrected(inputs, corrections, properties, *, consistent=False, counts=False, topology=None):
    flat = [*inputs, *corrections]
    config = (properties.operation.value, properties.supportMode.value,
              properties.operationMode.value, properties.kernel.value,
              properties.gradientMode.value, properties.laplacianMode.value,
              properties.divergenceDotMode, consistent, counts)
    _check_geometry(config, flat)
    differentiating = any(isinstance(value, torch.Tensor) and (
        (torch.is_grad_enabled() and value.requires_grad) or forward_ad.unpack_dual(value).tangent is not None)
        for value in flat)
    if not differentiating: return _core_primal(config, flat)
    if topology is None:
        topology = CSRTopology(AdjacencyList(None, inputs[12], inputs[11], inputs[10],
                                            len(inputs[0]), len(inputs[1])))
    return _Corrected.apply(config, topology, *flat)
