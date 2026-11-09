"""First-order native meshless products over a versioned fixed CSR topology."""
from dataclasses import dataclass
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import materialize_cotangent
from ...native._build import build
from ...native._topology import tensor_token

SLOTS = 37
SUPPORTED_TASKS = {0, 1, 2, 7}
SMOOTH_KERNELS = {0, 1, 2, 30, 32, 34}


def active(values):
    return any(isinstance(v, torch.Tensor) and v.is_floating_point() and
               ((torch.is_grad_enabled() and v.requires_grad) or forward_ad.unpack_dual(v).tangent is not None)
               for v in values)


@dataclass(frozen=True)
class Options:
    scalars: tuple
    integers: tuple
    task: int
    kernel: int
    topology: object
    sources: tuple


def contiguous(values):
    return [materialize_cotangent(v) if isinstance(v, torch.Tensor) else v for v in values]


def product(options, flat, bars, directions, mask, reverse):
    values = contiguous(flat)
    return tuple(build().meshless_products(values[:17], values[17:], list(options.scalars),
        list(options.integers), options.task, options.kernel, contiguous(bars), contiguous(directions),
        mask, reverse, torch.are_deterministic_algorithms_enabled()))


class _Jvp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, options, *values):
        return product(options, values[:SLOTS], (None,)*3, values[SLOTS:], 0, False)

    @staticmethod
    def backward(ctx, *bars):
        raise NotImplementedError('Native meshless products support first-order AD only')

    @staticmethod
    def jvp(ctx, *directions):
        raise NotImplementedError('Native meshless products support first-order AD only')


class _Meshless(torch.autograd.Function):
    @staticmethod
    def forward(ctx, options, *flat):
        ctx.options = options
        ctx.topology_token = options.topology.token()
        ctx.source_tokens = tuple(tensor_token(t) for t in options.sources)
        ctx.mask = sum(1 << k for k,v in enumerate(flat) if isinstance(v,torch.Tensor)
                       and v.is_floating_point() and ctx.needs_input_grad[k+1])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*flat, *options.sources)
        ctx.save_for_forward(*flat, *options.sources)
        result = tuple(build().meshless(list(flat[:17]), list(flat[17:]), list(options.scalars),
            list(options.integers), options.task, options.kernel))
        if options.task == 0:
            ctx.mark_non_differentiable(result[2])
        return result

    @staticmethod
    def check(ctx):
        if ctx.options.topology.token() != ctx.topology_token:
            raise RuntimeError('CSR topology changed before native meshless products')
        if tuple(tensor_token(t) for t in ctx.options.sources) != ctx.source_tokens:
            raise RuntimeError('A native meshless input was modified by an inplace operation')

    @staticmethod
    @once_differentiable
    def backward(ctx, *bars):
        _Meshless.check(ctx)
        flat = ctx.saved_tensors[:SLOTS]
        if any(v is not None and forward_ad.unpack_dual(v).tangent is not None for v in (*flat,*bars)):
            raise NotImplementedError('Native meshless reverse supports first-order AD only')
        return None, *product(ctx.options, flat, bars, (None,)*SLOTS, ctx.mask, True)

    @staticmethod
    def jvp(ctx, _options, *directions):
        _Meshless.check(ctx)
        return _Jvp.apply(ctx.options, *ctx.saved_tensors[:SLOTS], *directions)


def apply(options, geometry, fields):
    if options.task not in SUPPORTED_TASKS:
        raise NotImplementedError('This native meshless pass is forward-only')
    if options.kernel not in SMOOTH_KERNELS:
        raise NotImplementedError('Native meshless AD is not qualified for this kernel')
    return _Meshless.apply(options, *geometry, *fields)
