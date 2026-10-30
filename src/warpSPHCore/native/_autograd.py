"""First-order Torch bridge; numerical VJPs live in module-owned native kernels."""
import torch
import warnings
from torch.autograd.function import once_differentiable

# Kernels with a smooth radial origin. Cusp/singular force-kernel reverse paths
# remain explicit capability gaps until their nonsmooth contracts are qualified.
REVERSE_KERNELS = {0, 1, 2, 20, 30, 31, 32, 33, 34, 36, 41}


def _used_slots(name, scheme):
    used = {0, 1, 2, 3, 4, 7, 8}
    if name != 'density': used.add(15)
    if name not in ('density', 'covariance'): used.add(14)
    if name not in ('density', 'covariance', 'interpolate'):
        if scheme != 1 or name == 'laplacian': used.add(13)
        if scheme == 2: used.add(16)
    return used


class _NativeOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, support, direction, kernel, scheme, lap, dot, *inputs):
        from ._build import build
        ctx.parameters = (name, support, direction, kernel, scheme, lap, dot)
        ctx.save_for_backward(*inputs)
        used = _used_slots(name, scheme)
        ctx.mask = sum(1 << slot for slot in used if ctx.needs_input_grad[7+slot])
        operator = getattr(build(), name + '_forward')
        if name == 'density': return operator(list(inputs), support, direction, kernel)
        return operator(list(inputs), support, direction, kernel, scheme, lap, dot)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        from ._build import build
        name, support, direction, kernel, scheme, lap, dot = ctx.parameters
        gradients = getattr(build(), name + '_adjoint')(
            list(ctx.saved_tensors), cotangent.contiguous(), ctx.mask,
            support, direction, kernel, scheme, lap, dot)
        return (None,) * 7 + tuple(g if ctx.mask & (1 << slot) else None
                                   for slot, g in enumerate(gradients))


def apply_operator(name, inputs, properties):
    from ._build import build
    params = (properties.supportMode.value, properties.operationMode.value, properties.kernel.value,
              properties.gradientMode.value, properties.laplacianMode.value, properties.divergenceDotMode)
    used = _used_slots(name, properties.gradientMode.value)
    needs_reverse = torch.is_grad_enabled() and any(inputs[slot].requires_grad for slot in used)
    if needs_reverse:
        if torch.are_deterministic_algorithms_enabled():
            message = 'Native adjoints use scatter accumulation; deterministic reductions are not implemented'
            if torch.is_deterministic_algorithms_warn_only_enabled(): warnings.warn(message, RuntimeWarning)
            else: raise NotImplementedError(message)
        if properties.kernel.value not in REVERSE_KERNELS:
            raise NotImplementedError('Native reverse mode for cusp/singular force kernels is not qualified')
        return _NativeOperator.apply(name, *params, *inputs)
    operator = getattr(build(), name + '_forward')
    return operator(inputs, *params[:3]) if name == 'density' else operator(inputs, *params)
