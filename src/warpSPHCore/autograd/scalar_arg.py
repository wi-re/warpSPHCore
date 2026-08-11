import torch

from ..type_config import get_torch_precision


def asScalarArg(value, *, device: torch.device) -> torch.Tensor:
    """Normalize a scalar kernel argument into the length-1 tensor shape
    expected by a *differentiable-scalar* kernel parameter.

    Pair this with a kernel parameter declared ``wp.array(dtype=scalar_t)``
    (read in the kernel body as ``param[0]``) instead of a by-value
    ``scalar_t`` parameter -- that array-shaped declaration is what lets the
    argument flow through ``warpWrapper2``'s existing tensor branch
    (``StateAwareWarpFunction``) and pick up a real gradient. A kernel
    parameter still declared as plain ``scalar_t`` should keep using
    ``scalar_t(value)`` directly at the call site -- this helper is only for
    parameters a kernel author has opted into the array-shaped form.

    Unlike that default ``scalar_t(...)`` path, this never collapses a
    tensor to a Python float: a ``torch.Tensor`` (0-dim or 1-element) is
    reshaped -- not detached -- so gradients recorded by
    ``torch.autograd.gradcheck`` or a training loop still flow back to the
    original leaf through ``warpWrapper2``'s existing
    ``isinstance(arg, torch.Tensor)`` split. A plain Python float/int (or an
    already-detached tensor with ``requires_grad=False``) becomes a fresh,
    non-grad-tracked 1-element tensor at the active precision, so it still
    reaches the kernel through the same array-shaped parameter without
    tracking a gradient.

    Args:
        value:  Python float/int, or a 0-dim/1-element torch.Tensor.
        device: Target device for the plain float/int case. Ignored when
                *value* is already a tensor (its own device is kept).

    Returns:
        A shape-(1,) torch.Tensor suitable for warpWrapper2's
        additionalArguments.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"asScalarArg expects a 0-dim or 1-element tensor, got shape {tuple(value.shape)}"
            )
        return value.reshape(1)
    return torch.tensor([value], dtype=get_torch_precision(), device=device)
