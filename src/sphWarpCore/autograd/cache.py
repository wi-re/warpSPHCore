import warp as wp
import torch
from ..types import *


# Reusable cache for tiny default tensors passed to kernels as optional dummy args.
# Keyed by tensor role + shape + dtype + device to avoid per-call allocations.
_DUMMY_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def clearDummyTensorCache() -> None:
    """Clear all cached dummy tensors.

    Useful when explicitly reclaiming memory across long-lived sessions.
    """
    _DUMMY_TENSOR_CACHE.clear()


# ---------------------------------------------------------------------------
# wp.array identity cache -- REMOVED.
# ---------------------------------------------------------------------------
# This used to key a cached wp.array wrapper by (data_ptr, shape, strides,
# dtype) so that repeated calls on the same underlying tensor storage reused
# the same wp.array object instead of rebuilding it. The problem: the reused
# wp.array's .grad buffer is also reused and is never zeroed between unrelated
# forward/backward calls, so any two calls that happen to share tensor
# storage (the same leaf tensor called into repeatedly -- true of
# torch.autograd.gradcheck, and of any training loop that updates parameters
# in place) silently accumulate gradients from prior calls into the "new"
# one, making backward non-reentrant and gradients wrong from the second call
# onward. Found while adding torch.autograd.gradcheck coverage; see
# warpier_core.md. getCachedWarpArray now always builds a fresh wrapper.


from torch.profiler import record_function
from ..util.cast import castTorchToWarpAsBuiltins, castWarpToTorch


def getCachedWarpArray(t: torch.Tensor) -> "wp.array":
    """Return a fresh wp.array view of *t*.

    No longer caches the wrapper object -- see the module-level note above.
    Kept under its original name since it is part of the package's public
    surface and used throughout the operation backends.
    """
    return castTorchToWarpAsBuiltins(t.contiguous())


def clearWarpArrayCache() -> None:
    """No-op: the wp.array identity cache has been removed.

    Kept for backward compatibility with existing call sites.
    """
    pass

def getCachedDummyTensor(
    shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    fillValue: float = 0.0,
) -> torch.Tensor:
    # with record_function("[warpSPH] - getCachedDummyTensor"):
    """Return a shared cached tensor for optional kernel arguments.

    This avoids allocating new placeholder tensors on every kernel launch.
    """
    normalized_shape = tuple(int(s) for s in shape)
    key = ("dummy", normalized_shape, str(dtype), str(device), float(fillValue))

    tensor = _DUMMY_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.full(normalized_shape, fillValue, dtype=dtype, device=device).contiguous()
        _DUMMY_TENSOR_CACHE[key] = tensor

    return tensor


def getCachedIdentityMatrices(
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a shared cached tensor with shape (1, dim, dim) containing I."""
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")

    key = ("identity", int(dim), str(dtype), str(device))
    tensor = _DUMMY_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).contiguous()
        _DUMMY_TENSOR_CACHE[key] = tensor

    return tensor



# from wp_tensor import tensor


    
    
# ---------------------------------------------------------------------------
# The struct-bundle / wrapper-args caches that used to live here (keyed on
# tensor data_ptr, reusing Warp array wrapper objects -- and their attached
# .grad buffers -- across unrelated calls) have been removed. They were the
# same class of live-object-identity caching as getCachedWarpArray's removed
# cache (see wp_util.py): safe only by accident (gated to the no-grad path
# here), but the same pattern silently produced wrong, accumulating
# gradients elsewhere. Removed rather than reasoned about further. See
# warpier_core.md.


def clearKernelArgsCache() -> None:
    """No-op: the kernel-args/wrapper-args caches have been removed.

    Kept for backward compatibility with existing call sites.
    """
    pass