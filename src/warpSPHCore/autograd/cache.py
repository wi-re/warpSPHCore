import warp as wp
import torch
from ..type_config import *


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
#
# warpier_fields.md Step D reintroduces wrapper reuse, deliberately and
# behind an explicit opt-in (`use_cache=True`, wired up by
# StateAwareWarpFunction.forward on the no-grad path only) rather than as a
# global default: identity is now owned (a Field attached to one specific
# tensor object, via util.fieldRegistry.acquireView) instead of inferred from
# a storage address two unrelated tensors could share, and it ships gated to
# the workload -- plain simulation calls -- where the removed cache's failure
# mode (reused, unzeroed .grad buffers) cannot occur because there is no
# grad to reuse. Section 3.3 in warpier_fields.md has the full argument;
# Step E extends this to the grad path only after the twice-in-process
# gradcheck gate passes.


from ..profiling import record_function
from ..util.cast import castTorchToWarpAsBuiltins, castWarpToTorch
from ..util.fieldRegistry import acquireView


def getCachedWarpArray(t: torch.Tensor, use_cache: bool = False) -> "wp.array":
    """Return a wp.array view of *t*.

    ``use_cache=False`` (the default, and every call site outside
    StateAwareWarpFunction's no-grad path) preserves the fresh-wrapper
    semantics the module-level note above explains. ``use_cache=True`` reuses
    the view attached to *t* via the Field registry instead of rebuilding one
    every call (warpier_fields.md Step D) -- safe here because the caller is
    responsible for only ever passing True on a path with no gradient buffer
    to accidentally reuse unzeroed.

    *t* must be the caller's original tensor object, not an already-detached
    copy: the Field registry attaches its cache entry to whatever object it
    is given, and ``Tensor.detach()`` returns a distinct object on every
    call, so caching against a pre-detached tensor would never hit. Both
    branches below detach internally, at the point of actual conversion.
    """
    if use_cache:
        return acquireView(t)
    return castTorchToWarpAsBuiltins(t.detach().contiguous())


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