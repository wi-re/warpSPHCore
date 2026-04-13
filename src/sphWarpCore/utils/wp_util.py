import warp as wp
import torch


# Reusable cache for tiny default tensors passed to kernels as optional dummy args.
# Keyed by tensor role + shape + dtype + device to avoid per-call allocations.
_DUMMY_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}

def castTorchToWarp(x_torch):
    """
    Cast a PyTorch tensor to a Warp array, ensuring it's on the correct device and has the right dtype.
    """
    # Ensure tensor is contiguous and on the correct device
    x_torch = x_torch.contiguous()
    
    # Convert to Warp array
    x_warp = wp.from_torch(x_torch).contiguous()
    
    # this will return a flat Warp array, so we need to reshape it back to the original shape if necessary
    
    x_warp = x_warp.reshape(x_torch.shape)
    
    return x_warp

def castWarpToTorch(x_warp):
    """
    Cast a Warp array back to a PyTorch tensor, ensuring it's on the correct device and has the right dtype.
    """
    # Convert to PyTorch tensor
    x_torch = x_warp.contiguous().to_torch()
    
    x_torch = x_torch.reshape(x_warp.shape)  # Ensure the shape matches the original Warp array
    
    return x_torch


def clearDummyTensorCache() -> None:
    """Clear all cached dummy tensors.

    Useful when explicitly reclaiming memory across long-lived sessions.
    """
    _DUMMY_TENSOR_CACHE.clear()


def getCachedDummyTensor(
    shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    fillValue: float = 0.0,
) -> torch.Tensor:
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



from warp.types import vector, matrix
# from wp_tensor import tensor

def castTorchToWarpAsBuiltins(x_torch):
    """
    Cast a PyTorch tensor to a Warp array of built-in types (e.g., float32, int32), ensuring it's on the correct device and has the right dtype.
    
    This function also performs conversions to builtin warp types, e.g., vec2f for 2D float tensors, vec3i for 3D int tensors, etc.
    
    """
    x_torch = x_torch.contiguous()
    
    if len(x_torch.shape) == 1:
        # 1D tensor, return as is with appropriate dtype
        return wp.from_torch(x_torch)
    elif len(x_torch.shape) == 2:
        N, D = x_torch.shape
        return wp.from_torch(x_torch, dtype=vector(length=D, dtype=wp.from_torch(x_torch).dtype))
    elif len(x_torch.shape) == 3:
        N, M, D = x_torch.shape
        return wp.from_torch(x_torch, dtype=matrix(shape=(M, D), dtype=wp.from_torch(x_torch).dtype))
    elif len(x_torch.shape) == 4:
        # Warp's kernel type system only handles rank-1 (vector) and rank-2 (matrix).
        # Flatten the trailing three dims into a single vector dimension so the
        # array can be used as a kernel argument type.
        N, P, M, D = x_torch.shape
        scalar_wp = wp.from_torch(x_torch.reshape(N, P * M * D)).dtype
        return wp.from_torch(x_torch.reshape(N, P * M * D).contiguous(),
                             dtype=vector(length=P * M * D, dtype=scalar_wp))
    
    else:
        return wp.from_torch(x_torch)