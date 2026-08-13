import warp as wp
import torch
from ..type_config import *
from warp.types import vector, matrix


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


_TORCH_TO_WARP_SCALAR_DTYPE = {
    torch.float16: wp.float16,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
    torch.int8: wp.int8,
    torch.int16: wp.int16,
    torch.int32: wp.int32,
    torch.int64: wp.int64,
    torch.uint8: wp.uint8,
    torch.bool: wp.bool,
}

# Cache vector/matrix dtype objects to avoid rebuilding Warp type metadata
# for every tensor conversion call on hot paths.
_WARP_VECTOR_DTYPE_CACHE: dict[tuple[int, torch.dtype], object] = {}
_WARP_MATRIX_DTYPE_CACHE: dict[tuple[int, int, torch.dtype], object] = {}


def _torch_scalar_to_warp_dtype(dtype: torch.dtype):
    scalar = _TORCH_TO_WARP_SCALAR_DTYPE.get(dtype)
    if scalar is None:
        raise TypeError(f"Unsupported torch dtype for Warp conversion: {dtype}")
    return scalar


def _get_warp_vector_dtype(length: int, torch_dtype: torch.dtype):
    key = (int(length), torch_dtype)
    cached = _WARP_VECTOR_DTYPE_CACHE.get(key)
    if cached is None:
        cached = vector(length=length, dtype=_torch_scalar_to_warp_dtype(torch_dtype))
        _WARP_VECTOR_DTYPE_CACHE[key] = cached
    return cached


def _get_warp_matrix_dtype(rows: int, cols: int, torch_dtype: torch.dtype):
    key = (int(rows), int(cols), torch_dtype)
    cached = _WARP_MATRIX_DTYPE_CACHE.get(key)
    if cached is None:
        cached = matrix(shape=(rows, cols), dtype=_torch_scalar_to_warp_dtype(torch_dtype))
        _WARP_MATRIX_DTYPE_CACHE[key] = cached
    return cached

def allocateTorchWarp(shape, dtype, device, requires_grad: bool = False):
    """
    Allocate a Warp kernel output on torch's own caching allocator instead of
    Warp's (wp.zeros(...) + wp.to_torch(...)), so kernel outputs that end up
    as torch tensors don't grow a second, independent GPU memory pool.

    Returns (torch_tensor, warp_array), where warp_array is a zero-copy Warp
    view of torch_tensor suitable for passing straight into wp.launch(...).
    ``device`` is a Warp DeviceLike (a wp.Device or a device string Warp
    recognizes, e.g. "cuda:0"), matching what callers already have on hand
    from e.g. ``castTorchToWarp(x).device``.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    trailing_shape = getattr(dtype, "_shape_", ())
    scalar_dtype = getattr(dtype, "_wp_scalar_type_", dtype)
    torch_dtype = wp.dtype_to_torch(scalar_dtype)
    torch_device = wp.device_to_torch(device)
    output_torch = torch.zeros(shape + trailing_shape, dtype=torch_dtype, device=torch_device)
    output_warp = wp.from_torch(output_torch, dtype=dtype, requires_grad=requires_grad)
    return output_torch, output_warp


def castTorchToWarpAsBuiltins(x_torch):
    """
    Cast a PyTorch tensor to a Warp array of built-in types (e.g., float32, int32), ensuring it's on the correct device and has the right dtype.
    
    This function also performs conversions to builtin warp types, e.g., vec2f for 2D float tensors, vec3i for 3D int tensors, etc.
    
    """
    if not x_torch.is_contiguous():
        x_torch = x_torch.contiguous()

    ndim = x_torch.ndim

    if ndim == 1:
        # 1D tensor, return as is with appropriate dtype
        return wp.from_torch(x_torch)
    elif ndim == 2:
        D = x_torch.shape[1]
        return wp.from_torch(x_torch, dtype=_get_warp_vector_dtype(D, x_torch.dtype))
    elif ndim == 3:
        M = x_torch.shape[1]
        D = x_torch.shape[2]
        return wp.from_torch(x_torch, dtype=_get_warp_matrix_dtype(M, D, x_torch.dtype))
    elif ndim == 4:
        # Warp's kernel type system only handles rank-1 (vector) and rank-2 (matrix).
        # Flatten the trailing three dims into a single vector dimension so the
        # array can be used as a kernel argument type.
        N = x_torch.shape[0]
        P = x_torch.shape[1]
        M = x_torch.shape[2]
        D = x_torch.shape[3]
        flat = P * M * D
        reshaped = x_torch.reshape(N, flat)
        if not reshaped.is_contiguous():
            reshaped = reshaped.contiguous()
        return wp.from_torch(reshaped, dtype=_get_warp_vector_dtype(flat, x_torch.dtype))
    
    else:
        print(f"Warning: castTorchToWarpAsBuiltins received tensor with shape {x_torch.shape} which is not 1D, 2D, 3D, or 4D. Returning as a flat array of scalars.")
        return wp.from_torch(x_torch)