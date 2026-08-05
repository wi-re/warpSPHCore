import warp as wp
import torch
from ..types import *


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



from warp.types import vector, matrix
# from wp_tensor import tensor


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
    
# Convention Wise kind = 0 for fluid, 1 for boundary, 2 for ghost
# class OperationDirection(Enum):
#     AllToAll = 0
#     FluidToFluid = 1
#     FluidToBoundary = 2
#     BoundaryToFluid = 3
#     BoundaryToBoundary = 4
#     FluidToGhost = 5
#     GhostToFluid = 6
#     BoundaryToGhost = 7
#     GhostToBoundary = 8

@wp.func
def checkDirectionality_j(
    queryKind: wp.int32, opInt: wp.int32
):
    if opInt == 0: # No Ghost
        return queryKind != 2
    elif opInt == 9: # All to all
        return queryKind != 2
    elif opInt == 1: # fluid to fluid
        return queryKind == 0
    elif opInt == 2: # fluid to boundary
        return queryKind == 0
    elif opInt == 3: # boundary to fluid
        return queryKind == 1
    elif opInt == 4: # boundary to boundary
        return queryKind == 1
    elif opInt == 5: # fluid to ghost
        return queryKind == 0
    elif opInt == 6: # ghost to fluid
        return queryKind == 2
    elif opInt == 7: # boundary to ghost
        return queryKind == 1
    elif opInt == 8: # ghost to boundary
        return queryKind == 2
    elif opInt == 10: # all to ghost
        return queryKind != 2
    elif opInt == 11: # all to fluid
        return queryKind != 2
    elif opInt == 12: # all to boundary
        return queryKind != 2
    elif opInt == 13: # fluid to all
        return queryKind == 0
    elif opInt == 14: # boundary to all
        return queryKind == 1
    else:
        return False

@wp.func
def checkDirectionality_i(
    referenceKind: wp.int32, opInt: wp.int32
):
    if opInt == 0: # No Ghost
        return referenceKind != 2
    elif opInt == 9: # All to all
        return True
    elif opInt == 1: # fluid to fluid
        return referenceKind == 0
    elif opInt == 2: # fluid to boundary
        return referenceKind == 1
    elif opInt == 3: # boundary to fluid
        return referenceKind == 0
    elif opInt == 4: # boundary to boundary
        return referenceKind == 1
    elif opInt == 5: # fluid to ghost
        return referenceKind == 2
    elif opInt == 6: # ghost to fluid
        return referenceKind == 0
    elif opInt == 7: # boundary to ghost
        return referenceKind == 2
    elif opInt == 8: # ghost to boundary
        return referenceKind == 1
    elif opInt == 10: # all to ghost
        return referenceKind == 2
    elif opInt == 11: # all to fluid
        return referenceKind == 0
    elif opInt == 12: # all to boundary
        return referenceKind == 1
    elif opInt == 13: # fluid to all
        return referenceKind != 2
    elif opInt == 14: # boundary to all
        return referenceKind != 2
    else:
        return False
    
@wp.func
def checkDirectionality_Func(
    queryKind: wp.int32, referenceKind: wp.int32, opInt: wp.int32
):
    return checkDirectionality_i(queryKind, opInt) and checkDirectionality_j(referenceKind, opInt)

import torch
from ..math import volumeToSupport
from ..radiusSearch.radius_util import DomainDescription

def generateNeighborTestData(nx, targetNumNeighbors, dim, periodic, device):


    minDomain = torch.tensor([-1] * dim, dtype = torch.float32, device = device)
    maxDomain = torch.tensor([ 1] * dim, dtype = torch.float32, device = device)
    periodicity = torch.tensor([periodic] * dim, device = device, dtype = torch.bool)

    extent = maxDomain - minDomain
    shortExtent = torch.min(extent, dim = 0)[0].item()
    dx = (shortExtent / nx)
    ny = int(1 // dx)
    h = volumeToSupport(dx**dim, targetNumNeighbors, dim)
    dy = dx / 1.5
    ny = int(1 // dy)

    positions = []
    for d in range(dim):
        positions.append(torch.linspace(minDomain[d] + dx / 2, maxDomain[d] - dx / 2, int((extent[d] - dx) / dx) + 1, device = device))
    grid = torch.meshgrid(*positions, indexing = 'xy')
    
    positions = torch.stack(grid, dim = -1).reshape(-1,dim).to(device)
    supports = torch.ones(positions.shape[0], device = device) * h
    
    domain = DomainDescription(minDomain, maxDomain, periodicity, dim)
    
    return positions, supports, positions.shape[0], domain, dx



def getNextPrime(n):
    # Compute the next larger prime number greater than n
    # used primarily to set the hash map length to a prime number for better distribution of particles in the hash map
    
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    prime = n + 1
    while True:
        if is_prime(prime):
            return prime
        prime += 1

from typing import Any
from warp.types import vector, matrix


class vec1f(vector(length=1, dtype=wp.float32)):
    pass

class mat11f(matrix(shape=(1, 1), dtype=wp.float32)):
    pass

class vec1h(vector(length=1, dtype=wp.float16)):
    pass
class mat11h(matrix(shape=(1, 1), dtype=wp.float16)):
    pass

class vec1d(vector(length=1, dtype=wp.float64)):
    pass
class mat11d(matrix(shape=(1, 1), dtype=wp.float64)):
    pass

from warp import mat22f, mat33f, mat22h, mat33h, mat22d, mat33d
from warp import vec2f, vec3f, vec2h, vec3h, vec2d, vec3d, vec3i, vec2i

@wp.func
def zero_like(
    input: Any, # type: ignore
):
    return type(input)() * scalar_t(0.0)
@wp.func
def zero_like(
    input: wp.float32
):
    return wp.float32(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float32) # type: ignore
):
    return type(input)(0.0) * 0.0
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float32) # type: ignore
):
    return type(input)() * 0.0
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float32) # type: ignore
):
    return wp.float32(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = Any) # type: ignore
):
    return type(input[0])() * 0.0
@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float32)) # type: ignore
):
    return mat11f()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float32)) # type: ignore
):
    return mat22f()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float32)) # type: ignore
):
    return mat33f()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float32)) # type: ignore
):
    return vec1f(0.0) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float32)) # type: ignore
):
    return vec2f(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float32)) # type: ignore
):
    return vec3f(0.0)

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.int32)) # type: ignore
):
    return vector(length=1, dtype=wp.int32)(0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.int32)) # type: ignore
):
    return vec2i(0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.int32)) # type: ignore
):
    return vec3i(0)


@wp.func
def zero_like_warp(
    input: Any, # type: ignore
):
    return zero_like(input)


# Half precision versions
@wp.func
def zero_like(
    input: wp.float16
):
    return wp.float16(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float16) # type: ignore
):
    return type(input)(wp.float16(0.0)) * wp.float16(0.0)
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float16) # type: ignore
):
    return type(input)() * wp.float16(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float16) # type: ignore
):
    return wp.float16(0.0)

@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float16)) # type: ignore
):
    return mat11h()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float16)) # type: ignore
):
    return mat22h()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float16)) # type: ignore
):
    return mat33h()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float16)) # type: ignore
):
    return vec1h(wp.float16(0.0)) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float16)) # type: ignore
):
    return vec2h(wp.float16(0.0))
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float16)) # type: ignore
):
    return vec3h(wp.float16(0.0))

# Double precision versions
@wp.func
def zero_like(
    input: wp.float64
):
    return wp.float64(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float64) # type: ignore
):
    return type(input)(wp.float64(0.0)) * wp.float64(0.0)
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float64) # type: ignore
):
    return type(input)() * wp.float64(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float64) # type: ignore
):
    return wp.float64(0.0)

@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float64)) # type: ignore
):
    return mat11d()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float64)) # type: ignore
):
    return mat22d()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float64)) # type: ignore
):
    return mat33d()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float64)) # type: ignore
):
    return vec1d(wp.float64(0.0)) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float64)) # type: ignore
):
    return vec2d(wp.float64(0.0))
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float64)) # type: ignore
):
    return vec3d(wp.float64(0.0))   