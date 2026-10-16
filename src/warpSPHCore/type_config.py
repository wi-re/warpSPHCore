"""Central precision and dimension configuration for Warp SPH types.

Configure before importing `warpSPHCore` by setting environment variables:

- `warpSPHCore_PRECISION`: one of `float16|half|float32|single|float64|double`
- `warpSPHCore_DIM`: positive integer for fixed dimension, or `Any` for dynamic
- `warpSPHCore_INDEX_DTYPE`: `int32` (default) or `int64` for particle IDs
- `warpSPHCore_OFFSET_DTYPE`: `int32` (default) or `int64` for CSR addresses

Defaults:

- precision: `wp.float32`
- dimension: `Any`
- particle IDs and CSR offsets: `wp.int32`, independently configurable
"""

from __future__ import annotations

import os
from typing import Any
import importlib

import warp as wp


_PRECISION_ALIASES = {
    "half": wp.float16,
    "float16": wp.float16,
    "single": wp.float32,
    "float32": wp.float32,
    "double": wp.float64,
    "float64": wp.float64,
}


def _resolve_precision(value: str | None):
    if value is None:
        return wp.float32

    key = value.strip().lower()
    if key == "":
        return wp.float32

    if key not in _PRECISION_ALIASES:
        valid = ", ".join(sorted(_PRECISION_ALIASES.keys()))
        raise ValueError(
            f"Invalid warpSPHCore_PRECISION='{value}'. Expected one of: {valid}."
        )

    return _PRECISION_ALIASES[key]


def _resolve_dim(value: str | int | None):
    if value is None or (not isinstance(value, str) and not isinstance(value, int)):
        return Any

    if isinstance(value, int):
        dim = value
        if dim <= 0:
            raise ValueError(
                f"Invalid warpSPHCore_DIM={value}. Dimension must be a positive integer."
            )
        return dim
    key = value.strip()
    if key == "":
        return Any

    if key.lower() in {"any", "dynamic", "none", "*"}:
        return Any

    try:
        dim = int(key)
    except ValueError as exc:
        raise ValueError(
            f"Invalid warpSPHCore_DIM='{value}'. Expected a positive integer or 'Any'."
        ) from exc

    if dim <= 0:
        raise ValueError(
            f"Invalid warpSPHCore_DIM='{value}'. Dimension must be a positive integer."
        )

    return dim


def _resolve_integer_type(value: str | None, setting: str):
    key = 'int32' if value is None or not str(value).strip() else str(value).strip().lower()
    if key not in ('int32', 'int64'):
        raise ValueError(f"Invalid {setting}={value!r}; expected 'int32' or 'int64'.")
    return wp.int32 if key == 'int32' else wp.int64


scalar_t: type = _resolve_precision(os.getenv("warpSPHCore_PRECISION"))
scalar: type = scalar_t

dim_t: int | Any = _resolve_dim(os.getenv("warpSPHCore_DIM"))
index_t: type = _resolve_integer_type(os.getenv('warpSPHCore_INDEX_DTYPE'), 'warpSPHCore_INDEX_DTYPE')
offset_t: type = _resolve_integer_type(os.getenv('warpSPHCore_OFFSET_DTYPE'), 'warpSPHCore_OFFSET_DTYPE')
neighbor_count_t: type = wp.int32


def _load_pythonic_preconfig():
    try:
        cfg = importlib.import_module("warpSPHCore_config")
    except ModuleNotFoundError:
        return None, None, None, None

    precision = getattr(cfg, "precision", None)
    dim = getattr(cfg, "dim", None)
    return precision, dim, getattr(cfg, 'index_dtype', None), getattr(cfg, 'offset_dtype', None)


_pre_precision, _pre_dim, _pre_index, _pre_offset = _load_pythonic_preconfig()

if _pre_precision is not None:
    scalar_t = _resolve_precision(str(_pre_precision))
    scalar = scalar_t

if _pre_dim is not None:
    dim_t = _resolve_dim(_pre_dim)

if _pre_index is not None:
    index_t = _resolve_integer_type(_pre_index, 'index_dtype')
if _pre_offset is not None:
    offset_t = _resolve_integer_type(_pre_offset, 'offset_dtype')


def get_type_config() -> dict[str, object]:
    """Return the active precision, dimension and integer-role types."""
    return {
        "scalar_t": scalar_t,
        "dim_t": dim_t,
        "index_t": index_t,
        "offset_t": offset_t,
        "neighbor_count_t": neighbor_count_t,
    }

def get_precision() -> type:
    """Return the active scalar type."""
    return scalar_t

def get_dim() -> int | Any:
    """Return the active dimension type."""
    return dim_t

import torch


def get_torch_index_dtype() -> torch.dtype:
    return torch.int32 if index_t == wp.int32 else torch.int64


def get_torch_offset_dtype() -> torch.dtype:
    return torch.int32 if offset_t == wp.int32 else torch.int64

def get_torch_precision() -> type:
    """Return the corresponding PyTorch dtype for the active precision."""
    if scalar_t == wp.float16:
        return torch.float16
    elif scalar_t == wp.float32:
        return torch.float32
    elif scalar_t == wp.float64:
        return torch.float64
    else:
        raise ValueError(f"Unsupported scalar type: {scalar_t}")
    
def to_torch(value: scalar_t) -> torch.Tensor:
    """Convert a scalar value to a PyTorch tensor with the corresponding dtype."""
    return torch.tensor(value, dtype=get_torch_precision())
def to_numpy(value: scalar_t) -> float:
    """Convert a scalar value to a NumPy float with the corresponding dtype."""
    return float(value)


from warp.types import vector, matrix

vec_t: type = vector(length=dim_t, dtype=scalar_t)
mat_t: type = matrix(shape =(dim_t, dim_t), dtype=scalar_t)
int_t: type = wp.int32

vecArray_t: type = wp.array(dtype=vec_t)
matArray_t: type = wp.array(dtype=mat_t)
intArray_t: type = wp.array(dtype=int_t)
scalarArray_t: type = wp.array(dtype=scalar_t)

@wp.func
def scalar(value: float) -> scalar_t:
    """Convert a Python float to the active scalar type."""
    return scalar_t(value)

__all__ = ["scalar_t", "scalar", "dim_t", "get_type_config", "get_precision", "get_dim", "get_torch_precision", "to_torch", "to_numpy", "vec_t", "mat_t", "int_t", "vecArray_t", "matArray_t", "intArray_t", "scalarArray_t", "index_t", "offset_t", "neighbor_count_t", "get_torch_index_dtype", "get_torch_offset_dtype"]
