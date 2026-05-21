"""Central precision and dimension configuration for Warp SPH types.

Configure before importing `sphWarpCore` by setting environment variables:

- `SPHWARPCORE_PRECISION`: one of `float16|half|float32|single|float64|double`
- `SPHWARPCORE_DIM`: positive integer for fixed dimension, or `Any` for dynamic

Defaults:

- precision: `wp.float32`
- dimension: `Any`
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
            f"Invalid SPHWARPCORE_PRECISION='{value}'. Expected one of: {valid}."
        )

    return _PRECISION_ALIASES[key]


def _resolve_dim(value: str | int | None):
    if value is None or (not isinstance(value, str) and not isinstance(value, int)):
        return Any

    if isinstance(value, int):
        dim = value
        if dim <= 0:
            raise ValueError(
                f"Invalid SPHWARPCORE_DIM={value}. Dimension must be a positive integer."
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
            f"Invalid SPHWARPCORE_DIM='{value}'. Expected a positive integer or 'Any'."
        ) from exc

    if dim <= 0:
        raise ValueError(
            f"Invalid SPHWARPCORE_DIM='{value}'. Dimension must be a positive integer."
        )

    return dim


scalar_t: type = _resolve_precision(os.getenv("SPHWARPCORE_PRECISION"))
scalar: type = scalar_t

dim_t: int | Any = _resolve_dim(os.getenv("SPHWARPCORE_DIM"))


def _load_pythonic_preconfig() -> tuple[str | None, str | int | None]:
    try:
        cfg = importlib.import_module("sphWarpCore_config")
    except ModuleNotFoundError:
        return None, None

    precision = getattr(cfg, "precision", None)
    dim = getattr(cfg, "dim", None)
    return precision, dim


_pre_precision, _pre_dim = _load_pythonic_preconfig()

if _pre_precision is not None:
    scalar_t = _resolve_precision(str(_pre_precision))
    scalar = scalar_t

if _pre_dim is not None:
    dim_t = _resolve_dim(_pre_dim)


def get_type_config() -> dict[str, object]:
    """Return the active precision/dimension configuration."""
    return {
        "scalar_t": scalar_t,
        "dim_t": dim_t,
    }

def get_precision() -> type:
    """Return the active scalar type."""
    return scalar_t

def get_dim() -> int | Any:
    """Return the active dimension type."""
    return dim_t

import torch

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


__all__ = ["scalar_t", "scalar", "dim_t", "get_type_config", "get_precision", "get_dim", "get_torch_precision", "to_torch", "to_numpy"]
