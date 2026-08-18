"""Field acquisition, the null-field registry, and the struct-type lookup
table -- warpier_fields.md Step B. Not wired into any hot path yet; consumed
starting at Step C (null fields replace dummy tensors) and Step D/E (view
reuse on the no-grad, then grad, path).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import warp as wp

from ..dataTypes.field_t import Field, Role, ExecutionMode, FieldKind
from ..dataTypes.particleData import particleDataSoA_1, particleDataSoA_2, particleDataSoA_3
from ..dataTypes.corrections_t import correctionData_1, correctionData_2, correctionData_3
from ..type_config import get_torch_precision
from .cast import castTorchToWarpAsBuiltins


_ATTR = "_wsc_field"


# --------------------------------------------------------------------------
# acquireView -- the one entry point (Section 3.2). Replaces getCachedWarpArray.
# --------------------------------------------------------------------------

def _cache_disabled() -> bool:
    # Escape hatch: forces the miss path unconditionally, restoring exactly
    # today's (pre-Field) semantics. Bisecting a suspected caching bug is
    # then one environment variable, not a revert.
    return os.environ.get("WARPSPHCORE_DISABLE_FIELD_CACHE", "0") == "1"


def acquireView(t: torch.Tensor, role: Role = Role.PRIMAL) -> "wp.array":
    """Read t._wsc_field -> revalidate -> hit returns the cached view; miss
    (including a non-contiguous tensor, which is never cached -- see
    Field.matches's docstring and Section 3.1's staleness hazard) builds a
    fresh view and, if contiguous, attaches it for next time.

    Every branch below builds from ``t.detach()``, never bare ``t``: until
    warpier_fields.md Step E, this function was only ever called with a
    tensor that could not have requires_grad=True (null fields never
    require grad; Step D's caller-tensor caching was gated to the no-grad
    path only), so a missing detach() in the miss/escape-hatch branches was
    latent, not exercised. Step E is the first caller that can hand this a
    requires_grad=True tensor, and building an array off it undetached lets
    that array's conversion op (e.g. `.contiguous()`, a no-op that returns
    the same tensor when already contiguous) sit inside torch's own
    autograd graph in addition to warp's tape -- silently doubling the
    reported gradient. Caught by cross-checking a cached run against one
    with WARPSPHCORE_DISABLE_FIELD_CACHE=1, which hits exactly this branch.
    """
    if _cache_disabled():
        return castTorchToWarpAsBuiltins(t.detach().contiguous())

    if not t.is_contiguous():
        # castTorchToWarpAsBuiltins would .contiguous()-copy a non-contiguous
        # tensor; a later in-place write to the original storage would then
        # be invisible through that copy while data_ptr on the original stays
        # stable, so caching here would be a false hit. Rebuild every time.
        return castTorchToWarpAsBuiltins(t.detach())

    field: Optional[Field] = getattr(t, _ATTR, None)
    if field is not None and field.matches(t) and role in field.views:
        return field.views[role]

    # Build the view from t.detach(), not t itself: wp.from_torch() gives the
    # resulting wp.array a _tensor back-reference to whatever tensor it was
    # given. Cache that view as t._wsc_field and a view built from t directly
    # would close a t -> Field -> view -> t reference cycle -- only the
    # cyclic GC could break it, not refcounting (test 6 pins this). detach()
    # shares storage (same data_ptr/shape/strides/dtype) but is a distinct
    # tensor object, so the cycle never forms while provenance tracking
    # against t stays exact.
    view = castTorchToWarpAsBuiltins(t.detach())
    new_field = Field(
        view=view,
        ptr=t.data_ptr(),
        shape=tuple(t.shape),
        strides=tuple(t.stride()),
        dtype=t.dtype,
    )
    try:
        setattr(t, _ATTR, new_field)
    except Exception:
        # Some tensor subclasses/views may refuse arbitrary attributes;
        # degrade to "no cache" rather than fail the call.
        pass
    return view


# --------------------------------------------------------------------------
# Null fields (Section 3.4): a permanent registry keyed by
# (kind, dim, device, precision, fill mode), built once and never converted
# again.
# --------------------------------------------------------------------------

_INT_MIN = -2147483648

_NULL_REGISTRY: Dict[tuple, Field] = {}


def _null_fill_mode() -> str:
    mode = os.environ.get("WARPSPHCORE_NULL_FILL", "zeros")
    if mode not in ("zeros", "sentinel"):
        raise ValueError(f"Invalid WARPSPHCORE_NULL_FILL={mode!r}; expected 'zeros' or 'sentinel'.")
    return mode


def _shape_for_kind(kind: FieldKind, dim: int) -> Tuple[int, ...]:
    if kind == FieldKind.SCALAR:
        return (1,)
    if kind == FieldKind.VECTOR:
        return (1, dim)
    if kind == FieldKind.MATRIX:
        return (1, dim, dim)
    if kind == FieldKind.INT32:
        return (1,)
    if kind == FieldKind.INT64:
        return (1,)
    if kind == FieldKind.VEC2I:
        return (1, 2)
    if kind == FieldKind.VEC3I:
        return (1, 3)
    if kind == FieldKind.VEC3L:
        return (1, 3)
    if kind == FieldKind.BOOL:
        return (1,)
    raise ValueError(f"Unknown FieldKind: {kind}")


def _torch_dtype_for_kind(kind: FieldKind, precision_dtype: torch.dtype) -> torch.dtype:
    if kind in (FieldKind.SCALAR, FieldKind.VECTOR, FieldKind.MATRIX):
        return precision_dtype
    if kind in (FieldKind.INT32, FieldKind.VEC2I, FieldKind.VEC3I):
        return torch.int32
    if kind in (FieldKind.INT64, FieldKind.VEC3L):
        return torch.int64
    if kind == FieldKind.BOOL:
        return torch.bool
    raise ValueError(f"Unknown FieldKind: {kind}")


def nullField(
    kind: FieldKind,
    dim: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    *,
    tangent: bool = False,
) -> Field:
    """Permanent zero- (or sentinel-)filled Field for a disabled correction
    path, built once per (kind, dim, device, precision) and never converted
    again.

    ``tangent=True`` forces zero fill regardless of WARPSPHCORE_NULL_FILL: a
    zero tangent is semantically meaningful (the field simply isn't seeded),
    never a can't-happen placeholder -- Section 3.6, requirement 5. Sentinel
    fill is only ever appropriate for primal nulls, where every reader is
    supposed to be gated behind a `correctionData.useX` flag and a read
    despite that gate is a real bug to surface loudly.
    """
    precision_dtype = dtype if dtype is not None else get_torch_precision()
    fill_mode = "zeros" if tangent else _null_fill_mode()
    key = (kind, int(dim), str(device), str(precision_dtype), fill_mode)

    cached = _NULL_REGISTRY.get(key)
    if cached is not None:
        return cached

    shape = _shape_for_kind(kind, dim)
    torch_dtype = _torch_dtype_for_kind(kind, precision_dtype)

    if fill_mode == "sentinel" and torch_dtype != torch.bool:
        fill_value = float("nan") if torch_dtype.is_floating_point else _INT_MIN
        tensor = torch.full(shape, fill_value, dtype=torch_dtype, device=device)
    else:
        tensor = torch.zeros(shape, dtype=torch_dtype, device=device)
    tensor = tensor.contiguous()

    view = castTorchToWarpAsBuiltins(tensor)
    field = Field(
        view=view,
        ptr=tensor.data_ptr(),
        shape=tuple(tensor.shape),
        strides=tuple(tensor.stride()),
        dtype=tensor.dtype,
        owner=tensor,
    )
    _NULL_REGISTRY[key] = field
    return field


def clearNullFieldRegistry() -> None:
    """Test/debug hook: drop every cached null Field."""
    _NULL_REGISTRY.clear()


# --------------------------------------------------------------------------
# structFor (Section 3.5, requirement 3): a lookup table replacing the
# `dim == 1 if ... else ...` ternaries inline in arg_extract.py. Phase 6
# registers *_dual rows into this same table instead of rewriting every
# extractor. FORWARD rows alias REVERSE's struct classes rather than getting
# their own: Tier 1 forward mode (warpier_forward_mode_plan.md Phase 2) needs
# no new struct shape -- it relaunches the same kernels on a tangent array in
# place of the value array -- so there is nothing for a FORWARD-specific row
# to hold that REVERSE's rows don't already provide. A mode that needs an
# actual struct difference (e.g. a future Tier-2 dual-number layout) gets its
# own rows here instead of overloading this alias.
# --------------------------------------------------------------------------

_STRUCT_TABLE = {
    ("particleDataSoA", 1, ExecutionMode.NONE):    particleDataSoA_1,
    ("particleDataSoA", 1, ExecutionMode.REVERSE): particleDataSoA_1,
    ("particleDataSoA", 1, ExecutionMode.FORWARD): particleDataSoA_1,
    ("particleDataSoA", 2, ExecutionMode.NONE):    particleDataSoA_2,
    ("particleDataSoA", 2, ExecutionMode.REVERSE): particleDataSoA_2,
    ("particleDataSoA", 2, ExecutionMode.FORWARD): particleDataSoA_2,
    ("particleDataSoA", 3, ExecutionMode.NONE):    particleDataSoA_3,
    ("particleDataSoA", 3, ExecutionMode.REVERSE): particleDataSoA_3,
    ("particleDataSoA", 3, ExecutionMode.FORWARD): particleDataSoA_3,

    ("correctionData", 1, ExecutionMode.NONE):    correctionData_1,
    ("correctionData", 1, ExecutionMode.REVERSE): correctionData_1,
    ("correctionData", 1, ExecutionMode.FORWARD): correctionData_1,
    ("correctionData", 2, ExecutionMode.NONE):    correctionData_2,
    ("correctionData", 2, ExecutionMode.REVERSE): correctionData_2,
    ("correctionData", 2, ExecutionMode.FORWARD): correctionData_2,
    ("correctionData", 3, ExecutionMode.NONE):    correctionData_3,
    ("correctionData", 3, ExecutionMode.REVERSE): correctionData_3,
    ("correctionData", 3, ExecutionMode.FORWARD): correctionData_3,
}


def structFor(kind: str, dim: int, mode: ExecutionMode = ExecutionMode.REVERSE):
    try:
        return _STRUCT_TABLE[(kind, dim, mode)]
    except KeyError:
        raise ValueError(f"No struct registered for kind={kind!r}, dim={dim}, mode={mode!r}")
