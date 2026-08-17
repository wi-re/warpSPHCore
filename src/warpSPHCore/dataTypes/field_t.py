"""The `Field` abstraction: a cached dual torch/warp view of a tensor's storage.

Not wired into any hot path yet -- see warpier_fields.md Step B. This module and
`util/fieldRegistry.py` are unit-tested in isolation; `arg_extract.py` starts
consuming them in Step C (null fields) and Step D/E (view reuse).

See warpier_fields.md Section 3.1-3.4 for the full design rationale, in
particular why this is *not* a repeat of the data_ptr-keyed cache that was
deleted for silently accumulating gradients (Section 3.3): identity here is
owned (attached to one specific tensor object) rather than inferred from a
storage address that could be shared by unrelated tensors.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import torch
import warp as wp


class Role(Enum):
    """Which value a Field view represents. One role is live today; TANGENT
    is Phase 6's forward-mode affordance (Section 3.6, requirement 2) -- the
    slot costs one dict entry and is otherwise inert until something writes
    into it."""
    PRIMAL = 0
    TANGENT = 1


class ExecutionMode(Enum):
    """NONE | REVERSE | FORWARD, carried on a minimal execution context and
    folded into the StateBundle cache key (Step F). FORWARD is declared and
    deliberately unimplemented -- see warpier_fields.md Section 3.6,
    requirement 4 and Step G's readiness audit."""
    NONE = 0
    REVERSE = 1
    FORWARD = 2


class FieldKind(Enum):
    """The null-field registry's key axis (Section 3.4's table)."""
    SCALAR = 0
    VECTOR = 1
    MATRIX = 2
    INT32 = 3
    INT64 = 4
    VEC2I = 5
    VEC3I = 6
    VEC3L = 7
    BOOL = 8


def _no_field():
    return None


class Field:
    """Dual torch/warp view of one tensor's storage.

    Two ownership modes (Section 3.1):

    Attached fields (tensors owned by a caller, e.g. the frontend): stored on
    the tensor as ``t._wsc_field`` by ``fieldRegistry.acquireView``. Holds no
    strong reference back to ``t`` -- only ``owner=None`` here, so the tensor
    dies by refcount alone with no cycle for the GC to break on its own
    schedule (verified with gc.disable(); test 6 in the test suite pins this).

    Standalone fields (core-owned: null fields, adjacency, ...): ``owner`` is
    the torch.Tensor itself, since the whole point is that the Field outlives
    any particular call.
    """

    __slots__ = ("views", "tangent", "_ptr", "_shape", "_strides", "_dtype", "_owner")

    def __init__(
        self,
        view: "wp.array",
        ptr: int,
        shape: tuple,
        strides: tuple,
        dtype: torch.dtype,
        owner: Optional[torch.Tensor] = None,
    ):
        self.views: Dict[Role, "wp.array"] = {Role.PRIMAL: view}
        self.tangent: Optional["Field"] = None
        self._ptr = ptr
        self._shape = shape
        self._strides = strides
        self._dtype = dtype
        self._owner = owner

    def matches(self, t: torch.Tensor) -> bool:
        """Provenance check against a live tensor: data_ptr/shape/strides/dtype
        unchanged since this Field was built. Catches resize_, .data
        reassignment, and storage swaps (Section 3.1's "Revalidation")."""
        return (
            t.data_ptr() == self._ptr
            and tuple(t.shape) == self._shape
            and tuple(t.stride()) == self._strides
            and t.dtype == self._dtype
        )

    def view(self, role: Role = Role.PRIMAL) -> "wp.array":
        return self.views[role]

    # -- Copy / pickle safety (Section 2.4, 3.1) --------------------------
    # A Field wraps a ctypes-backed wp.array with a live device pointer.
    # deepcopy-ing or pickling one is at best meaningless and at worst a
    # dangling pointer once restored, so every path degrades to None. A
    # copied/restored tensor then arrives with `_wsc_field = None`, which
    # acquireView treats as a miss and rebuilds cleanly -- no cross-process
    # device pointers, no silent restart-file corruption.
    def __copy__(self):
        return None

    def __deepcopy__(self, memo):
        return None

    def __reduce__(self):
        return (_no_field, ())
