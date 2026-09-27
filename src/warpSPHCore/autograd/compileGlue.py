"""Fused elementwise glue via ``torch.compile`` -- opt-in, not bitwise.

Between the warp neighbour kernels, a small-problem step spends a large part
of its GPU time in short chains of torch elementwise/reduction ops (a clamp,
a blend, a norm, a max): each op is its own ~1-3 us kernel, and at ~10k
particles the kernels are pure launch/latency cost. Measured on the Marrone
3.1 probe (nx = 70, whole-step CUDA graph): ~740 kernels per step on the
simulation stream, ~1.1 ms of its 2.4 ms in torch glue.

Functions decorated with :func:`compileGlue` are *pure torch* helpers (no warp
launches, no host syncs) carved out of those regions. With the switch off
(the default) the decorator is a pass-through; with it on, the first call
compiles the function with ``torch.compile(fullgraph=True)`` (inductor fuses
each chain into one or a few Triton kernels) and later calls run the compiled
version. Compiled kernels are captured and replayed by CUDA graphs like any
other.

Not bitwise: fusion contracts multiply-adds (FMA) and may reorder
reductions, so results move at float rounding. Validate with tolerances and
physics runs, not bitwise comparisons against the eager code. Any compile
failure warns once and leaves that function eager.

``WARPSPHCORE_COMPILE_GLUE=1`` (or :func:`setCompileGlue`) turns it on; a
process-wide setting, read at call time.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Callable, Optional

__all__ = ['compileGlue', 'compileGlueEnabled', 'markDynamic', 'setCompileGlue']

_override: Optional[bool] = None
_env = os.environ.get('WARPSPHCORE_COMPILE_GLUE', '') not in ('', '0')


def compileGlueEnabled() -> bool:
    return _override if _override is not None else _env


def setCompileGlue(enabled: Optional[bool]) -> None:
    """Override the switch for this process; ``None`` reverts to the env."""
    global _override
    _override = None if enabled is None else bool(enabled)


def compileGlue(fn: Callable) -> Callable:
    """Decorator: run ``fn`` compiled when :func:`compileGlueEnabled`."""
    state = {'compiled': None, 'failed': False}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not compileGlueEnabled() or state['failed']:
            return fn(*args, **kwargs)
        if state['compiled'] is None:
            import torch
            state['compiled'] = torch.compile(fn, fullgraph=True, dynamic=None)
        try:
            return state['compiled'](*args, **kwargs)
        except Exception as ex:  # noqa: BLE001 -- any compile/trace failure -> eager
            state['failed'] = True
            warnings.warn(f'[warpSPHCore] compileGlue: {fn.__qualname__} runs eagerly '
                          f'({type(ex).__name__}: {str(ex).splitlines()[0][:200]})', stacklevel=2)
            return fn(*args, **kwargs)

    wrapper.eager = fn
    return wrapper


def markDynamic(tensor, dim: int = 0):
    """Declare `tensor`'s `dim` variable-sized for a `compileGlue` function
    about to see it (e.g. an edge list, whose length changes with every
    Verlet rebuild), so the first compile is already shape-generic instead of
    recompiling on the first rebuild. A no-op while the switch is off."""
    if compileGlueEnabled():
        import torch
        torch._dynamo.maybe_mark_dynamic(tensor, dim)
    return tensor
