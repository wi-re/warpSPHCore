"""Gated `record_function`, replacing the always-on `torch.profiler`
version used throughout the package (warpier_fields.md Section 3.7).

`torch.profiler.record_function`'s `__enter__`/`__exit__` call into its C++
RecordFunction machinery on every use regardless of whether a profiler is
actually attached -- Section 1.2 measured 12 enter/exit pairs per operator
call (94 across the package) paying that cost unconditionally. Set
`WARPSPHCORE_PROFILING=1` to get the real profiler-hooked version back;
otherwise every call site gets `contextlib.nullcontext`, whose
`__enter__`/`__exit__` are trivial.

Deliberately a top-level, zero-internal-dependency module (only stdlib
`os`/`contextlib`) imported as the very first thing in
`warpSPHCore/__init__.py`, and not inside e.g. `autograd/` -- every other
subpackage (`coreOperations`, `radiusSearch`, `crk`, `pinv`, `util`, ...)
needs this, and `autograd/__init__.py` itself pulls in `radiusSearch` (via
`arg_extract.py`) and other heavy submodules. Originally lived at
`autograd/profiling.py`; moved here after `from ..autograd.profiling import
record_function` in `util/wp_util.py` produced a genuine circular import --
that import forced `warpSPHCore.autograd`'s `__init__.py` to run, which
pulled in `radiusSearch`, which (via `compactHash/sort.py`'s `from
...util import *`) reentered `util` while it was still mid-import (still
inside its own first `from .wp_util import (...)` line), and Python's
reentrant-import handling returned that partially-initialized `util`
module -- silently missing names (`castTorchToWarp` among them) that
`util/__init__.py` hadn't reached yet. Being imported first here, before
`type_config` or anything else, means `warpSPHCore.profiling` is always
fully populated in `sys.modules` before any such reentrant chain can start,
so no submodule anywhere in the package can trigger this again by importing
it.
"""

import os
from contextlib import nullcontext

PROFILING = os.environ.get("WARPSPHCORE_PROFILING", "0") == "1"

if PROFILING:
    from torch.profiler import record_function
else:
    def record_function(name):  # noqa: ARG001 -- name kept for call-site compatibility
        return nullcontext()
