"""Readable Warp block inverse application."""
from typing import Any
import warp as wp

@wp.kernel(enable_backward=False)
def _blockApply(inverse: wp.array3d(dtype=Any), v: wp.array(dtype=Any),
                out: wp.array(dtype=Any), d: int, transpose: bool):
    row = wp.tid()
    particle, component = row // d, row % d
    value = v.dtype(0)
    for k in range(d):
        coefficient = inverse[particle, component, k]
        if transpose:
            coefficient = inverse[particle, k, component]
        value += coefficient * v[particle * d + k]
    out[row] = value
