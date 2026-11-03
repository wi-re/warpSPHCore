"""Readable Warp segmented reduction used by cell multigrid."""
import warp as wp
from ..type_config import scalar_t

@wp.kernel
def _segmentSum_Kernel(perm: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64), ends: wp.array(dtype=wp.int64),  # type: ignore
                       src: wp.array2d(dtype=scalar_t), out: wp.array2d(dtype=scalar_t)):  # type: ignore
    o, c = wp.tid()
    acc = out[o, c]
    for k in range(wp.int32(starts[o]), wp.int32(ends[o])):
        acc += src[perm[k], c]
    out[o, c] = acc
