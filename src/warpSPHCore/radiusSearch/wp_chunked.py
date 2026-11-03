"""Warp global-row materialization for owned whole-row storage."""
import warp as wp
from ..type_config import index_t,offset_t

@wp.kernel
def _fillGlobalRows(counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=offset_t),
                    rowStart: wp.int32, rows: wp.array(dtype=index_t)):
    row, lane = wp.tid()
    p = wp.int64(lane)
    while p < wp.int64(counts[row]):
        rows[offsets[row] + offset_t(p)] = index_t(rowStart + row)
        p += wp.int64(32)
