"""Checked CSR construction, independent particle-ID and edge-address widths."""
import torch
from ..type_config import get_torch_offset_dtype

# Warp array descriptors use signed int32 shapes, even with int64 element indices.
WARP_ARRAY_MAX = 2**31 - 1


def validateParticleCounts(numRows: int, numCols: int) -> None:
    """Bound per-row counts and the current int32 particle/grid traversal ABI."""
    if not (0 <= numRows <= WARP_ARRAY_MAX and 0 <= numCols <= WARP_ARRAY_MAX):
        raise OverflowError('Particle populations must fit Warp signed int32 array dimensions and per-row counts.')


def validateEdgeCount(total: int, dtype: torch.dtype | None = None) -> None:
    dtype = get_torch_offset_dtype() if dtype is None else dtype
    if dtype not in (torch.int32, torch.int64):
        raise TypeError('CSR offset dtype must be torch.int32 or torch.int64.')
    if total < 0 or total > torch.iinfo(dtype).max:
        raise OverflowError(f'CSR edge count {total} does not fit {dtype}; select int64 offsets before importing warpSPHCore.')


def validateEdgeCapacity(total: int, capacity: int) -> None:
    """Reject unsupported flat arrays before allocation, including headroom."""
    validateEdgeCount(total)
    if capacity < total or capacity < 0:
        raise ValueError('Neighbor capacity must be nonnegative and cover every live edge.')
    if capacity > WARP_ARRAY_MAX:
        raise OverflowError(f'Neighbor capacity {capacity} exceeds Warp flat-array dimension limit {WARP_ARRAY_MAX}; int64 offsets do not lift this backend limit.')


def buildCSROffsets(counts: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum/scan in int64; check before narrowing to configured offsets.

    Search kernels produce nonnegative int32 counts, bounded by the validated
    reference population. A wide total does not require wide per-row storage.
    """
    if counts.ndim != 1 or counts.dtype != torch.int32:
        raise TypeError('Neighbor counts must be a one-dimensional torch.int32 tensor.')
    total = int(counts.sum(dtype=torch.int64).item())
    validateEdgeCount(total)
    offsets = torch.zeros(counts.shape, dtype=get_torch_offset_dtype(), device=counts.device)
    offsets[1:] = counts[:-1].cumsum(0, dtype=torch.int64)
    return offsets, total
