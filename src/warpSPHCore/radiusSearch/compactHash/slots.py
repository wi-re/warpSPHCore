"""Fixed-shape, sync-free build of the compact hash map -- the grid the neighbour searches run on.

`buildCompactHashMap` (buildHashmap.py) sizes its outputs by the data: the grid cell size follows the largest
support, the occupied cells and the hash entries come out of `torch.unique_consecutive`, the stencil radius from
`hMax / hCell`. That costs ~160 torch ops and 4 host syncs per build and the shapes change from call to call, so it
can neither run every stage nor be captured in a CUDA graph.

Here the grid is built for a support RADIUS fixed at construction (`HashMapSlots`): the cell size, the cell counts
and the stencil are host constants, and every intermediate has the particle count's size (at most N occupied cells,
at most N hash groups; the unused tail is padded with a sentinel that sorts last and is never reached through the
hash table). A build is then a fixed sequence of torch ops and warp launches that reads the positions / supports
from static input slots and overwrites static output buffers, so `GraphedBuild` can capture it once and replay it:
no syncs, no allocation, the same device addresses every time.

The one data-dependent question is "does the largest support still fit the radius the grid was built for?"; it is a
device flag (`slots.fits`) read lazily by the caller. If it is false the caller makes a bigger `HashMapSlots`.

Neighbour order: the cell size is the one `compute_h(domain, radius)` gives, so a grid built for the radius the eager
build would have used (`hMax`) is the same grid and gives the same CSR order; any other radius is a different grid with
the same neighbour sets in a different order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import warp as wp

from ...type_config import *
from ...util import *
from ...dataTypes import *
from ...enumTypes import *
from ...profiling import record_function
from .grid import compute_h
from .wp_hashCells import hashCells
from .wp_index import indexCells

__all__ = ['HashMapSlots', 'makeHashMapSlots', 'buildFixed', 'GraphedBuild']


@dataclass
class HashMapSlots:
    """Static shape information and the buffers of one fixed-shape hash map. Plain data; `buildFixed` fills it."""
    n: int
    dim: int
    hashMapLength: int
    radius: float
    hCell: float
    numCells: Tuple[int, ...]
    periodic: Tuple[bool, ...]
    mode: SupportScheme
    # domain constants on the device
    qMin: torch.Tensor
    qMax: torch.Tensor
    numCellsT: torch.Tensor
    cellOffsets: torch.Tensor
    # input slots
    positions: torch.Tensor
    supports: torch.Tensor
    # output buffers (the CompactHashMap views these)
    sortedPositions: torch.Tensor
    sortedSupports: torch.Tensor
    sortIndex: torch.Tensor
    hashTableBuffer: torch.Tensor           # (hashMapLength + 1, 2): the last row is scratch for the padded groups
    sortedCellTable: torch.Tensor           # (n, 3) int64: cell linear id, first particle, particle count
    fits: torch.Tensor                      # 0-d bool: the largest support is within `radius`
    hashMap: CompactHashMap = None


def makeHashMapSlots(n: int, domain: DomainDescription, radius: float, *, device, dtype=None,
                     mode: SupportScheme = SupportScheme.SuperSymmetric, hashMapLength: Optional[int] = None) -> HashMapSlots:
    """Slots for `n` particles in `domain`, with the grid built for supports up to `radius`."""
    device = torch.device(device)
    dtype = dtype or get_torch_precision()
    dim = domain.dim
    qMin = domain.min.to(device=device, dtype=dtype)
    qMax = domain.max.to(device=device, dtype=dtype)
    # the same cell size and counts the eager build derives from hMax = radius (host constants from here on)
    periodic = tuple(bool(p) for p in domain.periodic.tolist())
    hCellT = compute_h(qMin, qMax, torch.tensor(radius, device=device, dtype=dtype))
    hCell = float(hCellT.cpu())
    hCell = float(torch.tensor(hCell, dtype=dtype))                  # rounded to the build precision as eager does
    extent = (qMax - qMin).double().cpu().numpy()
    raw = extent / hCell
    eps = 1e-6
    counts = [max(1, int(np.floor(r + eps)) if periodic[d] else int(np.ceil(r - eps))) for d, r in enumerate(raw)]
    length = hashMapLength or n + 1
    offsets = torch.cartesian_prod(*[torch.arange(-1, 2, device=device) for _ in range(dim)]).to(torch.int32)
    offsets = offsets.reshape(-1, dim) if offsets.dim() == 1 else offsets
    padded = torch.zeros((offsets.shape[0], 3), dtype=torch.int32, device=device)
    padded[:, :dim] = offsets
    f = dict(device=device, dtype=dtype)
    return HashMapSlots(
        n=n, dim=dim, hashMapLength=length, radius=float(radius), hCell=hCell, numCells=tuple(counts), periodic=periodic,
        mode=mode, qMin=qMin, qMax=qMax, numCellsT=torch.tensor(counts, dtype=torch.int32, device=device),
        cellOffsets=padded,
        positions=torch.zeros((n, dim), **f), supports=torch.zeros((n,), **f),
        sortedPositions=torch.zeros((n, dim), **f), sortedSupports=torch.zeros((n,), **f),
        sortIndex=torch.zeros((n,), dtype=torch.int64, device=device),
        hashTableBuffer=torch.zeros((length + 1, 2), dtype=torch.int32, device=device),
        sortedCellTable=torch.zeros((n, 3), dtype=torch.int64, device=device),
        fits=torch.ones((), dtype=torch.bool, device=device))


def _delinearize(linear: torch.Tensor, counts: Tuple[int, ...]) -> torch.Tensor:
    grid = torch.zeros((linear.shape[0], len(counts)), dtype=torch.int32, device=linear.device)
    for d, c in enumerate(counts):
        grid[:, d] = torch.remainder(linear, c).to(torch.int32)
        linear = torch.div(linear, c, rounding_mode='floor')
    return grid


def buildFixed(s: HashMapSlots) -> CompactHashMap:
    """Build the hash map for the positions / supports in `s`'s input slots, into `s`'s buffers. No host reads,
    no allocation that depends on the data; capturable."""
    n, H, dim = s.n, s.hashMapLength, s.dim
    dev = s.positions.device
    stream = wp.stream_from_torch(torch.cuda.current_stream(dev)) if dev.type == 'cuda' else None
    with record_function("neighborSearch - buildFixed"):
        x = torch.stack([torch.remainder(s.positions[:, d] - s.qMin[d], s.qMax[d] - s.qMin[d]) + s.qMin[d] if s.periodic[d]
                         else s.positions[:, d] for d in range(dim)], dim=1)
        s.fits.copy_(s.supports.max() <= s.radius * (1.0 + 1e-6))

        indices = torch.floor((x - s.qMin) / s.hCell).to(torch.int32)
        indices = torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), (s.numCellsT - 1).view(1, -1))
        linear = torch.zeros((n,), dtype=torch.int64, device=dev)
        wp.launch(indexCells, dim=n, inputs=[castTorchToWarp(indices), castTorchToWarp(s.numCellsT), castTorchToWarp(linear)],
                  device=castTorchToWarp(indices).device, stream=stream)
        order = torch.argsort(linear, stable=True)
        sortedLinear = linear[order]
        s.sortIndex.copy_(order)
        s.sortedPositions.copy_(x[order])
        s.sortedSupports.copy_(s.supports[order])

        # occupied cells, padded to n: run-length encoding of the sorted linear ids without unique_consecutive
        slotIds = torch.arange(n, device=dev)
        newCell = torch.ones((n,), dtype=torch.bool, device=dev)
        newCell[1:] = sortedLinear[1:] != sortedLinear[:-1]
        cellId = torch.cumsum(newCell, dim=0) - 1
        cellLinear = torch.full((n,), -1, dtype=torch.int64, device=dev).scatter_(0, cellId, sortedLinear)
        cellStart = torch.full((n,), n, dtype=torch.int64, device=dev).scatter_reduce_(0, cellId, slotIds, 'amin')
        cellCount = torch.zeros((n,), dtype=torch.int64, device=dev).scatter_add_(0, cellId, torch.ones_like(cellId))
        occupied = cellCount > 0

        grid = _delinearize(torch.where(occupied, cellLinear, torch.zeros_like(cellLinear)), s.numCells)
        hashed, hashedWarp = allocateTorchWarp(n, wp.uint32, castTorchToWarp(grid).device)   # (int32 view of uint32)
        wp.launch(hashCells, dim=n, inputs=[castTorchToWarp(grid), wp.uint32(H), hashedWarp],
                  device=castTorchToWarp(grid).device, stream=stream)
        key = torch.where(occupied, hashed.to(torch.int64) & 0xFFFFFFFF, torch.full_like(cellLinear, H))
        hashOrder = torch.argsort(key, stable=True)
        keySorted = key[hashOrder]
        s.sortedCellTable.copy_(torch.stack((cellLinear[hashOrder], cellStart[hashOrder], cellCount[hashOrder]), dim=1))

        # hash groups, padded to n; the group of unoccupied slots is key H and lands in the scratch row
        newGroup = torch.ones((n,), dtype=torch.bool, device=dev)
        newGroup[1:] = keySorted[1:] != keySorted[:-1]
        groupId = torch.cumsum(newGroup, dim=0) - 1
        groupKey = torch.full((n,), H, dtype=torch.int64, device=dev).scatter_(0, groupId, keySorted)
        groupStart = torch.full((n,), n, dtype=torch.int64, device=dev).scatter_reduce_(0, groupId, slotIds, 'amin')
        groupCount = torch.zeros((n,), dtype=torch.int64, device=dev).scatter_add_(0, groupId, torch.ones_like(groupId))
        s.hashTableBuffer[:, 0] = -1
        s.hashTableBuffer[:, 1] = 0
        s.hashTableBuffer[groupKey, 0] = groupStart.to(torch.int32)
        s.hashTableBuffer[groupKey, 1] = groupCount.to(torch.int32)

    if s.hashMap is None:
        s.hashMap = CompactHashMap(
            sortedPositions=s.sortedPositions, sortedSupports=s.sortedSupports, sortIndex=s.sortIndex,
            hashTable=s.hashTableBuffer[:H], sortedCellTable=s.sortedCellTable, qMin=s.qMin, qMax=s.qMax,
            hCell=s.hCell, numCells=s.numCellsT, mode_uint=s.mode.value, D=dim, searchRadius=1,
            numOffsets=s.cellOffsets.shape[0], cellOffsets=s.cellOffsets)
    return s.hashMap


class GraphedBuild:
    """`buildFixed` captured in a CUDA graph on first use (after one eager warm-up) and replayed after: the caller
    copies positions / supports into `slots.positions` / `slots.supports` and calls `replay()`. CPU: runs eagerly."""

    def __init__(self, slots: HashMapSlots):
        self.slots = slots
        self.graph = None
        self.calls = 0

    def replay(self) -> CompactHashMap:
        s = self.slots
        if s.positions.device.type != 'cuda':
            return buildFixed(s)
        if self.graph is None:
            if self.calls == 0:
                self.calls = 1
                return buildFixed(s)                                  # warm-up: compile kernels, size the allocator
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                buildFixed(s)
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                buildFixed(s)
            self.graph = graph
        self.graph.replay()
        return s.hashMap
