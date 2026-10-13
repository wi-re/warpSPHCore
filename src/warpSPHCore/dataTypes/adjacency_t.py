from .hashMap_t import CompactHashMap
from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..type_config import *

# A dataclass to hold the adjacency list information, including the indices of the neighbors, the number of neighbors for each query point, and the edge offsets for efficient access.
# This can be used as a coordinate formate (COO) for the adjacency list, where paddedI and paddedJ represent the row and column indices of the neighbors
# This can also be used as a compressed sparse row (CSR) format, where edgeOffsets can be used to quickly access the neighbors of each query point, and numNeighbors can be used to know how many neighbors each query point has.
# Notably this cannot be used as a compressed sparse column (CSC) format, since the neighbors are not sorted by the reference points, but rather by the query points.
# Because of the sorting we can reconstruct the row index from edgeOffsets and numNeighbors, but we keep it for convenience and to avoid having to reconstruct it every time.

# One unfortunate aspect is that the torch tensors need to be of dtype long to allow indexing within torch. warp could naturally handle int dtypes, consuming less memory, but torch does not allow indexing with int32 tensors, so we need to convert them to int64 (long) tensors, which consume more memory.

# CAPACITY PADDING (2026-10-13). The flat pair arrays are allocated with spare capacity so a rebuild can keep its shape
# (and, later, its buffers) and no consumer may rely on their length:
#   * `paddedI` / `paddedJ` have length >= `numEdges`; entries [numEdges:] are -1 in BOTH arrays (never a valid particle,
#     and NOT a dummy particle: nothing about the padding is visible in particle data or exported traces).
#   * the real entries are the velocity-Verlet SUPERSET of the interacting pairs, not the active pairs: a raw-pair
#     consumer must still test `dist < h` itself.
#   * a torch gather with index -1 silently wraps to the last element: slice with `livePairs()` or mask before indexing;
#     a warp kernel launched over the padded length must start with `if edgeI[tid] < 0: return`.
#   * the CSR view (`edgeOffsets` / `numNeighbors`, rows addressing `paddedJ`) never touches the padding.
# The members were renamed from `i` / `j` on purpose: every old `adjacency.i` / `adjacency.j` use now fails loudly and has to
# be revisited. Raw-pair interactions stay a development convenience; per-particle CSR kernels are the target.
@dataclass(frozen=True)
class CapacityPolicy:
    """How much spare room the flat pair arrays of a rebuilt `AdjacencyList` get.

    headroom     fractional extra capacity over the real edge count (0.1 = 10 %).
    shrinkBelow  with a prior capacity: keep that capacity (stable shapes) while the real count is within
                 [shrinkBelow * priorCapacity, priorCapacity]; reallocate when it overflows or falls below.
    inPlace      a rebuild that keeps the capacity also keeps the BUFFERS: it writes `paddedI` / `paddedJ` /
                 `numNeighbors` / `edgeOffsets` of the list it replaces (same shapes, same device addresses, so a
                 captured CUDA graph that reads them stays valid) and returns a new `AdjacencyList` object that
                 shares them. Off by default: the list that was replaced is then overwritten, so nothing may keep
                 it (or tensors taken from it) for later; opt in per run (a persistent graph), never for one-off
                 callers. A rebuild that outgrows or underfills the capacity allocates anew (the graph is recaptured).
    """
    headroom: float = 0.1
    shrinkBelow: float = 0.5
    inPlace: bool = False

    def capacityFor(self, numEdges: int, priorCapacity: Optional[int] = None) -> int:
        if priorCapacity is not None and self.shrinkBelow * priorCapacity <= numEdges <= priorCapacity:
            return int(priorCapacity)
        return int(numEdges + max(1, int(numEdges * self.headroom)))


DEFAULT_CAPACITY = CapacityPolicy()


@dataclass(slots=True)
class AdjacencyList:
    paddedI: torch.Tensor
    paddedJ: torch.Tensor
    numNeighbors: torch.Tensor
    edgeOffsets: torch.Tensor
    numRows: int
    numCols: int

    # Required for velocity verlet list, which stores not just the required neighbors but also some extra neighbors to avoid rebuilding the neighbor list every step. These positions are the original positions with which the list was built, NOT the current positions. These are used to determine when the list needs to be rebuilt, by checking if any particle has moved more than a certain threshold distance from its original position.
    queryPositions: torch.Tensor = None
    referencePositions: torch.Tensor = None
    querySupports: torch.Tensor = None
    referenceSupports: torch.Tensor = None

    hashMap: CompactHashMap = None

    # Number of real entries at the front of paddedI / paddedJ (the rest is -1 padding). Host int, known at build time.
    numEdges: Optional[int] = None

    def livePairs(self):
        """(i, j) views of the real entries only (the padding sliced off): the exact arrays the unpadded list used to
        have. Costs no sync and no copy; the shapes follow the rebuild, so this is for eager consumers only."""
        n = self.paddedI.shape[0] if self.numEdges is None else self.numEdges
        return self.paddedI[:n], self.paddedJ[:n]

@dataclass(slots=True)
class AdjacencyListWarp:
    paddedI: wp.array(dtype=wp.int64)
    paddedJ: wp.array(dtype=wp.int64)
    numNeighbors: wp.array(dtype=wp.int64)
    edgeOffsets: wp.array(dtype=wp.int64)



@wp.struct
class adjacencyData:
    neighborList: wp.array(dtype = wp.int64) # type: ignore
    neighborOffsets: wp.array(dtype = wp.int32) # type: ignore
    numNeighbors: wp.array(dtype = wp.int32) # type: ignore

@wp.struct
class gridData:
    sortIndex: wp.array(dtype = wp.int64) # type: ignore
    qMin: wp.array(dtype = scalar_t) # type: ignore
    qMax: wp.array(dtype = scalar_t) # type: ignore
    hCell: scalar_t
    numCells: wp.array(dtype = wp.int32) # type: ignore
    hashTable: wp.array(dtype = vector(length = 2, dtype = wp.int32)) # type: ignore
    cellTable: wp.array(dtype = vector(length = 3, dtype = wp.int64)) # type: ignore
    D: int
    numOffsets: int
    cellOffsets: wp.array(dtype = vector(length=3, dtype = wp.int32)) # type: ignore
    