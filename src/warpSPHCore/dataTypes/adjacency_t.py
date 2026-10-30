from .._models import CapacityPolicy, AdjacencyList, DEFAULT_CAPACITY
from .._backend import BACKEND

if BACKEND == "warp":
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

    # Particle-ID and CSR-offset widths are configured independently before import
    # (type_config.index_t / offset_t); per-row counts stay int32. Torch development
    # consumers may convert live pairs to long where required by their operation.

    # CAPACITY PADDING (2026-10-13). The flat pair arrays are allocated with spare capacity so a rebuild can keep its shape
    # (and, later, its buffers) and no consumer may rely on their length:
    #   * With storeI=False, paddedI is None; CSR remains complete. cooRows()/livePairs() explicitly reconstruct rows without caching.
    #   * Stored `paddedI` / `paddedJ` have length >= `numEdges`; entries [numEdges:] are -1 in BOTH arrays (never a valid particle,
    #     and NOT a dummy particle: nothing about the padding is visible in particle data or exported traces).
    #   * the real entries are the velocity-Verlet SUPERSET of the interacting pairs, not the active pairs: a raw-pair
    #     consumer must still test `dist < h` itself.
    #   * a torch gather with index -1 silently wraps to the last element: slice with `livePairs()` or mask before indexing;
    #     a warp kernel launched over the padded length must start with `if edgeI[tid] < 0: return`.
    #   * the CSR view (`edgeOffsets` / `numNeighbors`, rows addressing `paddedJ`) never touches the padding.
    # The members were renamed from `i` / `j` on purpose: every old `adjacency.i` / `adjacency.j` use now fails loudly and has to
    # be revisited. Raw-pair interactions stay a development convenience; per-particle CSR kernels are the target.

    @dataclass(slots=True)
    class AdjacencyListWarp:
        paddedI: wp.array(dtype=index_t)
        paddedJ: wp.array(dtype=index_t)
        numNeighbors: wp.array(dtype=neighbor_count_t)
        edgeOffsets: wp.array(dtype=offset_t)



    @wp.struct
    class adjacencyData:
        neighborList: wp.array(dtype = index_t) # type: ignore
        neighborOffsets: wp.array(dtype = offset_t) # type: ignore
        numNeighbors: wp.array(dtype = wp.int32) # type: ignore

    @wp.struct
    class gridData:
        sortIndex: wp.array(dtype = index_t) # type: ignore
        qMin: wp.array(dtype = scalar_t) # type: ignore
        qMax: wp.array(dtype = scalar_t) # type: ignore
        hCell: scalar_t
        numCells: wp.array(dtype = wp.int32) # type: ignore
        hashTable: wp.array(dtype = vector(length = 2, dtype = wp.int32)) # type: ignore
        cellTable: wp.array(dtype = vector(length = 3, dtype = wp.int64)) # type: ignore
        D: int
        numOffsets: int
        cellOffsets: wp.array(dtype = vector(length=3, dtype = wp.int32)) # type: ignore

