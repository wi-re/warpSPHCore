from .hashMap_t import CompactHashMap
from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union
from ..types import *

# A dataclass to hold the adjacency list information, including the indices of the neighbors, the number of neighbors for each query point, and the edge offsets for efficient access.
# This can be used as a coordinate formate (COO) for the adjacency list, where i and j represent the row and column indices of the neighbors
# This can also be used as a compressed sparse row (CSR) format, where edgeOffsets can be used to quickly access the neighbors of each query point, and numNeighbors can be used to know how many neighbors each query point has.
# Notably this cannot be used as a compressed sparse column (CSC) format, since the neighbors are not sorted by the reference points, but rather by the query points.
# Because of the sorting we can reconstruct i from edgeOffsets and numNeighbors, but we keep it for convenience and to avoid having to reconstruct it every time.

# One unfortunate aspect is that the torch tensors need to be of dtype long to allow indexing within torch. warp could naturally handle int dtypes, consuming less memory, but torch does not allow indexing with int32 tensors, so we need to convert them to int64 (long) tensors, which consume more memory.
# @torch.jit.script
@dataclass(slots=True)
class AdjacencyList:
    i: torch.Tensor
    j: torch.Tensor
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

@dataclass(slots=True)
class AdjacencyListWarp:
    i: wp.array(dtype=wp.int64)
    j: wp.array(dtype=wp.int64)
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
    