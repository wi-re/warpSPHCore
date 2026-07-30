from dataclasses import dataclass
import torch
import warp as wp

# A dataclass to hold the adjacency list information, including the indices of the neighbors, the number of neighbors for each query point, and the edge offsets for efficient access.
# This can be used as a coordinate formate (COO) for the adjacency list, where i and j represent the row and column indices of the neighbors
# This can also be used as a compressed sparse row (CSR) format, where edgeOffsets can be used to quickly access the neighbors of each query point, and numNeighbors can be used to know how many neighbors each query point has.
# Notably this cannot be used as a compressed sparse column (CSC) format, since the neighbors are not sorted by the reference points, but rather by the query points.
# Because of the sorting we can reconstruct i from edgeOffsets and numNeighbors, but we keep it for convenience and to avoid having to reconstruct it every time.


from ..type_config import *

@torch.jit.script
@dataclass
class CompactHashMap:
    sortedPositions: torch.Tensor
    sortedSupports: torch.Tensor
    sortIndex: torch.Tensor

    hashTable: torch.Tensor
    sortedCellTable: torch.Tensor

    qMin: torch.Tensor
    qMax: torch.Tensor
    hCell: float
    numCells: torch.Tensor
    mode_uint: int
    D: int
    searchRadius: int
    numOffsets: int
    cellOffsets: torch.Tensor

