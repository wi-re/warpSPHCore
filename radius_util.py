from dataclasses import dataclass
import torch
import warp as wp

# A dataclass to hold the adjacency list information, including the indices of the neighbors, the number of neighbors for each query point, and the edge offsets for efficient access.
# This can be used as a coordinate formate (COO) for the adjacency list, where i and j represent the row and column indices of the neighbors
# This can also be used as a compressed sparse row (CSR) format, where edgeOffsets can be used to quickly access the neighbors of each query point, and numNeighbors can be used to know how many neighbors each query point has.
# Notably this cannot be used as a compressed sparse column (CSC) format, since the neighbors are not sorted by the reference points, but rather by the query points.
# Because of the sorting we can reconstruct i from edgeOffsets and numNeighbors, but we keep it for convenience and to avoid having to reconstruct it every time.

# One unfortunate aspect is that the torch tensors need to be of dtype long to allow indexing within torch. warp could naturally handle int dtypes, consuming less memory, but torch does not allow indexing with int32 tensors, so we need to convert them to int64 (long) tensors, which consume more memory.
@dataclass(slots=True)
class AdjacencyList:
    i: torch.Tensor
    j: torch.Tensor
    numNeighbors: torch.Tensor
    edgeOffsets: torch.Tensor
    numRows: int
    numCols: int
    
@dataclass(slots=True)
class AdjacencyListWarp:
    i: wp.array(dtype=wp.int64)
    j: wp.array(dtype=wp.int64)
    numNeighbors: wp.array(dtype=wp.int64)
    edgeOffsets: wp.array(dtype=wp.int64)