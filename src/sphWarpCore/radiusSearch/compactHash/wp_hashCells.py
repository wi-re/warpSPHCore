import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *

def hashGridIndicesTorch(cellGridIndices: torch.Tensor, hashMapLength: int) -> torch.Tensor:
    """CPU/PyTorch reference hash used to validate Warp hash assignment."""
    D = cellGridIndices.shape[1]
    if D == 1:
        return torch.remainder(cellGridIndices[:, 0], hashMapLength).to(torch.int32)

    primes = torch.tensor([73856093, 19349663, 83492791], device=cellGridIndices.device, dtype=torch.int64)
    hashValue = torch.zeros(cellGridIndices.shape[0], device=cellGridIndices.device, dtype=torch.int64)
    for d in range(D):
        hashValue += cellGridIndices[:, d].to(torch.int64) * primes[d]
    return torch.remainder(hashValue, hashMapLength).to(torch.int32)
       
@wp.func
def hashGridIndex(cellIndex: wp.array(dtype=wp.int32), hashMapLength: wp.uint32) -> wp.uint32:
    if cellIndex.shape[0] == 1:
        return wp.uint32(cellIndex[0]) % hashMapLength
    else:
        primes = wp.vec3i(73856093, 19349663, 83492791)
        hashValue = wp.uint32(0)
        for d in range(cellIndex.shape[0]):
            hashValue += wp.uint32(cellIndex[d] * primes[d])
        return wp.uint32(hashValue) % hashMapLength
    
@wp.func
def hashGridVec3i(cellIndex: wp.vec3i, hashMapLength: wp.uint32, D: int) -> wp.uint32:
    if D == 1:
        return wp.uint32(cellIndex.x) % hashMapLength
    else:
        primes = wp.vec3i(73856093, 19349663, 83492791)
        hashValue = wp.uint32(0)
        hashValue += wp.uint32(cellIndex.x * primes.x)
        if D > 1:
            hashValue += wp.uint32(cellIndex.y * primes.y)
        if D > 2:
            hashValue += wp.uint32(cellIndex.z * primes.z)
        return wp.uint32(hashValue) % hashMapLength
    
@wp.kernel
def hashCells(
    cellGridIndices: wp.array2d(dtype=wp.int32),  # shape [C,D]
    hashMapLength: wp.uint32,
    hashValues: wp.array1d(dtype=wp.uint32)  # shape [C]
):
# template<std::size_t dim = 2>
# hostDeviceInline constexpr auto hashIndexing(std::array<int32_t, dim> cellIndices, uint32_t hashMapLength) {
#     // auto dim = cellIndices.size(0);
#     using unsignedType = uint32_t;
#     if constexpr (dim == 1) {
#         return ((unsignedType) cellIndices[0]) % (unsignedType) hashMapLength;
#     }else{
#         constexpr auto primes = std::array<unsignedType, 3>{73856093u, 19349663u, 83492791u};
#         unsignedType hash = 0;
#         for(int32_t i = 0; i < (int32_t) dim; i++){
#             hash += ((unsignedType) cellIndices[i]) * primes[i];
#         }
#         return (int32_t) (hash % (unsignedType) hashMapLength);
#     }
# }
    i = wp.tid()
    cellIndex = cellGridIndices[i]
    hashValue = hashGridIndex(cellIndex, hashMapLength)
    hashValues[i] = hashValue
    