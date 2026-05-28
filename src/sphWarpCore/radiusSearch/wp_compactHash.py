import warp as wp
import torch 
from ..types import *
from typing import Optional
from ..mathutil.wp_math import *
from ..utils.wp_util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from .radius_util import *
from ..enumTypes import *

def computeGridSupport(supportsX, supportsY, scheme: SupportScheme):
    # print(f'SupportsX: {supportsX}')
    # print(f'SupportsY: {supportsY}')
    if scheme == SupportScheme.Gather:  # gather
        return torch.max(supportsX)
    elif scheme == SupportScheme.Scatter:  # scatter
        return torch.max(supportsY)
    elif scheme == SupportScheme.MeanSymmetric: # partialSymmetric
        return torch.max((supportsX + supportsY) / 2)
    elif scheme == SupportScheme.KernelMeanSymmetric:  # symmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    elif scheme == SupportScheme.SuperSymmetric:  # superSymmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    elif scheme == SupportScheme.PartialSymmetric:  # partialSymmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    else:
        raise ValueError('Invalid scheme value. Must be a valid SupportScheme. Value is {}'.format(scheme))


def getDomainExtents(y, minDomain: Optional[torch.Tensor], maxDomain: Optional[torch.Tensor]):
    if minDomain is not None and maxDomain is not None:
        return minDomain, maxDomain
    elif minDomain is not None:
        maxD = torch.max(y, dim=0)[0]
        return minDomain, maxD
    elif maxDomain is not None:
        minD = torch.min(y, dim=0)[0]
        return minD, maxDomain
    else:
        minD = torch.min(y, dim=0)[0]
        maxD = torch.max(y, dim=0)[0]
        return minD, maxD



# @torch.jit.script
def compute_h(qMin, qMax, referenceSupport): 
    """
    Compute the smoothing length (h) based on the given minimum and maximum coordinates (qMin and qMax)
    and the reference support value. The smoothing length is used for grid operations and is determined
    by dividing the domain into cells based on the reference support value such that h > referenceSupport.

    Args:
        qMin (torch.Tensor): The minimum coordinates.
        qMax (torch.Tensor): The maximum coordinates.
        referenceSupport (scalar_t): The reference support value.

    Returns:
        torch.Tensor: The computed smoothing length (h).
    """
    qExtent = qMax - qMin
    qCells = qExtent / referenceSupport
    qfCells = torch.floor(qCells)
    # numCells = torch.where( qCells - qfCells < 1e-4, qfCells, qfCells+1)
    numCells = qfCells
    h = qExtent / (numCells)
    # print('Reference support:', referenceSupport)
    # print('Domain extent:', qExtent)

    # print('Reference Num Cells: ', qCells)
    # print('Computed Num Cells: ', numCells)

    # print('Reverse Count: ', qExtent / h - numCells)

    # print('Number of cells:', numCells)
    # print('Smoothing length:', h)
    # print('Resulting Cells: ', torch.floor(qExtent / h))

    if torch.any(qExtent / h - numCells > 0):
        # print('Warning: Reference support is not a multiple of the domain extent. Consider changing the reference support value.')
        numCells -= 1
        h = qExtent / numCells
        # print('New Num Cells: ', numCells)
        # print('New Smoothing length: ', h)


    if torch.any(torch.ceil(qExtent / h) > qExtent / h):
        h = h * (1e-4 + 1)

    # print(qfCells, qCells - qfCells)

    # print('Difference of support: ', torch.abs(h - referenceSupport))

    # if torch.any(torch.floor(qExtent / h) != torch.ceil(qExtent / h)):
    #     print(torch.floor(qExtent / h), torch.ceil(qExtent / h))
    #     print('Warning: Reference support is not a multiple of the domain extent. Consider changing the reference support value.')

    return torch.max(h)


@wp.func
def computeZOrderIndex(index: wp.vec3i) -> wp.int32:
    # Morton encoding (Z-order curve) for 3D indices
    x = wp.uint32(index.x)
    y = wp.uint32(index.y)
    z = wp.uint32(index.z)

    # Interleave bits of x, y, and z
    def splitBy3Bits(n):
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n
    
    return wp.vec3i(splitBy3Bits(x) | (splitBy3Bits(y) << 1) | (splitBy3Bits(z) << 2))

@wp.func
def mortonPattern(index: wp.int32) -> wp.uint64:
    # This function addresses a fundamental limitaton in warp. We cannot have 64 bit literal values as python does not support the ull suffix. But, we do know the patterns. We can generate the correct ones by building them as as 32 bit values and shifting them left by 32 bits. This allows us to generate the correct patterns for the splitBy3Bits function, which is the basis for the Morton encoding.
    # However
    # this also would require unsigned literals which python does not support so we need to assemble these values from 16 bit components instead
    bits_00_15 = wp.uint64(0)
    bits_16_31 = wp.uint64(0)
    bits_32_47 = wp.uint64(0)
    bits_48_63 = wp.uint64(0)
    
    if index == 0:
        # generate 0x1fffff:
        # lower = wp.uint64(0x1fffff)
        # upper = wp.uint64(0) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xffff)
        bits_16_31 = wp.uint64(0x1fff)
        bits_32_47 = wp.uint64(0)
        bits_48_63 = wp.uint64(0)
    elif index == 1:
        # generate 0x1f0000 0000ffff:
        # lower = wp.uint64(0x0000ffff)
        # upper = wp.uint64(0x001f0000) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xffff)
        bits_16_31 = wp.uint64(0x0000)
        bits_32_47 = wp.uint64(0x0000)
        bits_48_63 = wp.uint64(0x001f)
    elif index == 2:
        # generate 0x1f0000ff 0000ff:
        # lower = wp.uint64(0xff0000ff)
        # upper = wp.uint64(0x001f0000) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x00ff)
        bits_16_31 = wp.uint64(0xff00)
        bits_32_47 = wp.uint64(0x0000)
        bits_48_63 = wp.uint64(0x001f)
    elif index == 3:
        # generate 0x100f00f0 0f00f00f:
        # lower = wp.uint64(0x0f00f00f)
        # upper = wp.uint64(0x100f00f0) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xf00f)
        bits_16_31 = wp.uint64(0x0f00)
        bits_32_47 = wp.uint64(0x00f0)
        bits_48_63 = wp.uint64(0x100f)
    elif index == 4:
        # genrate 0x10c30c30 c30c30c3
        # lower = wp.uint64(0xc30c30c3)
        # upper = wp.uint64(0x10c30c30) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x30c3)
        bits_16_31 = wp.uint64(0xc30c)
        bits_32_47 = wp.uint64(0x0c30)
        bits_48_63 = wp.uint64(0x10c3)
    elif index == 5:
        # generate 0x12492492 49249249
        # lower = wp.uint64(0x49249249)
        # upper = wp.uint64(0x12492492) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x9249)
        bits_16_31 = wp.uint64(0x4924)
        bits_32_47 = wp.uint64(0x2492)
        bits_48_63 = wp.uint64(0x1249)

    return bits_00_15 | (bits_16_31 << wp.uint64(16)) | (bits_32_47 << wp.uint64(32)) | (bits_48_63 << wp.uint64(48))
    

@wp.func
def splitBy3Bits64(n: wp.uint64) -> wp.uint64:
    n = (n | (n << wp.uint64(32))) & mortonPattern(1)
    n = (n | (n << wp.uint64(16))) & mortonPattern(2)
    n = (n | (n << wp.uint64(8))) & mortonPattern(3)
    n = (n | (n << wp.uint64(4))) & mortonPattern(4)
    n = (n | (n << wp.uint64(2))) & mortonPattern(5)
    return n

@wp.func 
def computeZOrderIndex64(index: wp.vec3i) -> wp.int64:
    # Morton encoding (Z-order curve) for 3D indices into a 64-bit integer
    x = wp.uint64(index.x)
    y = wp.uint64(index.y)
    z = wp.uint64(index.z)
    
    return wp.int64(splitBy3Bits64(x) | (splitBy3Bits64(y) << wp.uint64(1)) | (splitBy3Bits64(z) << wp.uint64(2)))
    
    
@wp.func
def getLinearIndex64(
    cellIndex: wp.vec3i,
    cellCounts: wp.array(dtype=wp.int32),  # shape [D]
    D: int,
) -> wp.int64:
    # x-major (axis 0 first): x + Nx*y + Nx*Ny*z
    linearIndex = wp.int64(0)
    product = wp.int64(1)
    for d in range(D):
        linearIndex += wp.int64(cellIndex[d]) * product
        product = product * wp.int64(cellCounts[d])
    return linearIndex


@wp.func
def clampCellIndex(cellIndex: wp.vec3i, numCells: wp.array(dtype=wp.int32), D: int) -> wp.vec3i:
    for d in range(D):
        if cellIndex[d] < 0:
            cellIndex[d] = 0
        elif cellIndex[d] >= numCells[d]:
            cellIndex[d] = numCells[d] - 1
    return cellIndex


@wp.kernel
def indexCells(
    cellIndices: wp.array2d(dtype=wp.int32),
    cellCounts: wp.array(dtype=wp.int32),
    cellIndxes: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    numCells = cellIndices.shape[0]
    dim = cellIndices.shape[1]
    
    if i >= numCells:
        return
    
    cellIndex = wp.vec3i(0)
    for d in range(dim):
        cellIndex[d] = cellIndices[i, d]
        
    cellIndxes[i] = getLinearIndex64(cellIndex, cellCounts, dim)
    

@torch.jit.script
def linearIndexing(cellIndices, cellCounts):
    """
    Compute the linear index based on the given cell indices and cell counts.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        cellCounts (torch.Tensor): Tensor containing the cell counts.

    Returns:
        torch.Tensor: Tensor containing the linear indices.
    """
    dim = cellIndices.shape[1]
    linearIndex = torch.zeros(cellIndices.shape[0], dtype=cellIndices.dtype, device=cellIndices.device)
    product = 1
    for i in range(dim):
        linearIndex += cellIndices[:, i] * product
        product = product * cellCounts[i].item()
    return linearIndex

def sortReferenceParticles(referenceParticles, referenceSupport, domainMin, domainMax):
    """
    Sorts the reference particles based on their linear indices.

    Args:
        referenceParticles (torch.Tensor): The reference particles to be sorted.
        referenceSupport (scalar_t): The reference support value.
        domainMin: The minimum value of the domain.
        domainMax: The maximum value of the domain.

    Returns:
        sortedLinearIndices (torch.Tensor): The sorted linear indices of the reference particles.
        sortingIndices (torch.Tensor): The indices that sort the linear indices.
        cellCount (torch.Tensor): The number of cells in each dimension.
        domainMin: The minimum value of the domain.
        domainMax: The maximum value of the domain.
        hCell (scalar_t): The computed h value for the cells.
    """
    # with record_function("neighborSearch - sortReferenceParticles"): 
    # with record_function("neighborSearch - sortReferenceParticles[index Calculation]"): 
    hCell = compute_h(domainMin, domainMax, referenceSupport)
    qExtent = domainMax - domainMin
    # print('Domain Min:', domainMin, 'dtype:', domainMin.dtype)
    # print('Domain Max:', domainMax, 'dtype:', domainMax.dtype)
    # print('Reference Support:', referenceSupport, 'dtype:', referenceSupport.dtype)
    # print('Computed hCell:', hCell, 'dtype:', hCell.dtype)

    cellCount = torch.ceil(qExtent / (hCell)).to(torch.int32)
    indices = torch.floor((referenceParticles - domainMin) / hCell).to(torch.int32).view(-1, referenceParticles.shape[1])
    
    
    # print('Cell count:', cellCount, 'Cell size:', hCell, 'Domain extent:', qExtent)
    # print('indices:', indices.contiguous())
    warp_indices = castTorchToWarp(indices)
    
    out = wp.zeros((indices.shape[0],), dtype=wp.int64, device=warp_indices.device)
    warp_cell_count = castTorchToWarp(cellCount)
    wp.launch(
        indexCells,
        dim=indices.shape[0],
        inputs=[warp_indices, warp_cell_count, out],
        device=warp_indices.device,
    )
    linearIndices = wp.to_torch(out)
    # linearIndices = linearIndexing(indices, cellCount)
    # with record_function("neighborSearch - sortReferenceParticles[argsort]"): 
    sortingIndices = torch.argsort(linearIndices)
    # with record_function("neighborSearch - sortReferenceParticles[resort]"): 
    sortedLinearIndices = linearIndices[sortingIndices]
    return sortedLinearIndices, sortingIndices, \
            cellCount, domainMin, domainMax, scalar_t(hCell)
            
            
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
    
    
    
@wp.func 
def getLinearIndex(
    cellIndex: wp.vec3i,
    cellCounts: wp.array(dtype=wp.int32),  # shape [D]
    D: int
):
    linearIndex = wp.int32(0)
    product = wp.int32(1)
    for d in range(D):
        linearIndex += cellIndex[d] * product
        product = product * cellCounts[d]
    return linearIndex


@wp.kernel
def radiusSearchCountNeighborsCompactHashMap(
    queryPositions: wp.array2d(dtype=scalar_t),  # shape [N,D]
    querySupports: wp.array1d(dtype=scalar_t),  # shape [N]
    sortedPositions: wp.array2d(dtype=scalar_t),  # shape [M,D]
    sortedSupports: wp.array1d(dtype=scalar_t),  # shape [M]
    hashTable: wp.array2d(dtype=wp.int32),  # shape [hashMapLength,2]
    cellTable: wp.array2d(dtype=wp.int64),  # shape [C,3] with [cellIndex, cellStart, cellCount]
    qMin: wp.array1d(dtype=scalar_t),  # shape [D]
    hCell: scalar_t,
    D: int,
    cellOffsets: wp.array2d(dtype=wp.int32),  # shape [numCellOffsets, D]
    numCells: wp.array(dtype=wp.int32),  # shape [D]
    maxDomain: wp.array(dtype=scalar_t),  # shape [D]
    minDomain: wp.array(dtype=scalar_t),  # shape [D]
    periodicity: wp.array(dtype=wp.bool),  # shape [D]
    mode_uint: wp.uint32,  # 0 for gather, 1 for scatter, 2 for symmetric, 3 for superSymmetric
    
    # Outputs
    neighborCounts: wp.array1d(dtype=wp.int32)  # shape [N] 
    
#  * @param queryPositions_ The positions of the query particles.
#  * @param querySupport_ The support radii of the query particles.
#  * @param searchRange The search range.
#  * @param sortedPositions_ The sorted positions of the particles.
#  * @param sortedSupport_ The sorted support radii of the particles.
#  * @param hashTable_ The hash table.
#  * @param hashMapLength The length of the hash map.
#  * @param numCells_ The number of cells.
#  * @param cellTable_ The cell table.
#  * @param qMin_ The minimum domain bounds.
#  * @param hCell The cell size.
#  * @param maxDomain_ The maximum domain bounds.
#  * @param minDomain_ The minimum domain bounds.
#  * @param periodicity_ The periodicity flags.
#  * @param mode The support mode.
):
    # pass
    N = queryPositions.shape[0]
    # D = queryPositions.shape[1]
    M = sortedPositions.shape[0]
    i = wp.tid()
    hashMapLength = wp.uint32(hashTable.shape[0])
    if i >= N:
        return
    
    queryPos = queryPositions[i]
    querySupport = querySupports[i]
    
    # Determine the cell index of the query particle
    cellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
    for d in range(D):
        cellIndex[d] = wp.int32(wp.floor((queryPos[d] - qMin[d]) / hCell))
    cellIndex = clampCellIndex(cellIndex, numCells, D)
    # Compute the hash value for the cell index
    # hashValue = hashGridIndex(cellIndex, hashMapLength)
    numOffsets = cellOffsets.shape[0]
    # currentLinearIndex = getLinearIndex(cellIndex, numCells, D)
    count = wp.int32(0)
    
    # wp.printf("Thread %d: Query position (%f, %f) with support %f is in cell index (%d, %d, %d)\n", i, queryPos[0], queryPos[1], querySupport, cellIndex[0], cellIndex[1], cellIndex[2])
    for o in range(numOffsets):
        offset = cellOffsets[o]
        # wp.printf("\tThread %d: Checking offset (%d, %d, %d)\n", i, offset[0], offset[1], offset[2])
        
        currentCellIndex = wp.vec3i(0,0,0, dtype=wp.int32)
        for d in range(D):
            currentCellIndex[d] = cellIndex[d] + offset[d]
        # Handle periodic boundaries
        for d in range(D):
            if periodicity[d]:
                if currentCellIndex[d] < 0:
                    currentCellIndex[d] += numCells[d]
                elif currentCellIndex[d] >= numCells[d]:
                    currentCellIndex[d] -= numCells[d]
                    
        # linearIndex = getLinearIndex(currentCellIndex, numCells, D)
        linearIndex = getLinearIndex64(currentCellIndex, numCells, D)
        hashValue = wp.int32(hashGridVec3i(currentCellIndex, hashMapLength, D))
        # wp.printf("\tThread %d: Checking cell index (%d, %d, %d) with hash value %d\n", i, currentCellIndex[0], currentCellIndex[1], currentCellIndex[2], hashValue)
        hashEntry = hashTable[hashValue]
        
        # wp.printf("\tThread %d: Hash entry for hash value %d is (start: %d, count: %d)\n", i, hashValue, hashEntry[0], hashEntry[1])
        
        if hashEntry[1] == 0:
            continue  # No particles in this cell
        cellStart = hashEntry[0]
        cellCount = hashEntry[1]
        
        for c in range(cellCount):
            candidateIndex = cellTable[cellStart + c, 0]
            cellStartIndex = wp.int32(cellTable[cellStart + c, 1])
            cellParticleCount = wp.int32(cellTable[cellStart + c, 2])
            
            if linearIndex != candidateIndex:
                # wp.printf("\t\tThread %d: Skipping cell with linear index %d as it does not match the current cell index %d\n", i, candidateIndex, linearIndex)
                continue  # This cell does not match the current cell index
            
            # wp.printf("\t\tThread %d: Checking cell with linear index %d containing %d particles\n", i, candidateIndex, cellParticleCount)
            
            for p in range(wp.int32(cellParticleCount)):
                neighborIndex = cellStartIndex + p
                neighborPos = sortedPositions[neighborIndex]
                neighborSupport = sortedSupports[neighborIndex] if sortedSupports.shape[0] > 0 else scalar_t(0.0)
                
                dist = computeCartesianDistance(queryPos, neighborPos, minDomain, maxDomain, periodicity)
                
                # Determine threshold based on mode
                threshold = scalar_t(0.0)
                if mode_uint == wp.static(SupportScheme.Gather.value):  # gather
                    threshold = querySupport
                elif mode_uint == wp.static(SupportScheme.Scatter.value):  # scatter
                    threshold = neighborSupport
                elif mode_uint == wp.static(SupportScheme.MeanSymmetric.value):  # meanSymmetric
                    threshold = (querySupport + neighborSupport) / scalar_t(2.0)
                elif mode_uint == wp.static(SupportScheme.KernelMeanSymmetric.value):  # kernelMeanSymmetric
                    threshold = max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.SuperSymmetric.value):  # superSymmetric
                    threshold = wp.max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.PartialSymmetric.value):  # partialSymmetric
                    threshold = wp.max(querySupport, neighborSupport)
                
                # Count valid neighbors
                if dist <= threshold:
                    count += 1
    
                
            #     # Compute distance considering periodic boundaries
            #     distSquared = 0.0
            #     for d in range(D):
            #         delta = queryPos[d] - neighborPos[d]
            #         if periodicity[d]:
            #             domainSize = maxDomain[d] - minDomain[d]
            #             if abs(delta) > domainSize / 2:
            #                 delta -= wp.sign(delta) * domainSize
            #         distSquared += delta * delta
                
            #     radius = querySupport + neighborSupport if mode_uint == 2 else (querySupport if mode_uint == 0 else neighborSupport)
                
            #     if distSquared <= radius * radius:
            #         neighborCounts[i] += 1
    neighborCounts[i] = count
    
    
@wp.kernel
def radiusSearchCollectCompactHashMap(
    queryPositions: wp.array2d(dtype=scalar_t),  # shape [N,D]
    querySupports: wp.array1d(dtype=scalar_t),  # shape [N]
    sortedPositions: wp.array2d(dtype=scalar_t),  # shape [M,D]
    sortedSupports: wp.array1d(dtype=scalar_t),  # shape [M]
    hashTable: wp.array2d(dtype=wp.int32),  # shape [hashMapLength,2]
    cellTable: wp.array2d(dtype=wp.int64),  # shape [C,3] with [cellIndex, cellStart, cellCount]
    qMin: wp.array1d(dtype=scalar_t),  # shape [D]
    hCell: scalar_t,
    D: int,
    cellOffsets: wp.array2d(dtype=wp.int32),  # shape [numCellOffsets, D]
    numCells: wp.array(dtype=wp.int32),  # shape [D]
    maxDomain: wp.array(dtype=scalar_t),  # shape [D]
    minDomain: wp.array(dtype=scalar_t),  # shape [D]
    periodicity: wp.array(dtype=wp.bool),  # shape [D]
    mode_uint: wp.uint32,  # 0 for gather, 1 for scatter, 2 for symmetric, 3 for superSymmetric
    
    neighborCounts: wp.array1d(dtype=wp.int32),  # shape [N] 
    edge_offsets: wp.array(dtype=wp.int32), # Cumulative edge counts (N,)
    sortIndex: wp.array(dtype=wp.int64), # shape [M]
    # Outputs
    edge_i: wp.array1d(dtype=wp.int64),  # shape [total_edges]
    edge_j: wp.array1d(dtype=wp.int64)   # shape [total_edges]

#  * @param queryPositions_ The positions of the query particles.
#  * @param querySupport_ The support radii of the query particles.
#  * @param searchRange The search range.
#  * @param sortedPositions_ The sorted positions of the particles.
#  * @param sortedSupport_ The sorted support radii of the particles.
#  * @param hashTable_ The hash table.
#  * @param hashMapLength The length of the hash map.
#  * @param numCells_ The number of cells.
#  * @param cellTable_ The cell table.
#  * @param qMin_ The minimum domain bounds.
#  * @param hCell The cell size.
#  * @param maxDomain_ The maximum domain bounds.
#  * @param minDomain_ The minimum domain bounds.
#  * @param periodicity_ The periodicity flags.
#  * @param mode The support mode.
):
    # pass
    N = queryPositions.shape[0]
    # D = queryPositions.shape[1]
    M = sortedPositions.shape[0]
    i = wp.tid()
    hashMapLength = wp.uint32(hashTable.shape[0])
    if i >= N:
        return
    
    queryPos = queryPositions[i]
    querySupport = querySupports[i]
    edge_offset = edge_offsets[i]
    edge_index = edge_offset
    
    # Determine the cell index of the query particle
    cellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
    for d in range(D):
        cellIndex[d] = wp.int32(wp.floor((queryPos[d] - qMin[d]) / hCell))
    cellIndex = clampCellIndex(cellIndex, numCells, D)
    # Compute the hash value for the cell index
    # hashValue = hashGridIndex(cellIndex, hashMapLength)
    numOffsets = cellOffsets.shape[0]
    currentLinearIndex = getLinearIndex64(cellIndex, numCells, D)
    
    # getLinearIndex(cellIndex, numCells, D)
    count = wp.int32(0)
    
    # wp.printf("Thread %d: Query position (%f, %f) with support %f is in cell index (%d, %d, %d)\n", i, queryPos[0], queryPos[1], querySupport, cellIndex[0], cellIndex[1], cellIndex[2])
    for o in range(numOffsets):
        offset = cellOffsets[o]
        # wp.printf("\tThread %d: Checking offset (%d, %d, %d)\n", i, offset[0], offset[1], offset[2])
        
        currentCellIndex = wp.vec3i(0,0,0, dtype=wp.int32)
        for d in range(D):
            currentCellIndex[d] = cellIndex[d] + offset[d]
        # Handle periodic boundaries
        for d in range(D):
            if periodicity[d]:
                if currentCellIndex[d] < 0:
                    currentCellIndex[d] += numCells[d]
                elif currentCellIndex[d] >= numCells[d]:
                    currentCellIndex[d] -= numCells[d]
                    
        # linearIndex = getLinearIndex(currentCellIndex, numCells, D)
        linearIndex = getLinearIndex64(currentCellIndex, numCells, D)
        hashValue = wp.int32(hashGridVec3i(currentCellIndex, hashMapLength, D))
        # wp.printf("\tThread %d: Checking cell index (%d, %d, %d) with hash value %d\n", i, currentCellIndex[0], currentCellIndex[1], currentCellIndex[2], hashValue)
        hashEntry = hashTable[hashValue]
        
        # wp.printf("\tThread %d: Hash entry for hash value %d is (start: %d, count: %d)\n", i, hashValue, hashEntry[0], hashEntry[1])
        
        if hashEntry[1] == 0:
            continue  # No particles in this cell
        cellStart = hashEntry[0]
        cellCount = hashEntry[1]
        
        for c in range(cellCount):
            candidateIndex = cellTable[cellStart + c, 0]
            cellStartIndex = wp.int32(cellTable[cellStart + c, 1])
            cellParticleCount = wp.int32(cellTable[cellStart + c, 2])
            
            if linearIndex != candidateIndex:
                # wp.printf("\t\tThread %d: Skipping cell with linear index %d as it does not match the current cell index %d\n", i, candidateIndex, linearIndex)
                continue  # This cell does not match the current cell index
            
            # wp.printf("\t\tThread %d: Checking cell with linear index %d containing %d particles\n", i, candidateIndex, cellParticleCount)
            
            for p in range(wp.int32(cellParticleCount)):
                neighborIndex = cellStartIndex + p
                neighborPos = sortedPositions[neighborIndex]
                neighborSupport = sortedSupports[neighborIndex] if sortedSupports.shape[0] > 0 else scalar_t(0.0)
                
                dist = computeCartesianDistance(queryPos, neighborPos, minDomain, maxDomain, periodicity)
                
                # Determine threshold based on mode
                threshold = scalar_t(0.0)
                if mode_uint == wp.static(SupportScheme.Gather.value):  # gather
                    threshold = querySupport
                elif mode_uint == wp.static(SupportScheme.Scatter.value):  # scatter
                    threshold = neighborSupport
                elif mode_uint == wp.static(SupportScheme.MeanSymmetric.value):  # meanSymmetric
                    threshold = (querySupport + neighborSupport) / scalar_t(2.0)
                elif mode_uint == wp.static(SupportScheme.KernelMeanSymmetric.value):  # kernelMeanSymmetric
                    threshold = max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.SuperSymmetric.value):  # superSymmetric
                    threshold = wp.max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.PartialSymmetric.value):  # partialSymmetric
                    threshold = wp.max(querySupport, neighborSupport)
                
                # Count valid neighbors
                if dist <= threshold:
                    edge_i[edge_index] = wp.int64(i)
                    edge_j[edge_index] = sortIndex[neighborIndex]
                    edge_index += 1
    
                
            #     # Compute distance considering periodic boundaries
            #     distSquared = 0.0
            #     for d in range(D):
            #         delta = queryPos[d] - neighborPos[d]
            #         if periodicity[d]:
            #             domainSize = maxDomain[d] - minDomain[d]
            #             if abs(delta) > domainSize / 2:
            #                 delta -= wp.sign(delta) * domainSize
            #         distSquared += delta * delta
                
            #     radius = querySupport + neighborSupport if mode_uint == 2 else (querySupport if mode_uint == 0 else neighborSupport)
                
            #     if distSquared <= radius * radius:
            #         neighborCounts[i] += 1
    # neighborCounts[i] = count
    
import numpy as np
    
from torch.profiler import record_function

# For some operations we want to be able to run the operations directly on the hash map, e.g., to do a particle to grid transfer. In this case we only access the neighborhood once per query set. In this case, it is not worth it to build the full neighbor list as the overhead of building the neighbor list is larger than the cost of just doing the search.
# For this purpose we want to be able to wrap the hash map information into a datastructure that allows us to do the search directly on the hash map without building the full neighbor list. This is what the CompactHashMap class is for. It contains all the information about the hash map and the cell table, as well as the sorted positions and supports, and allows us to do the search directly on this information without building the full neighbor list.
@dataclass
class CompactHashMap:
    sortedPositions: torch.Tensor
    sortedSupports: torch.Tensor
    sortIndex: torch.Tensor

    hashTable: torch.Tensor
    sortedCellTable: torch.Tensor

    qMin: torch.Tensor
    qMax: torch.Tensor
    hCell: scalar_t
    numCells: torch.Tensor
    mode_uint: int
    D: int
    searchRadius: int
    numOffsets: int
    cellOffsets: torch.Tensor


def buildCompactHashMap(
    queryPositions: torch.Tensor,
    referencePositions: torch.Tensor,
    querySupports: torch.Tensor,
    referenceSupports: torch.Tensor,
    periodicity: torch.Tensor,
    domainDescription: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLength: int = 4096
):
    with record_function("warpNeighborSearch - buildCompactHashMap"):
        with record_function("neighborSearch - preprocess"):
            mode_uint = supportSchemeToUint(mode)
            # mode_map = {'gather': 1, 'scatter': 2, 'symmetric': 3, 'superSymmetric': 4, ''}
            # mode_uint = mode_map.get(mode, 0)
            # if mode_uint == 0:
                # raise ValueError(f"Invalid mode: {mode}. Supported modes are: {list(mode_map.keys())}")
                
            minDomain = domainDescription.min if domainDescription.min is not None else None
            maxDomain = domainDescription.max if domainDescription.max is not None else None
            hMax = computeGridSupport(querySupports, referenceSupports, mode)
            minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)

        with record_function("neighborSearch - sortParticles"):
            x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
            y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT

            sortedLinear, sortIndex, numCells, qMin, qMax, hCell = sortReferenceParticles(x, hMax, minD, maxD)



            # Do the actual resort
            sortedPositions = x[sortIndex,:]
            # sortedSupports = xSupport[sortIndex]
        with record_function("neighborSearch - buildCells"):
            # compact teh list of occupied cells
            cellIndices, cellCounters = torch.unique_consecutive(sortedLinear, return_counts=True, return_inverse=False)
            cellCounters = cellCounters.to(torch.int32)
            # Needs to zero padded for the indexing to work properly as the 0th cell is valid and cumsum doesn't include the first element

            cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1].to(torch.int32)

            # We can now use the cumCell to index into the sortedIndices to get the cell index for each particle
            # We could have reversed the linear indices to get the cell index for each cell, but this is more reliable and avoids inverse computations
            sortedIndices = torch.floor((sortedPositions - qMin) / to_numpy(hCell)).to(torch.int32)
            for d in range(sortedIndices.shape[1]):
                sortedIndices[:, d] = torch.clamp(sortedIndices[:, d], 0, int(numCells[d].item()) - 1)
            cellGridIndices = sortedIndices[cumCell,:]
            # Cell indices contains the linear indices of the particles in each cell
            # cellCounters contains the number of particles in each cell
            # cumCell contains the cumulative sum of the number of particles in each cell, i.e., the offset into the cell
            # With this information we can build a datastructure with [begin, end) for each cell using cellCounters and cumCell!

            # print('cellIndices', cellIndices.device, cellIndices.dtype, cellIndices.shape)
            # print('cumCell', cumCell.device, cumCell.dtype, cumCell.shape)
            # print('cellCounters', cellCounters.device, cellCounters.dtype, cellCounters.shape)


            cellTable = torch.stack((cellIndices, cumCell, cellCounters), dim = 1)
    
        with record_function("neighborSearch - buildHashMap"):
            warpDevice = castTorchToWarp(queryPositions).device
            torchDevice = queryPositions.device

            cellGridIndices_warp = castTorchToWarp(cellGridIndices)
            hashedIndices_warp = wp.zeros(cellGridIndices.shape[0], dtype=wp.uint32, device=warpDevice)
            wp.launch(hashCells, dim=cellGridIndices.shape[0], inputs=[cellGridIndices_warp, wp.uint32(hashMapLength), hashedIndices_warp], device=warpDevice)
            hashedIndices = wp.to_torch(hashedIndices_warp).to(torch.int32)
            # print(hashedIndices_warp)

            sortedSupports = referenceSupports[sortIndex] if referenceSupports is not None else None


            hashIndexSorting = torch.argsort(hashedIndices)
            hashMap, hashMapCounters = torch.unique_consecutive(hashedIndices[hashIndexSorting], return_counts=True, return_inverse=False)
            hashMapCounters = hashMapCounters.to(torch.int32)

        with record_function("neighborSearch - sort cells"):
            # Resort the entries based on the hashIndexSorting so they can be accessed through the hashmap
            sortedCellIndices = cellIndices[hashIndexSorting]
            sortedCellTable = torch.stack([c[hashIndexSorting] for c in cellTable.unbind(1)], dim = 1)
        with record_function("neighborSearch - buildHashTable"):
            # Same construction as for the cell list but this time we create a more direct table
            # The table contains the start and length for each cell in the hash table and -1 if the cell is empty
            hashTable = hashMap.new_ones(hashMapLength,2, dtype = torch.int32) * -1
            hashTable[:,1] = 0
            hashMap64 = hashMap.to(torch.int64)
            hashTable[hashMap64,0] = torch.hstack((torch.tensor([0], device = sortedCellIndices.device, dtype=torch.int32),torch.cumsum(hashMapCounters,dim=0)))[:-1].to(torch.int32) #torch.cumsum(hashMapCounters, dim = 0) #torch.arange(hashMap.shape[0], device=hashMap.device)

            hashTable[hashMap64,1] = hashMapCounters
        
        with record_function("neighborSearch - precomputeOffsets"):
            # we precompute the offset we want to iterate over based on the searchradius parameter
            # in 3D for a search radius of n we will iterate over (2n+1)^3 cells, in 2D we will iterate over (2n+1)^2 cells, and in 1D we will iterate over 2n+1 cells
            searchRadius = 1
            numOffsets = (2 * searchRadius + 1) ** domainDescription.dim
            offsets = torch.cartesian_prod(*[torch.arange(-searchRadius, searchRadius+1, device = queryPositions.device) for _ in range(domainDescription.dim)]).to(torch.int32)
            if len(offsets.shape) == 1:
                offsets = offsets.unsqueeze(1)
            # print('Offsets to iterate over: ', offsets.shape, numOffsets, offsets)
            # pad to always have the same dimension for the kernel
            # offsets = torch.cat((offsets, torch.zeros(numOffsets, 3 - domainDescription.dim, dtype=torch.int32, device=offsets.device)), dim = 1)
            paddedOffsets = torch.zeros((numOffsets, 3), dtype=torch.int32, device=offsets.device)
            paddedOffsets[:,:offsets.shape[1]] = offsets
            offsets = paddedOffsets
            # print('Offsets to iterate over: ', offsets.shape, numOffsets, offsets)
            D = domainDescription.dim

        hashMap = CompactHashMap(
            sortedPositions=sortedPositions,
            sortedSupports=sortedSupports,
            sortIndex=sortIndex,
            hashTable=hashTable,
            sortedCellTable=sortedCellTable,
            qMin=qMin,
            qMax=qMax,
            hCell=hCell,
            numCells=numCells,
            mode_uint=mode_uint,
            D = D,
            searchRadius = 1,
            numOffsets = numOffsets,
            cellOffsets = offsets
        )
        return hashMap
    

def radiusSearchOnCompactHashMap(
    datastructure: CompactHashMap,
    queryPositions: torch.Tensor,
    referencePositions: torch.Tensor,
    querySupports: torch.Tensor,
    referenceSupports: torch.Tensor,
    periodicity: torch.Tensor,
    domainDescription: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLength: int = 4096,
):                
    mode_uint = datastructure.mode_uint
    minDomain = domainDescription.min if domainDescription.min is not None else None
    maxDomain = domainDescription.max if domainDescription.max is not None else None
    hMax = computeGridSupport(querySupports, referenceSupports, mode)
    minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
    # periodicity = domainDescription.periodicity if domainDescription.periodicity is not None else [False] * referencePositions.shape[1]

    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT
    
    sortedPositions = datastructure.sortedPositions
    sortedSupports = datastructure.sortedSupports
    sortIndex = datastructure.sortIndex
    hashTable = datastructure.hashTable
    sortedCellTable = datastructure.sortedCellTable
    qMin = datastructure.qMin
    qMax = datastructure.qMax
    hCell = datastructure.hCell
    numCells = datastructure.numCells
    D = datastructure.D
    offsets = datastructure.cellOffsets

    warpDevice = castTorchToWarp(queryPositions).device
    
    with record_function("neighborSearch - count neighbors"):
        N = queryPositions.shape[0]
        M = sortedPositions.shape[0]
        edge_count = wp.zeros(queryPositions.shape[0], dtype=wp.int32, device=warpDevice)
        wp.launch(radiusSearchCountNeighborsCompactHashMap, dim=queryPositions.shape[0], inputs=[
            castTorchToWarp(y),
            castTorchToWarp(querySupports),
            castTorchToWarp(sortedPositions),
            castTorchToWarp(sortedSupports),
            castTorchToWarp(hashTable),
            castTorchToWarp(sortedCellTable),
            castTorchToWarp(qMin),
            hCell,
            D,
            castTorchToWarp(offsets),
            castTorchToWarp(numCells),
            castTorchToWarp(qMax),
            castTorchToWarp(qMin),
            castTorchToWarp(periodicity),
            wp.uint32(mode_uint),
            edge_count
        ], device=warpDevice)
    with record_function("neighborSearch - allocate neighbors"):


        # Convert counts to host (only the counts, not the main data)
        edge_count_t = wp.to_torch(edge_count)
        total_edges = torch.sum(edge_count_t).cpu().item()

        # Compute cumulative offsets
        edge_offsets = torch.zeros(N, dtype=torch.int32, device = queryPositions.device)
        edge_offsets[1:] = torch.cumsum(edge_count_t[:-1], dim=0)
        edge_offsets_warp = wp.from_torch(edge_offsets)

        # Allocate output arrays on GPU
        edge_i = wp.zeros(total_edges, dtype=wp.int64, device=warpDevice)
        edge_j = wp.zeros(total_edges, dtype=wp.int64, device=warpDevice)
    with record_function("neighborSearch - collect neighbors"):
        wp.launch(radiusSearchCollectCompactHashMap, dim=queryPositions.shape[0], inputs=[
            castTorchToWarp(y),
            castTorchToWarp(querySupports),
            castTorchToWarp(sortedPositions),
            castTorchToWarp(sortedSupports),
            castTorchToWarp(hashTable),
            castTorchToWarp(sortedCellTable),
            castTorchToWarp(qMin),
            hCell,
            D,
            castTorchToWarp(offsets),
            castTorchToWarp(numCells),
            castTorchToWarp(qMax),
            castTorchToWarp(qMin),
            castTorchToWarp(periodicity),
            wp.uint32(mode_uint),   
            edge_count,
            edge_offsets_warp,
            castTorchToWarp(sortIndex),
            edge_i,
            edge_j
        ], device=warpDevice)


    with record_function("neighborSearch - build adjacency"):
        i_torch = wp.to_torch(edge_i)
        j_torch = wp.to_torch(edge_j)

        adjacencyCH = AdjacencyList(
            i=i_torch,  # Ensure dtype is long for indexing
            j=j_torch,
            numNeighbors=wp.to_torch(edge_count).to(dtype=torch.int32),
            edgeOffsets=wp.to_torch(edge_offsets_warp).to(dtype=torch.int32),
            numRows=N,
            numCols=M,
            queryPositions = queryPositions,
            referencePositions = referencePositions,
            querySupports = querySupports,
            referenceSupports = referenceSupports
        )
    return adjacencyCH
    

def radiusSearchCompactHashMap_(
    queryPositions: torch.Tensor,
    referencePositions: torch.Tensor,
    querySupports: torch.Tensor,
    referenceSupports: torch.Tensor,
    periodicity: torch.Tensor,
    domainDescription: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLength: int = 4096,
    returnCompactHashMap: bool = False
):
    with record_function("warpNeighborSearch - radiusSearchCompactHashMap"):
        datastructure = buildCompactHashMap(
            queryPositions,
            referencePositions,
            querySupports,
            referenceSupports,
            periodicity,
            domainDescription,
            mode,
            hashMapLength
        ) 
        adjacencyCH = radiusSearchOnCompactHashMap(
            datastructure,
            queryPositions,
            referencePositions,
            querySupports,
            referenceSupports,
            periodicity,
            domainDescription,
            mode,
            hashMapLength
        )
        
        return adjacencyCH if not returnCompactHashMap else (adjacencyCH, datastructure)

    
from ..state import *

def radiusSearchCompactHashMap(
    queryParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLengthMode: HashMapLengthMode = HashMapLengthMode.NextPrime,
    fixedHashMapLength: int = 4096,
    returnCompactHashMap: bool = False,
    referenceParticles: Optional[ParticleState] = None
):
    referenceParticles = queryParticles if referenceParticles is None else referenceParticles
    queryPositions = queryParticles.positions
    referencePositions = referenceParticles.positions
    querySupports = queryParticles.supports if queryParticles.supports is not None else torch.zeros(queryParticles.positions.shape[0], device=queryParticles.positions.device)
    referenceSupports = referenceParticles.supports if referenceParticles.supports is not None else torch.zeros(referenceParticles.positions.shape[0], device=referenceParticles.positions.device)
    periodicity = torch.tensor(domain.periodic if domain.periodic is not None else [False] * queryParticles.positions.shape[1], device=queryParticles.positions.device)

    if hashMapLengthMode == HashMapLengthMode.Fixed:
        hashMapLength = fixedHashMapLength
    elif hashMapLengthMode == HashMapLengthMode.NumberOfParticles:
        hashMapLength = queryPositions.shape[0]
        if hashMapLength % 2 == 0:
            hashMapLength += 1  # Ensure it's odd to reduce collisions
    elif hashMapLengthMode == HashMapLengthMode.NextPrime:
        hashMapLength = getNextPrime(queryPositions.shape[0])
    else:
        raise ValueError("Invalid hash map length mode")

    return radiusSearchCompactHashMap_(
        queryPositions,
        referencePositions,
        querySupports,
        referenceSupports,
        periodicity,
        domain,
        mode,
        hashMapLength,
        returnCompactHashMap
    )