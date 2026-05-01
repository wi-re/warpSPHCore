
import warp as wp
from ..enumTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap, getLinearIndex64, hashGridVec3i
from warp.types import vector, matrix
from typing import Any

@wp.func
def iterateCell(
    linearIndex: wp.int64,
    cellCount: wp.int32, 
                cellStart: wp.int32, 
                cellTable: wp.array(dtype=vector(length=3, dtype=wp.int64)) # shape [C,3] with [cellIndex, cellStart, cellCount] # type: ignore
                ):    
    for c in range(cellCount):
        candidateIndex = cellTable[cellStart + c][0]
        cellStartIndex = wp.int32(cellTable[cellStart + c][1])
        cellParticleCount = wp.int32(cellTable[cellStart + c][2])
        
        if linearIndex != candidateIndex:
            # wp.printf("\t\tThread %d: Skipping cell with linear index %d as it does not match the current cell index %d\n", i, candidateIndex, linearIndex)
            continue  # Keep scanning the hash bucket for a matching linear index
        return cellStartIndex, cellParticleCount
    return -1, -1  # No matching cell found

@wp.func 
def checkOffset(
    i: wp.int32, queryPositions: wp.array(dtype=vector(length=Any, dtype = wp.float32)), # shape [N,D] # type: ignore
    numCells: wp.array(dtype=wp.int32), D: int,# type: ignore
    
    o: wp.int32,
    cellOffsets: wp.array(dtype=vector(length=3, dtype=wp.int32)), # shape [numOffsets, 3] # type: ignore

    hashTable: wp.array(dtype=vector(length = 2, dtype = wp.int32)), # shape [hashMapLength,2] # type: ignore
    cellTable: wp.array(dtype=vector(length = 3, dtype = wp.int64)), # shape [C,3] with [cellIndex, cellStart, cellCount] # type: ignore

    periodicity: wp.array(dtype = wp.bool), qMin: wp.array(dtype = wp.float32), qMax: wp.array(dtype = wp.float32), # type: ignore
    hCell: float
):
    # pass
    N = queryPositions.shape[0]
    # D = queryPositions.shape[1]
    # M = sortedPositions.shape[0]
    hashMapLength = wp.uint32(hashTable.shape[0])
    
    queryPos = queryPositions[i]
    # querySupport = querySupports[i]
                        
    # Determine the cell index of the query particle
    cellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
    for d in range(D):
        cellIndex[d] = wp.int32(wp.floor((queryPos[d] - qMin[d]) / hCell))
    # Compute the hash value for the cell index
    # hashValue = hashGridIndex(cellIndex, hashMapLength)
    numOffsets = cellOffsets.shape[0]
    currentLinearIndex = getLinearIndex64(cellIndex, numCells, D)
    
    # getLinearIndex(cellIndex, numCells, D)
    count = wp.int32(0)

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
        return -1, -1  # No particles in this cell
    cellStart = hashEntry[0]
    cellCount = hashEntry[1]
    
    return iterateCell(linearIndex, cellCount, cellStart, cellTable)