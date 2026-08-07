
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..type_config import *
from ..enumTypes import *
from ..dataTypes import *
from .compactHash.indexing import getLinearIndex64
from .compactHash.wp_hashCells import hashGridVec3i
from warp.types import vector, matrix
from typing import Any


@wp.func
def wrapCellComponentPeriodic(cell: wp.int32, numCell: wp.int32) -> wp.int32:
    # Wrap integer cell indices for periodic domains, even if they are more than one box out-of-range.
    if numCell <= 0:
        return wp.int32(0)
    return cell - wp.int32(wp.floor(scalar_t(cell) / scalar_t(numCell))) * numCell

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
    i: wp.int32, queryPositions: vecArray_t, # shape [N,D] # type: ignore
    numCells: wp.array(dtype=wp.int32), D: int,# type: ignore

    o: wp.int32,
    cellOffsets: wp.array(dtype=vector(length=3, dtype=wp.int32)), # shape [numOffsets, 3] # type: ignore

    hashTable: wp.array(dtype=vector(length = 2, dtype = wp.int32)), # shape [hashMapLength,2] # type: ignore
    cellTable: wp.array(dtype=vector(length = 3, dtype = wp.int64)), # shape [C,3] with [cellIndex, cellStart, cellCount] # type: ignore

    periodicity: wp.array(dtype = wp.bool), qMin: wp.array(dtype = scalar_t), qMax: wp.array(dtype = scalar_t), # type: ignore
    hCell: scalar_t
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
        rawCell = wp.int32(wp.floor((queryPos[d] - qMin[d]) / hCell))
        if periodicity[d]:
            cellIndex[d] = wrapCellComponentPeriodic(rawCell, numCells[d])
        else:
            if rawCell < 0:
                cellIndex[d] = 0
            elif rawCell >= numCells[d]:
                cellIndex[d] = numCells[d] - 1
            else:
                cellIndex[d] = rawCell
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
    validCell = wp.bool(True)
    for d in range(D):
        if periodicity[d]:
            currentCellIndex[d] = wrapCellComponentPeriodic(currentCellIndex[d], numCells[d])
        else:
            if currentCellIndex[d] < 0 or currentCellIndex[d] >= numCells[d]:
                validCell = False

    if not validCell:
        return -1, -1

    # In periodic domains, multiple offsets may map to the same wrapped cell.
    # Skip duplicates so a cell is processed once per query and offset sweep.
    duplicateCell = wp.bool(False)
    for p in range(o):
        prevOffset = cellOffsets[p]
        prevCellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
        for d in range(D):
            prevCellIndex[d] = cellIndex[d] + prevOffset[d]

        prevValid = wp.bool(True)
        for d in range(D):
            if periodicity[d]:
                prevCellIndex[d] = wrapCellComponentPeriodic(prevCellIndex[d], numCells[d])
            else:
                if prevCellIndex[d] < 0 or prevCellIndex[d] >= numCells[d]:
                    prevValid = False

        if not prevValid:
            continue

        sameCell = wp.bool(True)
        for d in range(D):
            if prevCellIndex[d] != currentCellIndex[d]:
                sameCell = False

        if sameCell:
            duplicateCell = True

    if duplicateCell:
        return -1, -1

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


@wp.func
def getIndexRange(
    i: wp.int32,
    o: wp.int32,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    queryState: Any, # particleDataSoA_1/2/3
    domainState: domainData,
):
    if useAdjacency:
        beginIndex = adjacencyState.neighborOffsets[i]
        numIndices = adjacencyState.numNeighbors[i]
        return beginIndex, numIndices
    else:
        beginIndex, numIndices = checkOffset(
            i, queryState.positions, gridState.numCells, gridState.D,
            o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
            domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
        )
        return beginIndex, numIndices