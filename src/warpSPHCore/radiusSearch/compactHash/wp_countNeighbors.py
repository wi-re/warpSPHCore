import warp as wp
import torch
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *
from ...dataTypes import *
from ...enumTypes import *
from .grid import *
from .indexing import *
from .wp_index import wrapCellComponentPeriodic, clampCellIndex, indexCells
from .wp_hashCells import hashGridVec3i

@wp.func
def _countNeighborsCompactHashMapRow(queryPositions: wp.array2d(dtype=scalar_t), querySupports: wp.array1d(dtype=scalar_t), sortedPositions: wp.array2d(dtype=scalar_t), sortedSupports: wp.array1d(dtype=scalar_t), hashTable: wp.array2d(dtype=wp.int32), cellTable: wp.array2d(dtype=wp.int64), qMin: wp.array1d(dtype=scalar_t), hCell: scalar_t, D: int, cellOffsets: wp.array2d(dtype=wp.int32), numCells: wp.array(dtype=wp.int32), maxDomain: wp.array(dtype=scalar_t), minDomain: wp.array(dtype=scalar_t), periodicity: wp.array(dtype=wp.bool), mode_uint: wp.uint32, queryIndex: int, rowIndex: int):
    N = queryPositions.shape[0]
    M = sortedPositions.shape[0]
    i = rowIndex
    hashMapLength = wp.uint32(hashTable.shape[0])
    if i >= N:
        return wp.int32(0)
    queryPos = queryPositions[queryIndex]
    querySupport = querySupports[i]
    domainState = domainData()
    domainState.domainMin = minDomain
    domainState.domainMax = maxDomain
    domainState.periodicity = periodicity
    domainState.dim = D
    cellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
    for d in range(D):
        rawCell = wp.int32(wp.floor((queryPos[d] - qMin[d]) / hCell))
        if periodicity[d]:
            cellIndex[d] = wrapCellComponentPeriodic(rawCell, numCells[d])
        elif rawCell < 0:
            cellIndex[d] = 0
        elif rawCell >= numCells[d]:
            cellIndex[d] = numCells[d] - 1
        else:
            cellIndex[d] = rawCell
    numOffsets = cellOffsets.shape[0]
    count = wp.int32(0)
    for o in range(numOffsets):
        offset = cellOffsets[o]
        currentCellIndex = wp.vec3i(0, 0, 0, dtype=wp.int32)
        for d in range(D):
            currentCellIndex[d] = cellIndex[d] + offset[d]
        validCell = wp.bool(True)
        for d in range(D):
            if periodicity[d]:
                currentCellIndex[d] = wrapCellComponentPeriodic(currentCellIndex[d], numCells[d])
            elif currentCellIndex[d] < 0 or currentCellIndex[d] >= numCells[d]:
                validCell = False
        if not validCell:
            continue
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
                elif prevCellIndex[d] < 0 or prevCellIndex[d] >= numCells[d]:
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
            continue
        linearIndex = getLinearIndex64(currentCellIndex, numCells, D)
        hashValue = wp.int32(hashGridVec3i(currentCellIndex, hashMapLength, D))
        hashEntry = hashTable[hashValue]
        if hashEntry[1] == 0:
            continue
        cellStart = hashEntry[0]
        cellCount = hashEntry[1]
        for c in range(cellCount):
            candidateIndex = cellTable[cellStart + c, 0]
            cellStartIndex = wp.int32(cellTable[cellStart + c, 1])
            cellParticleCount = wp.int32(cellTable[cellStart + c, 2])
            if linearIndex != candidateIndex:
                continue
            for p in range(wp.int32(cellParticleCount)):
                neighborIndex = cellStartIndex + p
                neighborPos = sortedPositions[neighborIndex]
                neighborSupport = sortedSupports[neighborIndex] if sortedSupports.shape[0] > 0 else scalar_t(0.0)
                dist = computeCartesianDistance(queryPos, neighborPos, domainState)
                threshold = scalar_t(0.0)
                if mode_uint == wp.static(SupportScheme.Gather.value):
                    threshold = querySupport
                elif mode_uint == wp.static(SupportScheme.Scatter.value):
                    threshold = neighborSupport
                elif mode_uint == wp.static(SupportScheme.MeanSymmetric.value):
                    threshold = (querySupport + neighborSupport) / scalar_t(2.0)
                elif mode_uint == wp.static(SupportScheme.KernelMeanSymmetric.value):
                    threshold = max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.SuperSymmetric.value):
                    threshold = wp.max(querySupport, neighborSupport)
                elif mode_uint == wp.static(SupportScheme.PartialSymmetric.value):
                    threshold = wp.max(querySupport, neighborSupport)
                if dist <= threshold:
                    count += 1
    return count

@wp.kernel
def radiusSearchCountNeighborsCompactHashMap(queryPositions: wp.array2d(dtype=scalar_t), querySupports: wp.array1d(dtype=scalar_t), sortedPositions: wp.array2d(dtype=scalar_t), sortedSupports: wp.array1d(dtype=scalar_t), hashTable: wp.array2d(dtype=wp.int32), cellTable: wp.array2d(dtype=wp.int64), qMin: wp.array1d(dtype=scalar_t), hCell: scalar_t, D: int, cellOffsets: wp.array2d(dtype=wp.int32), numCells: wp.array(dtype=wp.int32), maxDomain: wp.array(dtype=scalar_t), minDomain: wp.array(dtype=scalar_t), periodicity: wp.array(dtype=wp.bool), mode_uint: wp.uint32, neighborCounts: wp.array1d(dtype=wp.int32)):
    neighborCounts[wp.tid()] = _countNeighborsCompactHashMapRow(queryPositions, querySupports, sortedPositions, sortedSupports, hashTable, cellTable, qMin, hCell, D, cellOffsets, numCells, maxDomain, minDomain, periodicity, mode_uint, wp.tid(), wp.tid())

@wp.kernel
def radiusSearchCountNeighborsSpatial(queryPositions: wp.array2d(dtype=scalar_t), querySupports: wp.array1d(dtype=scalar_t), sortedPositions: wp.array2d(dtype=scalar_t), sortedSupports: wp.array1d(dtype=scalar_t), hashTable: wp.array2d(dtype=wp.int32), cellTable: wp.array2d(dtype=wp.int64), qMin: wp.array1d(dtype=scalar_t), hCell: scalar_t, D: int, cellOffsets: wp.array2d(dtype=wp.int32), numCells: wp.array(dtype=wp.int32), maxDomain: wp.array(dtype=scalar_t), minDomain: wp.array(dtype=scalar_t), periodicity: wp.array(dtype=wp.bool), mode_uint: wp.uint32, neighborCounts: wp.array1d(dtype=wp.int32), sortIndex: wp.array(dtype=index_t)):
    neighborCounts[wp.int32(sortIndex[wp.tid()])] = _countNeighborsCompactHashMapRow(sortedPositions, querySupports, sortedPositions, sortedSupports, hashTable, cellTable, qMin, hCell, D, cellOffsets, numCells, maxDomain, minDomain, periodicity, mode_uint, wp.tid(), wp.int32(sortIndex[wp.tid()]))
