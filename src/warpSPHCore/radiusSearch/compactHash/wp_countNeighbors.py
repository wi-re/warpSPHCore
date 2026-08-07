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

from .grid import *
from .indexing import *
from .wp_index import wrapCellComponentPeriodic, clampCellIndex, indexCells
from .wp_hashCells import hashGridVec3i


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

    domainState = domainData()
    domainState.domainMin = minDomain
    domainState.domainMax = maxDomain
    domainState.periodicity = periodicity
    domainState.dim = D

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
        validCell = wp.bool(True)
        for d in range(D):
            if periodicity[d]:
                currentCellIndex[d] = wrapCellComponentPeriodic(currentCellIndex[d], numCells[d])
            else:
                if currentCellIndex[d] < 0 or currentCellIndex[d] >= numCells[d]:
                    validCell = False

        if not validCell:
            continue

        # In periodic domains, different offsets can wrap to the same cell.
        # Skip duplicates so a cell contributes at most once per query particle.
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
            continue
                    
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
                
                dist = computeCartesianDistance(queryPos, neighborPos, domainState)
                
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
    