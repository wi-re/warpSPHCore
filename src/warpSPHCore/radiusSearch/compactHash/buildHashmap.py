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

from torch.profiler import record_function
from .grid import *
from .sort import *
from .indexing import *
from .wp_hashCells import *
import numpy as np

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
            mode_uint = wp.uint32(mode.value)
            # mode_map = {'gather': 1, 'scatter': 2, 'symmetric': 3, 'superSymmetric': 4, ''}
            # mode_uint = mode_map.get(mode, 0)
            # if mode_uint == 0:
                # raise ValueError(f"Invalid mode: {mode}. Supported modes are: {list(mode_map.keys())}")
                
            minDomain = domainDescription.min if domainDescription.min is not None else None
            maxDomain = domainDescription.max if domainDescription.max is not None else None
            hMax = computeGridSupport(querySupports, referenceSupports, mode)
            minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)

        with record_function("neighborSearch - sortParticles"):
            # print(f'Periodicity: {periodicity}')
            x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
            y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT

            sortedLinear, sortIndex, numCells, qMin, qMax, hCell = sortReferenceParticles(x, hMax, minD, maxD, periodicity)



            # Do the actual resort
            sortedPositions = x[sortIndex,:]
            # sortedSupports = xSupport[sortIndex]
        with record_function("neighborSearch - buildCells"):
            # compact teh list of occupied cells
            cellIndices, cellCounters = torch.unique_consecutive(sortedLinear, return_counts=True, return_inverse=False)
            cellCounters = cellCounters.to(torch.int32)
            # Needs to zero padded for the indexing to work properly as the 0th cell is valid and cumsum doesn't include the first element

            cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1].to(torch.int32)

            # Derive grid indices directly from linear cell ids to avoid any floating-point
            # boundary mismatch between build and lookup paths.
            cellGridIndices = delinearizeIndices(cellIndices, numCells, domainDescription.dim)
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
            # wp.dtype_to_torch(wp.uint32) is torch.int32 (torch has no native
            # uint32), so this already gives us an int32 tensor directly --
            # matches the .to(torch.int32) the old wp.to_torch() path needed.
            hashedIndices, hashedIndices_warp = allocateTorchWarp(cellGridIndices.shape[0], wp.uint32, warpDevice)
            wp.launch(hashCells, dim=cellGridIndices.shape[0], inputs=[cellGridIndices_warp, wp.uint32(hashMapLength), hashedIndices_warp], device=warpDevice)
            wp.synchronize()  # ensure hashCells is done before PyTorch reads on its own stream
            referenceHashedIndices = hashGridIndicesTorch(cellGridIndices, hashMapLength)
            # if not torch.equal(hashedIndices, referenceHashedIndices):
            #     mismatch = torch.nonzero(hashedIndices != referenceHashedIndices, as_tuple=False).flatten()
            #     sample = mismatch[:8]
            #     details = []
            #     for idx in sample:
            #         i = int(idx.item())
            #         c = cellGridIndices[i].tolist()
            #         got = int(hashedIndices[i].item())
            #         exp = int(referenceHashedIndices[i].item())
            #         details.append(f"cell={c}, got={got}, expected={exp}")
            #     raise RuntimeError(
            #         "Warp hash assignment mismatch detected for occupied cells. "
            #         f"mismatches={int(mismatch.numel())}, sample=[" + "; ".join(details) + "]"
            #     )
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
            hMaxValue = float(hMax.item()) if torch.is_tensor(hMax) else float(hMax)
            hCellValue = float(hCell.item()) if torch.is_tensor(hCell) else float(hCell)
            # If hCell is smaller than the interaction support, we must expand the cell stencil.
            # Small epsilon avoids promoting exactly-1 ratios to radius=2 from FP noise.
            searchRadius = max(1, int(np.ceil(hMaxValue / max(hCellValue, 1e-12) - 1e-6)))
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
            hCell=hCellValue,
            numCells=numCells,
            mode_uint=mode_uint,
            D = D,
            searchRadius = searchRadius,
            numOffsets = numOffsets,
            cellOffsets = offsets
        )
        return hashMap
    