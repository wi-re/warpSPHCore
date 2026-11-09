"""Compact hash-map construction with unchanged Torch layout/order operations."""
from __future__ import annotations
from typing import Any, Optional, Union, Tuple, List
from ...math.periodic import periodic_positions
import numpy as np
import torch
from ..._backend import OPERATORS
from ...type_config import get_torch_precision, get_torch_index_dtype
from ...dataTypes.hashMap_t import CompactHashMap
from ...dataTypes import DomainDescription
from ...enumTypes import SupportScheme
from ...profiling import record_function
from ..csr import validateParticleCounts
from .grid import computeGridSupport, getDomainExtents, delinearizeIndices
from .sort import sortReferenceParticles

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
    if OPERATORS == 'warp':
        import warp as wp
        from ...util import castTorchToWarp, allocateTorchWarp
        from .wp_hashCells import hashCells
    validateParticleCounts(queryPositions.shape[0], referencePositions.shape[0])
    if hashMapLength <= 0:
        raise ValueError('hashMapLength must be positive, including for empty populations.')
    with record_function("warpNeighborSearch - buildCompactHashMap"):
        with record_function("neighborSearch - preprocess"):
            mode_uint = int(mode.value) if OPERATORS == 'native' else wp.uint32(mode.value)
            # mode_map = {'gather': 1, 'scatter': 2, 'symmetric': 3, 'superSymmetric': 4, ''}
            # mode_uint = mode_map.get(mode, 0)
            # if mode_uint == 0:
                # raise ValueError(f"Invalid mode: {mode}. Supported modes are: {list(mode_map.keys())}")
                
            minDomain = domainDescription.min if domainDescription.min is not None else None
            maxDomain = domainDescription.max if domainDescription.max is not None else None
            hMax = computeGridSupport(querySupports, referenceSupports, mode)
            if queryPositions.shape[0] == 0 or referencePositions.shape[0] == 0:
                # No interacting pairs exist. Keep a valid, positive cell size
                # even if the selected role's radius bound is vacuously zero.
                hMax = torch.where(hMax > 0, hMax, torch.ones_like(hMax))
            minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)

        with record_function("neighborSearch - sortParticles"):
            x = periodic_positions(referencePositions, periodicity, minD, maxD)
            y = periodic_positions(queryPositions, periodicity, minD, maxD)

            sortedLinear, sortIndex, numCells, qMin, qMax, hCell = sortReferenceParticles(x, hMax, minD, maxD, periodicity)
            # The grid scalars the host needs (search radius, cell counts,
            # hCell as a kernel argument) in ONE transfer instead of a sync
            # each. float64 holds the float32 values exactly; hCell is rounded
            # to the build precision as `scalar_t(hCell)` used to do.
            _host = torch.cat([torch.as_tensor(hMax, device=numCells.device).reshape(1).double(),
                               torch.as_tensor(hCell, device=numCells.device).reshape(1).to(get_torch_precision()).double(),
                               numCells.double()]).cpu().tolist()
            hMaxValue, hCellValue, numCellsHost = _host[0], _host[1], [int(c) for c in _host[2:]]



            # Do the actual resort
            sortedPositions = x[sortIndex,:]
            # sortedSupports = xSupport[sortIndex]
        with record_function("neighborSearch - buildCells"):
            # compact teh list of occupied cells
            cellIndices, cellCounters = torch.unique_consecutive(sortedLinear, return_counts=True, return_inverse=False)
            cellCounters = cellCounters.to(torch.int32)
            # Needs to zero padded for the indexing to work properly as the 0th cell is valid and cumsum doesn't include the first element

            cumCell = torch.hstack((torch.zeros(1, device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1].to(torch.int32)

            # Derive grid indices directly from linear cell ids to avoid any floating-point
            # boundary mismatch between build and lookup paths.
            cellGridIndices = delinearizeIndices(cellIndices, numCells, domainDescription.dim, cellCountsHost=numCellsHost)
            # Cell indices contains the linear indices of the particles in each cell
            # cellCounters contains the number of particles in each cell
            # cumCell contains the cumulative sum of the number of particles in each cell, i.e., the offset into the cell
            # With this information we can build a datastructure with [begin, end) for each cell using cellCounters and cumCell!

            # print('cellIndices', cellIndices.device, cellIndices.dtype, cellIndices.shape)
            # print('cumCell', cumCell.device, cumCell.dtype, cumCell.shape)
            # print('cellCounters', cellCounters.device, cellCounters.dtype, cellCounters.shape)


            cellTable = torch.stack((cellIndices, cumCell, cellCounters), dim = 1)
    
        with record_function("neighborSearch - buildHashMap"):
            if OPERATORS == 'native':
                from ...native._build import build
                hashed64 = torch.empty(len(cellGridIndices), device=cellGridIndices.device, dtype=torch.int64)
                build().fixed_grid_keys(cellGridIndices.contiguous(), numCells, hashed64, hashMapLength)
                hashedIndices = hashed64.to(torch.int32)
            else:
                warpDevice = castTorchToWarp(queryPositions).device
                torchDevice = queryPositions.device

                cellGridIndices_warp = castTorchToWarp(cellGridIndices)
                # wp.dtype_to_torch(wp.uint32) is torch.int32 (torch has no native
                # uint32), so this already gives us an int32 tensor directly --
                # matches the .to(torch.int32) the old wp.to_torch() path needed.
                hashedIndices, hashedIndices_warp = allocateTorchWarp(cellGridIndices.shape[0], wp.uint32, warpDevice)
                # launched on torch's current stream, so the torch ops below are
                # ordered after it without the device-wide wp.synchronize() this
                # used to need (nor the unused torch reference hash it computed)
                wp.launch(hashCells, dim=cellGridIndices.shape[0], inputs=[cellGridIndices_warp, wp.uint32(hashMapLength), hashedIndices_warp], device=warpDevice,
                          stream=wp.stream_from_torch(torch.cuda.current_stream(torchDevice)) if torchDevice.type == 'cuda' else None)
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
            hashTable[hashMap64,0] = torch.hstack((torch.zeros(1, device = sortedCellIndices.device, dtype=torch.int32),torch.cumsum(hashMapCounters,dim=0)))[:-1].to(torch.int32) #torch.cumsum(hashMapCounters, dim = 0) #torch.arange(hashMap.shape[0], device=hashMap.device)

            hashTable[hashMap64,1] = hashMapCounters
        
        with record_function("neighborSearch - precomputeOffsets"):
            # we precompute the offset we want to iterate over based on the searchradius parameter
            # in 3D for a search radius of n we will iterate over (2n+1)^3 cells, in 2D we will iterate over (2n+1)^2 cells, and in 1D we will iterate over 2n+1 cells
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
            sortIndex=sortIndex.to(dtype=get_torch_index_dtype()),
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

