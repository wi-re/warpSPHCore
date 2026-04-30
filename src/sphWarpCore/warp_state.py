import warp as wp
from warp.types import vector, matrix
from typing import Any, Union
import torch

@wp.struct
class particleDataSoA_1:
    positions: wp.array(dtype=vector(length=1, dtype = wp.float32)) # type: ignore
    supports: wp.array(dtype = wp.float32) # type: ignore
    masses: wp.array(dtype = wp.float32) # type: ignore
    densities: wp.array(dtype = wp.float32) # type: ignore
    kinds: wp.array(dtype = wp.int32) # type: ignore
@wp.struct
class particleDataSoA_2:
    positions: wp.array(dtype=vector(length=2, dtype = wp.float32)) # type: ignore
    supports: wp.array(dtype = wp.float32) # type: ignore
    masses: wp.array(dtype = wp.float32) # type: ignore
    densities: wp.array(dtype = wp.float32) # type: ignore
    kinds: wp.array(dtype = wp.int32) # type: ignore
@wp.struct
class particleDataSoA_3:
    positions: wp.array(dtype=vector(length=3, dtype = wp.float32)) # type: ignore
    supports: wp.array(dtype = wp.float32) # type: ignore
    masses: wp.array(dtype = wp.float32) # type: ignore
    densities: wp.array(dtype = wp.float32) # type: ignore
    kinds: wp.array(dtype = wp.int32) # type: ignore

@wp.struct
class adjacencyData:
    neighborList: wp.array(dtype = wp.int64) # type: ignore
    neighborOffsets: wp.array(dtype = wp.int32) # type: ignore
    numNeighbors: wp.array(dtype = wp.int32) # type: ignore

@wp.struct
class gridData:
    sortIndex: wp.array(dtype = wp.int64) # type: ignore
    qMin: wp.array(dtype = wp.float32) # type: ignore
    qMax: wp.array(dtype = wp.float32) # type: ignore
    hCell: float
    numCells: wp.array(dtype = wp.int32) # type: ignore
    hashTable: wp.array(dtype = vector(length = 2, dtype = wp.int32)) # type: ignore
    cellTable: wp.array(dtype = vector(length = 3, dtype = wp.int64)) # type: ignore
    D: int
    numOffsets: int
    cellOffsets: wp.array(dtype = vector(length=3, dtype = wp.int32)) # type: ignore
    
@wp.struct
class domainData:
    domainMin: wp.array(dtype = wp.float32) # type: ignore
    domainMax: wp.array(dtype = wp.float32) # type: ignore
    periodicity: wp.array(dtype = wp.bool) # type: ignore
    dim: wp.int32

@wp.func
def getParticle(
    SoA: Any, # particleDataSoA_1 or particleDataSoA_2 or particleDataSoA_3
    i: wp.int32
):
    return (SoA.positions[i], SoA.supports[i], SoA.masses[i], SoA.densities[i], SoA.kinds[i])




@wp.struct
class correctionData_1:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(1,1), dtype=wp.float32)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = wp.float32) # type: ignore
    referenceVolumes: wp.array(dtype = wp.float32) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = wp.float32)  # type: ignore
    referenceOmegas: wp.array(dtype = wp.float32) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype = wp.float32)  # type: ignore
    queryB: wp.array(dtype=vector(length=1, dtype=wp.float32))  # type: ignore
    queryGradA: wp.array(dtype = vector(length=1, dtype=wp.float32))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(1,1), dtype=wp.float32)) # type: ignore
    referenceA: wp.array(dtype = wp.float32)  # type: ignore
    referenceB: wp.array(dtype=vector(length=1, dtype=wp.float32))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=1, dtype=wp.float32))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(1,1), dtype=wp.float32)) # type: ignore

@wp.struct
class correctionData_2:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(2,2), dtype=wp.float32)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = wp.float32) # type: ignore
    referenceVolumes: wp.array(dtype = wp.float32) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = wp.float32) # type: ignore
    referenceOmegas: wp.array(dtype = wp.float32) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype = wp.float32)  # type: ignore
    queryB: wp.array(dtype=vector(length=2, dtype=wp.float32))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=2, dtype=wp.float32))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(2,2), dtype=wp.float32)) # type: ignore
    referenceA: wp.array(dtype = wp.float32)  # type: ignore
    referenceB: wp.array(dtype=vector(length=2, dtype=wp.float32))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=2, dtype=wp.float32))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(2,2), dtype=wp.float32)) # type: ignore

@wp.struct
class correctionData_3:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(3,3), dtype=wp.float32)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = wp.float32)  # type: ignore
    referenceVolumes: wp.array(dtype = wp.float32) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = wp.float32) # type: ignore
    referenceOmegas: wp.array(dtype = wp.float32) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype=wp.float32)  # type: ignore
    queryB: wp.array(dtype=vector(length=1, dtype=wp.float32))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=3, dtype=wp.float32))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(3,3), dtype=wp.float32)) # type: ignore
    referenceA: wp.array(dtype=wp.float32)  # type: ignore
    referenceB: wp.array(dtype=vector(length=3, dtype=wp.float32))  # type: ignore
    referenceGradA: wp.array(dtype=vector(length=3, dtype=wp.float32))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(3,3), dtype=wp.float32)) # type: ignore

from .utils.wp_util import zero_like_warp


@wp.func
def getL_i(
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3
    i: wp.int32
):
    if correctionData.useGradientRenormalization:
        return True, correctionData.renormalizationMatrices[i]
    else:
        return False, zero_like_warp(correctionData.renormalizationMatrices)
    
@wp.func
def getVolume_i(correctionData: Any, i: wp.int32):
    if correctionData.useVolume:
        return True, correctionData.queryVolumes[i]
    else:
        return False, zero_like_warp(correctionData.queryVolumes)
@wp.func
def getVolume_j(
    correctionData: Any, j: wp.int32
):
    if correctionData.useVolume:
        return True, correctionData.referenceVolumes[j]
    else:
        return False, zero_like_warp(correctionData.referenceVolumes)
@wp.func
def getGradH_i(
    correctionData: Any, i: wp.int32
):
    if correctionData.useGradHTerms:
        return True, correctionData.queryOmegas[i]
    else:
        return False, zero_like_warp(correctionData.queryOmegas)
@wp.func
def getGradH_j(
    correctionData: Any, j: wp.int32
):
    if correctionData.useGradHTerms:
        return True, correctionData.referenceOmegas[j]
    else:
        return False, zero_like_warp(correctionData.referenceOmegas)
@wp.func
def getCRK_i(
    correctionData: Any, i: wp.int32
):
    if correctionData.useCRK:
        return True, correctionData.queryA[i], correctionData.queryB[i], correctionData.queryGradA[i], correctionData.queryGradB[i]
    else:
        return False, zero_like_warp(correctionData.queryA), zero_like_warp(correctionData.queryB), zero_like_warp(correctionData.queryGradA), zero_like_warp(correctionData.queryGradB)

@wp.func
def getCRK_j(
    correctionData: Any, j: wp.int32
):
    if correctionData.useCRK:
        return True, correctionData.referenceA[j], correctionData.referenceB[j], correctionData.referenceGradA[j], correctionData.referenceGradB[j]
    else:
        return False, zero_like_warp(correctionData.referenceA), zero_like_warp(correctionData.referenceB), zero_like_warp(correctionData.referenceGradA), zero_like_warp(correctionData.referenceGradB)



