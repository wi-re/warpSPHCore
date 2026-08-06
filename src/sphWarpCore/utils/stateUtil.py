import warp as wp
from warp.types import vector, matrix
from typing import Any, Union
import torch
from ..types import *
from ..dataTypes import *

from .wp_util import zero_like_warp

@wp.func
def getParticle(
    SoA: Any, # particleDataSoA_1 or particleDataSoA_2 or particleDataSoA_3
    i: wp.int32
):
    return (SoA.positions[i], SoA.supports[i], SoA.masses[i], SoA.densities[i], SoA.kinds[i])

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



