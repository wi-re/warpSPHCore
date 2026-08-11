import warp as wp
from warp.types import vector, matrix
from typing import Any, Union
import torch
from ..type_config import *
from ..dataTypes import *

from ..math import zero_like_warp

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


# 1D:
@wp.func
def getParticleData(
    SoA: particleDataSoA_1, # particleDataSoA_1 or particleDataSoA_2 or particleDataSoA_3
    i: wp.int32
):
    return WarpParticle_1(SoA.positions[i], SoA.supports[i], SoA.masses[i], SoA.densities[i], SoA.kinds[i])

@wp.func
def getParticleCorrectionData_i(
    correctionData: correctionData_1, # correctionData_1 or correctionData_2 or correctionData_3
    i: wp.int32
):
    gradCorrection, L_i = getL_i(correctionData, i)
    volumeCorrection, V_i = getVolume_i(correctionData, i)
    gradHCorrection, omega_i = getGradH_i(correctionData, i)
    crkCorrection, A_i, B_i, gradA_i, gradB_i = getCRK_i(correctionData, i)
    return ParticleCorrectionData_1(
        L_i,
        V_i,
        omega_i,
        A_i, B_i, gradA_i, gradB_i
    )
@wp.func
def getParticleCorrectionData_j(
    correctionData: correctionData_1, # correctionData_1 or correctionData_2 or correctionData_3
    j: wp.int32
):
    gradCorrection, L_j = False, zero_like_warp(correctionData.renormalizationMatrices)
    volumeCorrection, V_j = getVolume_j(correctionData, j)
    gradHCorrection, omega_j = getGradH_j(correctionData, j)
    crkCorrection, A_j, B_j, gradA_j, gradB_j = getCRK_j(correctionData, j)
    return ParticleCorrectionData_1(
        L_j,
        V_j,
        omega_j,
        A_j, B_j, gradA_j, gradB_j
    )

#2D:
@wp.func
def getParticleData(
    SoA: particleDataSoA_2, # particleDataSoA_1 or particleDataSoA_2 or particleDataSoA_3
    i: wp.int32
):
    return WarpParticle_2(SoA.positions[i], SoA.supports[i], SoA.masses[i], SoA.densities[i], SoA.kinds[i])

@wp.func
def getParticleCorrectionData_i(
    correctionData: correctionData_2, # correctionData_1 or correctionData_2 or correctionData_3
    i: wp.int32
):
    gradCorrection, L_i = getL_i(correctionData, i)
    volumeCorrection, V_i = getVolume_i(correctionData, i)
    gradHCorrection, omega_i = getGradH_i(correctionData, i)
    crkCorrection, A_i, B_i, gradA_i, gradB_i = getCRK_i(correctionData, i)
    return ParticleCorrectionData_2(
        L_i,
        V_i,
        omega_i,
        A_i, B_i, gradA_i, gradB_i
    )
@wp.func
def getParticleCorrectionData_j(
    correctionData: correctionData_2, # correctionData_1 or correctionData_2 or correctionData_3
    j: wp.int32
):
    gradCorrection, L_j = False, zero_like_warp(correctionData.renormalizationMatrices)
    volumeCorrection, V_j = getVolume_j(correctionData, j)
    gradHCorrection, omega_j = getGradH_j(correctionData, j)
    crkCorrection, A_j, B_j, gradA_j, gradB_j = getCRK_j(correctionData, j)
    return ParticleCorrectionData_2(
        L_j,
        V_j,
        omega_j,
        A_j, B_j, gradA_j, gradB_j
    )

#3D:
@wp.func
def getParticleData(
    SoA: particleDataSoA_3, # particleDataSoA_1 or particleDataSoA_2 or particleDataSoA_3
    i: wp.int32
):
    return WarpParticle_3(SoA.positions[i], SoA.supports[i], SoA.masses[i], SoA.densities[i], SoA.kinds[i])

@wp.func
def getParticleCorrectionData_i(
    correctionData: correctionData_3, # correctionData_1 or correctionData_2 or correctionData_3
    i: wp.int32
):
    gradCorrection, L_i = getL_i(correctionData, i)
    volumeCorrection, V_i = getVolume_i(correctionData, i)
    gradHCorrection, omega_i = getGradH_i(correctionData, i)
    crkCorrection, A_i, B_i, gradA_i, gradB_i = getCRK_i(correctionData, i)
    return ParticleCorrectionData_3(
        L_i,
        V_i,
        omega_i,
        A_i, B_i, gradA_i, gradB_i
    )
@wp.func
def getParticleCorrectionData_j(
    correctionData: correctionData_3, # correctionData_1 or correctionData_2 or correctionData_3
    j: wp.int32
):
    gradCorrection, L_j = False, zero_like_warp(correctionData.renormalizationMatrices) # gradRenorm is always using i never j
    volumeCorrection, V_j = getVolume_j(correctionData, j)
    gradHCorrection, omega_j = getGradH_j(correctionData, j)
    crkCorrection, A_j, B_j, gradA_j, gradB_j = getCRK_j(correctionData, j)
    return ParticleCorrectionData_3(
        L_j,
        V_j,
        omega_j,
        A_j, B_j, gradA_j, gradB_j
    )


# @wp.func
# def access_optional(arr: wp.array(dtype = Any), index: wp.int32, defaultValue: Any): # type: ignore
#     if arr.shape[0] > 1:
#         return arr[index]
#     else:
#         return defaultValue

@wp.func
def access_optional(arr: wp.array(dtype = Any), index: wp.int32, condition: wp.bool, defaultValue: Any): # type: ignore
    if condition:
        return arr[index]
    else:
        return defaultValue

@wp.func    
def ternary_helper(condition: wp.bool, true_value: Any, false_value: Any): # type: ignore
    if condition:
        return true_value
    else:
        return false_value