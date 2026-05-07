import torch

import warp as wp
from enum import Enum
from ..kernels.wp_kernel import sphKernel_xi
from ..mathutil import computeDistanceVec, safe_sqrt
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *

from ..enumTypes import *
from .util import *


@wp.struct
class DiffusionParameters:
    c_s: wp.float32 # Speed of sound, used in some formulations to compute the signal velocity
    C_l: wp.float32 # Linear viscosity coefficient, also referred to as alpha in some formulations
    C_q: wp.float32 # Quadratic viscosity coefficient, also referred to as beta in some formulations
    Cu_l: wp.float32 # Linear thermal conductivity coefficient, also referred to as alpha_u in some formulations
    Cu_q: wp.float32 # Quadratic thermal conductivity coefficient, also referred to as beta_u in some formulations
    
    K: wp.float32 # Overall viscosity scaling factor
    thermalConductivity: wp.float32 # Overall thermal conductivity scaling factor
    viscosityTerm: wp.int32 # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    thermalConducitiyTerm: wp.int32 # Thermal conductivity formulation to use, e.g. Monaghan1997 thermal conductivity term, Cleary1998 thermal conductivity term etc.
    scaleBeta: wp.bool # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    monaghanSwitch: wp.bool # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    correctXi: wp.bool # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    


# Note this function returns the term multiplied by rhoj!!
# This is to enable the computation with mj/rhoj as the apparrent volume so the rhoj cancels out for those formulations. This is different from diffSPH which does not multiply by rhoj.
@wp.func
def computePi_actual(
    x_i: vector(dtype = wp.float32, length=Any), x_j:  vector(dtype = wp.float32, length=Any), # type: ignore
    h_i: wp.float32, h_j: wp.float32, # type: ignore
    m_i: wp.float32, m_j: wp.float32, # type: ignore
    rho_i: wp.float32, rho_j: wp.float32, # type: ignore
    explicitPressure: wp.bool, P_i: wp.float32, P_j: wp.float32, # type: ignore
    v_i: vector(dtype = wp.float32, length=Any), v_j: vector(dtype = wp.float32, length=Any), # type: ignore
    
    domainState: domainData, 
    kernel_int : wp.int32,
    c_i: wp.float32, c_j: wp.float32,
    alpha_i: wp.float32, alpha_j: wp.float32,

    viscosityParams: DiffusionParameters,
    # c_s: wp.float32, # Speed of sound, used in some formulations to compute the signal velocity
    # C_l: wp.float32, C_q: wp.float32, # Viscosity coefficients also referred to as alpha and beta in some formulations
    # K_: wp.float32, # Overall viscosity scaling factor
    # viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    # scaleBeta : wp.bool = False, # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    # switch : wp.bool = True, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    # correctXi : wp.bool = False, # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    useJ : wp.bool = False, # Whether to use the properties of the j particle instead of the i particle in the viscosity computation. This can be relevant for certain formulations and scenarios, such as when computing the viscosity force on particle i due to particle j, it might make sense to use the properties of particle j in the computation. This is also related to the use of rho_bar, c_bar and h_bar which are typically computed as averages of the i and j particle properties.
    thermalConductivity : wp.bool = False, # Whether this viscosity computation is being used for thermal conductivity. This can be relevant for certain formulations that use different coefficients or terms for thermal conductivity compared to momentum viscosity, such as in the case of the Monaghan1997 formulation where the thermal conductivity term has a different form and coefficients compared to the momentum viscosity term.
):
    rho_bar = 1.0/2.0 * (rho_i + rho_j)

    # c_i = viscosityParams.c_s
    # c_j = viscosityParams.c_s
    # c_bar = viscosityParams.c_s
    c_bar = 1.0/2.0 * (c_i + c_j)
    # alpha_i = 1.0
    # alpha_j = 1.0 # No viscosity switch here

    h_bar = 1.0/2.0 * (h_i + h_j)

    xi = sphKernel_xi(kernel_int, domainState.dim) if viscosityParams.correctXi else 1.0

    C_l_ = viscosityParams.C_l if not thermalConductivity else viscosityParams.Cu_l
    C_q_ = viscosityParams.C_q if not thermalConductivity else viscosityParams.Cu_q

    C_l = 1.0/2.0 * (alpha_i + alpha_j) * C_l_
    C_q = 1.0/2.0 * (alpha_i + alpha_j) * C_q_
    if viscosityParams.scaleBeta:
        C_q = C_q * C_l

    x_ij = computeDistanceVec(x_i, x_j, domainState.periodicity, domainState.domainMin, domainState.domainMax)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    u_ij = v_i - v_j
    ux_ij = wp.dot(u_ij, x_ij)

    viscosityTerm = viscosityParams.viscosityTerm if not thermalConductivity else viscosityParams.thermalConducitiyTerm

    mu_ij, scalingFactor = compute_mu_ij(ux_ij, r_ij, h_bar, viscosityTerm, xi)

    # if viscosityParams.monaghanSwitch and ux_ij > 0:
    #     mu_ij = 0.0

    v_sig = wp.float32(0.0)
    K = wp.float32(viscosityParams.K)


    rho, c, h = compute_bars(
        rho_i, rho_j, rho_bar, 
        c_i, c_j, c_bar, 
        h_i, h_j, h_bar, 
        viscosityTerm, useJ)

    if viscosityTerm == 1: # MonaghanGingold1983
        # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
        # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
        # nu = alpha h_bar c_bar / rho_bar
        # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
        # combined with using c_bar, rho_bar and h_bar. 
        # Consequently this uses 
        # v_sig = c_bar
        # K = 1
        v_sig = c
        K = 1.0
    elif viscosityTerm == 2: # Cleary1998
        # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
        # mu_a = 1/8 alpha_a h_a c_a rho_a
        # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1.0/(2.0*(wp.float32(domainState.dim)+2.0)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
        K = 1.0
    elif viscosityTerm == 3: # Monaghan1992         
        # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
        # mu = h / rho ( alpha c - beta mu_ij)
        # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c  - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 4: # Monaghan1997a
        # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
        # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
        # in the 1997 paper and has a strange wording in 2005 of using 1.0/2.0 instead of 1 for K
        # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
        # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 5: # Monaghan1997b
        # Note that the C_q here is not the usual quadratic coefficient. For lim C_q -> 0 the term collapses to c_i + c_j, i.e., 2 c_bar. This is equivalent to C_l = 2 and C_q = 1 in the 1997a formulation. 
        # For this formulation as C_q increases the viscosity increases, however, for large C_q this term becomes quickly unstable. In the paper the statement is 'where beta [this is our C_q] is a parameter that could be determined by numerical experiments' with no specific suggestion of value.
        # Based on Monaghan 1997 eq 4.7:
        v_sig = safe_sqrt(c_i*c_i + C_q * mu_ij*mu_ij) + safe_sqrt(c_j*c_j + C_q * mu_ij*mu_ij) - mu_ij
        K = 1.0
    elif viscosityTerm == 6: # Dukowicz
        # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3.0/4.0 * C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 7: # Price2012_98
        # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 8: # Price2012
        # Based on equation 103
        v_sig = C_l * c - C_q / 2.0 * mu_ij
        K = 1.0
    elif viscosityTerm == 9: # Price2008
        # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        # P_i = queryPressures[i]
        # P_j = referencePressures[j]
        # rho_bar = (rho_i + rho_j) / 2
        # v_sig = C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
        # K = 1.0
        # Since we don't have access to the pressures here, we can use an approximation based on the ideal gas law, P = rho * c^2
        P_i_ = rho_i * c_i * c_i if not explicitPressure else P_i
        P_j_ = rho_j * c_j * c_j if not explicitPressure else P_j
        rho_bar = (rho_i + rho_j) / 2.0
        v_sig = C_l * safe_sqrt(wp.abs(P_i_ - P_j_) / (rho_bar + 1e-14 * h))
        K = 1.0
    elif viscosityTerm == 10: # Wadsley2008
        v_sig = C_l * wp.abs(mu_ij)
        K = 1.0
    elif viscosityTerm == 11 or viscosityTerm == 0: # DeltaSPH / Default
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0

    val = rho_j * K / rho * v_sig * scalingFactor #* mu_ij

    if viscosityParams.monaghanSwitch and not thermalConductivity:
        if ux_ij > 0:
            val = 0.0

    return val

from ..operations.wp_gradient import computeSPHGradientTensor_Func


@wp.func
def computePiViscosityKernel_Func(   
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValue: vector(length=Any, dtype=wp.float32), # type: ignore

    c_s: wp.float32,
    C_l: wp.float32, C_q: wp.float32,
    K_: wp.float32,
    viscosityTerm: wp.int32,
    scaleBeta: wp.bool,
    switch: wp.bool,
    correctXi: wp.bool,
    useJ: wp.bool,
):
    
    f_interpolated = type(fi)(0.0)
    dim = wp.int32(xi.length)
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj

        Pi_ij = computePi_actual(
            xi, positions[j],
            hi, supports[j],
            mi, mj,
            rhoi, rhoj,
            fi, values[j],
            domainMin, domainMax, periodicity,
            dim,
            kernel_int,
            c_s, C_l, C_q, K_, viscosityTerm, scaleBeta, switch, correctXi, useJ
        )
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        f_interpolated += apparentVolume * Pi_ij * kernelGradient

    
    return f_interpolated

@wp.kernel
def computeViscosityKernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32),  # type: ignore
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    c_s: wp.float32, # Speed of sound, used in some formulations to compute the signal velocity
    C_l: wp.float32, C_q: wp.float32, # Viscosity coefficients also referred to as alpha and beta in some formulations
    K_: wp.float32, # Overall viscosity scaling factor
    viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    scaleBeta : wp.bool, # If true then the quadratic viscosity term is scaled by the linear viscosity term
    switch : wp.bool, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles
    correctXi : wp.bool, # Whether to apply the xi correction factor
    useJ : wp.bool, # Whether to use the properties of the j particle instead of the i particle
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)), # type: ignore
):
                                                                      
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]

    outputValues[i] = computePiViscosityKernel_Func(
        xi, hi, mi, rhoi, fi,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        periodicity, domainMin, domainMax,
        mode_uint, kernel_int,
        neighborList, neighborListRowOffsets[i], numNeighbors[i],
        numDims, flatInputShape, flatOutputShape,
        type(outputValues[i])(0.0),
        c_s, C_l, C_q, K_, viscosityTerm, scaleBeta, switch, correctXi, useJ
    )

def computePiViscosity(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    adjacency: AdjacencyListWarp,
    
    c_s: float,
    C_l: float, C_q: float,
    K: float,
    viscosityTerm: ViscosityTerms,
    scaleBeta: bool,
    switch: bool,
    correctXi: bool,
    useJ: bool,        
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    modeUint = wp.uint32(mode.value)
    kernelInt = wp.int32(kernel.value)    

    inputShape = queryValues.shape[1:]
    flatInputShape = 1
    for dim in inputShape:
        flatInputShape *= dim
        
    # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
    outputSize = (queryPositions.shape[0])
    # For the output shape we keep the same shape as the input as the laplacian of a scalar field is still a scalar field, and the laplacian of a vector field is still a vector field. We just need to make sure to flatten the inner dimensions for the warp kernel.
    outputShape = inputShape

    flatOutputShape = 1
    for dim in outputShape:
        flatOutputShape *= dim
    numDims = len(inputShape)

    warp_interpolation = warpWrapper(
        launch_kernel, computeViscosityKernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues,
        
        domainMin, domainMax, periodicity,
        modeUint,
        kernelInt,
        
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
        wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
        
        c_s, C_l, C_q, K, wp.int32(viscosityTerm.value), scaleBeta, switch, correctXi, useJ

    )

    return warp_interpolation

