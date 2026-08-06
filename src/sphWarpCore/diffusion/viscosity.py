import torch

import warp as wp
from enum import Enum
from ..kernels import sphKernel_xi
from ..math import computeDistanceVec, safe_sqrt
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..math import *
from ..kernels import *

from ..enumTypes import *
from .util import *

from dataclasses import dataclass, field
from ..types import *

@wp.struct
class DiffusionParameters:
    c_s: scalar_t = field(default=scalar_t(1.0)) # Speed of sound, used in some formulations to compute the signal velocity
    C_l: scalar_t = field(default=scalar_t(1.0)) # Linear viscosity coefficient, also referred to as alpha in some formulations
    C_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic viscosity coefficient, also referred to as beta in some formulations
    Cu_l: scalar_t = field(default=scalar_t(1.0)) # Linear thermal conductivity coefficient, also referred to as alpha_u in some formulations
    Cu_q: scalar_t = field(default=scalar_t(2.0)) # Quadratic thermal conductivity coefficient, also referred to as beta_u in some formulations
    
    K: scalar_t = field(default=scalar_t(1.0)) # Overall viscosity scaling factor
    thermalConductivity: scalar_t = field(default=scalar_t(0.5)) # Overall thermal conductivity scaling factor
    viscosityTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    thermalConducitiyTerm: wp.int32 = field(default=ViscosityTerms.Price2012_98.value) # Thermal conductivity formulation to use, e.g. Monaghan1997 thermal conductivity term, Cleary1998 thermal conductivity term etc.
    scaleBeta: wp.bool = field(default=False) # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    monaghanSwitch: wp.bool = field(default=True) # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    correctXi: wp.bool = field(default=True) # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    


# Note this function returns the term multiplied by rhoj!!
# This is to enable the computation with mj/rhoj as the apparrent volume so the rhoj cancels out for those formulations. This is different from diffSPH which does not multiply by rhoj.
@wp.func
def computePi_actual(
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    m_i: scalar_t, m_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, P_j: scalar_t, # type: ignore
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
    
    domainState: domainData, 
    kernel_int : wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,

    viscosityParams: DiffusionParameters,
    # c_s: scalar_t, # Speed of sound, used in some formulations to compute the signal velocity
    # C_l: scalar_t, C_q: scalar_t, # Viscosity coefficients also referred to as alpha and beta in some formulations
    # K_: scalar_t, # Overall viscosity scaling factor
    # viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    # scaleBeta : wp.bool = False, # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    # switch : wp.bool = True, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    # correctXi : wp.bool = False, # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    useJ : wp.bool = False, # Whether to use the properties of the j particle instead of the i particle in the viscosity computation. This can be relevant for certain formulations and scenarios, such as when computing the viscosity force on particle i due to particle j, it might make sense to use the properties of particle j in the computation. This is also related to the use of rho_bar, c_bar and h_bar which are typically computed as averages of the i and j particle properties.
    thermalConductivity : wp.bool = False, # Whether this viscosity computation is being used for thermal conductivity. This can be relevant for certain formulations that use different coefficients or terms for thermal conductivity compared to momentum viscosity, such as in the case of the Monaghan1997 formulation where the thermal conductivity term has a different form and coefficients compared to the momentum viscosity term.
):
    rho_bar = scalar_t(1.0)/scalar_t(2.0) * (rho_i + rho_j)

    # c_i = viscosityParams.c_s
    # c_j = viscosityParams.c_s
    # c_bar = viscosityParams.c_s
    c_bar = scalar_t(1.0)/scalar_t(2.0) * (c_i + c_j)
    # alpha_i = scalar_t(1.0)
    # alpha_j = scalar_t(1.0) # No viscosity switch here

    h_bar = scalar_t(1.0)/scalar_t(2.0) * (h_i + h_j)

    # xi = sphKernelScale(kernel_int, domainState.dim) if viscosityParams.correctXi else scalar_t(1.0)
    xi = sphKernel_xi(kernel_int, domainState.dim) if viscosityParams.correctXi else scalar_t(1.0)

    C_l_ = viscosityParams.C_l if not thermalConductivity else viscosityParams.Cu_l
    C_q_ = viscosityParams.C_q if not thermalConductivity else viscosityParams.Cu_q

    C_l = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_l_
    C_q = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_q_
    if viscosityParams.scaleBeta:
        C_q = C_q * C_l

    x_ij = computeDistanceVec(x_i, x_j, domainState.periodicity, domainState.domainMin, domainState.domainMax)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    u_ij = v_i - v_j
    ux_ij = wp.dot(u_ij, x_ij)

    viscosityTerm = viscosityParams.viscosityTerm if not thermalConductivity else viscosityParams.thermalConducitiyTerm

    mu_ij, scalingFactor = compute_mu_ij(ux_ij, r_ij, h_bar, viscosityTerm, xi)

    # if viscosityParams.monaghanSwitch and ux_ij > 0:
    #     mu_ij = scalar_t(0.0)

    v_sig = scalar_t(scalar_t(0.0))
    K = scalar_t(viscosityParams.K)


    rho, c, h = compute_bars(
        rho_i, rho_j, rho_bar, 
        c_i, c_j, c_bar, 
        h_i, h_j, h_bar, 
        viscosityTerm, useJ)

    if viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        # Monaghan and Gingold 1983: The terms are given in (scalar_t(8.3)) and (scalar_t(8.4))  of Monaghan 2005 and are
        # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
        # nu = alpha h_bar c_bar / rho_bar
        # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
        # combined with using c_bar, rho_bar and h_bar. 
        # Consequently this uses 
        # v_sig = c_bar
        # K = 1
        v_sig = c
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        # Cleary 1998: The terms are given in (scalar_t(8.8)) and (scalar_t(8.9)) of Monaghan 2005 and are
        # mu_a = 1/8 alpha_a h_a c_a rho_a
        # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = scalar_t(1.0)/(scalar_t(2.0)*(scalar_t(domainState.dim)+scalar_t(2.0))) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # scalar_t(19.8) based on Cleary and Ha 2002
        v_sig = scalar_t(19.8) * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + scalar_t(1e-14) * h)
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992         
        # Monaghan 1992: The term is given in (scalar_t(8.10)) of Monaghan 2005 and is
        # mu = h / rho ( alpha c - beta mu_ij)
        # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c  - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        # Monaghan 1997: The term is given in (scalar_t(8.11)) of Monaghan 2005 and is very similar
        # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
        # in the 1997 paper and has a strange wording in 2005 of using scalar_t(1.0)/scalar_t(2.0) instead of 1 for K
        # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
        # c_bar term with alpha = 1 and beta = 4! This is also eq scalar_t(3.7) in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        # Note that the C_q here is not the usual quadratic coefficient. For lim C_q -> 0 the term collapses to c_i + c_j, i.e., 2 c_bar. This is equivalent to C_l = 2 and C_q = 1 in the 1997a formulation. 
        # For this formulation as C_q increases the viscosity increases, however, for large C_q this term becomes quickly unstable. In the paper the statement is 'where beta [this is our C_q] is a parameter that could be determined by numerical experiments' with no specific suggestion of value.
        # Based on Monaghan 1997 eq scalar_t(4.7):
        v_sig = safe_sqrt(c_i*c_i + C_q * mu_ij*mu_ij) + safe_sqrt(c_j*c_j + C_q * mu_ij*mu_ij) - mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        # The term is given in (scalar_t(4.8)) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - scalar_t(3.0)/scalar_t(4.0) * C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        # Based on equation 103
        v_sig = C_l * c - C_q / scalar_t(2.0) * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        # P_i = queryPressures[i]
        # P_j = referencePressures[j]
        # rho_bar = (rho_i + rho_j) / 2
        # v_sig = C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + scalar_t(1e-14) * h))
        # K = scalar_t(1.0)
        # Since we don't have access to the pressures here, we can use an approximation based on the ideal gas law, P = rho * c^2
        P_i_ = rho_i * c_i * c_i if not explicitPressure else P_i
        P_j_ = rho_j * c_j * c_j if not explicitPressure else P_j
        rho_bar = (rho_i + rho_j) / scalar_t(2.0)
        v_sig = C_l * safe_sqrt(wp.abs(P_i_ - P_j_) / (rho_bar + scalar_t(1e-14) * h))
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        v_sig = C_l * wp.abs(mu_ij)
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value) or viscosityTerm == wp.static(ViscosityTerms.Default.value): # DeltaSPH / Default
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)

    # rhoTerm = rho_j if not useJ else rho_i

    val = rho_j * K / rho * v_sig * scalingFactor #* mu_ij

    if viscosityParams.monaghanSwitch and not thermalConductivity:
        if ux_ij > 0:
            val = scalar_t(0.0)

    return val


@wp.func
def computePiViscosityKernel_Func(   
    xi: vector(dtype = scalar_t, length=Any), hi : scalar_t, mi: scalar_t, rhoi: scalar_t, fi : vector(dtype = scalar_t, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = scalar_t)), supports : wp.array(dtype = scalar_t), masses: wp.array(dtype = scalar_t), densities: wp.array(dtype = scalar_t), values: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValue: vector(length=Any, dtype=scalar_t), # type: ignore

    c_s: scalar_t,
    C_l: scalar_t, C_q: scalar_t,
    K_: scalar_t,
    viscosityTerm: wp.int32,
    scaleBeta: wp.bool,
    switch: wp.bool,
    correctXi: wp.bool,
    useJ: wp.bool,
):
    
    f_interpolated = type(fi)(scalar_t(0.0))
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
    queryPositions : wp.array(dtype = vector(length=Any, dtype=scalar_t)), referencePositions : wp.array(dtype=vector(length=Any, dtype=scalar_t)), # type: ignore
    querySupports : wp.array(dtype = scalar_t), referenceSupports : wp.array(dtype = scalar_t), # type: ignore
    queryMasses: wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore
    queryDensities: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = scalar_t, length=Any)), referenceValues: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32),  # type: ignore
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    c_s: scalar_t, # Speed of sound, used in some formulations to compute the signal velocity
    C_l: scalar_t, C_q: scalar_t, # Viscosity coefficients also referred to as alpha and beta in some formulations
    K_: scalar_t, # Overall viscosity scaling factor
    viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    scaleBeta : wp.bool, # If true then the quadratic viscosity term is scaled by the linear viscosity term
    switch : wp.bool, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles
    correctXi : wp.bool, # Whether to apply the xi correction factor
    useJ : wp.bool, # Whether to use the properties of the j particle instead of the i particle
    outputValues : wp.array(dtype = vector(length=Any, dtype = scalar_t)), # type: ignore
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
        type(outputValues[i])(scalar_t(0.0)),
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
    
    c_s: scalar_t,
    C_l: scalar_t, C_q: scalar_t,
    K: scalar_t,
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
        launch_kernel, computeViscosityKernel, outputSize, vector(length=flatOutputShape, dtype = scalar_t),
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

