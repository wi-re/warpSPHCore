import torch


import warp as wp
from enum import Enum
from ..types import *
from ..kernels.wp_kernel import sphKernel_xi
from ..math import computeDistanceVec, safe_sqrt
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..math import *
from ..kernels.wp_kernel import *

from ..enumTypes import *


@wp.func
def compute_mu_ij(
    ux_ij: scalar_t, r_ij: scalar_t, h: scalar_t, viscosityTerm: wp.int32, xi: scalar_t
):
    mu_ij = ux_ij / (r_ij + scalar_t(1e-14) * h) # Always start with this as the base

    scalingFactor = h / xi / (r_ij + scalar_t(1e-14) * h)
    scaled_mu_ij = mu_ij * scalingFactor
    
    if viscosityTerm == wp.static(ViscosityTerms.Default.value): # Default to Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        return scaled_mu_ij, h / xi
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992
        return scaled_mu_ij, scalar_t(1.0) * scalingFactor
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        return mu_ij, scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value): # DeltaSPH
        return scaled_mu_ij, h / xi
    else:
        return scaled_mu_ij, h / xi

@wp.func
def compute_bars(
    rho_i : scalar_t, rho_j : scalar_t, rho_bar : scalar_t, 
    c_i : scalar_t, c_j : scalar_t, c_bar : scalar_t,
    h_i : scalar_t, h_j : scalar_t, h_bar : scalar_t,
    viscosityTerm: wp.int32, useJ: bool
):
    use_rho_bar = wp.bool(False)
    use_c_bar = wp.bool(False)
    use_h_bar = wp.bool(False)

    if viscosityTerm == wp.static(ViscosityTerms.Default.value): # Default
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        use_rho_bar = False
        use_c_bar = False
        use_h_bar = False
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992
        use_rho_bar = True
        use_c_bar = False
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value): # DeltaSPH
        use_rho_bar = False
        use_c_bar = True
        use_h_bar = True
    else:        
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True

    rho = scalar_t(scalar_t(0.0))
    c = scalar_t(scalar_t(0.0))
    h = scalar_t(scalar_t(0.0))

    if use_rho_bar:
        rho = rho_bar
    else:
        if useJ:
            rho = rho_j
        else:            
            rho = rho_i    

    if use_c_bar:
        c = c_bar
    else:
        if useJ:
            c = c_j
        else:            
            c = c_i

    if use_h_bar:
        h = h_bar
    else:
        if useJ:
            h = h_j
        else:            
            h = h_i

    return rho, c, h
