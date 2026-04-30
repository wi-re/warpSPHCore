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


@wp.func
def compute_mu_ij(
    ux_ij: wp.float32, r_ij: wp.float32, h: wp.float32, viscosityTerm: wp.int32, xi: wp.float32
):
    mu_ij = ux_ij / (r_ij + 1e-14 * h) # Always start with this as the base

    scalingFactor = h / xi / (r_ij + 1e-14 * h)
    scaled_mu_ij = mu_ij * scalingFactor
    
    if viscosityTerm == 0: # Default to Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 1: # MonaghanGingold1983
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 2: # Cleary1998
        return mu_ij, r_ij
    elif viscosityTerm == 3: # Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 4: # Monaghan1997a
        return mu_ij, r_ij
    elif viscosityTerm == 5: # Monaghan1997b
        return mu_ij, r_ij
    elif viscosityTerm == 6: # Dukowicz
        return mu_ij, r_ij
    elif viscosityTerm == 7: # Price2012_98
        return mu_ij, r_ij
    elif viscosityTerm == 8: # Price2012
        return mu_ij, r_ij
    elif viscosityTerm == 9: # Price2008
        return mu_ij, r_ij
    elif viscosityTerm == 10: # Wadsley2008
        return mu_ij, r_ij
    elif viscosityTerm == 11: # DeltaSPH
        return scaled_mu_ij, h / xi
    else:
        return scaled_mu_ij, h / xi

@wp.func
def compute_bars(
    rho_i : wp.float32, rho_j : wp.float32, rho_bar : wp.float32, 
    c_i : wp.float32, c_j : wp.float32, c_bar : wp.float32,
    h_i : wp.float32, h_j : wp.float32, h_bar : wp.float32,
    viscosityTerm: wp.int32, useJ: bool
):
    use_rho_bar = wp.bool(False)
    use_c_bar = wp.bool(False)
    use_h_bar = wp.bool(False)

    if viscosityTerm == 0:
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 1: # MonaghanGingold1983
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 2: # Cleary1998
        use_rho_bar = False
        use_c_bar = False
        use_h_bar = False
    elif viscosityTerm == 3: # Monaghan1992
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 4: # Monaghan1997a
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 5: # Monaghan1997b
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 6: # Dukowicz
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 7: # Price2012_98
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 8: # Price2012    
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 9: # Price2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 10: # Wadsley2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 11: # DeltaSPH
        use_rho_bar = False
        use_c_bar = True
        use_h_bar = True
    else:        
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True

    rho = wp.float32(0.0)
    c = wp.float32(0.0)
    h = wp.float32(0.0)

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
