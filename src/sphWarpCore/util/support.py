from warp.types import vector, matrix

from sphWarpCore.math.wp_distance import computeDistanceVec
from ..math import safe_sqrt
from typing import Optional, Any, Union, List, Tuple
import numpy as np
import warp as wp

import torch
from ..type_config import scalar_t
from ..enumTypes import SupportScheme
from ..math import vectorNorm_warp

@torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : int, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)


def n_h_to_nH(n_h: float, dim: int) -> float:
    """Converts n_h (particles per smoothing length, per axis -- the
    resolution knob that actually stays comparable across dimensions) into
    the target neighbor count N_h expected by volumeToSupport/
    generateNeighborTestData, using the same volume-ratio convention (a
    particle "cell" of side 1/n_h has volume (1/n_h)**dim; N_h is how many of
    those fit inside the support region, vH = 2 for a 1D segment, pi for a 2D
    disc, 4/3*pi for a 3D ball). A flat, dimension-agnostic target neighbor
    count (e.g. the same literal "55" for 1D/2D/3D) is not comparable across
    dimensions -- for a fixed particle spacing, the same N_h implies a wildly
    different h per dimension, which is why a flat count produces a
    disproportionately large (or small) support radius outside of the
    dimension it happened to be tuned for. n_h=4 is a reasonable default
    resolution across all three dimensions.
    """
    spacing = 1.0 / n_h
    v = spacing ** dim
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4.0 / 3.0) * np.pi)
    return vH / v


@wp.func
def volumeToSupport_warp(volume : scalar_t, targetNeighbors : wp.int32, dim : wp.int32):
    if dim == 1:
        return targetNeighbors * volume / scalar_t(2.0)
    elif dim == 2:
        return safe_sqrt(targetNeighbors * volume / scalar_t(np.pi))
    else:
        return wp.pow(targetNeighbors * volume / scalar_t(np.pi * 3.0 /4.0), scalar_t(1.0/3.0))



@wp.func
def computePairwiseSupport(hx: scalar_t, hy: scalar_t, mode: wp.uint32):
    if mode == wp.static(SupportScheme.Gather.value): # gather
        return hx
    elif mode == wp.static(SupportScheme.Scatter.value): # scatter
        return hy
    elif mode == wp.static(SupportScheme.MeanSymmetric.value): # meanSymmetric
        return (hx + hy) / scalar_t(2.0)
    else:
        return wp.max(hx, hy)
    

@wp.func
def isInSupport(
    xi: vector(dtype=scalar_t, length=3), xj: vector(dtype=scalar_t, length=3), 
    hi: scalar_t, hj: scalar_t, mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool), minDomain: wp.array(dtype = scalar_t), maxDomain: wp.array(dtype = scalar_t)
    ):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    r = vectorNorm_warp(xij)
    q = r / hij
    return q <= scalar_t(1.0)