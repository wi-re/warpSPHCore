"""Host/Torch support conversions shared by both execution backends."""
import math
import numpy as np
import torch

def volumeToSupport_tensor(volume : torch.Tensor, targetNeighbors : int, dim : int) -> torch.Tensor:
    """Tensor-aware counterpart of volumeToSupport (uses torch.sqrt instead of
    math.sqrt so it works elementwise on a tensor of per-particle volumes)."""
    if dim == 1:
        return targetNeighbors * volume / 2
    elif dim == 2:
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        return (targetNeighbors * volume / np.pi * 3 / 4) ** (1 / 3)


def volumeToSupport(volume, targetNeighbors : int, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float or torch.Tensor): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    float or torch.Tensor: The support radius.
    """
    if isinstance(volume, torch.Tensor):
        return volumeToSupport_tensor(volume, targetNeighbors, dim)
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return math.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return (targetNeighbors * volume / np.pi * 3 / 4) ** (1 / 3)


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

def nH_to_n_h(nH: float, dim: int) -> float:
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    v = vH / nH
    return (1 / v)**(1/dim)
