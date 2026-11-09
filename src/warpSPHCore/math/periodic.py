"""Tensor minimum-image arithmetic, independent of the execution backend.

Displacements use [-L/2, L/2); +L/2 maps to -L/2. Local differences retain
all their stored bits. This is distinct from wrapping positions for grid cells.
Domain extents must be positive; flags and extents broadcast over pair axes.
"""
import torch


def minimum_image_displacement(delta, periodic, length):
    """Wrap displacement tensors without recentering small differences."""
    half = length / 2
    reduced = torch.fmod(delta, length)
    wrapped = torch.where(reduced >= half, reduced - length,
                          torch.where(reduced < -half, reduced + length, reduced))
    return torch.where(periodic.to(device=delta.device, dtype=torch.bool), wrapped, delta)


def periodic_difference(x, y, periodic, length):
    """Subtract positions, retaining local differences and arbitrary images.

    Reducing each nonlocal coordinate separately avoids losing a small y when
    a much larger x would round it away in x-y. No device values reach the host.
    """
    delta = x - y
    half = length / 2
    reduced = torch.fmod(x, length) - torch.fmod(y, length)
    wrapped = minimum_image_displacement(reduced, periodic, length)
    needs_image = periodic.to(device=delta.device, dtype=torch.bool) & ((delta >= half) | (delta < -half))
    return torch.where(needs_image, wrapped, delta)


def periodic_positions(positions, periodic, lower, upper):
    """Wrap positions for indexing, preserving coordinates already in bounds.

    A subtract-origin/add-origin round trip can lose bits even if no image is
    needed. Keep that operation out of the returned in-domain coordinates.
    """
    lower = torch.as_tensor(lower, dtype=positions.dtype, device=positions.device)
    upper = torch.as_tensor(upper, dtype=positions.dtype, device=positions.device)
    periodic = torch.as_tensor(periodic, dtype=torch.bool, device=positions.device)
    wrapped = torch.remainder(positions - lower, upper - lower) + lower
    needs_image = periodic.to(device=positions.device, dtype=torch.bool) & ((positions < lower) | (positions >= upper))
    return torch.where(needs_image, wrapped, positions)
