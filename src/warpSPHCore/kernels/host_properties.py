"""Host packing/support constants for native setup and timestep selection.

Values match the existing Warp leaf functions, including their configured scalar
precision. These are support properties, not density-normalization corrections.
"""
import numpy as np
import torch
from ..type_config import get_torch_precision

# KernelFunctions value -> (scales in 1D/2D/3D, packing ratio).
# Keep aligned with kernelFunctions/*_kernelScale and *_packingRatio.
_PROPERTIES = {20: ((1.0, 1.0, 1.0), 1.0),
 30: ((1.732051, 1.778002, 1.825742), (1.292, 1.0175)),
 31: ((1.936492, 1.977173, 2.018932), 1.203),
 32: ((2.121321, 2.158131, 2.195775), (1.595, 1.1425)),
 34: ((2.291288, 2.32517, 2.3597), (1.595, 1.1425)),
 33: ((2.44949, 2.481044, 2.513123), (1.595, 1.1425)),
 35: ((2.224323, 2.192463, 2.18968441), 2.158),
 36: ((8.0, 8.0, 8.0), 1.337),
 0: ((1.620185, 1.897367, 1.936492), 1.487),
 1: ((1.936492, 2.171239, 2.20794), 1.643),
 2: ((2.20794, 2.41523, 2.44949), 1.866),
 21: ((1.0, 1.0, 1.0), 1.0),
 40: ((1.0, 1.0, 1.0), 1.0),
 41: ((1.0, 1.0, 1.0), 1.0),
 42: ((1.0, 1.0, 1.0), 1.0)}


def _scalar(value):
    dtype = {torch.float16: np.float16, torch.float32: np.float32,
             torch.float64: np.float64}[get_torch_precision()]
    return dtype(value)


def sphKernelScale(kernel, dim):
    properties = _PROPERTIES.get(kernel)
    if properties is None:
        return _scalar(float('nan'))
    # Warp host calls return the leaf scalar's original Python value here.
    return properties[0][0 if dim == 1 else 1 if dim == 2 else 2]


def sphKernel_xi(kernel, dim):
    properties = _PROPERTIES.get(kernel)
    if properties is None:
        return _scalar(float('nan'))
    packing = properties[1]
    if isinstance(packing, tuple):
        # Keep the Warp leaf's multiplication order and intermediate rounding.
        packing = _scalar(_scalar(packing[0]) * _scalar(packing[1]))
    return _scalar(_scalar(packing) * _scalar(sphKernelScale(kernel, dim)))
