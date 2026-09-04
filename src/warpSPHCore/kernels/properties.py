from typing import Any
from ..type_config import *
import warp as wp
import numpy as np
from ..math import computeDistanceVec
from ..type_config import scalar_t, dim_t

from ..dataTypes.kernelState_t import kernelState
from .kernelFunctions import *
from .eval_kernel import *

@wp.func
def sphKernelScale(kernel: wp.int32, dim: wp.int32):
    return eval_kernelScale(kernel, dim)

@wp.func
def sphKernelC_d(kernel: wp.int32, dim: wp.int32):
    return eval_C_d(dim, kernel)

@wp.func
def sphKernelN_H(kernel: wp.int32, dim: wp.int32):
    packingRatio = eval_packing(kernel)
    fac = scalar_t(2.0) if dim == 1 else (np.pi if dim == 2 else 4 * np.pi / 3)
    N = fac * packingRatio**dim * eval_kernelScale(kernel, dim)**dim
    return N

@wp.func
def sphKernel_xi(kernel: wp.int32, dim: wp.int32):
    return eval_packing(kernel) * eval_kernelScale(kernel, dim)


@wp.func
def resolveNormalization(kernelProperties: kernelState):
    """The scalar every kernel value and derivative is multiplied by.

    `1.0` unless the lattice-normalisation correction is switched on, in which
    case `1 / L(kernel, dim, n_h)` -- see `kernelState` and
    `warpSPHCore.util.latticeDensity`. Applied at the public boundary of the
    kernel functions rather than at each of the 13 `eval_C_d` sites: every one
    of those functions is linear in `C_d`, so scaling the return value is
    identical arithmetic and cannot be applied to the value while being
    forgotten for the gradient. Deliberately NOT applied to `sphKernelScale`,
    `sphKernel_xi` or `sphKernelN_H`, which are packing/support properties
    rather than kernel values.
    """
    if kernelProperties.calibrateNormalization:
        return kernelProperties.normalizationCoefficient
    return scalar_t(1.0)
