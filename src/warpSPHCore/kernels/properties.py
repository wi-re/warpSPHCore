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

# Unit-ball volumes per dimension. Kept as module-level Python floats and
# wrapped as single scalar_t leaves: warp traces Python constant BinOps as
# int32/float32 mixes that fail to compile under a float64 build.
_NU1_VOLUME = 2.0
_NU2_VOLUME = float(np.pi)
_NU3_VOLUME = float(4 * np.pi / 3.0)

@wp.func
def sphKernelN_H(kernel: wp.int32, dim: wp.int32):
    packingRatio = eval_packing(kernel)
    fac = scalar_t(_NU1_VOLUME) if dim == 1 else (scalar_t(_NU2_VOLUME) if dim == 2 else scalar_t(_NU3_VOLUME))
    # wp.pow with scalar_t exponents: a runtime `**dim` (int32) has no
    # float64 overload (only the float32 build ever compiled this function
    # before the kernel audit, which runs float64).
    N = fac * wp.pow(packingRatio, scalar_t(dim)) * wp.pow(eval_kernelScale(kernel, dim), scalar_t(dim))
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
