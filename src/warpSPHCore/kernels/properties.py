from typing import Any
from ..type_config import *
import warp as wp
import numpy as np
from ..math import computeDistanceVec
from ..type_config import scalar_t, dim_t

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