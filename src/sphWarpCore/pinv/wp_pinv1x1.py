import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.util.wp_autograd import *

from ..dataTypes import *
from sphWarpCore.math import *
from sphWarpCore.kernels import *
from sphWarpCore.util.wp_util import getCachedDummyTensor
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.util.arg_check import *
from typing import Optional


@wp.func
def pseudoInverse1x1(
    m: matrix(shape = (1, 1), dtype = scalar_t) # type: ignore
):
    if m[0, 0] > 1e-10:
        return wp.mat11f(1.0 / m[0, 0]), wp.vec1f(m[0, 0])
    else:
        return wp.mat11f(0.0), wp.vec1f(m[0, 0])
    

@wp.kernel
def pseudoInverse1x1Kernel(
    input_matrices: wp.array(dtype = matrix(shape = (1, 1), dtype = scalar_t)), # type: ignore
    output_matrices: wp.array(dtype = matrix(shape = (1, 1), dtype = scalar_t)), # type: ignore
    output_evals: wp.array(dtype = vector(length = 1, dtype = scalar_t)), # type: ignore
):
    i = wp.tid()
    if i >= input_matrices.shape[0]:
        return
    out_m, out_ev = pseudoInverse1x1(input_matrices[i])
    output_matrices[i] = out_m
    output_evals[i] = out_ev

def pinv1x1(m: torch.Tensor) -> torch.Tensor:
    outputSize = m.shape[0]


    warp_result = warpWrapper(
        launch_kernel, pseudoInverse1x1Kernel, outputSize, (wp.mat11f, wp.vec1f),
        m
    )

    return warp_result[0], warp_result[1]