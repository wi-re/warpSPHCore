import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.autograd import *

from ..dataTypes import *
from sphWarpCore.math import *
from sphWarpCore.kernels import *
from sphWarpCore.autograd.cache import getCachedDummyTensor
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.autograd.arg_check import *
from typing import Optional

# mat11f/vec1f (etc.) are named wp.matrix/wp.vector subclasses, one per precision
# (see math/wp_vec1.py) -- Warp's generic matrix(shape=(1,1), dtype=...)/vector(length=1,
# dtype=...) factory types work fine as *parameter* types (resolved once at import time
# via scalar_t, same as every other kernel in this codebase) but wp.zeros(..., dtype=...)
# (used by launch_kernel to allocate this function's outputs) needs a concrete, named
# type it can hash/cache -- hence picking the matching named subclass here rather than
# passing the generic factory type through, and picking the one that matches scalar_t
# rather than hardcoding float32 (which silently mismatches the kernel's actual output
# dtype -- and doesn't even exist as `wp.mat11f`/`wp.vec1f` on the warp module itself,
# only as sphWarpCore's own subclasses -- under any other SPHWARPCORE_PRECISION).
if scalar_t == wp.float32:
    _mat11_t, _vec1_t = mat11f, vec1f
elif scalar_t == wp.float64:
    _mat11_t, _vec1_t = mat11d, vec1d
elif scalar_t == wp.float16:
    _mat11_t, _vec1_t = mat11h, vec1h
else:
    raise ValueError(f"Unsupported scalar type: {scalar_t}")


@wp.func
def pseudoInverse1x1(
    m: matrix(shape = (1, 1), dtype = scalar_t) # type: ignore
):
    if m[0, 0] > scalar_t(1e-10):
        return _mat11_t(scalar_t(1.0) / m[0, 0]), _vec1_t(m[0, 0])
    else:
        return _mat11_t(scalar_t(0.0)), _vec1_t(m[0, 0])


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
        launch_kernel, pseudoInverse1x1Kernel, outputSize, (_mat11_t, _vec1_t),
        m
    )

    return warp_result[0], warp_result[1]