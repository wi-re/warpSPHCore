import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.utils.wp_autograd import *
from sphWarpCore.radiusSearch.radius_util import convertModeToUint

from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.mathutil.wp_math import *
from sphWarpCore.kernels.wp_kernel import *
from sphWarpCore.utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.utils.arg_check import *
from typing import Optional


@wp.func
def square(
    x: float
):
    return x * x

@wp.func
def clamp(
    x: float, minVal: float, maxVal: float
):
    return max(min(x, maxVal), minVal)

@wp.func
def sign(
    x: float
):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

@wp.func
def diag_embed(
    x: vector(length = 2, dtype = wp.float32), # type: ignore
):
    out = wp.mat22f() * 0.0
    out[0,0] = x[0]
    out[1,1] = x[1]
    return out

@wp.func
def matmul(
    A: matrix(shape = (2, 2), dtype = wp.float32), # type: ignore
    B: matrix(shape = (2, 2), dtype = wp.float32), # type: ignore
):
    out = wp.mat22f() * 0.0
    for i in range(2):
        for j in range(2):
            out[i,j] = A[i,0] * B[0,j] + A[i,1] * B[1,j]
    return out

@wp.func
def pseudoInverse2x2(
    M: matrix(shape = (2, 2), dtype = wp.float32) # type: ignore
):
    a = M[0,0]
    b = M[0,1]
    c = M[1,0]
    d = M[1,1]

    theta = 0.5 * wp.atan2(2.0 * a * c + 2.0 * b * d, square(a) + square(b) - square(c) - square(d))
    cosTheta = wp.cos(theta)
    sinTheta = wp.sin(theta)
    U = type(M)() * 0.0
    U[0,0] = cosTheta
    U[0,1] = - sinTheta
    U[1,0] = sinTheta
    U[1,1] = cosTheta

    S1 = square(a) + square(b) + square(c) + square(d)
    S2 = safe_sqrt(square(square(a) + square(b) - square(c) - square(d)) + 4.0 * square(a * c + b *d))

    o1 = safe_sqrt((S1 + S2) / 2.0)
    o2 = safe_sqrt(max(S1 - S2 + 1e-7, 1e-7) / 2.0)

    phi = 0.5 * wp.atan2(2.0 * a * b + 2.0 * c * d, square(a) - square(b) + square(c) - square(d))
    cosPhi = wp.cos(phi)
    sinPhi = wp.sin(phi)
    s11 = sign((a * cosTheta + c * sinTheta) * cosPhi + ( b * cosTheta + d * sinTheta) * sinPhi)
    s22 = sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)

    V = type(M)() * 0.0
    V[0,0] = cosPhi * s11
    V[0,1] = - sinPhi * s22
    V[1,0] = sinPhi * s11
    V[1,1] = cosPhi * s22

    o1_1 = wp.float32(0.0)
    o2_1 = wp.float32(0.0)

    if wp.abs(o1) > 1e-5:
        o1_1 = 1.0 / o1
    if wp.abs(o2) > 1e-5:
        o2_1 = 1.0 / o2

    o = wp.vec2f(o1_1, o2_1)

    S_1 = diag_embed(o)

    eigVals = wp.vec2f(o1, o2) if wp.abs(o2) < wp.abs(o1) else wp.vec2f(o2, o1)

    return matmul(matmul(V, S_1), wp.transpose(U)), eigVals
    
    

@wp.kernel
def pseudoInverse2x2Kernel(
    input_matrices: wp.array(dtype = matrix(shape = (2, 2), dtype = wp.float32)), # type: ignore
    output_matrices: wp.array(dtype = matrix(shape = (2, 2), dtype = wp.float32)), # type: ignore
    output_evals: wp.array(dtype = vector(length = 2, dtype = wp.float32)), # type: ignore
):
    i = wp.tid()
    if i >= input_matrices.shape[0]:
        return
    out_m, out_ev = pseudoInverse2x2(input_matrices[i])
    output_matrices[i] = out_m
    output_evals[i] = out_ev

def pinv2x2(m: torch.Tensor) -> torch.Tensor:
    outputSize = m.shape[0]


    warp_result = warpWrapper(
        launch_kernel, pseudoInverse2x2Kernel, outputSize, (wp.mat22f, wp.vec2f),
        m
    )

    return warp_result[0], warp_result[1]