

# import warp as wp
# from warp.types import vector, matrix
# # from wp_tensor import tensor
# from typing import Any
# import torch
# from sphWarpCore.utils.wp_autograd import *
# from sphWarpCore.radiusSearch.radius_util import convertModeToUint

# from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
# from sphWarpCore.mathutil.wp_math import *
# from sphWarpCore.kernels.wp_kernel import *
# from sphWarpCore.utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
# from torch.profiler import profile, record_function, ProfilerActivity
# from sphWarpCore.enumTypes import *
# from sphWarpCore.utils.arg_check import *
# from typing import Optional

# @wp.func
# def matmul(
#     A: matrix(shape = (3, 3), dtype = wp.float32), # type: ignore
#     B: matrix(shape = (3, 3), dtype = wp.float32), # type: ignore
# ):
#     out = wp.mat33f() * 0.0
#     for i in range(3):
#         for j in range(3):
#             out[i,j] = A[i,0] * B[0,j] + A[i,1] * B[1,j] + A[i,2] * B[2,j]
#     return out
# @wp.func
# def fmaf(
#     a: float, b: float, c: float
# ):
#     return a * b + c
# )

# @wp.func
# def approximateGivensQuaternion(
#     A: matrix(shape = (3, 3), dtype = wp.float32), # type: ignore
# ):
#     ch, sh = 2.f * (A[0,0] - A[1,1]), A.m[1,0]
#     b = _gamma * sh * sh < ch * ch
#     w = rsqrt(fmaf(ch, ch, sh * sh))
#     if w != w:
#         b = 0.0

# @wp.func
# def jacobiConjugation(
#     x: wp.int32, y: wp.int32, z: wp.int32,
#     S: matrix(shape = (3, 3), dtype = wp.float32), # type: ignore
#     q: vector(length = 4, dtype = wp.float32) # type: ignore
# )

# @wp.func
# def jacobiEigenAnalysis(
#     S: matrix(shape = (3, 3), dtype = wp.float32) # type: ignore
# ):
#     q = wp.vec4f()
#     for i in range(JACOBI_STEPS):
#         jacobiConjugation(0, 1, 2, S, q)
#         jacobiConjugation(1, 2, 0, S, q)
#         jacobiConjugation(2, 0, 1, S, q)
#     return q


# @wp.func
# def svd3x3(
#     M: wp.mat33f # type: ignore
# ):
#     V = jacobiEigenAnalysis(matmul(M, wp.transpose(M)))

#     B = matmul(A, V)
#     # sortSingularValues(B, V)
#     # qr = QRDecomposition(B)

#     return qr[0], qr[1], V # returns Q, R, V



# @wp.kernel
# def pseudoInverse3x3Kernel(
#     input_matrices: wp.array(dtype = matrix(shape = (3, 3), dtype = wp.float32)), # type: ignore
#     output_matrices: wp.array(dtype = matrix(shape = (3, 3), dtype = wp.float32)), # type: ignore
#     output_evals: wp.array(dtype = vector(length = 3, dtype = wp.float32)), # type: ignore
# ):
#     i = wp.tid()
#     if i >= input_matrices.shape[0]:
#         return
#     S, V, D = svd3x3(input_matrices[i])
#     output_matrices[i] = out_m
#     output_evals[i] = out_ev

# def pinv3x3(m: torch.Tensor) -> torch.Tensor:
#     outputSize = m.shape[0]


#     warp_result = warpWrapper(
#         launch_kernel, pseudoInverse3x3Kernel, outputSize, (wp.mat33f, wp.vec3f),
#         m
#     )

#     return warp_result[0], warp_result[1]