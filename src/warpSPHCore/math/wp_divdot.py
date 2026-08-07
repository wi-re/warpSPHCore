from ..type_config import *
import warp as wp
from warp.types import vector, matrix
from typing import Any


# Unified Divergence kernel: same design as the unified Gradient/Interpolate/Density
# kernels (see warpier_core.md's "Working Prototype -> Production" section). Divergence
# shares essentially all of Gradient's correction-path machinery (CRK, grad-h, volume,
# renormalization) and differs only in the neighbor-level contraction: `divergenceProduct`
# (contracts the input's last/first flattened axis against the kernel gradient) in place
# of Gradient's `outerTensorProduct` (which appends a new axis instead).


# In dot mode we compute torch.einsum('nd..., nd -> n...', q, k)
# Otherwise we compute torch.einsum('n...d, nd -> n...', q, k)
# the inputs are the flattened versions of the original tensors, i.e.,
# if q initially was of shape [n, d, d, d] it is now [n, d^3]
# The output is always of shape [n, d^(N-1)] where N is the rank of the input tensor q. So if q was originally [n, d, d, d] the output will be [n, d^2]
# The kernelGradient is always of shape [n, d]
# this also allows us to overload the function based on d! We can then use the numDims parameter to do the correct indexing inside the kernel without having to write separate kernels for different dimensions.
@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=3), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.

    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]

    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=2), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.

    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]

    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = scalar_t, length=Any),  # type: ignore
    kernelGradient: vector(dtype = scalar_t, length=1), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(scalar_t(0.0))
    # in 1D the divergence product is just a simple multiplication
    res[0] = fij[0] * kernelGradient[0]
    return res

