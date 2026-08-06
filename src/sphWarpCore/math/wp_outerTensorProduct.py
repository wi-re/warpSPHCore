import warp as wp
from warp.types import vector, matrix
from ..types import *
from typing import Optional, Any, Union, List, Tuple

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = scalar_t, length=Any), # type: ignore
    vec : vector(dtype = scalar_t, length=3), # type: ignore
    out : vector(dtype = scalar_t, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    dim = wp.int32(3) # hardcoded as this is only implemented for 3D vectors currently.
    
    # the output is stored as a flattened vector, so we need to compute the correct index for accumulation
    res = type(out)(scalar(0.0))
    for i in range(flatInputShape): # loop over elements of input tensor
        for j in range(dim): # loop over dimensions of output gradient
            outIndex = j * flatInputShape + i # compute flattened index for output
            res[outIndex] += vec[j] * tensor[i] # accumulate outer product into output
            
    return res

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = scalar_t, length=Any), # type: ignore
    vec : vector(dtype = scalar_t, length=2), # type: ignore
    out : vector(dtype = scalar_t, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    dim = wp.int32(2) # hardcoded as this is only implemented for 2D vectors currently.
    
    # the output is stored as a flattened vector, so we need to compute the correct index for accumulation
    res = type(out)(scalar_t(0.0))
    for i in range(flatInputShape): # loop over elements of input tensor
        for j in range(dim): # loop over dimensions of output gradient
            outIndex = j  + i * dim# compute flattened index for output
            res[outIndex] += vec[j] * tensor[i] # accumulate outer product into output
            
    return res

@wp.func
def outerTensorProduct(
    tensor: vector(dtype = scalar_t, length=Any), # type: ignore
    vec : vector(dtype = scalar_t, length=1), # type: ignore
    out : vector(dtype = scalar_t, length=Any), # type: ignore
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32
):
    # for 1D vectors the outer product is just a scalar multiplication, so we can skip the indexing logic
    res = type(out)(scalar_t(0.0))
    for i in range(flatInputShape):
        res[i] += vec[0] * tensor[i]
    return res
    