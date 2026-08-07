
from ..type_config import *
import warp as wp
from warp.types import vector, matrix
from typing import Any


@wp.func
def positiveDotProduct(
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    fq_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    f_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    dim: wp.int32
):
    dot = scalar_t(scalar_t(0.0))
    for d in range(dim):
        dot += x_ij[d] * fq_ij[d]

    result = type(f_ij)(scalar_t(0.0))
    if dot >= scalar_t(0.0):
        for d in range(dim):
            result[d] = f_ij[d]
    return result
