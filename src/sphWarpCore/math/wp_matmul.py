import warp as wp
from warp.types import vector, matrix
from ..types import *
from typing import Optional, Any, Union, List, Tuple
         
@wp.func
def matmul(
    mat: matrix(shape=(1, 1), dtype=scalar_t), # type: ignore
    vec: vector(dtype = scalar_t, length=1), # type: ignore
):
    numRows = 1
    numCols = 1
    
    res = type(vec)(scalar_t(0.0))
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res

@wp.func
def matmul(
    mat: matrix(shape=(2, 2), dtype=scalar_t), # type: ignore
    vec: vector(dtype = scalar_t, length=2), # type: ignore
):
    numRows = 2
    numCols = 2
    
    res = type(vec)(scalar_t(0.0))
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res

@wp.func
def matmul(
    mat: matrix(shape=(3, 3), dtype=scalar_t), # type: ignore
    vec: vector(dtype = scalar_t, length=3), # type: ignore
):
    numRows = 3
    numCols = 3
    
    res = type(vec)(scalar_t(0.0))
    for i in range(numRows):
        for j in range(numCols):
            res[i] += mat[i, j] * vec[j]

    return res
