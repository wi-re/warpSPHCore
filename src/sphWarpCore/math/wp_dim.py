import warp as wp
from warp.types import vector, matrix
from ..types import *

@wp.func
def get_dim(v: vector(length=1, dtype = scalar_t)): # type: ignore
    return 1
@wp.func
def get_dim(v: vector(length=2, dtype = scalar_t)): # type: ignore
    return 2
@wp.func
def get_dim(v: vector(length=3, dtype = scalar_t)): # type: ignore
    return 3

@wp.func
def get_dim(v: wp.array(dtype = vector(length=1, dtype = scalar_t))): # type: ignore
    return 1
@wp.func
def get_dim(v: wp.array(dtype = vector(length=2, dtype = scalar_t))): # type: ignore
    return 2
@wp.func
def get_dim(v: wp.array(dtype = vector(length=3, dtype = scalar_t))): # type: ignore
    return 3

@wp.func
def get_dim(v: matrix(shape=(1,1), dtype = scalar_t)): # type: ignore
    return 1
@wp.func
def get_dim(v: matrix(shape=(2,2), dtype = scalar_t)): # type: ignore
    return 2    
@wp.func
def get_dim(v: matrix(shape=(3,3), dtype = scalar_t)): # type: ignore
    return 3
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(1,1), dtype = scalar_t))): # type: ignore
    return 1    
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(2,2), dtype = scalar_t))): # type: ignore
    return 2
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(3,3), dtype = scalar_t))): # type: ignore
    return 3
