from ..type_config import *
from typing import Any
import warp as wp
from .wp_vec1 import *
from warp import mat22f, mat33f, mat22h, mat33h, mat22d, mat33d
from warp import vec2f, vec3f, vec2h, vec3h, vec2d, vec3d, vec3i, vec2i


@wp.func
def zero_like(
    input: Any, # type: ignore
):
    return type(input)() * scalar_t(0.0)
@wp.func
def zero_like(
    input: wp.float32
):
    return wp.float32(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float32) # type: ignore
):
    return type(input)(0.0) * 0.0
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float32) # type: ignore
):
    return type(input)() * 0.0
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float32) # type: ignore
):
    return wp.float32(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = Any) # type: ignore
):
    return type(input[0])() * 0.0
@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float32)) # type: ignore
):
    return mat11f()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float32)) # type: ignore
):
    return mat22f()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float32)) # type: ignore
):
    return mat33f()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float32)) # type: ignore
):
    return vec1f(0.0) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float32)) # type: ignore
):
    return vec2f(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float32)) # type: ignore
):
    return vec3f(0.0)

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.int32)) # type: ignore
):
    return vector(length=1, dtype=wp.int32)(0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.int32)) # type: ignore
):
    return vec2i(0)
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.int32)) # type: ignore
):
    return vec3i(0)


@wp.func
def zero_like_warp(
    input: Any, # type: ignore
):
    return zero_like(input)


# Half precision versions
@wp.func
def zero_like(
    input: wp.float16
):
    return wp.float16(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float16) # type: ignore
):
    return type(input)(wp.float16(0.0)) * wp.float16(0.0)
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float16) # type: ignore
):
    return type(input)() * wp.float16(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float16) # type: ignore
):
    return wp.float16(0.0)

@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float16)) # type: ignore
):
    return mat11h()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float16)) # type: ignore
):
    return mat22h()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float16)) # type: ignore
):
    return mat33h()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float16)) # type: ignore
):
    return vec1h(wp.float16(0.0)) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float16)) # type: ignore
):
    return vec2h(wp.float16(0.0))
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float16)) # type: ignore
):
    return vec3h(wp.float16(0.0))

# Double precision versions
@wp.func
def zero_like(
    input: wp.float64
):
    return wp.float64(0.0)
@wp.func
def zero_like(
    input: vector(length=Any, dtype=wp.float64) # type: ignore
):
    return type(input)(wp.float64(0.0)) * wp.float64(0.0)
@wp.func
def zero_like(
    input: matrix(shape=(Any, Any), dtype=wp.float64) # type: ignore
):
    return type(input)() * wp.float64(0.0)
@wp.func
def zero_like(
    input: wp.array(dtype = wp.float64) # type: ignore
):
    return wp.float64(0.0)

@wp.func 
def zero_like(
    input: wp.array(dtype=matrix(shape=(1, 1), dtype=wp.float64)) # type: ignore
):
    return mat11d()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(2, 2), dtype=wp.float64)) # type: ignore
):
    return mat22d()
@wp.func
def zero_like(
    input: wp.array(dtype=matrix(shape=(3, 3), dtype=wp.float64)) # type: ignore
):
    return mat33d()

@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=1, dtype=wp.float64)) # type: ignore
):
    return vec1d(wp.float64(0.0)) 
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=2, dtype=wp.float64)) # type: ignore
):
    return vec2d(wp.float64(0.0))
@wp.func
def zero_like(
    input: wp.array(dtype=vector(length=3, dtype=wp.float64)) # type: ignore
):
    return vec3d(wp.float64(0.0))   