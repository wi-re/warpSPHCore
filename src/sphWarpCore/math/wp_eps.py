import warp as wp
from ..type_config import scalar_t
from typing import Any
@wp.func
def get_epsilon(val: Any):
    return scalar_t(1e-7)

@wp.func
def get_epsilon(val: wp.float32) -> wp.float32:
    return wp.float32(1e-7)

@wp.func
def get_epsilon(val: wp.float64) -> wp.float64:
    return wp.float64(1e-15)

@wp.func
def get_epsilon(val: wp.float16) -> wp.float16:
    return wp.float16(1e-3)