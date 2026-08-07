import warp as wp
from warp.types import vector, matrix
from ..type_config import *
from .wp_sqrt import *
from typing import Optional, Any, Union, List, Tuple

@wp.func
def safe_sqrt(x: scalar_t):
    return wp.sqrt(x)
@wp.func_grad(safe_sqrt)
def adj_safe_sqrt(x: scalar_t, adj_ret: scalar_t):
    if x > 0.0:
        wp.adjoint[x] += scalar(1.0) / (scalar(2.0) * wp.sqrt(x)) * adj_ret
