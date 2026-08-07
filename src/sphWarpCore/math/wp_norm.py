import warp as wp
from ..type_config import vec_t, scalar_t
from .wp_sqrt import safe_sqrt
from .wp_eps import get_epsilon
from warp.types import vector, matrix
from .wp_eye import warp_eye
from .wp_pow import iPow
from typing import Any

@wp.func
def norm_warp(input: vec_t):
    return safe_sqrt(wp.dot(input, input))

@wp.func 
def norm_grad_warp(input: vec_t):
    length = norm_warp(input)
    float_eps = get_epsilon(length)
    length = wp.max(length, float_eps)
    return input / length

@wp.func
def norm_hess_warp(input: vector(dtype=scalar_t, length=Any)):
    eps = get_epsilon(input[0])
    r = norm_warp(input) + eps

    outerProd = wp.outer(input, input)
    diagTerm = warp_eye(input) * (iPow(r, 2) + iPow(eps, 3))

    tensor = scalar_t(1.0)/(iPow(r, 3) + iPow(eps, 3)) * (diagTerm - outerProd)

    return tensor