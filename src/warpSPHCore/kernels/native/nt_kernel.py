"""Tensor-facing native kernel evaluation and explicit derivative primitives."""
from typing import NamedTuple
import torch
from ...enumTypes import KernelFunctions
from ...native._checks import _forward_only


class KernelEvaluation(NamedTuple):
    value: torch.Tensor
    gradient: torch.Tensor
    hessian: torch.Tensor
    d_h: torch.Tensor
    gradient_d_h: torch.Tensor
    laplacian: torch.Tensor
    radial: torch.Tensor  # f, df/dq, d2f/dq2, d3f/dq3, C_d, scale, packing


def evaluateKernel(displacement, supports, kernel=KernelFunctions.Wendland2):
    """Evaluate [N,D] displacements and [N] support radii, D=1/2/3.

    This explicit primitive is forward-only in Torch autograd. SPH operators
    reuse its analytic derivatives in their manual first-order adjoints.
    """
    from ...native._build import build
    if not isinstance(kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    _forward_only(displacement, supports)
    return KernelEvaluation(*build().kernel_forward(displacement, supports, kernel.value))
