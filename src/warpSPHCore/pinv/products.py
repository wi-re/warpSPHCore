"""Real square Moore-Penrose product rules matching Torch's selected-rank AD.

These general matrix rules retain nonsymmetric perturbations and do not
require differentiating singular vectors. Rank/cutoff decisions are discrete.
For a selected spectral function of symmetric matrices (including nonzero
rejected eigenvalues), use ``symmetric._symmetric_pinv_jvp`` instead.
"""
import torch


def pinv_jvp(matrix,inverse,tangent):
    identity=torch.eye(matrix.shape[-1],device=matrix.device,dtype=matrix.dtype)
    return (-inverse@tangent@inverse
            +(inverse@inverse.mT)@tangent.mT@(identity-matrix@inverse)
            +(identity-inverse@matrix)@tangent.mT@(inverse.mT@inverse))


def pinv_vjp(matrix,inverse,cotangent):
    identity=torch.eye(matrix.shape[-1],device=matrix.device,dtype=matrix.dtype)
    return (-inverse.mT@cotangent@inverse.mT
            +(identity-matrix@inverse)@cotangent.mT@(inverse@inverse.mT)
            +(inverse.mT@inverse)@cotangent.mT@(identity-inverse@matrix))
