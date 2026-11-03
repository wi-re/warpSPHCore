"""Torch backward orchestration for the native laplacian operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('laplacian', inputs, properties)
