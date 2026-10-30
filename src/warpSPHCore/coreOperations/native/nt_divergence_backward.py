"""Torch backward orchestration for the native divergence operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('divergence', inputs, properties)
