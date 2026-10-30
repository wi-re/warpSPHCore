"""Torch backward orchestration for the native gradient operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('gradient', inputs, properties)
