"""Torch backward orchestration for the native density operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('density', inputs, properties)
