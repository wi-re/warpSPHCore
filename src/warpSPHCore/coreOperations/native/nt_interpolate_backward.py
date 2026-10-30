"""Torch backward orchestration for the native interpolate operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('interpolate', inputs, properties)
