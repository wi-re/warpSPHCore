"""Torch backward orchestration for the native curl operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('curl', inputs, properties)
