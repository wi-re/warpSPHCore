"""Torch backward orchestration for the native covariance operator."""
from ...native._autograd import apply_operator


def apply(inputs, properties):
    return apply_operator('covariance', inputs, properties)
