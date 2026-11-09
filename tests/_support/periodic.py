"""Exact rational geometry oracle; independent of all production wrappers."""
from fractions import Fraction
import torch


def cases(dtype, dim, device='cpu'):
    a = torch.tensor(.25, dtype=dtype)
    b = torch.nextafter(a, torch.full_like(a, float('inf')))
    half = torch.tensor(.5, dtype=dtype)
    below = torch.nextafter(half, torch.zeros_like(half))
    tiny = torch.finfo(dtype).eps / 4
    # Local ulps, both seam directions, ties/near-ties, multiple image shifts,
    # and disparate coordinates whose direct subtraction loses .25.
    x = torch.tensor([b, a, tiny, -tiny, .99, .01, .5, -.5, below,
                      -below, 4.25, -7.75, 2.**55, -(2.**55)], dtype=dtype)
    y = torch.tensor([a, b, 0, 0, .01, .99, 0, 0, 0, 0,
                      -.125, .125, .25, .25], dtype=dtype)
    # Different physical box origins must not alter pair displacement.
    length = torch.tensor([1., 2., .75][:dim], dtype=dtype)
    x = x[:, None].expand(-1, dim).clone()
    y = y[:, None].expand(-1, dim).clone()
    return x.to(device), y.to(device), length.to(device)


def exact(x, y, periodic, length):
    out = torch.empty_like(x)
    for i in range(len(x)):
        for d in range(x.shape[1]):
            value = Fraction(float(x[i, d])) - Fraction(float(y[i, d]))
            if periodic[d]:
                span = Fraction(float(length[d]))
                value -= ((value / span + Fraction(1, 2)) // 1) * span
            out[i, d] = float(value)
    return out
