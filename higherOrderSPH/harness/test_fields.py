#!/usr/bin/env python3
"""Test fields for the convergence harness.

Two families, per the parent plan:

* **Monomials** x^a y^b (z^c) up to a configurable degree (default 4) for
  patch tests -- consistency order is read off which monomial degrees an
  operator reproduces exactly. Scalar monomials plus one linear vector
  field (for vector-gradient probes).
* **Smooth C-infinity fields** for convergence-rate tests: sinusoid modes
  (commensurate with the domain box, so they are also valid on periodic
  domains), a wrapped Gaussian (periodic, built from 3^dim images), and an
  ordinary Gaussian for open domains.

Every field carries its analytic value, gradient and Laplacian, so the
harness needs no numerical differentiation. Fields are pure torch
functions of the (device) position tensor -- no warp, no warpSPHCore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = ["Field", "monomial_fields", "vector_field", "smooth_periodic_fields",
           "smooth_open_fields"]


@dataclass(frozen=True)
class Field:
    name: str
    kind: str                    # "monomial" | "smooth"
    degree: int                  # polynomial degree; -1 for smooth
    is_vector: bool
    value: Callable[[torch.Tensor], torch.Tensor]      # (N,) or (N, dim)
    grad: Callable[[torch.Tensor], torch.Tensor]       # (N, dim) or (N, dim, dim)
    lap: Callable[[torch.Tensor], torch.Tensor] | None  # (N,) or (N, dim); None if not needed


def _monomial(x: torch.Tensor, exps) -> torch.Tensor:
    out = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    for d, e in enumerate(exps):
        if e > 0:
            out = out * x[:, d] ** e
    return out


def _monomial_grad(x: torch.Tensor, exps) -> torch.Tensor:
    N = x.shape[0]
    g = torch.zeros(N, len(exps), dtype=x.dtype, device=x.device)
    for d, e in enumerate(exps):
        if e > 0:
            rest = torch.ones(N, dtype=x.dtype, device=x.device)
            for d2, e2 in enumerate(exps):
                if d2 != d and e2 > 0:
                    rest = rest * x[:, d2] ** e2
            g[:, d] = e * x[:, d] ** (e - 1) * rest
    return g


def _monomial_lap(x: torch.Tensor, exps) -> torch.Tensor:
    N = x.shape[0]
    out = torch.zeros(N, dtype=x.dtype, device=x.device)
    for d, e in enumerate(exps):
        if e >= 2:
            term = e * (e - 1) * x[:, d] ** (e - 2)
            for d2, e2 in enumerate(exps):
                if d2 != d and e2 > 0:
                    term = term * x[:, d2] ** e2
            out = out + term
    return out


def _monomial_name(exps) -> str:
    parts = []
    for sym, e in zip(("x", "y", "z"), exps):
        if e == 1:
            parts.append(sym)
        elif e > 1:
            parts.append(f"{sym}^{e}")
    return "".join(parts) if parts else "1"


def _compositions(total: int, dim: int) -> list[tuple[int, ...]]:
    """All tuples of `dim` non-negative ints summing to `total`, in
    lexicographic (first-exponent ascending) order."""
    if dim == 1:
        return [(total,)]
    if dim == 2:
        return [(a, total - a) for a in range(total + 1)]
    # dim == 3
    return [(a, b, total - a - b)
            for a in range(total + 1)
            for b in range(total - a + 1)]


def monomial_fields(dim: int, degree_max: int = 4) -> list[Field]:
    """All scalar monomials x^a y^b (z^c) with a+b+c <= degree_max (each
    exponent vector appearing exactly once), in ascending degree then
    lexicographic exponent order."""
    fields: list[Field] = []
    for total in range(degree_max + 1):
        for exps in _compositions(total, dim):
            f = _monomial
            g = _monomial_grad
            l = _monomial_lap
            fields.append(Field(
                name=_monomial_name(exps),
                kind="monomial",
                degree=sum(exps),
                is_vector=False,
                value=lambda x, e=exps: f(x, e),
                grad=lambda x, e=exps: g(x, e),
                lap=lambda x, e=exps: l(x, e),
            ))
    return fields


# Fixed constant matrix for the linear vector field (first `dim` rows).
_VEC_M = ((1.3, -0.7, 0.4),
          (0.4, 1.1, -0.9),
          (-0.2, 0.6, 1.5))


def vector_field(dim: int) -> Field:
    """f(x) = M x with constant M -- exact linear vector field: gradient is
    M everywhere, Laplacian is zero."""
    M = torch.tensor([_VEC_M[d][:dim] for d in range(dim)],
                     dtype=torch.float64)

    def value(x: torch.Tensor) -> torch.Tensor:
        return x @ M.to(x).T

    def grad(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return M.to(x).repeat(N, 1, 1)

    def lap(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.zeros(N, dim, dtype=x.dtype, device=x.device)

    return Field(name="Mx", kind="monomial", degree=1, is_vector=True,
                 value=value, grad=grad, lap=lap)


def _mode(x: torch.Tensor, box, k: tuple, phase: float) -> torch.Tensor:
    """sin of a commensurate plane wave: 2*pi*sum(k_d x_d / L_d) + phase."""
    arg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d, kd in enumerate(k):
        if kd:
            arg = arg + kd * x[:, d] / box[d]
    return torch.sin(2.0 * math.pi * arg + phase)


def _mode_grad(x: torch.Tensor, box, k: tuple, phase: float) -> torch.Tensor:
    """Gradient of _mode: 2*pi*k_d/L_d * cos(...) per axis."""
    arg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d, kd in enumerate(k):
        if kd:
            arg = arg + kd * x[:, d] / box[d]
    c = torch.cos(2.0 * math.pi * arg + phase)
    N = x.shape[0]
    g = torch.zeros(N, len(k), dtype=x.dtype, device=x.device)
    for d, kd in enumerate(k):
        if kd:
            g[:, d] = 2.0 * math.pi * kd / box[d] * c
    return g


def _mode_lap(x: torch.Tensor, box, k: tuple, phase: float) -> torch.Tensor:
    arg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d, kd in enumerate(k):
        if kd:
            arg = arg + kd * x[:, d] / box[d]
    k2 = sum(kd * kd * (2.0 * math.pi / box[d]) ** 2 for d, kd in enumerate(k))
    return -k2 * torch.sin(2.0 * math.pi * arg + phase)


def smooth_periodic_fields(dim: int, box) -> list[Field]:
    """Smooth fields that are periodic with the lattice box (valid on
    periodic domains -- no seam contamination)."""
    box = [float(b) for b in box]
    sigma = 0.15 * min(box)
    center = [b / 2.0 for b in box]

    fields: list[Field] = []

    # single-axis modes
    for d in range(min(dim, 2)):
        k = tuple(int(i == d) for i in range(dim))
        fields.append(Field(
            name=f"sin_{chr(97 + d)}", kind="smooth", degree=-1,
            is_vector=False,
            value=lambda x, k=k: _mode(x, box, k, 0.0),
            grad=lambda x, k=k: _mode_grad(x, box, k, 0.0),
            lap=lambda x, k=k: _mode_lap(x, box, k, 0.0),
        ))

    # product of two modes (2D+): sin(k1.x) * cos(k2.x)
    if dim >= 2:
        k1 = tuple(1 if i == 0 else 0 for i in range(dim))
        k2 = tuple(1 if i == 1 else 0 for i in range(dim))

        def prod_value(x):
            return _mode(x, box, k1, 0.0) * _mode(x, box, k2, math.pi / 2.0)

        def prod_grad(x):
            # d/dx [s1 c2] = g1 c2 + s1 g2, with g the mode gradients
            s1, c2 = _mode(x, box, k1, 0.0), _mode(x, box, k2, math.pi / 2.0)
            g1, g2 = _mode_grad(x, box, k1, 0.0), _mode_grad(x, box, k2, math.pi / 2.0)
            return g1 * c2.unsqueeze(1) + s1.unsqueeze(1) * g2

        def prod_lap(x):
            # lap[s1 c2] = lap(s1) c2 + s1 lap(c2)
            s1, c2 = _mode(x, box, k1, 0.0), _mode(x, box, k2, math.pi / 2.0)
            return _mode_lap(x, box, k1, 0.0) * c2 + \
                s1 * _mode_lap(x, box, k2, math.pi / 2.0)

        fields.append(Field(name="sin_cos", kind="smooth", degree=-1,
                            is_vector=False, value=prod_value,
                            grad=prod_grad, lap=prod_lap))

    # wrapped Gaussian: wrap the input into [0, box) (exact periodicity by
    # construction) then sum 3^dim images -- with sigma = 0.15 min(box) the
    # s=+-1 images cover the edge tails to ~1e-3, s=+-2 to ~1e-21.
    box_t = torch.tensor(box, dtype=torch.float64)

    def _wrap(x):
        return x - box_t.to(x) * torch.floor(x / box_t.to(x))

    def _centers(sh):
        return torch.tensor([c + s * L for c, s, L in
                             zip(center, sh, box)],
                            dtype=torch.float64)  # moved to x's device by callers

    def gauss_value(x):
        x = _wrap(x)
        out = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for sh in _image_shifts(dim):
            dxv = x - _centers(sh).to(x)
            r2 = (dxv * dxv).sum(dim=1)
            out = out + torch.exp(-r2 / (2.0 * sigma * sigma))
        return out

    def gauss_grad(x):
        x = _wrap(x)
        N = x.shape[0]
        out = torch.zeros(N, dim, dtype=x.dtype, device=x.device)
        for sh in _image_shifts(dim):
            dxv = x - _centers(sh).to(x)
            w = torch.exp(-(dxv * dxv).sum(dim=1) / (2.0 * sigma * sigma))
            out = out - dxv * (w / (sigma * sigma)).unsqueeze(1)
        return out

    def gauss_lap(x):
        # lap g = (|dx|^2/sigma^4 - d/sigma^2) g  (per image)
        x = _wrap(x)
        N = x.shape[0]
        out = torch.zeros(N, dtype=x.dtype, device=x.device)
        for sh in _image_shifts(dim):
            dxv = x - _centers(sh).to(x)
            r2 = (dxv * dxv).sum(dim=1)
            g = torch.exp(-r2 / (2.0 * sigma * sigma))
            out = out + (r2 / (sigma ** 4) - dim / (sigma ** 2)) * g
        return out

    fields.append(Field(name="gauss_periodic", kind="smooth", degree=-1,
                        is_vector=False, value=gauss_value,
                        grad=gauss_grad, lap=gauss_lap))
    return fields


def _image_shifts(dim: int):
    import itertools
    return list(itertools.product((-1, 0, 1), repeat=dim))


def smooth_open_fields(dim: int, box) -> list[Field]:
    """Smooth fields for open (non-periodic) domains: an ordinary Gaussian
    centred in the box plus the same mode product as the periodic set."""
    box = [float(b) for b in box]
    sigma = 0.15 * min(box)
    center = torch.tensor([b / 2.0 for b in box], dtype=torch.float64)
    fields: list[Field] = []

    def gauss_value(x):
        dxv = x - center.to(x)
        r2 = (dxv * dxv).sum(dim=1)
        return torch.exp(-r2 / (2.0 * sigma * sigma))

    def gauss_grad(x):
        dxv = x - center.to(x)
        w = torch.exp(-(dxv * dxv).sum(dim=1) / (2.0 * sigma * sigma))
        return -dxv * (w / (sigma * sigma)).unsqueeze(1)

    def gauss_lap(x):
        dxv = x - center.to(x)
        r2 = (dxv * dxv).sum(dim=1)
        g = torch.exp(-r2 / (2.0 * sigma * sigma))
        return (r2 / (sigma ** 4) - dim / (sigma ** 2)) * g

    fields.append(Field(name="gauss", kind="smooth", degree=-1,
                        is_vector=False, value=gauss_value,
                        grad=gauss_grad, lap=gauss_lap))

    if dim >= 2:
        k1 = tuple(1 if i == 0 else 0 for i in range(dim))
        k2 = tuple(1 if i == 1 else 0 for i in range(dim))

        def prod_value(x):
            return _mode(x, box, k1, 0.0) * _mode(x, box, k2, math.pi / 2.0)

        def prod_grad(x):
            s1, c2 = _mode(x, box, k1, 0.0), _mode(x, box, k2, math.pi / 2.0)
            g1, g2 = _mode_grad(x, box, k1, 0.0), _mode_grad(x, box, k2, math.pi / 2.0)
            return g1 * c2.unsqueeze(1) + s1.unsqueeze(1) * g2

        def prod_lap(x):
            s1, c2 = _mode(x, box, k1, 0.0), _mode(x, box, k2, math.pi / 2.0)
            return _mode_lap(x, box, k1, 0.0) * c2 + \
                s1 * _mode_lap(x, box, k2, math.pi / 2.0)

        fields.append(Field(name="sin_cos", kind="smooth", degree=-1,
                            is_vector=False, value=prod_value,
                            grad=prod_grad, lap=prod_lap))
    return fields
