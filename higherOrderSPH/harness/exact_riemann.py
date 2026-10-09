"""Exact Riemann solution for the 1-D Euler equations of an ideal gas (Toro,
ch. 4): sampling of the self-similar solution at x/t. Used to score the
shock-tube runs."""

from __future__ import annotations

import math

import numpy as np


def _fK(p, rho, P, g):
    a = math.sqrt(g * P / rho)
    if p > P:
        A, B = 2.0 / ((g + 1) * rho), (g - 1) / (g + 1) * P
        q = math.sqrt(A / (p + B))
        return (p - P) * q, q * (1 - (p - P) / (2 * (p + B)))
    f = 2 * a / (g - 1) * ((p / P) ** ((g - 1) / (2 * g)) - 1)
    return f, 1.0 / (rho * a) * (p / P) ** (-(g + 1) / (2 * g))


def star(rL, uL, pL, rR, uR, pR, g):
    p = 0.5 * (pL + pR)
    for _ in range(100):
        fL, dL = _fK(p, rL, pL, g)
        fR, dR = _fK(p, rR, pR, g)
        dp = (fL + fR + uR - uL) / (dL + dR)
        p = max(p - dp, 1e-12)
        if abs(dp) < 1e-12 * p:
            break
    fL, _ = _fK(p, rL, pL, g)
    fR, _ = _fK(p, rR, pR, g)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(xi, rL, uL, pL, rR, uR, pR, g):
    """(rho, u, P) at similarity coordinate ``xi = (x - x0)/t`` (array)."""
    ps, us = star(rL, uL, pL, rR, uR, pR, g)
    aL, aR = math.sqrt(g * pL / rL), math.sqrt(g * pR / rR)
    out = np.empty((len(xi), 3))
    for k, s in enumerate(np.asarray(xi, dtype=float)):
        if s <= us:       # left of the contact
            if ps > pL:   # left shock
                SL = uL - aL * math.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                if s < SL:
                    out[k] = (rL, uL, pL)
                else:
                    out[k] = (rL * ((ps / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pL + 1)), us, ps)
            else:         # left rarefaction
                aS = aL * (ps / pL) ** ((g - 1) / (2 * g))
                if s < uL - aL:
                    out[k] = (rL, uL, pL)
                elif s > us - aS:
                    out[k] = (rL * (ps / pL) ** (1 / g), us, ps)
                else:
                    u = 2 / (g + 1) * (aL + (g - 1) / 2 * uL + s)
                    c = 2 / (g + 1) * (aL + (g - 1) / 2 * (uL - s))
                    out[k] = (rL * (c / aL) ** (2 / (g - 1)), u, pL * (c / aL) ** (2 * g / (g - 1)))
        else:
            if ps > pR:   # right shock
                SR = uR + aR * math.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                if s > SR:
                    out[k] = (rR, uR, pR)
                else:
                    out[k] = (rR * ((ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1)), us, ps)
            else:
                aS = aR * (ps / pR) ** ((g - 1) / (2 * g))
                if s > uR + aR:
                    out[k] = (rR, uR, pR)
                elif s < us + aS:
                    out[k] = (rR * (ps / pR) ** (1 / g), us, ps)
                else:
                    u = 2 / (g + 1) * (-aR + (g - 1) / 2 * uR + s)
                    c = 2 / (g + 1) * (aR - (g - 1) / 2 * (uR - s))
                    out[k] = (rR * (c / aR) ** (2 / (g - 1)), u, pR * (c / aR) ** (2 * g / (g - 1)))
    return out
