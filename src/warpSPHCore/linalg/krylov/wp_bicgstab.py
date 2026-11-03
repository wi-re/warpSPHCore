"""Readable Warp inference update equations; orchestration is shared."""
from typing import Any
import warp as wp

@wp.func
def _divide(a: Any, b: Any):
    if wp.isfinite(b) and b != 0:
        return a / b
    return a


@wp.kernel(enable_backward=False)
def _initialize(rr: wp.array(dtype=Any), target: wp.array(dtype=Any),
                r0norm: wp.array(dtype=Any), rnorm: wp.array(dtype=Any),
                state: wp.array(dtype=wp.int32), threshold: wp.array(dtype=wp.bool)):
    norm = wp.sqrt(rr[0])
    r0norm[0] = norm
    rnorm[0] = norm
    conv = wp.isfinite(norm) and norm <= target[0]
    state[0] = wp.int32(not conv and not threshold[0])
    state[1] = -14
    if conv:
        state[1] = 0
    if threshold[0]:
        state[1] = -12
    for i in range(2, 9):
        state[i] = 0


@wp.kernel(enable_backward=False)
def _alpha(rho: wp.array(dtype=Any), rv: wp.array(dtype=Any), aa: wp.array(dtype=Any),
           r0norm: wp.array(dtype=Any), rnorm: wp.array(dtype=Any), alpha: wp.array(dtype=Any),
           state: wp.array(dtype=wp.int32)):
    eps = rho.dtype(1.0e-6)
    badRho = not wp.isfinite(rho[0]) or rho[0] == rho.dtype(0) or wp.abs(rho[0]) < eps * r0norm[0] * rnorm[0]
    badRv = not wp.isfinite(rv[0]) or rv[0] == rv.dtype(0) or wp.abs(rv[0]) < eps * r0norm[0] * wp.sqrt(aa[0])
    alpha[0] = _divide(rho[0], rv[0])
    state[4] = wp.int32(state[0] != 0 and not badRho and not badRv)
    if state[0] != 0:
        if badRho:
            state[1] = -10
        elif badRv:
            state[1] = -11


@wp.kernel(enable_backward=False)
def _halfCandidate(x: wp.array(dtype=Any), r: wp.array(dtype=Any), z: wp.array(dtype=Any),
                   ap: wp.array(dtype=Any), alpha: wp.array(dtype=Any), candidate: wp.array(dtype=Any),
                   s: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
                   threshold: Any, hasThreshold: bool, dim: int):
    i = wp.tid()
    value = x[i] + alpha[0] * z[i]
    candidate[i] = value
    s[i] = r[i] - alpha[0] * ap[i]
    if not wp.isfinite(value):
        wp.atomic_max(flags, 0, 1)
    if hasThreshold and i % dim == 0:
        length2 = x.dtype(0)
        for k in range(dim):
            v = x[i+k] + alpha[0] * z[i+k]
            length2 += v * v
        if wp.sqrt(length2) > x.dtype(threshold):
            wp.atomic_max(flags, 1, 1)


@wp.kernel(enable_backward=False)
def _halfDecision(ss: wp.array(dtype=Any), target: wp.array(dtype=Any),
                  flags: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32)):
    go = state[4] != 0 and flags[0] == 0
    norm = wp.sqrt(ss[0])
    half = go and wp.isfinite(norm) and norm <= target[0]
    state[5] = wp.int32(half)
    state[6] = wp.int32(go and not half)
    if state[4] != 0 and flags[0] != 0:
        state[1] = -11
    if half:
        state[1] = state[3]
        if flags[1] != 0:
            state[1] = -12


@wp.kernel(enable_backward=False)
def _omega(ts: wp.array(dtype=Any), tt: wp.array(dtype=Any), omega: wp.array(dtype=Any)):
    omega[0] = _divide(ts[0], tt[0])


@wp.kernel(enable_backward=False)
def _fullCandidate(x: wp.array(dtype=Any), z: wp.array(dtype=Any), zs: wp.array(dtype=Any),
                   ast: wp.array(dtype=Any), s: wp.array(dtype=Any), candidate: wp.array(dtype=Any),
                   alpha: wp.array(dtype=Any), omega: wp.array(dtype=Any),
                   flags: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32),
                   threshold: Any, hasThreshold: bool, dim: int):
    i = wp.tid()
    value = (x[i] + alpha[0] * z[i]) + omega[0] * zs[i]
    if state[6] != 0:
        candidate[i] = value
    s[i] = s[i] - omega[0] * ast[i]
    if not wp.isfinite(value):
        wp.atomic_max(flags, 0, 1)
    if hasThreshold and i % dim == 0:
        length2 = x.dtype(0)
        for k in range(dim):
            v = (x[i+k] + alpha[0] * z[i+k]) + omega[0] * zs[i+k]
            length2 += v * v
        if wp.sqrt(length2) > x.dtype(threshold):
            wp.atomic_max(flags, 1, 1)


@wp.kernel(enable_backward=False)
def _fullDecision(rr: wp.array(dtype=Any), rhon: wp.array(dtype=Any), rho: wp.array(dtype=Any),
                  alpha: wp.array(dtype=Any), omega: wp.array(dtype=Any), tt: wp.array(dtype=Any),
                  beta: wp.array(dtype=Any), target: wp.array(dtype=Any),
                  flags: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32)):
    norm = wp.sqrt(rr[0])
    bad = not wp.isfinite(omega[0]) or omega[0] == omega.dtype(0) or not wp.isfinite(tt[0]) or tt[0] == tt.dtype(0)
    bad = bad or flags[0] != 0 or not wp.isfinite(norm)
    full = state[6] != 0 and not bad
    conv = full and norm <= target[0]
    state[7] = wp.int32(full)
    state[8] = wp.int32(full and flags[1] == 0 and not conv)
    beta[0] = _divide(rhon[0], rho[0]) * _divide(alpha[0], omega[0])
    if state[6] != 0 and bad:
        state[1] = -11
    if full:
        if flags[1] != 0:
            state[1] = -12
        elif conv:
            state[1] = state[3] + 1


@wp.kernel(enable_backward=False)
def _direction(x: wp.array(dtype=Any), candidate: wp.array(dtype=Any), s: wp.array(dtype=Any),
               p: wp.array(dtype=Any), ap: wp.array(dtype=Any), pn: wp.array(dtype=Any),
               beta: wp.array(dtype=Any), omega: wp.array(dtype=Any),
               flags: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    value = s[i] + beta[0] * (p[i] - omega[0] * ap[i])
    pn[i] = value
    if state[8] != 0 and (not wp.isfinite(value) or not wp.isfinite(beta[0])):
        wp.atomic_max(flags, 2, 1)
    if state[5] != 0 or state[7] != 0:
        x[i] = candidate[i]


@wp.kernel(enable_backward=False)
def _finish(rr: wp.array(dtype=Any), rnorm: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
            state: wp.array(dtype=wp.int32), norms: wp.array(dtype=Any), valid: wp.array(dtype=wp.bool), history: bool):
    if state[8] != 0 and flags[2] != 0:
        state[1] = -11
    state[0] = wp.int32(state[8] != 0 and flags[2] == 0)
    state[2] += wp.int32(state[5] != 0 or state[7] != 0)
    if history:
        norms[state[3]] = wp.sqrt(rr[0])
        valid[state[3]] = state[7] != 0
    if state[0] != 0:
        rnorm[0] = wp.sqrt(rr[0])
    state[3] += 1


@wp.kernel(enable_backward=False)
def _commit(r: wp.array(dtype=Any), p: wp.array(dtype=Any), s: wp.array(dtype=Any),
            pn: wp.array(dtype=Any), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if state[0] != 0:
        r[i] = s[i]
        p[i] = pn[i]


@wp.kernel(enable_backward=False)
def _diagonal(v: wp.array(dtype=Any), inverse: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = v[i] * inverse[i]
