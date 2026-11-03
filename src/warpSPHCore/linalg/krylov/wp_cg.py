"""Readable Warp inference update equations; orchestration is shared."""
from typing import Any
import warp as wp

@wp.kernel(enable_backward=False)
def _initialize(rr: wp.array(dtype=Any), target: wp.array(dtype=Any),
                state: wp.array(dtype=wp.int32), norms: wp.array(dtype=Any),
                valid: wp.array(dtype=wp.bool), initialThreshold: wp.array(dtype=wp.bool),
                zeroRhs: wp.array(dtype=wp.bool), history: bool):
    norm = wp.sqrt(rr[0])
    conv = wp.isfinite(norm) and norm <= target[0]
    live = not conv and not initialThreshold[0]
    state[0] = wp.int32(live)
    state[1] = -14
    if conv:
        state[1] = 0
    if initialThreshold[0]:
        state[1] = -12
    state[2] = 0  # accepted updates
    state[3] = 0  # scheduled updates
    state[4] = 0  # accept current candidate
    if history:
        norms[0] = norm
        valid[0] = not (zeroRhs[0] and conv)


@wp.kernel(enable_backward=False)
def _candidate(x: wp.array(dtype=Any), r: wp.array(dtype=Any), p: wp.array(dtype=Any),
               ap: wp.array(dtype=Any), rz: wp.array(dtype=Any), pap: wp.array(dtype=Any),
               xn: wp.array(dtype=Any), rn: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
               threshold: Any, hasThreshold: bool, dim: int):
    i = wp.tid()
    alpha = x.dtype(0)
    if wp.isfinite(pap[0]) and pap[0] != x.dtype(0):
        alpha = rz[0] / pap[0]
    xn[i] = x[i] + alpha * p[i]
    rn[i] = r[i] - alpha * ap[i]
    if not wp.isfinite(xn[i]) or not wp.isfinite(rn[i]):
        wp.atomic_max(flags, 0, 1)
    if hasThreshold and i % dim == 0:
        length2 = x.dtype(0)
        for j in range(dim):
            value = x[i + j] + alpha * p[i + j]
            length2 += value * value
        if wp.sqrt(length2) > x.dtype(threshold):
            wp.atomic_max(flags, 1, 1)


@wp.kernel(enable_backward=False)
def _decide(rz: wp.array(dtype=Any), pap: wp.array(dtype=Any), rr: wp.array(dtype=Any),
            target: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
            state: wp.array(dtype=wp.int32), norms: wp.array(dtype=Any),
            valid: wp.array(dtype=wp.bool), history: bool):
    norm = wp.sqrt(rr[0])
    live = state[0] != 0
    ok = (wp.isfinite(rz[0]) and rz[0] > rz.dtype(0) and wp.isfinite(pap[0])
          and pap[0] > pap.dtype(0) and wp.isfinite(norm) and flags[0] == 0)
    go = live and ok
    conv = go and norm <= target[0] and flags[1] == 0
    if live and not ok:
        state[1] = -16
    elif go and flags[1] != 0:
        state[1] = -12
    elif conv:
        state[1] = state[3]
    state[0] = wp.int32(go and not conv and flags[1] == 0)
    state[4] = wp.int32(go)
    state[2] += wp.int32(go)
    state[3] += 1
    if history:
        norms[state[3]] = norm
        valid[state[3]] = go


@wp.kernel(enable_backward=False)
def _commit(x: wp.array(dtype=Any), r: wp.array(dtype=Any), xn: wp.array(dtype=Any),
            rn: wp.array(dtype=Any), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if state[4] != 0:
        x[i] = xn[i]
        r[i] = rn[i]


@wp.kernel(enable_backward=False)
def _diagonal(r: wp.array(dtype=Any), inverse: wp.array(dtype=Any), z: wp.array(dtype=Any)):
    i = wp.tid()
    z[i] = r[i] * inverse[i]


@wp.kernel(enable_backward=False)
def _direction(z: wp.array(dtype=Any), p: wp.array(dtype=Any), rz: wp.array(dtype=Any),
               rzn: wp.array(dtype=Any), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if state[0] != 0:
        beta = p.dtype(0)
        if wp.isfinite(rz[0]) and rz[0] != p.dtype(0):
            beta = rzn[0] / rz[0]
        p[i] = z[i] + beta * p[i]
