#!/usr/bin/env python3
"""Shared helpers for the Dehnen & Aly (2012) replication.

Two jobs, both deliberate:

1. **Pin the notation once.** The paper's h (eq. 8, the 2-sigma resolution
   scale) and the code's h (the support radius H) are different quantities
   that every figure in the paper uses interchangeably. Every script in this
   folder gets the conversion from here instead of re-deriving it:

       h_paper = h_code / kernelScale(kernel, dim)

   where ``kernelScale`` is the shipped ``eval_kernelScale`` (support radius
   in units of the 2-sigma resolution scale, i.e. the paper's H/h).

2. **Evaluate the SHIPPED kernels on the host, not a re-transcription.**
   Phase 1 audits the kernels in ``src/warpSPHCore/kernels/`` *as they are*;
   re-typing the formulas into a numpy reference would make the audit
   circular or, worse, silently audit a transcription instead of the code.
   So the shape functions, C_d, kernelScale and packing ratio are evaluated
   by launching the actual ``eval_*`` dispatch functions on ``device="cpu"``
   through the same ``@wp.kernel`` wrappers ``scripts/gradcheck/
   kernel_sanity_native.py`` uses (float64, one wrapper per dimension,
   dtypes baked in by name at module level).

Reference values (paper tables, ICs, SPHS equations) live in
``data/da2012_reference.yaml`` -- the single source of truth for every
number quoted from the paper in this folder.

Run a smoke test:
    python scripts/common.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

from pathlib import Path

import numpy as np
import warp as wp
import yaml

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.eval_kernel import (
    eval_k,
    eval_C_d,
    eval_kernelScale,
    eval_packing,
)

DEVICE = "cpu"
_HERE = Path(__file__).resolve().parent
REFERENCE_YAML = _HERE.parent / "data" / "da2012_reference.yaml"

# Paper-table name -> shipped KernelFunctions member. HOCT4 and Gaussian join
# this map in Phase 3, once they exist in the core library; until then the
# replication references them through the verified definitions in
# data/da2012_reference.yaml and a local numpy shape function.
KERNEL_BY_NAME = {
    "cubic_b4": KernelFunctions.CubicSpline,
    "quartic_b5": KernelFunctions.QuarticSpline,
    "quintic_b6": KernelFunctions.QuinticSpline,
    "wendland_C2": KernelFunctions.Wendland2,
    "wendland_C4": KernelFunctions.Wendland4,
    "wendland_C6": KernelFunctions.Wendland6,
}

_initialized = False


def init() -> None:
    """wp.init() exactly once per process (kernels are lazily compiled)."""
    global _initialized
    if not _initialized:
        wp.init()
        _initialized = True


def load_reference() -> dict:
    """The paper's reference data (tables, ICs, constants) as a dict."""
    with open(REFERENCE_YAML) as f:
        return yaml.safe_load(f)


def kernel_id(name: str) -> int:
    try:
        return KERNEL_BY_NAME[name].value
    except KeyError:
        raise KeyError(
            f"unknown kernel name {name!r}; known: {sorted(KERNEL_BY_NAME)}"
        ) from None


# ---------------------------------------------------------------------------
# Host-side evaluation of the SHIPPED warp functions (cpu, float64).
# One small wrapper per (function, dimension), mirroring
# scripts/gradcheck/kernel_sanity_native.py.
# ---------------------------------------------------------------------------

@wp.kernel
def _k_1(q: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], wp.int32(1), kernel_id)


@wp.kernel
def _k_2(q: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], wp.int32(2), kernel_id)


@wp.kernel
def _k_3(q: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], wp.int32(3), kernel_id)


_K_KERNEL = {1: _k_1, 2: _k_2, 3: _k_3}


@wp.kernel
def _Cd_1(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_C_d(wp.int32(1), kernel_id)


@wp.kernel
def _Cd_2(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_C_d(wp.int32(2), kernel_id)


@wp.kernel
def _Cd_3(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_C_d(wp.int32(3), kernel_id)


_CD_KERNEL = {1: _Cd_1, 2: _Cd_2, 3: _Cd_3}


@wp.kernel
def _scale_1(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_kernelScale(kernel_id, wp.int32(1))


@wp.kernel
def _scale_2(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_kernelScale(kernel_id, wp.int32(2))


@wp.kernel
def _scale_3(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_kernelScale(kernel_id, wp.int32(3))


_SCALE_KERNEL = {1: _scale_1, 2: _scale_2, 3: _scale_3}


@wp.kernel
def _packing_kernel(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_packing(kernel_id)


def _launch_scalar(fn, inputs: list) -> float:
    out = wp.zeros(1, dtype=scalar_t, device=DEVICE)
    wp.launch(fn, dim=1, inputs=inputs, outputs=[out], device=DEVICE)
    return float(out.numpy()[0])


def _launch_array(fn, q_np: np.ndarray, inputs_tail: list) -> np.ndarray:
    q = wp.array(q_np, dtype=scalar_t, device=DEVICE)
    out = wp.zeros(len(q_np), dtype=scalar_t, device=DEVICE)
    wp.launch(fn, dim=len(q_np), inputs=[q, *inputs_tail], outputs=[out], device=DEVICE)
    return out.numpy()


def shape(q, dim: int, name: str) -> np.ndarray:
    """The dimensionless shape f(q) of the shipped kernel, q = r/H."""
    init()
    q_np = np.atleast_1d(np.asarray(q, dtype=float))
    return _launch_array(_K_KERNEL[dim], q_np, [kernel_id(name)])


def C_d(dim: int, name: str) -> float:
    """Normalisation constant of the shipped kernel (W = C_d f(q)/H^nu)."""
    init()
    return _launch_scalar(_CD_KERNEL[dim], [kernel_id(name)])


def kernel_scale(dim: int, name: str) -> float:
    """Support radius in units of the paper's h (= 2 sigma), i.e. H/h."""
    init()
    return _launch_scalar(_SCALE_KERNEL[dim], [kernel_id(name)])


def packing_ratio(name: str) -> float:
    """The code's effective packing ratio (note: CubicSpline and B7/B8
    carry deliberate multiplicative corrections -- Price 2012 / CRKSPH --
    so this is NOT the densest-packing ratio for every kernel)."""
    init()
    return _launch_scalar(_packing_kernel, [kernel_id(name)])


# ---------------------------------------------------------------------------
# Notation pin (paper <-> code).
# ---------------------------------------------------------------------------

def h_paper(h_code, dim: int, name: str):
    """Paper resolution scale h (= 2 sigma) for a code smoothing length.

    h_code IS the paper's support radius H; the paper's h is
    H / kernelScale.
    """
    return np.asarray(h_code, dtype=float) / kernel_scale(dim, name)


def N_H_of_N_h(N_h: float, dim: int, name: str) -> float:
    return float(N_h) * kernel_scale(dim, name) ** dim


def N_h_of_N_H(N_H: float, dim: int, name: str) -> float:
    return float(N_H) / kernel_scale(dim, name) ** dim


def N_H_volume(dim: int, name: str) -> float:
    """V_nu * H^nu in units of (rho/m) at unit spacing: fac * packing^dim *
    scale^dim, exactly the code's sphKernelN_H."""
    fac = {1: 2.0, 2: np.pi, 3: 4.0 * np.pi / 3.0}[dim]
    return fac * packing_ratio(name) ** dim * kernel_scale(dim, name) ** dim


# ---------------------------------------------------------------------------
# Integration (composite Simpson, same convention as kernel_sanity_native.py)
# ---------------------------------------------------------------------------

_OMEGA = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}

_SIMPSON_N = 200001  # 200000 intervals (even), float64 on smooth f


def simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = len(x) - 1
    assert n % 2 == 0, "composite Simpson needs an even number of intervals"
    h = (x[-1] - x[0]) / n
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return float(s * h / 3.0)


def q_moment(order: int, dim: int, name: str) -> float:
    """int_0^1 f(q) q^order dq for the shipped kernel (Simpson)."""
    q_np = np.linspace(0.0, 1.0, _SIMPSON_N)
    f = shape(q_np, dim, name)
    return simpson(f * q_np**order, q_np)


def sigma2_over_H2(dim: int, name: str) -> float:
    """sigma^2 / H^2 per paper eq. 8:  sigma^2 = nu^-1 int |x|^2 W d^nu x.

    The 1/nu is in the paper's printed definition (sigma is the standard
    deviation of ONE Cartesian component, not of |x|). Forgetting it
    overestimates sigma^2/H^2 by exactly a factor of nu (dim 1: 2: 3 =
    1 : 2 : 3) -- the cubic-spline 3D value is 3/40, not 9/40.
    """
    return C_d(dim, name) * _OMEGA[dim] * q_moment(dim + 1, dim, name) / dim


def W0(name: str, dim: int, h_code: float) -> float:
    """W(0, h) for the shipped kernel (used by the eq. 18 density
    correction rho_hat_corr = rho_hat - eps m W(0, H))."""
    f0 = float(shape(np.array([0.0]), dim, name)[0])
    return C_d(dim, name) * f0 / h_code**dim


def eps_density_correction(name: str, N_H: float, ref: dict | None = None) -> float:
    """Eq. 19: eps = eps_100 (N_H/100)^(-alpha). Only the three Wendland
    kernels have fitted constants in the paper."""
    ref = ref if ref is not None else load_reference()
    try:
        c = ref["density_correction"][name]
    except KeyError:
        raise KeyError(
            f"no eq.-19 constants for {name!r} (paper fits only the "
            f"Wendland C2/C4/C6 kernels)"
        ) from None
    return c["eps_100"] * (N_H / 100.0) ** (-c["alpha"])


# ---------------------------------------------------------------------------
# Check helper (scalar- or array-valued), same tolerance logic as
# kernel_sanity_native.py.
# ---------------------------------------------------------------------------

def check(name: str, actual, expected, atol: float = 1e-9, rtol: float = 1e-7) -> bool:
    a = np.atleast_1d(np.asarray(actual, dtype=float))
    e = np.atleast_1d(np.asarray(expected, dtype=float))
    diff = np.abs(a - e)
    denom = np.maximum(np.abs(e), atol)
    rel = diff / denom
    ok = bool(np.all((diff <= atol) | (rel <= rtol)))
    worst = float(np.max(diff)) if diff.size else 0.0
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<55} max|Δ|={worst:.3e}")
    return ok


# ---------------------------------------------------------------------------
# Smoke test: pin the shipped kernels against the paper's Table 1.
# ---------------------------------------------------------------------------

def _table1_value(v) -> float:
    """Table 1 entries are exact symbolic strings ('16/pi') or numbers."""
    if isinstance(v, str):
        return float(eval(v, {"__builtins__": {}}, {"pi": np.pi}))
    return float(v)


def _smoke_test() -> bool:
    print("common.py smoke test: shipped kernels vs D&A 2012 Table 1 (dim 3)")
    ref = load_reference()
    ok = True
    for name, row in ref["table1"].items():
        if name not in KERNEL_BY_NAME:
            print(f"  skip  {name:<14} (not yet in the core library)")
            continue
        ok &= check(f"{name} C_3", C_d(3, name), _table1_value(row["C"][2]), atol=1e-10, rtol=1e-12)
        ok &= check(f"{name} sigma2/H2_3", sigma2_over_H2(3, name), _table1_value(row["sigma2_over_H2"][2]), atol=1e-8, rtol=1e-9)
        ok &= check(f"{name} H/h_3", kernel_scale(3, name), row["H_over_h"][2], atol=1e-5, rtol=1e-6)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _smoke_test() else 1)
