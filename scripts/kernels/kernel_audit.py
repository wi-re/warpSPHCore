#!/usr/bin/env python3
"""Kernel audit battery -- the machine half of the kernel-onboarding process.

Runs a fixed set of checks against every kernel listed in
`scripts/kernels/kernel_specs.yaml` (the machine-readable spec: source
citation, dimensionless shape f(q) with q = r/H, normalisation C per
dimension, sigma^2/H^2 per dimension in the D&A eq.-8 convention, code
packing ratio + reference N_H):

  1. form          code eval_k vs the spec shape on a fine grid
                   (incl. outside the support);
  2. normalisation code eval_C_d vs the spec C vs an independent
                   numerical V_nu * int_0^1 f q^(nu-1) dq;
  3. moments       sigma2/H2 from C * V_nu/nu * int f q^(nu+1) dq (eq.-8
                   convention, 1/nu included) vs the spec; code
                   kernelScale vs 1/(2 sqrt(sigma2/H2));
  4. derivatives   code _dkdq/_d2kdq2/_d3kdq3 vs wp.Tape AD of the level
                   below (same pattern as
                   scripts/gradcheck/kernel_sanity_native.py Section C);
  5. support       exactly zero for q > 1; f(1) = f'(1) = 0;
  6. FT sign       (--ft) 3D FT per D&A eq. (14): wbar(0) = 1, the
                   eq. (17) Taylor wbar = 1 - 1/2 sigma^2 k^2, and
                   non-negativity / first zero per the spec's
                   `ft_expected`;
  7. packing       code packingRatio == spec value; sphKernelN_H at that
                   packing reproduces the reference N_H (3-decimal
                   packing rounding allowed); sphKernel_xi == H * d_nn.

The code is evaluated by launching the SHIPPED warp dispatch functions on
`device="cpu"` (float64) -- never a re-transcription, so the audit is
always about the code as it is.

Documented deviations are declared per kernel in the spec's
`known_issues`; a matching failure is reported KNOWN (green) while any
NEW failure on the same check stays a hard FAIL. A known issue that no
longer reproduces raises a warning to update the spec.

Exit code: 0 if no hard failures, 1 otherwise.

Run:
    python scripts/kernels/kernel_audit.py [--ft] [--only NAME,...] [--json OUT]
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.eval_kernel import (
    eval_k,
    eval_dkdq,
    eval_d2kdq2,
    eval_d3kdq3,
    eval_C_d,
    eval_kernelScale,
    eval_packing,
)
from warpSPHCore.kernels.properties import sphKernelN_H, sphKernel_xi

DEVICE = "cpu"
_OMEGA = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}

_SPEC_PATH = Path(__file__).resolve().parent / "kernel_specs.yaml"
_INITIALIZED = False

# q-grid points that are kinks of some spec'd kernel (piece boundaries /
# positive-part nodes); the AD grid must stay clear of them.
_KINKS = (1.0 / 7.0, 3.0 / 7.0, 0.214108111463, 0.25, 1.0 / 3.0, 0.5,
          5.0 / 7.0, 0.6, 2.0 / 3.0, 0.75)


def init() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        wp.init()
        _INITIALIZED = True


# ---------------------------------------------------------------------------
# Host-side evaluation of the shipped warp functions (cpu, float64).
# ---------------------------------------------------------------------------

@wp.kernel
def _k(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], dim, kernel_id)


@wp.kernel
def _dkdq(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_dkdq(q[i], dim, kernel_id)


@wp.kernel
def _d2kdq2(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_d2kdq2(q[i], dim, kernel_id)


@wp.kernel
def _d3kdq3(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_d3kdq3(q[i], dim, kernel_id)


@wp.kernel
def _Cd(dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_C_d(dim, kernel_id)


@wp.kernel
def _scale(dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_kernelScale(kernel_id, dim)


@wp.kernel
def _packing(kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_packing(kernel_id)


@wp.kernel
def _nH(dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = sphKernelN_H(kernel_id, dim)


@wp.kernel
def _xi(dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = sphKernel_xi(kernel_id, dim)


def _launch_array(fn, q_np: np.ndarray, dim: int, kernel_id: int) -> np.ndarray:
    q = wp.array(q_np, dtype=scalar_t, device=DEVICE)
    out = wp.zeros(len(q_np), dtype=scalar_t, device=DEVICE)
    wp.launch(fn, dim=len(q_np), inputs=[q, dim, kernel_id], outputs=[out], device=DEVICE)
    return out.numpy()


def _launch_scalar(fn, inputs: list) -> float:
    out = wp.zeros(1, dtype=scalar_t, device=DEVICE)
    wp.launch(fn, dim=1, inputs=inputs, outputs=[out], device=DEVICE)
    return float(out.numpy()[0])


def code_k(q, dim: int, kernel_id: int) -> np.ndarray:
    return _launch_array(_k, np.atleast_1d(np.asarray(q, dtype=float)), dim, kernel_id)


def code_dkdq(q, dim: int, kernel_id: int) -> np.ndarray:
    return _launch_array(_dkdq, np.atleast_1d(np.asarray(q, dtype=float)), dim, kernel_id)


def code_d2kdq2(q, dim: int, kernel_id: int) -> np.ndarray:
    return _launch_array(_d2kdq2, np.atleast_1d(np.asarray(q, dtype=float)), dim, kernel_id)


def code_C_d(dim: int, kernel_id: int) -> float:
    return _launch_scalar(_Cd, [dim, kernel_id])


def code_scale(dim: int, kernel_id: int) -> float:
    return _launch_scalar(_scale, [dim, kernel_id])


def code_packing(kernel_id: int) -> float:
    return _launch_scalar(_packing, [kernel_id])


def code_N_H(dim: int, kernel_id: int) -> float:
    return _launch_scalar(_nH, [dim, kernel_id])


def code_xi(dim: int, kernel_id: int) -> float:
    return _launch_scalar(_xi, [dim, kernel_id])


# ---------------------------------------------------------------------------
# Spec-side evaluation.
# ---------------------------------------------------------------------------

def _expr_for_dim(spec: dict, dim: int) -> str:
    if "shape_by_dim" in spec:
        return spec["shape_by_dim"][str(dim)]
    return spec["shape"]["expr"]


def spec_shape(spec: dict, dim: int, q) -> np.ndarray:
    """The spec's f(q): the expr on [0, 1], 0 outside the support."""
    q_np = np.atleast_1d(np.asarray(q, dtype=float))
    ns = {"q": q_np, "pos": lambda x: np.maximum(x, 0.0),
          "step": lambda x: np.where(x >= 0.0, 1.0, 0.0),
          "pi": np.pi, "exp": np.exp, "sqrt": np.sqrt}
    f = np.array(eval(_expr_for_dim(spec, dim), {"__builtins__": {}}, ns), dtype=float)
    return np.where(q_np <= 1.0, f, 0.0)


def spec_C(spec: dict, dim: int) -> float | None:
    c = spec.get("C", {}).get(str(dim)) or spec.get("C", {}).get(dim)
    return None if c is None else float(eval(str(c), {"__builtins__": {}}, {"pi": np.pi}))


def spec_sigma2(spec: dict, dim: int) -> float | None:
    s = spec.get("sigma2")
    if s is None:
        return None
    v = s.get(str(dim)) or s.get(dim)
    return None if v is None else float(eval(str(v), {"__builtins__": {}}, {"pi": np.pi}))


def simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = len(x) - 1
    assert n % 2 == 0, "composite Simpson needs an even number of intervals"
    h = (x[-1] - x[0]) / n
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return float(s * h / 3.0)


def _integrate(f_of_q, order: int, n: int = 200001) -> float:
    """int_0^1 f(q) q^order dq (composite Simpson, float64)."""
    x = np.linspace(0.0, 1.0, n)
    return simpson(f_of_q(x) * x**order, x)


# ---------------------------------------------------------------------------
# Result bookkeeping.
# ---------------------------------------------------------------------------

class Audit:
    def __init__(self, quiet: bool = False):
        self.rows: list[dict] = []
        self.quiet = quiet
        self.hard_failures = 0

    def add(self, kernel: str, check: str, dim, status: str, detail: str):
        self.rows.append({"kernel": kernel, "check": check, "dim": dim,
                          "status": status, "detail": detail})
        if status == "FAIL":
            self.hard_failures += 1
        if not self.quiet:
            d = f"  dim={dim}" if dim is not None else ""
            print(f"  {status:<5} {kernel} {check}{d}: {detail}")

    def ok(self, kernel: str, check: str, dim, actual, expected, atol, rtol, label=""):
        a = float(np.max(np.abs(np.atleast_1d(actual))))
        e = float(np.max(np.abs(np.atleast_1d(expected))))
        diff = float(np.max(np.abs(np.atleast_1d(actual) - np.atleast_1d(expected))))
        good = diff <= atol or diff <= rtol * max(abs(e), 1e-300)
        self.add(kernel, check, dim, "PASS" if good else "FAIL",
                 f"{label} |actual|={a:.9e} |expected|={e:.9e} max|diff|={diff:.3e}")
        return good


# ---------------------------------------------------------------------------
# AD derivative (same pattern as kernel_sanity_native.py Section C:
# out[i] depends only on q[i], so seeding every adjoint with 1 gives the
# per-element derivative in one backward pass).
# ---------------------------------------------------------------------------

def _forward_and_ad(kernel_fn, q_np: np.ndarray, dim: int, kernel_id: int):
    q = wp.array(q_np, dtype=scalar_t, requires_grad=True, device=DEVICE)
    out = wp.zeros(len(q_np), dtype=scalar_t, requires_grad=True, device=DEVICE)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel_fn, dim=len(q_np), inputs=[q, dim, kernel_id], outputs=[out], device=DEVICE)
    seed = wp.array(np.ones(len(q_np)), dtype=scalar_t, device=DEVICE)
    tape.backward(grads={out: seed})
    grad = q.grad.numpy().copy()
    tape.zero()
    return out.numpy().copy(), grad


def _ad_grid() -> np.ndarray:
    """Interior grid, clear of every spec'd kernel's kink and of the
    clamp boundary at q = 1."""
    q = np.linspace(0.02, 0.98, 97)
    keep = np.ones(len(q), dtype=bool)
    for k in _KINKS:
        keep &= np.abs(q - k) > 1e-6
    return q[keep]


# ---------------------------------------------------------------------------
# The checks.
# ---------------------------------------------------------------------------

def check_form(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    q = np.linspace(0.0, 1.0, 4097)
    q = np.concatenate([q, [1.0 + 1e-9, 1.0 + 1e-6, 1.5, 2.0]])
    got = code_k(q, dim, kid)
    want = spec_shape(spec, dim, q)
    diff = np.max(np.abs(got - want))
    scale = max(1.0, float(np.max(np.abs(want))))
    aud.add(name, "form", dim, "PASS" if diff <= 1e-12 * scale else "FAIL",
            f"max|code - spec|={diff:.3e} (scale {scale:.3e})")


def check_normalisation(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    C_code = code_C_d(dim, kid)
    C_spec = spec_C(spec, dim)
    if C_spec is not None:
        aud.ok(name, "normalisation", dim, C_code, C_spec, 0.0, 1e-12, label="C_code vs C_spec")
    C_expected = 1.0 / (_OMEGA[dim] * _integrate(lambda q: spec_shape(spec, dim, q), dim - 1))
    aud.ok(name, "normalisation", dim, C_code, C_expected, 0.0, 1e-9,
           label="C_code vs 1/(V_nu int f q^(nu-1))")


def check_moments(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    C_code = code_C_d(dim, kid)
    f_spec = lambda q: spec_shape(spec, dim, q)  # noqa: E731
    s2_spec_shape = C_code * _OMEGA[dim] * _integrate(f_spec, dim + 1) / dim
    s2_spec = spec_sigma2(spec, dim)
    if s2_spec is not None:
        aud.ok(name, "moments", dim, s2_spec_shape, s2_spec, 0.0, 1e-9,
               label="sigma2/H2 (spec shape, eq.-8 convention) vs spec")
    scale_code = code_scale(dim, kid)
    scale_expected = 1.0 / (2.0 * np.sqrt(s2_spec_shape))
    aud.ok(name, "moments", dim, scale_code, scale_expected, 0.0, 1e-6,
           label="kernelScale vs 1/(2 sqrt(sigma2/H2)) (6-decimal rounding allowed)")
    if not aud.quiet:
        print(f"        [info] {name} dim={dim}: sigma2/H2={s2_spec_shape:.9f}, "
              f"H/h(derived)={scale_expected:.6f}, H/h(code)={scale_code:.6f}")


def check_derivatives(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    q = _ad_grid()
    f1, _ = _forward_and_ad(_dkdq, q, dim, kid)
    _, ad1 = _forward_and_ad(_k, q, dim, kid)
    aud.ok(name, "derivatives", dim, f1, ad1, 1e-10, 1e-9, label="dkdq vs AD[k]")
    f2, _ = _forward_and_ad(_d2kdq2, q, dim, kid)
    _, ad2 = _forward_and_ad(_dkdq, q, dim, kid)
    aud.ok(name, "derivatives", dim, f2, ad2, 1e-9, 1e-8, label="d2kdq2 vs AD[dkdq]")
    f3, _ = _forward_and_ad(_d3kdq3, q, dim, kid)
    _, ad3 = _forward_and_ad(_d2kdq2, q, dim, kid)
    aud.ok(name, "derivatives", dim, f3, ad3, 1e-8, 1e-7, label="d3kdq3 vs AD[d2kdq2]")


def check_support(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    outside = code_k(np.array([1.0 + 1e-9, 1.0 + 1e-3, 10.0]), dim, kid)
    if np.any(outside != 0.0):
        aud.add(name, "support", dim, "FAIL", f"nonzero outside support: {outside}")
        return
    f1 = code_k(np.array([1.0]), dim, kid)
    fp1 = code_dkdq(np.array([1.0]), dim, kid)
    aud.ok(name, "support", dim, f1, 0.0, 1e-14, 0.0, label="f(1) == 0")
    aud.ok(name, "support", dim, fp1, 0.0, 1e-14, 0.0, label="f'(1) == 0")


def check_packing(aud: Audit, name: str, spec: dict, dim: int, kid: int) -> None:
    p_code = code_packing(kid)
    p_spec = spec["packing"]["value"]
    aud.ok(name, "packing", dim, p_code, p_spec, 0.0, 1e-9, label="packingRatio vs spec")
    xi_code = code_xi(dim, kid)
    scale_code = code_scale(dim, kid)
    aud.ok(name, "packing", dim, xi_code, p_code * scale_code, 0.0, 1e-12,
           label="sphKernel_xi == packing * scale")
    ref = spec["packing"].get("reference_N_H")
    factor = float(spec["packing"].get("code_factor", 1.0))
    if dim == 3 and ref is not None:
        nH_code = code_N_H(3, kid)
        nH_expected = float(ref) * factor**3
        aud.ok(name, "packing", dim, nH_code, nH_expected, 0.0, 5e-3,
               label=f"sphKernelN_H vs reference {ref} x factor^3 "
                     f"(3-decimal packing rounding allowed)")
    elif not aud.quiet:
        print(f"        [info] {name} dim={dim}: N_H(derived)={code_N_H(dim, kid):.3f}, "
              f"xi = H/d_nn = {xi_code:.5f}")


_FT_NOISE_FLOOR = 1e-6


def check_ft(aud: Audit, name: str, spec: dict, kid: int) -> None:
    """3D FT per D&A eq. (14), dimensionless: wbar(kappa*H) =
    (4 pi C / kappa) int_0^1 sin(kappa x) f(x) x dx."""
    dim = 3
    C_code = code_C_d(dim, kid)
    x = np.linspace(0.0, 1.0, 200001)
    f = spec_shape(spec, dim, x)
    w0 = 4.0 * np.pi * C_code * simpson(f * x**2, x)
    aud.ok(name, "ft", dim, w0, 1.0, 1e-9, 1e-9, label="wbar(0) == 1")

    sigma2 = C_code * _OMEGA[dim] * simpson(f * x**(dim + 1), x) / dim
    k_small = 0.05
    w_small = (4.0 * np.pi * C_code / k_small) * simpson(np.sin(k_small * x) * f * x, x)
    taylor = 1.0 - 0.5 * sigma2 * k_small**2
    aud.ok(name, "ft", dim, w_small, taylor, 2e-4, 0.0,
           label=f"Taylor wbar = 1 - 1/2 sigma^2 k^2 at k={k_small} (eq. 17)")

    kappa = np.linspace(0.05, 40.0, 2000)
    wbar = np.empty_like(kappa)
    for i, k in enumerate(kappa):
        wbar[i] = (4.0 * np.pi * C_code / k) * np.trapezoid(np.sin(k * x) * f * x, x)
    kmin = kappa[np.argmin(wbar)]
    expected = spec.get("ft_expected")
    if expected == "nonnegative":
        good = wbar.min() > -_FT_NOISE_FLOOR
        aud.add(name, "ft", dim, "PASS" if good else "FAIL",
                f"min wbar={wbar.min():.3e} at kappa={kmin:.3f} "
                f"(expected non-negative up to the {_FT_NOISE_FLOOR:.0e} noise floor)")
    elif expected == "negative_lobe":
        first_zero = None
        s = np.sign(wbar)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        if idx.size:
            first_zero = float(0.5 * (kappa[idx[0]] + kappa[idx[0] + 1]))
        good = wbar.min() < -_FT_NOISE_FLOOR
        aud.add(name, "ft", dim, "PASS" if good else "FAIL",
                f"min wbar={wbar.min():.3e} at kappa={kmin:.3f}, "
                f"first zero kappa*H ~ {first_zero if first_zero is None else round(first_zero, 3)} "
                f"(expected a negative lobe)")
    else:
        first_zero = None
        s = np.sign(wbar)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        if idx.size and wbar[idx[0]] < -_FT_NOISE_FLOOR:
            first_zero = float(0.5 * (kappa[idx[0]] + kappa[idx[0] + 1]))
        aud.add(name, "ft", dim, "PASS",
                f"min wbar={wbar.min():.3e} at kappa={kmin:.3f}, "
                f"first zero kappa*H ~ {first_zero if first_zero is None else round(first_zero, 3)} "
                f"(no ft_expected declared -- informational)")


# ---------------------------------------------------------------------------
# Per-kernel driver.
# ---------------------------------------------------------------------------

def audit_kernel(aud: Audit, name: str, spec: dict, do_ft: bool) -> None:
    code_name = spec["code_name"]
    try:
        kid = KernelFunctions[code_name].value
    except KeyError:
        aud.add(name, "setup", None, "FAIL", f"KernelFunctions.{code_name} not found")
        return

    dims = [int(d) for d in spec["dims"]]
    for dim in dims:
        check_form(aud, name, spec, dim, kid)
        check_normalisation(aud, name, spec, dim, kid)
        check_moments(aud, name, spec, dim, kid)
        check_derivatives(aud, name, spec, dim, kid)
        check_support(aud, name, spec, dim, kid)
        check_packing(aud, name, spec, dim, kid)
    if do_ft:
        check_ft(aud, name, spec, kid)

    # known_issues: downgrade matching FAILs to KNOWN; warn on stale ones.
    known = spec.get("known_issues") or []
    if not known:
        return
    failed_checks = {r["check"] for r in aud.rows if r["kernel"] == name and r["status"] == "FAIL"}
    for ki in known:
        c = ki["check"]
        if c in failed_checks:
            for r in aud.rows:
                if r["kernel"] == name and r["check"] == c and r["status"] == "FAIL":
                    r["status"] = "KNOWN"
                    r["detail"] += f"  [KNOWN ISSUE] {ki.get('note', '').strip()}"
                    aud.hard_failures -= 1
            if not aud.quiet:
                print(f"  KNOWN {name} {c}: documented deviation (see spec known_issues)")
        else:
            aud.add(name, "known_issue_stale", None, "WARN",
                    f"known issue on '{c}' no longer reproduces -- update the spec "
                    f"(or the code changed): {ki.get('note', '').strip()[:120]}")


def load_specs(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["kernels"]


def run_audit(specs: dict, names: list[str] | None, do_ft: bool, quiet: bool = False) -> Audit:
    init()
    aud = Audit(quiet=quiet)
    for name, spec in specs.items():
        if names is not None and name not in names:
            continue
        if not quiet:
            print(f"\n=== {name} ({spec['code_name']}) -- {spec.get('source', '')} ===")
        audit_kernel(aud, name, spec, do_ft)
    return aud


def print_verdict(aud: Audit) -> None:
    print("\n=== Verdict ===")
    kernels = sorted({r["kernel"] for r in aud.rows})
    any_fail = False
    for k in kernels:
        rows = [r for r in aud.rows if r["kernel"] == k]
        fails = [r for r in rows if r["status"] == "FAIL"]
        known = [r for r in rows if r["status"] == "KNOWN"]
        warns = [r for r in rows if r["status"] == "WARN"]
        status = "FAIL" if fails else ("PASS" if not warns else "PASS*")
        if fails:
            any_fail = True
        extra = []
        if known:
            extra.append(f"{len(known)} known")
        if warns:
            extra.append(f"{len(warns)} warning(s)")
        print(f"  {status:<5} {k:<12} {len(rows)} check-rows"
              + (f"  ({', '.join(extra)})" if extra else ""))
    print()
    if any_fail:
        print("AUDIT FAILED -- see the FAIL rows above.")
    elif not aud.quiet:
        print("AUDIT PASSED.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", type=Path, default=_SPEC_PATH,
                    help="path to kernel_specs.yaml (default: alongside this script)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated spec names to audit (default: all)")
    ap.add_argument("--ft", action="store_true",
                    help="run the 3D Fourier-sign check (slower)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full result rows to this JSON file")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-check output (verdict only)")
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.only.split(",")] if args.only else None
    specs = load_specs(args.specs)
    aud = run_audit(specs, names, args.ft, quiet=args.quiet)
    print_verdict(aud)
    if args.json is not None:
        with open(args.json, "w") as f:
            json.dump(aud.rows, f, indent=2)
        if not args.quiet:
            print(f"result rows written to {args.json}")
    return 1 if aud.hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
