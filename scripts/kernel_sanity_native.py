#!/usr/bin/env python3
"""Sanity checks for the raw SPH kernel functions themselves (src/warpSPHCore/kernels/).

Every existing gradcheck_*.py script validates a full *operator* (Density,
Interpolate, Gradient, ...) end to end. None of them isolate the smoothing
kernel math underneath: eval_k/eval_dkdq/eval_d2kdq2/eval_d3kdq3 (the
per-kernel-family dispatch in eval_kernel.py), sphKernel_/sphGradient_/
sphKernelDerivative_ (the raw pairwise kernel + its derivatives, before any
domain/support-scheme wrapping), and the twelve KernelFunctions families
themselves (Wendland2/4/6, CubicSpline, QuarticSpline, QuinticSpline, B7,
Poly6, Spiky, ViscosityKernel, CohesionKernel, AdhesionKernel).

This is Step 0 of the forward-mode/adjoint plan in warpier_fields.md Section
3.6: positions enter every SPH operator through a genuinely nonlinear kernel
function of |x_i - x_j|/h, so Tier 2 (tangents w.r.t. positions/supports)
needs a hand-derived adjoint per kernel family. Before deriving those
adjoints, this script establishes that the forward kernels are what they
claim to be:

1. **Dispatch identity** -- eval_k(q, dim, K.value) must equal K's own
   directly-called _k/_dkdq/_d2kdq2/_d3kdq3 function, for every
   KernelFunctions member. This is a pure code-path check, no math
   involved, and it is the check that catches wiring bugs (see below).
2. **Normalization** -- every genuine SPH interpolation kernel (the eight
   families meant to be used as W(r,h) itself: Wendland2/4/6, CubicSpline,
   QuarticSpline, QuinticSpline, B7, Poly6, Spiky) must integrate to 1 over
   its support, in dim 1/2/3. ViscosityKernel/CohesionKernel/AdhesionKernel
   are deliberately excluded: they are Monaghan/Akinci-style special-purpose
   kernels (viscosity dissipation, surface-tension cohesion/adhesion) that
   are not normalized interpolation kernels by design.
3. **Derivative-chain sanity** -- the hand-written eval_dkdq/eval_d2kdq2/
   eval_d3kdq3 must equal the automatic (reverse-mode, wp.Tape) derivative
   of eval_k/eval_dkdq/eval_d2kdq2 respectively, for every kernel family.
4. **Kernel-gradient sanity** -- sphGradient_(x, h, kernel), the hand-written
   vector gradient w.r.t. position, must equal the automatic gradient of
   sphKernel_(x, h, kernel) w.r.t. x (a genuine wp.Tape backward through the
   position argument, not just through q). This is the actual quantity a
   Tier-2 adjoint has to reproduce, so it is the most direct rehearsal for
   deriving one. sphKernelDerivative_ (the scalar dW/d|x|) is cross-checked
   against the norm of that same gradient as a free extra assertion.
5. **dK/dh sanity** -- sphKernelDkDh_(x, h, kernel), the hand-written
   derivative w.r.t. the smoothing length (the grad-h / Omega correction
   term), must equal the automatic derivative of sphKernel_(x, h, kernel)
   w.r.t. h.
6. **Hessian sanity** -- sphKernelHessian_(x, h, kernel), the hand-written
   d(dim x dim) matrix, must equal the AD Jacobian of sphGradient_ w.r.t. x
   (one wp.Tape backward per matrix row, seeded on the already-AD-verified
   gradient kernel -- so this is the second derivative of an
   analytically-exact first derivative, not a finite-difference
   approximation, and stays tight even at float64). sphKernelLaplacian_ is
   cross-checked as trace(Hessian) for the same samples -- Laplacian is
   otherwise the only one of the three actually wired into a production
   operator (coreOperations/wp_laplacian.py's Brookshaw scheme), so this is
   also an indirect regression guard for that operator's math.
7. **Compact-support boundary** -- outside q=|x|/h > 1, sphKernel_ and
   sphGradient_ both explicitly return zero. sphKernelHessian_ and
   sphKernelDkDh_ did not carry the same explicit check (see the bug list
   below); sphKernelLaplacian_ already did.
8. **d(grad W)/dh sanity** -- sphGradientDkDh_(x, h, kernel) (new: the mixed
   partial d(sphGradient_)/dh, kernels/gradH.py) must equal the automatic
   derivative of sphGradient_(x, h, kernel) w.r.t. h. sphKernelDkDh_ already
   covers d(kernel value)/dh; this is the piece needed to propagate a
   support-length tangent through any operator built on the kernel
   *gradient* rather than the value (Gradient/Divergence/Curl/Laplacian),
   which none of the existing kernels/ functions provided before this.

**Bugs found writing this script and fixed alongside it** (all confirmed
dead/unreachable in this repo and in ~/dev/warpSPH before being touched --
see git history for the full before/after):

* `KernelFunctions.ViscosityKernel/CohesionKernel/AdhesionKernel` collided
  with `CubicSpline/QuarticSpline/QuinticSpline` (all shared 30/31/32).
  `eval_k`'s `elif` chain checks CubicSpline before ViscosityKernel, so any
  future caller passing `KernelFunctions.ViscosityKernel` would silently get
  CubicSpline's formula instead. Fixed by renumbering the three special
  kernels to 40/41/42. Check 1 above is the standing regression guard.
* `sphKernelC_d` (src/warpSPHCore/kernels/properties.py) called
  `eval_C_d(kernel, dim)`, but `eval_C_d`'s signature is `(dim, kernel)` --
  the two arguments were swapped, silently returning a different kernel
  family's constant for the wrong dimension. Fixed by swapping the call
  site to match `eval_C_d`'s actual signature (which `kernel.py`'s
  `sphKernel_` already called correctly).
* `Poly6_C_d`/`Spiky_C_d` had wrong normalization constants in 1D and 2D
  (right in 3D, the two kernels' original -- Muller et al. 2003 -- domain).
  Verified against `sympy.integrate` on the actual kernel formula and fixed:
  Poly6 1D 35/16 -> 35/32, 2D 35/(32*pi) -> 4/pi; Spiky 1D 0.25 -> 2,
  2D 2/pi -> 10/pi. Check 2 above is the standing regression guard.
* `adhesionKernel_d3kdq3`'s analytic formula was simply wrong (off by up to
  O(1000), including sign flips, verified against both `sympy`'s exact third
  derivative and finite differences). Re-derived symbolically and fixed.
  Check 3 above is the standing regression guard.
* Poly6's `_k`/`_dkdq`/`_d2kdq2` use raw `iPow`, not the clamping
  `cpow_warp`/`bpow_warp` every other family uses for its base term, so
  Poly6 alone does not naturally vanish for q>1 -- every other family's
  compact support beyond q=1 was, until now, an accident of `cpow_warp`
  clamping rather than a guaranteed property. `sphKernelHessian_` and
  `sphKernelDkDh_` (kernels/hessian.py, kernels/gradH.py) had no explicit
  `q > 1` cutoff (unlike sphKernel_/sphGradient_/sphKernelLaplacian_, which
  all have one), so calling either with Poly6 outside the support radius
  returned a nonzero, wrong value. In production this is masked by the
  neighbor search only ever presenting pairs within the support radius, but
  it is exactly the kind of implicit invariant that an adjoint/perturbation
  method (Tier-2 forward mode, or gradcheck's own finite-difference probing)
  can silently step outside of. Fixed by adding an explicit `q > 1.0`
  cutoff to both functions, matching the other three.
* `sphKernelLaplacian_`/`sphKernelHessian_` both regularized `r` with a
  hardcoded `eps = 1e-5` (`r_eps = r + eps*h`, etc.) applied unconditionally
  to every evaluation, not just the r~0 case it was meant to guard. That
  leaks an O(eps) absolute error into every s/t (Laplacian) or
  factorA/factorB (Hessian) term -- negligible where the true value is
  large, but a multi-percent *relative* error wherever the true Laplacian
  happens to be small (e.g. near an inflection point), which Section G/H's
  sweep hit repeatedly across kernels/dims before the fix. Confirmed by
  scaling eps down and watching the discrepancy shrink linearly. Fixed by
  switching to `get_epsilon(r)` (src/warpSPHCore/math/wp_eps.py, already
  used by norm_grad_warp/norm_hess_warp for exactly this purpose): 1e-15 at
  float64 instead of 1e-5, so the regularization only bites where it is
  actually needed -- true r~0 -- and the `q < eps` self-interaction branch
  (unaffected in intent, since q=0 is still `< eps` at any eps magnitude)
  keeps working exactly as before.

**New function added, not a bug fix**: `sphGradientDkDh_` (kernels/gradH.py)
did not exist before this script -- kernels/ had d(kernel value)/dh
(sphKernelDkDh_) but not d(kernel gradient)/dh, the piece Tier-2 forward
mode needs to propagate a support tangent through Gradient/Divergence/
Curl/Laplacian (which consume grad W, not W itself). Derived by
differentiating sphGradient_'s closed form w.r.t. h (product + chain rule
through q(h) = r/h) and verified against wp.Tape in Section J below, not
trusted from the derivation alone -- same discipline as everything else in
this file.

Run:
    python scripts/kernel_sanity_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import numpy as np
import warp as wp
from warp.types import vector, matrix

from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.eval_kernel import eval_k, eval_dkdq, eval_d2kdq2, eval_d3kdq3, eval_C_d
from warpSPHCore.kernels.kernel import sphKernel_
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.derivative import sphKernelDerivative_
from warpSPHCore.kernels.properties import sphKernelC_d
from warpSPHCore.kernels.gradH import sphKernelDkDh_, sphGradientDkDh_
from warpSPHCore.kernels.hessian import sphKernelHessian_
from warpSPHCore.kernels.laplacian import sphKernelLaplacian_, sphKernelLaplacianGradient_, sphKernelLaplacianDkDh_
from warpSPHCore.kernels.kernelFunctions import (
    wendland2_k, wendland2_dkdq, wendland2_d2kdq2, wendland2_d3kdq3,
    wendland4_k, wendland4_dkdq, wendland4_d2kdq2, wendland4_d3kdq3,
    wendland6_k, wendland6_dkdq, wendland6_d2kdq2, wendland6_d3kdq3,
    cubicSpline_k, cubicSpline_dkdq, cubicSpline_d2kdq2, cubicSpline_d3kdq3,
    quarticSpline_k, quarticSpline_dkdq, quarticSpline_d2kdq2, quarticSpline_d3kdq3,
    quinticSpline_k, quinticSpline_dkdq, quinticSpline_d2kdq2, quinticSpline_d3kdq3,
    B7_k, B7_dkdq, B7_d2kdq2, B7_d3kdq3,
    poly6_k, poly6_dkdq, poly6_d2kdq2, poly6_d3kdq3,
    spiky_k, spiky_dkdq, spiky_d2kdq2, spiky_d3kdq3,
    viscosityKernel_k, viscosityKernel_dkdq, viscosityKernel_d2kdq2, viscosityKernel_d3kdq3,
    cohesionKernel_k, cohesionKernel_dkdq, cohesionKernel_d2kdq2, cohesionKernel_d3kdq3,
    adhesionKernel_k, adhesionKernel_dkdq, adhesionKernel_d2kdq2, adhesionKernel_d3kdq3,
)

DEVICE = "cpu"

# Kernels meant to be used as a normalized interpolation kernel W(r,h).
# ViscosityKernel/CohesionKernel/AdhesionKernel are excluded from this list
# (and from the normalization check) -- they are special-purpose weighting
# functions, not normalized SPH kernels; see the module docstring.
SPH_KERNELS = [
    KernelFunctions.Wendland2,
    KernelFunctions.Wendland4,
    KernelFunctions.Wendland6,
    KernelFunctions.CubicSpline,
    KernelFunctions.QuarticSpline,
    KernelFunctions.QuinticSpline,
    KernelFunctions.B7,
    KernelFunctions.Poly6,
    KernelFunctions.Spiky,
]

ALL_KERNELS = SPH_KERNELS + [
    KernelFunctions.ViscosityKernel,
    KernelFunctions.CohesionKernel,
    KernelFunctions.AdhesionKernel,
]

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)
vec3_t = vector(dtype=scalar_t, length=3)
VEC_T = {1: vec1_t, 2: vec2_t, 3: vec3_t}

mat1_t = matrix(shape=(1, 1), dtype=scalar_t)
mat2_t = matrix(shape=(2, 2), dtype=scalar_t)
mat3_t = matrix(shape=(3, 3), dtype=scalar_t)
MAT_T = {1: mat1_t, 2: mat2_t, 3: mat3_t}


# --------------------------------------------------------------------------
# Section A: dispatch identity -- eval_k/eval_dkdq/eval_d2kdq2/eval_d3kdq3
# vs. each family's own directly-named function.
# --------------------------------------------------------------------------

@wp.kernel
def _dispatch4(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t, ndim=2)):
    i = wp.tid()
    out[i, 0] = eval_k(q[i], dim, kernel_id)
    out[i, 1] = eval_dkdq(q[i], dim, kernel_id)
    out[i, 2] = eval_d2kdq2(q[i], dim, kernel_id)
    out[i, 3] = eval_d3kdq3(q[i], dim, kernel_id)


def _make_direct4(k_fn, dkdq_fn, d2kdq2_fn, d3kdq3_fn):
    @wp.kernel
    def _direct4(q: wp.array(dtype=scalar_t), dim: wp.int32, out: wp.array(dtype=scalar_t, ndim=2)):
        i = wp.tid()
        out[i, 0] = k_fn(q[i], dim)
        out[i, 1] = dkdq_fn(q[i], dim)
        out[i, 2] = d2kdq2_fn(q[i], dim)
        out[i, 3] = d3kdq3_fn(q[i], dim)
    return _direct4


_DIRECT4 = {
    KernelFunctions.Wendland2: _make_direct4(wendland2_k, wendland2_dkdq, wendland2_d2kdq2, wendland2_d3kdq3),
    KernelFunctions.Wendland4: _make_direct4(wendland4_k, wendland4_dkdq, wendland4_d2kdq2, wendland4_d3kdq3),
    KernelFunctions.Wendland6: _make_direct4(wendland6_k, wendland6_dkdq, wendland6_d2kdq2, wendland6_d3kdq3),
    KernelFunctions.CubicSpline: _make_direct4(cubicSpline_k, cubicSpline_dkdq, cubicSpline_d2kdq2, cubicSpline_d3kdq3),
    KernelFunctions.QuarticSpline: _make_direct4(quarticSpline_k, quarticSpline_dkdq, quarticSpline_d2kdq2, quarticSpline_d3kdq3),
    KernelFunctions.QuinticSpline: _make_direct4(quinticSpline_k, quinticSpline_dkdq, quinticSpline_d2kdq2, quinticSpline_d3kdq3),
    KernelFunctions.B7: _make_direct4(B7_k, B7_dkdq, B7_d2kdq2, B7_d3kdq3),
    KernelFunctions.Poly6: _make_direct4(poly6_k, poly6_dkdq, poly6_d2kdq2, poly6_d3kdq3),
    KernelFunctions.Spiky: _make_direct4(spiky_k, spiky_dkdq, spiky_d2kdq2, spiky_d3kdq3),
    KernelFunctions.ViscosityKernel: _make_direct4(viscosityKernel_k, viscosityKernel_dkdq, viscosityKernel_d2kdq2, viscosityKernel_d3kdq3),
    KernelFunctions.CohesionKernel: _make_direct4(cohesionKernel_k, cohesionKernel_dkdq, cohesionKernel_d2kdq2, cohesionKernel_d3kdq3),
    KernelFunctions.AdhesionKernel: _make_direct4(adhesionKernel_k, adhesionKernel_dkdq, adhesionKernel_d2kdq2, adhesionKernel_d3kdq3),
}


def check(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-9, rtol: float = 1e-7) -> bool:
    diff = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), atol)
    rel = diff / denom
    ok = bool(np.all((diff <= atol) | (rel <= rtol)))
    worst = float(np.max(diff)) if diff.size else 0.0
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<55} max|Δ|={worst:.3e}")
    return ok


def section_a_dispatch_identity() -> bool:
    print("\n=== Section A: dispatch identity (eval_k/... vs. each family's direct function) ===")
    # q values avoiding q=0 (viscosityKernel's 0.5/q) and the immediate
    # neighborhood of adhesionKernel's q=0.5 branch boundary (see module
    # docstring on why exact q=0.5 itself is safe: the ">" comparison takes
    # the constant-zero branch there).
    q_np = np.linspace(0.1, 0.95, 18)
    q = wp.array(q_np, dtype=scalar_t, device=DEVICE)
    ok = True
    for dim in (1, 2, 3):
        for kf in ALL_KERNELS:
            direct_out = wp.zeros((len(q_np), 4), dtype=scalar_t, device=DEVICE)
            wp.launch(_DIRECT4[kf], dim=len(q_np), inputs=[q, dim], outputs=[direct_out], device=DEVICE)
            dispatch_out = wp.zeros((len(q_np), 4), dtype=scalar_t, device=DEVICE)
            wp.launch(_dispatch4, dim=len(q_np), inputs=[q, dim, kf.value], outputs=[dispatch_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name}", dispatch_out.numpy(), direct_out.numpy())
    return ok


# --------------------------------------------------------------------------
# Section B: normalization -- integral of W(r,h) dV == 1, h-independent
# after the r = h*q substitution, so only q in [0, 1] is swept.
# --------------------------------------------------------------------------

@wp.kernel
def _dispatch_k_only(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], dim, kernel_id)


@wp.kernel
def _eval_Cd_kernel(dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = eval_C_d(dim, kernel_id)


def _composite_simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = len(x) - 1
    assert n % 2 == 0, "composite Simpson needs an even number of intervals"
    h = (x[-1] - x[0]) / n
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return float(s * h / 3.0)


_OMEGA = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}


def section_b_normalization() -> bool:
    print("\n=== Section B: normalization (integral of W(r,h) dV == 1) ===")
    n = 200001  # 200000 intervals, even, for composite Simpson
    q_np = np.linspace(0.0, 1.0, n)
    q = wp.array(q_np, dtype=scalar_t, device=DEVICE)
    ok = True
    for dim in (1, 2, 3):
        for kf in SPH_KERNELS:
            out = wp.zeros(n, dtype=scalar_t, device=DEVICE)
            wp.launch(_dispatch_k_only, dim=n, inputs=[q, dim, kf.value], outputs=[out], device=DEVICE)
            k_vals = out.numpy()
            weighted = k_vals * (q_np ** (dim - 1))
            moment = _composite_simpson(weighted, q_np)

            cd_out = wp.zeros(1, dtype=scalar_t, device=DEVICE)
            wp.launch(_eval_Cd_kernel, dim=1, inputs=[dim, kf.value], outputs=[cd_out], device=DEVICE)
            cd = float(cd_out.numpy()[0])

            integral = cd * _OMEGA[dim] * moment
            ok &= check(f"dim={dim} {kf.name} ∫W dV", np.array([integral]), np.array([1.0]), atol=1e-6, rtol=1e-6)
    return ok


# --------------------------------------------------------------------------
# Section C: derivative-chain sanity via reverse-mode AD (wp.Tape).
# eval_dkdq must equal d/dq[eval_k], eval_d2kdq2 must equal d/dq[eval_dkdq],
# eval_d3kdq3 must equal d/dq[eval_d2kdq2] -- for every kernel family.
# --------------------------------------------------------------------------

@wp.kernel
def _dispatch_k(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_k(q[i], dim, kernel_id)


@wp.kernel
def _dispatch_dkdq(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_dkdq(q[i], dim, kernel_id)


@wp.kernel
def _dispatch_d2kdq2(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_d2kdq2(q[i], dim, kernel_id)


@wp.kernel
def _dispatch_d3kdq3(q: wp.array(dtype=scalar_t), dim: wp.int32, kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = eval_d3kdq3(q[i], dim, kernel_id)


def _forward_and_ad_derivative(kernel_fn, q_np: np.ndarray, dim: int, kernel_id: int):
    """Elementwise: since out[i] depends only on q[i], seeding every output
    adjoint with 1.0 gives d(out[i])/d(q[i]) directly, for every i in one
    backward pass -- no cross terms to worry about."""
    q = wp.array(q_np, dtype=scalar_t, requires_grad=True, device=DEVICE)
    out = wp.zeros(len(q_np), dtype=scalar_t, requires_grad=True, device=DEVICE)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel_fn, dim=len(q_np), inputs=[q, dim, kernel_id], outputs=[out], device=DEVICE)
    seed = wp.array(np.ones(len(q_np)), dtype=scalar_t, device=DEVICE)
    tape.backward(grads={out: seed})
    grad = q.grad.numpy().copy()
    forward = out.numpy().copy()
    tape.zero()
    return forward, grad


def section_c_derivative_chain() -> bool:
    print("\n=== Section C: derivative chain (analytic dkdq/d2kdq2/d3kdq3 vs. AD of the level below) ===")
    q_np = np.linspace(0.1, 0.95, 18)
    ok = True
    for dim in (1, 2, 3):
        for kf in ALL_KERNELS:
            _, ad_dkdq = _forward_and_ad_derivative(_dispatch_k, q_np, dim, kf.value)
            analytic_dkdq, _ = _forward_and_ad_derivative(_dispatch_dkdq, q_np, dim, kf.value)
            ok &= check(f"dim={dim} {kf.name} dkdq == d/dq[k]", analytic_dkdq, ad_dkdq)

            _, ad_d2kdq2 = _forward_and_ad_derivative(_dispatch_dkdq, q_np, dim, kf.value)
            analytic_d2kdq2, _ = _forward_and_ad_derivative(_dispatch_d2kdq2, q_np, dim, kf.value)
            ok &= check(f"dim={dim} {kf.name} d2kdq2 == d/dq[dkdq]", analytic_d2kdq2, ad_d2kdq2)

            _, ad_d3kdq3 = _forward_and_ad_derivative(_dispatch_d2kdq2, q_np, dim, kf.value)
            analytic_d3kdq3, _ = _forward_and_ad_derivative(_dispatch_d3kdq3, q_np, dim, kf.value)
            ok &= check(f"dim={dim} {kf.name} d3kdq3 == d/dq[d2kdq2]", analytic_d3kdq3, ad_d3kdq3)
    return ok


# --------------------------------------------------------------------------
# Section D: kernel-gradient sanity. sphGradient_(x, h, kernel) -- the
# hand-written vector gradient w.r.t. position -- must equal the automatic
# gradient of sphKernel_(x, h, kernel) w.r.t. x. This is the exact quantity
# a Tier-2 (position-tangent) adjoint has to reproduce.
# --------------------------------------------------------------------------

# Explicit, one-kernel-per-dim (not a closure factory): warp resolves a
# @wp.kernel's array-dtype annotations by re-evaluating the annotation
# string against the function's globals/closure at decoration time, which
# does not reliably see a dtype captured only via an enclosing function's
# local variable. Top-level definitions with the dtype baked in by name are
# the safe, established pattern here (matches struct-per-dim elsewhere in
# this repo, e.g. particleDataSoA_1/_2/_3).

@wp.kernel
def _eval_sphKernel_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernel_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernel_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernel_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernel_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernel_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphGradient_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec1_t)):
    i = wp.tid()
    out[i] = sphGradient_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphGradient_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec2_t)):
    i = wp.tid()
    out[i] = sphGradient_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphGradient_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec3_t)):
    i = wp.tid()
    out[i] = sphGradient_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelDerivative_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDerivative_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelDerivative_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDerivative_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelDerivative_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDerivative_(x[i], h[i], kernel_id)


_SPHKERNEL_BY_DIM = {1: _eval_sphKernel_1d, 2: _eval_sphKernel_2d, 3: _eval_sphKernel_3d}
_SPHGRADIENT_BY_DIM = {1: _eval_sphGradient_1d, 2: _eval_sphGradient_2d, 3: _eval_sphGradient_3d}
_SPHDERIVATIVE_BY_DIM = {1: _eval_sphKernelDerivative_1d, 2: _eval_sphKernelDerivative_2d, 3: _eval_sphKernelDerivative_3d}


def _directions(dim: int) -> np.ndarray:
    """A handful of unit directions per dim: axis-aligned plus one diagonal."""
    dirs = list(np.eye(dim))
    if dim > 1:
        diag = np.ones(dim) / np.sqrt(dim)
        dirs.append(diag)
    return np.array(dirs)


def section_d_kernel_gradient() -> bool:
    print("\n=== Section D: sphGradient_ vs. AD gradient of sphKernel_ w.r.t. position ===")
    h_val = 1.0
    q_samples = np.linspace(0.05, 0.95, 10)
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        dirs = _directions(dim)
        for kf in SPH_KERNELS:
            for direction in dirs:
                x_np = np.array([q_samples[j] * h_val * direction for j in range(len(q_samples))])
                h_np = np.full(len(q_samples), h_val)

                x = wp.array(x_np, dtype=vec_t, requires_grad=True, device=DEVICE)
                h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                out = wp.zeros(len(q_samples), dtype=scalar_t, requires_grad=True, device=DEVICE)
                tape = wp.Tape()
                with tape:
                    wp.launch(_SPHKERNEL_BY_DIM[dim], dim=len(q_samples), inputs=[x, h, kf.value], outputs=[out], device=DEVICE)
                seed = wp.array(np.ones(len(q_samples)), dtype=scalar_t, device=DEVICE)
                tape.backward(grads={out: seed})
                ad_grad = x.grad.numpy().copy()
                tape.zero()

                manual_grad_out = wp.zeros(len(q_samples), dtype=vec_t, device=DEVICE)
                wp.launch(_SPHGRADIENT_BY_DIM[dim], dim=len(q_samples), inputs=[x, h, kf.value], outputs=[manual_grad_out], device=DEVICE)
                manual_grad = manual_grad_out.numpy()

                label = f"dim={dim} {kf.name} dir={np.array2string(direction, precision=2)}"
                ok &= check(f"{label} ∇W (AD vs manual)", ad_grad, manual_grad)

                deriv_out = wp.zeros(len(q_samples), dtype=scalar_t, device=DEVICE)
                wp.launch(_SPHDERIVATIVE_BY_DIM[dim], dim=len(q_samples), inputs=[x, h, kf.value], outputs=[deriv_out], device=DEVICE)
                ok &= check(f"{label} |∇W| == |dW/dr|", np.linalg.norm(manual_grad, axis=-1), np.abs(deriv_out.numpy()))
    return ok


# --------------------------------------------------------------------------
# Section E: sphKernelC_d regression guard (properties.py's argument-order
# bug, fixed alongside this script -- see module docstring).
# --------------------------------------------------------------------------

@wp.kernel
def _eval_sphKernelC_d(kernel_id: wp.int32, dim: wp.int32, out: wp.array(dtype=scalar_t)):
    out[0] = sphKernelC_d(kernel_id, dim)


def section_e_sphKernelC_d() -> bool:
    print("\n=== Section E: sphKernelC_d(kernel, dim) vs. eval_C_d(dim, kernel) directly ===")
    ok = True
    for dim in (1, 2, 3):
        for kf in ALL_KERNELS:
            wrapper_out = wp.zeros(1, dtype=scalar_t, device=DEVICE)
            wp.launch(_eval_sphKernelC_d, dim=1, inputs=[kf.value, dim], outputs=[wrapper_out], device=DEVICE)
            direct_out = wp.zeros(1, dtype=scalar_t, device=DEVICE)
            wp.launch(_eval_Cd_kernel, dim=1, inputs=[dim, kf.value], outputs=[direct_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name} sphKernelC_d", wrapper_out.numpy(), direct_out.numpy())
    return ok


# --------------------------------------------------------------------------
# Section F: dK/dh sanity. sphKernelDkDh_(x, h, kernel) -- the hand-written
# derivative w.r.t. smoothing length -- must equal the automatic derivative
# of sphKernel_(x, h, kernel) w.r.t. h.
# --------------------------------------------------------------------------

@wp.kernel
def _eval_sphKernelDkDh_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelDkDh_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelDkDh_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelDkDh_(x[i], h[i], kernel_id)


_SPHDKDH_BY_DIM = {1: _eval_sphKernelDkDh_1d, 2: _eval_sphKernelDkDh_2d, 3: _eval_sphKernelDkDh_3d}


def section_f_dkdh() -> bool:
    print("\n=== Section F: sphKernelDkDh_ vs. AD d(sphKernel_)/dh ===")
    q_samples = np.linspace(0.05, 0.95, 10)
    h_val = 1.0
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        for kf in SPH_KERNELS:
            for direction in _directions(dim):
                x_np = np.array([q_samples[j] * h_val * direction for j in range(len(q_samples))])
                h_np = np.full(len(q_samples), h_val)

                x = wp.array(x_np, dtype=vec_t, device=DEVICE)
                h = wp.array(h_np, dtype=scalar_t, requires_grad=True, device=DEVICE)
                out = wp.zeros(len(q_samples), dtype=scalar_t, requires_grad=True, device=DEVICE)
                tape = wp.Tape()
                with tape:
                    wp.launch(_SPHKERNEL_BY_DIM[dim], dim=len(q_samples), inputs=[x, h, kf.value], outputs=[out], device=DEVICE)
                seed = wp.array(np.ones(len(q_samples)), dtype=scalar_t, device=DEVICE)
                tape.backward(grads={out: seed})
                ad_dWdh = h.grad.numpy().copy()
                tape.zero()

                manual_out = wp.zeros(len(q_samples), dtype=scalar_t, device=DEVICE)
                wp.launch(_SPHDKDH_BY_DIM[dim], dim=len(q_samples), inputs=[x, h, kf.value], outputs=[manual_out], device=DEVICE)

                label = f"dim={dim} {kf.name} dir={np.array2string(direction, precision=2)}"
                ok &= check(f"{label} dW/dh (AD vs manual)", ad_dWdh, manual_out.numpy())
    return ok


# --------------------------------------------------------------------------
# Section G/H: Hessian sanity (sphKernelHessian_ vs. the AD Jacobian of
# sphGradient_ w.r.t. x -- one wp.Tape backward per matrix row, seeded on
# the already-AD-verified gradient kernel from Section D) and Laplacian
# sanity (sphKernelLaplacian_ vs. trace of that same AD Hessian).
# --------------------------------------------------------------------------

@wp.kernel
def _eval_sphKernelHessian_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=mat1_t)):
    i = wp.tid()
    out[i] = sphKernelHessian_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelHessian_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=mat2_t)):
    i = wp.tid()
    out[i] = sphKernelHessian_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelHessian_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=mat3_t)):
    i = wp.tid()
    out[i] = sphKernelHessian_(x[i], h[i], kernel_id)


_SPHHESSIAN_BY_DIM = {1: _eval_sphKernelHessian_1d, 2: _eval_sphKernelHessian_2d, 3: _eval_sphKernelHessian_3d}


@wp.kernel
def _eval_sphKernelLaplacian_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacian_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacian_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacian_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacian_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacian_(x[i], h[i], kernel_id)


_SPHLAPLACIAN_BY_DIM = {1: _eval_sphKernelLaplacian_1d, 2: _eval_sphKernelLaplacian_2d, 3: _eval_sphKernelLaplacian_3d}


def section_gh_hessian_laplacian() -> bool:
    print("\n=== Section G/H: sphKernelHessian_ (AD Jacobian of sphGradient_) and sphKernelLaplacian_ (trace) ===")
    q_samples = np.linspace(0.05, 0.95, 8)
    h_val = 1.0
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        n = len(q_samples)
        for kf in SPH_KERNELS:
            for direction in _directions(dim):
                x_np = np.array([q_samples[j] * h_val * direction for j in range(n)])
                h_np = np.full(n, h_val)

                ad_hessian = np.zeros((n, dim, dim))
                for c in range(dim):
                    x = wp.array(x_np, dtype=vec_t, requires_grad=True, device=DEVICE)
                    h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                    out = wp.zeros(n, dtype=vec_t, requires_grad=True, device=DEVICE)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(_SPHGRADIENT_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[out], device=DEVICE)
                    seed_np = np.zeros((n, dim))
                    seed_np[:, c] = 1.0
                    seed = wp.array(seed_np, dtype=vec_t, device=DEVICE)
                    tape.backward(grads={out: seed})
                    ad_hessian[:, c, :] = x.grad.numpy().copy()
                    tape.zero()

                x = wp.array(x_np, dtype=vec_t, device=DEVICE)
                h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                manual_out = wp.zeros(n, dtype=MAT_T[dim], device=DEVICE)
                wp.launch(_SPHHESSIAN_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[manual_out], device=DEVICE)
                manual_hessian = manual_out.numpy()

                label = f"dim={dim} {kf.name} dir={np.array2string(direction, precision=2)}"
                ok &= check(f"{label} Hessian (AD vs manual)", ad_hessian, manual_hessian)

                laplacian_out = wp.zeros(n, dtype=scalar_t, device=DEVICE)
                wp.launch(_SPHLAPLACIAN_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[laplacian_out], device=DEVICE)
                ad_trace = np.trace(ad_hessian, axis1=1, axis2=2)
                ok &= check(f"{label} Laplacian == trace(AD Hessian)", laplacian_out.numpy(), ad_trace)
    return ok


# --------------------------------------------------------------------------
# Section I: compact-support boundary. Outside q=|x|/h > 1, every one of
# sphKernel_/sphGradient_/sphKernelLaplacian_/sphKernelHessian_/
# sphKernelDkDh_ must return zero -- see the module docstring's bug list for
# why this was not previously guaranteed for the latter two.
# --------------------------------------------------------------------------

def section_i_compact_support_boundary() -> bool:
    print("\n=== Section I: compact-support boundary (q>1 => Hessian/Laplacian/DkDh/GradientDkDh == 0) ===")
    q_outside = np.array([1.05, 1.2, 1.5, 2.0])
    h_val = 1.0
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        direction = np.eye(dim)[0]
        x_np = np.array([q * h_val * direction for q in q_outside])
        h_np = np.full(len(q_outside), h_val)
        x = wp.array(x_np, dtype=vec_t, device=DEVICE)
        h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
        for kf in SPH_KERNELS:
            hess_out = wp.zeros(len(q_outside), dtype=MAT_T[dim], device=DEVICE)
            wp.launch(_SPHHESSIAN_BY_DIM[dim], dim=len(q_outside), inputs=[x, h, kf.value], outputs=[hess_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name} Hessian(q>1)==0", hess_out.numpy(), np.zeros_like(hess_out.numpy()), atol=1e-12)

            lap_out = wp.zeros(len(q_outside), dtype=scalar_t, device=DEVICE)
            wp.launch(_SPHLAPLACIAN_BY_DIM[dim], dim=len(q_outside), inputs=[x, h, kf.value], outputs=[lap_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name} Laplacian(q>1)==0", lap_out.numpy(), np.zeros_like(lap_out.numpy()), atol=1e-12)

            dkdh_out = wp.zeros(len(q_outside), dtype=scalar_t, device=DEVICE)
            wp.launch(_SPHDKDH_BY_DIM[dim], dim=len(q_outside), inputs=[x, h, kf.value], outputs=[dkdh_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name} DkDh(q>1)==0", dkdh_out.numpy(), np.zeros_like(dkdh_out.numpy()), atol=1e-12)

            graddkdh_out = wp.zeros(len(q_outside), dtype=vec_t, device=DEVICE)
            wp.launch(_SPHGRADIENTDKDH_BY_DIM[dim], dim=len(q_outside), inputs=[x, h, kf.value], outputs=[graddkdh_out], device=DEVICE)
            ok &= check(f"dim={dim} {kf.name} GradientDkDh(q>1)==0", graddkdh_out.numpy(), np.zeros_like(graddkdh_out.numpy()), atol=1e-12)
    return ok


# --------------------------------------------------------------------------
# Section J: d(grad W)/dh sanity. sphGradientDkDh_(x, h, kernel) -- the new
# mixed-partial function -- must equal the automatic derivative of
# sphGradient_(x, h, kernel) w.r.t. h. Same per-row-of-a-Jacobian technique
# as Section G's Hessian check, but differentiating w.r.t. the scalar h
# instead of the vector x: for each output component c, one wp.Tape
# backward seeded on that component gives d(grad_c[i])/dh[i] for every
# sample i at once (elementwise dependence, no cross terms).
# --------------------------------------------------------------------------

@wp.kernel
def _eval_sphGradientDkDh_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec1_t)):
    i = wp.tid()
    out[i] = sphGradientDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphGradientDkDh_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec2_t)):
    i = wp.tid()
    out[i] = sphGradientDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphGradientDkDh_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec3_t)):
    i = wp.tid()
    out[i] = sphGradientDkDh_(x[i], h[i], kernel_id)


_SPHGRADIENTDKDH_BY_DIM = {1: _eval_sphGradientDkDh_1d, 2: _eval_sphGradientDkDh_2d, 3: _eval_sphGradientDkDh_3d}


def section_j_gradient_dkdh() -> bool:
    print("\n=== Section J: sphGradientDkDh_ vs. AD d(sphGradient_)/dh ===")
    q_samples = np.linspace(0.05, 0.95, 8)
    h_val = 1.0
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        n = len(q_samples)
        for kf in SPH_KERNELS:
            for direction in _directions(dim):
                x_np = np.array([q_samples[j] * h_val * direction for j in range(n)])
                h_np = np.full(n, h_val)

                ad_dgraddh = np.zeros((n, dim))
                for c in range(dim):
                    x = wp.array(x_np, dtype=vec_t, device=DEVICE)
                    h = wp.array(h_np, dtype=scalar_t, requires_grad=True, device=DEVICE)
                    out = wp.zeros(n, dtype=vec_t, requires_grad=True, device=DEVICE)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(_SPHGRADIENT_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[out], device=DEVICE)
                    seed_np = np.zeros((n, dim))
                    seed_np[:, c] = 1.0
                    seed = wp.array(seed_np, dtype=vec_t, device=DEVICE)
                    tape.backward(grads={out: seed})
                    ad_dgraddh[:, c] = h.grad.numpy().copy()
                    tape.zero()

                x = wp.array(x_np, dtype=vec_t, device=DEVICE)
                h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                manual_out = wp.zeros(n, dtype=vec_t, device=DEVICE)
                wp.launch(_SPHGRADIENTDKDH_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[manual_out], device=DEVICE)

                label = f"dim={dim} {kf.name} dir={np.array2string(direction, precision=2)}"
                ok &= check(f"{label} d(grad W)/dh (AD vs manual)", ad_dgraddh, manual_out.numpy())
    return ok


# --------------------------------------------------------------------------
# Section K: Tier 2.3 (warpier_adjoint.md) -- d(sphKernelLaplacian_)/dx and
# /dh, the LaplacianScheme.Naive JVP building blocks, vs. the automatic
# (wp.Tape) derivative of the already-validated sphKernelLaplacian_ itself.
# Not on any performance-relevant path (Brookshaw, covered by Section G/H's
# Hessian-trace relationship plus Tier 2.2, is what wp_laplacian.py's own
# comments treat as the consistent estimator) -- derived for methodological
# completeness of the adjoint SPH scheme (Naive is nonetheless a real, wired
# -in LaplacianScheme, not a hypothetical one; see the module docstrings on
# kernels/laplacian.py's two new functions). Simpler than Section G/H's per-
# Jacobian-row loop: sphKernelLaplacian_'s output is a scalar, so a single
# backward (seed=1, elementwise-independent samples) gives the whole d/dx
# vector directly, and a second single backward gives the scalar d/dh --
# exactly Section D/F's pattern, not Section G/H's.
# --------------------------------------------------------------------------

@wp.kernel
def _eval_sphKernelLaplacianGradient_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec1_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianGradient_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacianGradient_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec2_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianGradient_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacianGradient_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=vec3_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianGradient_(x[i], h[i], kernel_id)


_SPHLAPLACIANGRAD_BY_DIM = {1: _eval_sphKernelLaplacianGradient_1d, 2: _eval_sphKernelLaplacianGradient_2d, 3: _eval_sphKernelLaplacianGradient_3d}


@wp.kernel
def _eval_sphKernelLaplacianDkDh_1d(x: wp.array(dtype=vec1_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacianDkDh_2d(x: wp.array(dtype=vec2_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianDkDh_(x[i], h[i], kernel_id)


@wp.kernel
def _eval_sphKernelLaplacianDkDh_3d(x: wp.array(dtype=vec3_t), h: wp.array(dtype=scalar_t), kernel_id: wp.int32, out: wp.array(dtype=scalar_t)):
    i = wp.tid()
    out[i] = sphKernelLaplacianDkDh_(x[i], h[i], kernel_id)


_SPHLAPLACIANDKDH_BY_DIM = {1: _eval_sphKernelLaplacianDkDh_1d, 2: _eval_sphKernelLaplacianDkDh_2d, 3: _eval_sphKernelLaplacianDkDh_3d}


def section_k_laplacian_jvp_buildingblocks() -> bool:
    print("\n=== Section K: sphKernelLaplacianGradient_/sphKernelLaplacianDkDh_ vs. AD of sphKernelLaplacian_ (Tier 2.3) ===")
    q_samples = np.linspace(0.05, 0.95, 8)
    h_val = 1.0
    ok = True
    for dim in (1, 2, 3):
        vec_t = VEC_T[dim]
        n = len(q_samples)
        for kf in SPH_KERNELS:
            for direction in _directions(dim):
                x_np = np.array([q_samples[j] * h_val * direction for j in range(n)])
                h_np = np.full(n, h_val)
                seed = wp.array(np.ones(n), dtype=scalar_t, device=DEVICE)

                x = wp.array(x_np, dtype=vec_t, requires_grad=True, device=DEVICE)
                h = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                out = wp.zeros(n, dtype=scalar_t, requires_grad=True, device=DEVICE)
                tape = wp.Tape()
                with tape:
                    wp.launch(_SPHLAPLACIAN_BY_DIM[dim], dim=n, inputs=[x, h, kf.value], outputs=[out], device=DEVICE)
                tape.backward(grads={out: seed})
                ad_dLdx = x.grad.numpy().copy()
                tape.zero()

                x2 = wp.array(x_np, dtype=vec_t, device=DEVICE)
                h2 = wp.array(h_np, dtype=scalar_t, requires_grad=True, device=DEVICE)
                out2 = wp.zeros(n, dtype=scalar_t, requires_grad=True, device=DEVICE)
                tape2 = wp.Tape()
                with tape2:
                    wp.launch(_SPHLAPLACIAN_BY_DIM[dim], dim=n, inputs=[x2, h2, kf.value], outputs=[out2], device=DEVICE)
                tape2.backward(grads={out2: seed})
                ad_dLdh = h2.grad.numpy().copy()
                tape2.zero()

                x3 = wp.array(x_np, dtype=vec_t, device=DEVICE)
                h3 = wp.array(h_np, dtype=scalar_t, device=DEVICE)
                manual_dx = wp.zeros(n, dtype=vec_t, device=DEVICE)
                wp.launch(_SPHLAPLACIANGRAD_BY_DIM[dim], dim=n, inputs=[x3, h3, kf.value], outputs=[manual_dx], device=DEVICE)
                manual_dh = wp.zeros(n, dtype=scalar_t, device=DEVICE)
                wp.launch(_SPHLAPLACIANDKDH_BY_DIM[dim], dim=n, inputs=[x3, h3, kf.value], outputs=[manual_dh], device=DEVICE)

                label = f"dim={dim} {kf.name} dir={np.array2string(direction, precision=2)}"
                ok &= check(f"{label} d(Laplacian)/dx (AD vs manual)", ad_dLdx, manual_dx.numpy())
                ok &= check(f"{label} d(Laplacian)/dh (AD vs manual)", ad_dLdh, manual_dh.numpy())
    return ok


def main() -> None:
    wp.init()

    ok = True
    ok &= section_a_dispatch_identity()
    ok &= section_b_normalization()
    ok &= section_c_derivative_chain()
    ok &= section_d_kernel_gradient()
    ok &= section_e_sphKernelC_d()
    ok &= section_f_dkdh()
    ok &= section_gh_hessian_laplacian()
    ok &= section_j_gradient_dkdh()
    ok &= section_i_compact_support_boundary()
    ok &= section_k_laplacian_jvp_buildingblocks()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see the individual section(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
