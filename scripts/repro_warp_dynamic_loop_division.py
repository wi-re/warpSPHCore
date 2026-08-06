#!/usr/bin/env python3
"""Minimal reproduction: a nonlinear op (division) applied to a dynamic-loop
accumulator, *inside the same @wp.func that contains the loop*, produces NaN
gradients -- even though Warp's own documented workaround pattern for dynamic
loops (move the loop into a separate @wp.func) is followed correctly.

Environment this was observed on:
    warp-lang == 1.15.0
    torch     == 2.13.0+cu130
    Python    == 3.14.6
    Linux (WSL2), both CPU and CUDA devices available in the process

Run:
    python repro_warp_dynamic_loop_division.py

--------------------------------------------------------------------------
Status: OPEN. Found while gradient-checking sphWarpCore's CRK-volume kernel
(src/sphWarpCore/crk/crk_volume.py, since fixed by restructuring around this
-- see warpier_core.md's "Landing CRK's dual-path rework"). Not yet reported
upstream to warp-lang. This script isolates it down to ~20 lines with no
sphWarpCore dependency at all, for that report and as a standing regression
guard once/if it's fixed.
--------------------------------------------------------------------------

Background (Warp's own docs, "Limitations and Workarounds in Differentiability",
https://nvidia.github.io/warp/stable/user_guide/differentiability.html):
dynamic loops (loop trip count a runtime value, not a compile-time constant)
are "not replayed or unrolled in the backward pass, meaning intermediate
values that are meant to be computed in the loop ... are not updated." The
documented workaround is to move the loop body into a separate @wp.func and
consume its *result* from the calling kernel:

    @wp.func
    def loop(x, weights, iters):
        sum = float(0.0)
        norm = float(0.0)
        for i in range(iters):
            w = weights[i]
            norm += w
            sum += x[i] * w
        return sum, norm

    @wp.kernel
    def dynamic_loop_sum(...):
        sum, norm = loop(x, weights, iters)
        l = sum / norm                        # division in the KERNEL, not in loop()
        wp.atomic_add(loss, 0, l)

This script follows that pattern to the letter -- the loop lives in its own
@wp.func, and the reduction it computes (a plain sum, only ever combined via
+=) is exactly the kind of thing the workaround is supposed to make safe.
The ONLY variable is *where the division by that sum happens*:

    variant "func"   -- division happens inside the SAME @wp.func as the loop
                        (return sum / count)
    variant "kernel" -- division happens in the calling @wp.kernel, one
                        level up, exactly like the docs' own l = sum / norm

Both variants compute the identical forward value (y = 1 / sum(x)) and, by
hand, the identical analytic gradient (dy/dx_i = -1 / sum(x)**2 for every i).
"func" gets -inf gradients; "kernel" gets the correct ones. This holds even
for a loop with a SINGLE iteration (n=1) -- so it isn't about a long loop
needing "replay" of many per-iteration intermediates (the division doesn't
touch any per-iteration value at all, only the loop's final accumulated
total) -- which is why this doesn't look like a straightforward instance of
the documented per-iteration-intermediate limitation above.

Two extra variants (below) pin down the actual shape of the bug precisely:
"scale by constant" (a LINEAR read of the accumulator, `total * 3.0`, still
inside the looped function) is PASS -- linear post-loop ops inside the
looped function are fine. "square" (`total * total`, NONLINEAR, no division
at all) is FAIL -- but silently wrong (returns 0.0) rather than -inf/nan.

That -inf-for-reciprocal / 0.0-for-square pairing is the tell -- and it's
directly CONFIRMED below, not just inferred: d(1/S)/dS diverges to -inf as
S->0, and is exactly -1.0 at S=1. `sum_then_divide_INSIDE_init0` and
`sum_then_divide_INSIDE_init1` are identical except for the accumulator's
pre-loop initial value (0.0 vs. 1.0) -- if the backward pass reads that
pre-loop value instead of the loop's true final sum, the observed gradient
should be exactly -1/init_value**2, independent of x, n, or the loop at all.
That is exactly what's observed (see the "mechanism confirmation" section of
this script's output): init0 gives -inf, init1 gives exactly -1.0, for every
n in {1, 3, 7}. So this isn't "dynamic loops are sometimes flaky under
autodiff" -- it's a precise, reproducible statement: a NONLINEAR op reading
a dynamic-loop accumulator, inside the same @wp.func as the loop, has its
local backward derivative evaluated against the accumulator's PRE-LOOP value,
not its post-loop value. A linear op's local derivative is a constant
independent of the accumulator's value, so it can't be affected by that
value being wrong -- which is exactly why "scale by constant" passes.
Returning the raw accumulator and doing the nonlinear op one level up (a
different @wp.func / @wp.kernel scope) sidesteps it entirely, which is also
why the documented workaround (move the loop into its own @wp.func, consume
the *result* from the caller) already happens to dodge this -- as long as
the caller does something nonlinear with that result, not the function
containing the loop itself.
"""

import torch
import warp as wp

wp.init()
wp.set_device("cpu")


# --------------------------------------------------------------------------
# Variant "func": division INSIDE the same @wp.func as the dynamic loop.
# Two sub-variants, differing ONLY in the accumulator's pre-loop initial
# value (0.0 vs. 1.0), to directly confirm (not just infer) that the
# backward pass reads that pre-loop value rather than the loop's true final
# accumulated total: init0 -> d(1/S)/dS blows up at S=0 (-inf observed);
# init1 -> d(1/S)/dS = -1/S**2 = -1.0 at S=1 (-1.0 observed, exactly).
# --------------------------------------------------------------------------
@wp.func
def sum_then_divide_INSIDE_init0(x: wp.array(dtype=wp.float64), n: wp.int32):
    total = wp.float64(0.0)
    for i in range(n):
        total += x[i]
    return wp.float64(1.0) / total


@wp.kernel
def kernel_divide_inside_init0(x: wp.array(dtype=wp.float64), n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    out[0] = sum_then_divide_INSIDE_init0(x, n[0])


@wp.func
def sum_then_divide_INSIDE_init1(x: wp.array(dtype=wp.float64), n: wp.int32):
    total = wp.float64(1.0)  # only difference from init0: pre-loop initial value
    for i in range(n):
        total += x[i]
    return wp.float64(1.0) / total


@wp.kernel
def kernel_divide_inside_init1(x: wp.array(dtype=wp.float64), n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    out[0] = sum_then_divide_INSIDE_init1(x, n[0])


# --------------------------------------------------------------------------
# Variant "kernel": loop returns the raw sum; division happens one level up,
# in the @wp.kernel -- the pattern the Warp docs' own example uses.
# --------------------------------------------------------------------------
@wp.func
def sum_only(x: wp.array(dtype=wp.float64), n: wp.int32):
    total = wp.float64(0.0)
    for i in range(n):
        total += x[i]
    return total


@wp.kernel
def kernel_divide_outside(x: wp.array(dtype=wp.float64), n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    total = sum_only(x, n[0])
    out[0] = wp.float64(1.0) / total  # division here, outside sum_only


# --------------------------------------------------------------------------
# Two probes: how far does "nonlinear op on the accumulator, inside the
# looped function" reach? Multiply-by-external-scalar (linear, doesn't need
# the accumulator's own value nonlinearly) vs. square (nonlinear, no
# division involved) -- both still fully inside the looped @wp.func.
# --------------------------------------------------------------------------
@wp.func
def sum_then_scale_INSIDE(x: wp.array(dtype=wp.float64), n: wp.int32, scale: wp.float64):
    total = wp.float64(0.0)
    for i in range(n):
        total += x[i]
    return total * scale  # linear in `total`


@wp.kernel
def kernel_scale_inside(x: wp.array(dtype=wp.float64), n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    out[0] = sum_then_scale_INSIDE(x, n[0], wp.float64(3.0))


@wp.func
def sum_then_square_INSIDE(x: wp.array(dtype=wp.float64), n: wp.int32):
    total = wp.float64(0.0)
    for i in range(n):
        total += x[i]
    return total * total  # nonlinear in `total`, but no division


@wp.kernel
def kernel_square_inside(x: wp.array(dtype=wp.float64), n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    out[0] = sum_then_square_INSIDE(x, n[0])


def run(kernel, x_vals, analytic_grad_fn):
    """Launches `kernel` with a dynamic (runtime-valued) loop count, seeds the
    output adjoint to 1.0, and returns (forward_value, dx, analytic_dx)."""
    n = len(x_vals)
    x = wp.array(x_vals, dtype=wp.float64, requires_grad=True)
    n_arr = wp.array([n], dtype=wp.int32)  # runtime value, not a Python literal -- genuinely dynamic trip count
    out = wp.zeros(1, dtype=wp.float64, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[x, n_arr, out])
    out.grad.fill_(1.0)
    tape.backward(grads={out: out.grad})

    forward_value = wp.to_torch(out).clone()
    dx = wp.to_torch(x.grad).clone()
    analytic_dx = analytic_grad_fn(x_vals)
    return forward_value, dx, analytic_dx


def analytic_reciprocal_grad(x_vals):
    s = sum(x_vals)
    return torch.tensor([-1.0 / (s * s)] * len(x_vals), dtype=torch.float64)


def stale_accumulator_reciprocal_grad(init_value, x_vals):
    """The gradient you'd get if the backward pass evaluated d(1/S)/dS at the
    accumulator's PRE-LOOP initial value instead of its true final sum --
    the (now confirmed, not just hypothesized) failure mechanism. dS/dx_i is
    unaffected (the += accumulation itself is fine per Warp's docs), so this
    is just -1/init_value**2 repeated for every x_i, independent of x_i or n."""
    stale = float("-inf") if init_value == 0.0 else -1.0 / (init_value * init_value)
    return torch.tensor([stale] * len(x_vals), dtype=torch.float64)


def analytic_scale_grad(x_vals):
    return torch.tensor([3.0] * len(x_vals), dtype=torch.float64)


def analytic_square_grad(x_vals):
    s = sum(x_vals)
    return torch.tensor([2.0 * s] * len(x_vals), dtype=torch.float64)


def check(label, kernel, x_vals, analytic_fn):
    """Correctness check: does the observed gradient match the TRUE analytic one?"""
    forward_value, dx, analytic_dx = run(kernel, x_vals, analytic_fn)
    ok = bool(torch.isfinite(dx).all()) and torch.allclose(dx, analytic_dx, atol=1e-8)
    status = "PASS" if ok else ("FAIL (nan/inf)" if not torch.isfinite(dx).all() else "FAIL (wrong value)")
    print(f"{label:<55} n={len(x_vals):<3} dx={dx.tolist()}  expected={analytic_dx.tolist()}  {status}")
    return ok


def check_confirms_stale_accumulator(label, kernel, x_vals, predicted_stale_dx):
    """Mechanism check (not a correctness check): does the observed WRONG gradient
    match the value predicted by "backward reads the accumulator's pre-loop initial
    value instead of its true final sum"? A match here doesn't mean the kernel is
    correct -- it means the failure mode is understood precisely, not just NaN noise."""
    forward_value, dx, _ = run(kernel, x_vals, lambda _: predicted_stale_dx)
    confirms = torch.allclose(dx, predicted_stale_dx, equal_nan=True)
    print(f"{label:<55} n={len(x_vals):<3} dx={dx.tolist()}  predicted(stale)={predicted_stale_dx.tolist()}  {'CONFIRMS staleness theory' if confirms else 'does NOT match prediction'}")
    return confirms


def main():
    print(f"{'variant':<55}{'':<5}")
    print("-" * 100)

    for n in (1, 3, 7):
        x_vals = [1.0 + 0.3 * i for i in range(n)]
        check("reciprocal, division INSIDE looped func (init 0.0)", kernel_divide_inside_init0, x_vals, analytic_reciprocal_grad)
        check("reciprocal, division INSIDE looped func (init 1.0)", kernel_divide_inside_init1, x_vals, analytic_reciprocal_grad)
        check("reciprocal, division OUTSIDE (in kernel)", kernel_divide_outside, x_vals, analytic_reciprocal_grad)
        check("scale by constant, INSIDE looped func", kernel_scale_inside, x_vals, analytic_scale_grad)
        check("square, INSIDE looped func (nonlinear, no division)", kernel_square_inside, x_vals, analytic_square_grad)
        print()

    print("=" * 100)
    print("Mechanism confirmation: only the accumulator's pre-loop initial value changes below.")
    print("If the observed gradient tracks the prediction exactly, the failure mode is pinned down,")
    print("not just 'sometimes NaN'.")
    print("-" * 100)
    for n in (1, 3, 7):
        x_vals = [1.0 + 0.3 * i for i in range(n)]
        check_confirms_stale_accumulator("division INSIDE, init 0.0", kernel_divide_inside_init0, x_vals, stale_accumulator_reciprocal_grad(0.0, x_vals))
        check_confirms_stale_accumulator("division INSIDE, init 1.0", kernel_divide_inside_init1, x_vals, stale_accumulator_reciprocal_grad(1.0, x_vals))
        print()

    print("=" * 100)
    print("Expected result: both 'division INSIDE' rows FAIL correctness (init 0.0 -> -inf,")
    print("init 1.0 -> exactly -1.0); 'division OUTSIDE' and 'scale by constant INSIDE' PASS;")
    print("'square INSIDE' FAILs with a silent 0.0. The mechanism-confirmation rows below that")
    print("should all read CONFIRMS staleness theory. If so, this is the confirmed minimal repro")
    print("-- see this script's docstring for what it means.")
    print("=" * 100)


if __name__ == "__main__":
    main()
