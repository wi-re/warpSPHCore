#!/usr/bin/env python3
"""Minimal reproduction of the "ternary array-read zeroing" adjoint bug.

Environment this was observed on:
    warp-lang == 1.12.0
    Python    == 3.13.12
    Linux (WSL2), both CPU and CUDA devices available in the process

Run:
    python repro_ternary_adjoint_zeroing.py

--------------------------------------------------------------------------
Status: CONFIRMED, workaround known (use explicit if/else). Not filed
upstream yet. See docs/lessons_learned.md and warpier_core.md's "Landing
Gradient in production" section for where this actually bit real code
(Interpolate originally, then reintroduced while porting Gradient to its
new unified kernel).
--------------------------------------------------------------------------

The bug: a ternary expression assigned to a local variable, where *both*
branches read the same array at the same index, compiles fine, runs the
correct branch at runtime, and still produces a silently-zero adjoint for
that array -- no error, no warning, correct forward value.

    val = arr[i] / term[i] if flag else arr[i]   # <-- both branches read arr[i]

vs. the safe, equivalent form:

    if flag:
        val = arr[i] / term[i]
    else:
        val = arr[i]

With flag=False at every element, both kernels compute the exact same
forward output (val = arr[i]) -- so d(val)/d(arr) should be 1.0 everywhere
in both cases. Below, the ternary version reports an all-zero gradient for
`arr`; the if/else version reports the correct all-ones gradient.

No torch involved -- this is pure Warp: a single wp.Tape() around a
wp.launch(), reading .grad back directly. `flag` is passed as an ordinary
runtime wp.bool kernel argument (not wp.constant / wp.static), matching how
production code carries it (a bool field on a struct, e.g.
correctionData.useGradHTerms) -- this is not a compile-time-specialization
issue, it reproduces with a genuinely dynamic branch.
"""

import warp as wp

wp.init()


@wp.kernel
def ternary_kernel(
    arr: wp.array(dtype=wp.float64),
    term: wp.array(dtype=wp.float64),
    flag: wp.bool,
    out: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    val = arr[i] / term[i] if flag else arr[i]  # both branches read arr[i] -- the dangerous shape
    out[i] = val


@wp.kernel
def ifelse_kernel(
    arr: wp.array(dtype=wp.float64),
    term: wp.array(dtype=wp.float64),
    flag: wp.bool,
    out: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    if flag:
        val = arr[i] / term[i]
    else:
        val = arr[i]
    out[i] = val


def run(kernel, flag: bool, n: int = 4):
    device = "cpu"
    arr = wp.array(list(range(1, n + 1)), dtype=wp.float64, requires_grad=True, device=device)
    term = wp.array([2.0] * n, dtype=wp.float64, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.float64, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[arr, term, flag], outputs=[out], device=device)

    # Seed every output adjoint with 1.0, i.e. compute d(sum(out))/d(arr).
    seed = wp.array([1.0] * n, dtype=wp.float64, device=device)
    tape.backward(grads={out: seed})

    forward = out.numpy().copy()
    grad_arr = arr.grad.numpy().copy()
    tape.zero()
    return forward, grad_arr


def main():
    n = 4
    expected_forward = [float(v) for v in range(1, n + 1)]  # flag=False -> val = arr[i]
    expected_grad = [1.0] * n  # d(arr[i])/d(arr[i]) = 1

    print(f"{'kernel':<16}{'forward matches arr':<22}{'grad(arr) == 1.0 everywhere'}")
    print("-" * 70)

    for name, kernel in (("ternary", ternary_kernel), ("if/else", ifelse_kernel)):
        forward, grad_arr = run(kernel, flag=False, n=n)
        forward_ok = list(forward) == expected_forward
        grad_ok = list(grad_arr) == expected_grad
        print(f"{name:<16}{str(forward_ok):<22}{str(grad_ok)} (grad(arr) = {list(grad_arr)})")

    print()
    print("Both kernels compute an identical forward pass (flag=False everywhere).")
    print("The ternary kernel nonetheless reports a zero adjoint for `arr` -- the")
    print("read inside the untaken `arr[i] / term[i]` branch poisons the adjoint")
    print("for the read in the taken branch, because both index the same array.")


if __name__ == "__main__":
    main()
