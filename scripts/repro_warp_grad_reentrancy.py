#!/usr/bin/env python3
"""Minimal reproduction of TWO independent, non-overlapping bugs in a
torch.autograd.Function that wraps a Warp kernel -- both silently return a
WRONG gradient (no error) on repeated use, but each needs a different fix,
and neither fix substitutes for the other.

Environment this was observed on:
    warp-lang == 1.15.0
    torch     == 2.13.0+cu130
    Python    == 3.14.6
    Linux (WSL2), both CPU and CUDA devices available in the process

Run:
    python repro_warp_grad_reentrancy.py

--------------------------------------------------------------------------
Bug 1: torch.autograd.gradcheck's own reentrancy self-check fails
--------------------------------------------------------------------------
gradcheck calls backward() more than once against the same retained graph
(comparing the two results for determinism). That second call returns a
wrong (typically all-zero) gradient unless the array's gradient is (a)
copied out of Warp with .clone() rather than returned as a view, and (b)
the array's .grad buffer is reset via tape.zero() afterward.

--------------------------------------------------------------------------
Bug 2: a reused grad_outputs tensor across separate calls fails
--------------------------------------------------------------------------
If a caller reuses the same torch tensor object as grad_outputs across
separate, independent torch.autograd.grad(...) calls (a fresh forward pass
and a fresh wp.Tape() every time -- a realistic pattern for e.g. a
preallocated "ones" gradient-seed buffer reused every training step), the
result is correct on the first call and silently wrong (all zero) on every
call after that -- UNLESS the incoming grad_outputs tensor is forced to
fresh storage (e.g. .clone()) before being wrapped and assigned to the
output array's .grad.

--------------------------------------------------------------------------
The two fixes do not substitute for each other
--------------------------------------------------------------------------
This is the part worth emphasizing: applying ONLY the Bug 1 fix does not
fix Bug 2, and applying ONLY the Bug 2 fix does not fix Bug 1. All four
combinations are exercised explicitly below. Only "both fixes applied"
passes both checks.

Separately (not exercised by the fixed code path below, but part of the
diagnosis): two independent wp.from_torch() calls on the same underlying
tensor storage return different Python wp.array objects (different id())
whose .grad buffers nonetheless share the exact same underlying pointer
(confirmed via array.grad.ptr). That is consistent with Warp keying some
internal grad-buffer/tape-adjoint bookkeeping off the tensor's memory
address rather than off Python object identity, and not fully refreshing
that bookkeeping the next time the same address is reused as a gradient
source -- which would explain both bugs as two manifestations of the same
underlying mechanism, triggered via two different call patterns. We don't
have visibility into Warp's internals to confirm that further.

Both bugs reproduce identically on "cpu" and "cuda".
"""

import torch
import warp as wp

wp.init()


@wp.kernel
def square_kernel(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = x[i] * x[i]


def make_function(write_clone: bool, read_clone: bool, tape_zero: bool):
    """Builds a torch.autograd.Function for y = x**2 with the two candidate
    fixes independently toggleable:
      write_clone -- clone grad_output before seeding the output array's .grad
      read_clone  -- clone the gradient read back off the input array
      tape_zero   -- reset the tape's gradient buffers after backward
    """

    class SquareFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_warp = wp.from_torch(x.detach().contiguous(), dtype=wp.float64)
            x_warp.requires_grad = True
            y_warp = wp.zeros(x.shape[0], dtype=wp.float64, device=x_warp.device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(square_kernel, dim=x.shape[0], inputs=[x_warp], outputs=[y_warp], device=x_warp.device)
            ctx.tape = tape
            ctx.x_warp = x_warp
            ctx.y_warp = y_warp
            return wp.to_torch(y_warp)

        @staticmethod
        def backward(ctx, grad_output):
            g = grad_output.contiguous()
            if write_clone:
                g = g.clone()
            ctx.y_warp.grad = wp.from_torch(g, dtype=wp.float64)
            ctx.tape.backward()

            grad = wp.to_torch(ctx.x_warp.grad)
            if read_clone:
                grad = grad.clone()
            if tape_zero:
                ctx.tape.zero()
            return grad

    return SquareFunction



def check_bug1_gradcheck_reentrancy(write_clone: bool, read_clone: bool, tape_zero: bool) -> bool:
    print("-" * 80)
    print(f"Checking bug1 - reentrancy - for Write Clone {write_clone}, Read Clone {read_clone}, Tape Zero {tape_zero}...")
    """Bug 1: does torch.autograd.gradcheck's own reentrancy self-check pass?"""
    Fn = make_function(write_clone, read_clone, tape_zero)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    try:
        return bool(torch.autograd.gradcheck(Fn.apply, (x,), eps=1e-6, atol=1e-5))
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a diagnostic check
        print(f"    (gradcheck raised {type(exc).__name__}: {str(exc).splitlines()[0]})")
        return False


def check_bug2_reused_grad_outputs(write_clone: bool, read_clone: bool, tape_zero: bool, n_trials: int = 4) -> bool:
    print("-" * 80)
    print(f"Checking bug2 - reused grad_outputs - for Write Clone {write_clone}, Read Clone {read_clone}, Tape Zero {tape_zero}...")
    """Bug 2: does reusing the same grad_outputs tensor object across
    separate, independent calls (fresh forward + fresh tape every time)
    keep returning the correct gradient?"""
    Fn = make_function(write_clone, read_clone, tape_zero)
    grad_out_shared = torch.ones(3, dtype=torch.float64)  # same object, reused every trial, on purpose
    ok = True
    for _ in range(n_trials):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        y = Fn.apply(x)
        (grad,) = torch.autograd.grad(y, (x,), grad_outputs=grad_out_shared, retain_graph=False)
        expected = torch.tensor([2.0 * v for v in x.tolist()], dtype=torch.float64)
        print(f"    Equal: {torch.allclose(grad, expected)} | grad: {grad.tolist()} | expected: {expected.tolist()}")
        ok &= torch.allclose(grad, expected)
    return ok


combinations = [
    ("naive (no fixes)", False, False, False),
    ("Bug-1 fix only  (read_clone + tape_zero)", False, True, True),
    ("Bug-2 fix only  (write_clone)", True, False, False),
    ("both fixes", True, True, True),
]

table = []
for label, write_clone, read_clone, tape_zero in combinations:
    bug1_ok = check_bug1_gradcheck_reentrancy(write_clone, read_clone, tape_zero)
    bug2_ok = check_bug2_reused_grad_outputs(write_clone, read_clone, tape_zero)
    table.append((bug1_ok, bug2_ok))

header = f"{'configuration':<42}{'Bug 1 (gradcheck reentrancy)':<32}{'Bug 2 (reused grad_outputs)'}"
print(header)
print("-" * len(header))
for (bug1_ok, bug2_ok), (label, write_clone, read_clone, tape_zero) in zip(table, combinations):

    print(f"{label:<42}{('PASS' if bug1_ok else 'FAIL'):<32}{'PASS' if bug2_ok else 'FAIL'}")
