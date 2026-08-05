#!/usr/bin/env python3
"""Minimal reproduction of TWO independent bugs in a torch.autograd.Function
wrapping a Warp kernel, and the confirmed fix for both.

Environment this was observed on:
    warp-lang == 1.15.0
    torch     == 2.13.0+cu130
    Python    == 3.14.6
    Linux (WSL2), both CPU and CUDA devices available in the process

Run:
    python repro_warp_grad_reentrancy.py

--------------------------------------------------------------------------
Status: RESOLVED. Reported upstream and confirmed by the warp-lang team --
reproduced on both CPU and CUDA on current Warp. This script is kept as a
regression guard: it should always print PASS/PASS on the last row below.
--------------------------------------------------------------------------

Bug 1 -- torch.autograd.gradcheck's own reentrancy self-check fails.
gradcheck calls backward() more than once against the same retained graph
(to check determinism). The second call returns a wrong (all-zero)
gradient unless the gradient read back off Warp is (a) copied out with
.clone() rather than returned as a view (wp.to_torch() is zero-copy), and
(b) the array's .grad buffer is reset via tape.zero() afterward.

Bug 2 -- a reused grad_outputs tensor across separate calls fails.
If a caller reuses the same torch tensor object as grad_outputs across
separate, independent torch.autograd.grad(...) calls (fresh forward pass,
fresh wp.Tape() every time -- e.g. a preallocated gradient-seed buffer
reused every training step), the result is correct on the first call and
silently wrong (all zero) on every call after.

Root cause of Bug 2 (per the warp-lang team): wp.from_torch() is
zero-copy, so directly assigning `some_array.grad = wp.from_torch(g)`
makes the torch tensor `g` itself the live output-adjoint buffer. Warp's
backward pass *consumes* output adjoints by reading them and then zeroing
them -- so `g` gets zeroed out from under the caller. Confirmed identical
mechanism explains Bug 1 too: repeated backward() on a retained graph
reads and zeros the same buffer a second time, this time inconsistently
with what gradcheck's reentrancy check expects unless a fresh copy was
taken on the first read.

--------------------------------------------------------------------------
The fix (confirmed upstream as the intended pattern)
--------------------------------------------------------------------------
Don't assign array.grad directly. Seed gradients via
Tape.backward(grads={...}) instead:

    @staticmethod
    def backward(ctx, grad_output):
        seed = wp.from_torch(grad_output.contiguous(), dtype=wp.float64)
        ctx.tape.backward(grads={ctx.y_warp: seed})
        grad = wp.to_torch(ctx.x_warp.grad).clone()
        ctx.tape.zero()
        return grad

Tape.backward(grads=...) copies values into each array's own persistent
.grad buffer (array.grad.assign(seed)) rather than aliasing the seed
itself when that buffer already exists (which it does here, since
wp.zeros(..., requires_grad=True) allocates one up front) -- so the
caller's tensor is never mutated. This fixes Bug 2 outright (no .clone()
of grad_output needed at all). Bug 1 still needs its own fix on the read
side: .clone() the gradient read off Warp, and call tape.zero() afterward.

Applied to sphWarpCore's real AD bridge in
src/sphWarpCore/utils/wp_autograd.py (WarpFunctionWrapper.backward and
StateAwareWarpFunction.backward) -- see warpier_core.md for the
production-code writeup.
"""

import torch
import warp as wp

wp.init()


@wp.kernel
def square_kernel(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = x[i] * x[i]


def make_function(use_tape_grads: bool, read_clone: bool, tape_zero: bool):
    """Builds a torch.autograd.Function for y = x**2 with the two candidate
    fixes independently toggleable:
      use_tape_grads -- seed via tape.backward(grads={...}) (the confirmed
                        fix) instead of assigning array.grad directly
      read_clone      -- clone the gradient read back off the input array
      tape_zero       -- reset the tape's gradient buffers after backward
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
            seed = wp.from_torch(grad_output.contiguous(), dtype=wp.float64)
            if use_tape_grads:
                ctx.tape.backward(grads={ctx.y_warp: seed})
            else:
                ctx.y_warp.grad = seed  # the broken pattern
                ctx.tape.backward()

            grad = wp.to_torch(ctx.x_warp.grad)
            if read_clone:
                grad = grad.clone()
            if tape_zero:
                ctx.tape.zero()
            return grad

    return SquareFunction


def check_bug1_gradcheck_reentrancy(use_tape_grads: bool, read_clone: bool, tape_zero: bool) -> bool:
    """Bug 1: does torch.autograd.gradcheck's own reentrancy self-check pass?"""
    Fn = make_function(use_tape_grads, read_clone, tape_zero)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    try:
        return bool(torch.autograd.gradcheck(Fn.apply, (x,), eps=1e-6, atol=1e-5))
    except Exception:  # noqa: BLE001 - deliberately broad, this is a diagnostic check
        return False


def check_bug2_reused_grad_outputs(use_tape_grads: bool, read_clone: bool, tape_zero: bool, n_trials: int = 4) -> bool:
    """Bug 2: does reusing the same grad_outputs tensor object across
    separate, independent calls (fresh forward + fresh tape every time)
    keep returning the correct gradient?"""
    Fn = make_function(use_tape_grads, read_clone, tape_zero)
    grad_out_shared = torch.ones(3, dtype=torch.float64)  # same object, reused every trial, on purpose
    ok = True
    for _ in range(n_trials):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        y = Fn.apply(x)
        (grad,) = torch.autograd.grad(y, (x,), grad_outputs=grad_out_shared, retain_graph=False)
        expected = torch.tensor([2.0 * v for v in x.tolist()], dtype=torch.float64)
        ok &= torch.allclose(grad, expected)
    return ok


def main():
    combinations = [
        ("naive (no fixes)", False, False, False),
        ("Bug-1 fix only  (read_clone + tape_zero)", False, True, True),
        ("Bug-2 fix only  (tape.backward(grads=...))", True, False, False),
        ("both fixes", True, True, True),
    ]

    header = f"{'configuration':<45}{'Bug 1 (gradcheck reentrancy)':<32}{'Bug 2 (reused grad_outputs)'}"
    print(header)
    print("-" * len(header))
    for label, use_tape_grads, read_clone, tape_zero in combinations:
        bug1_ok = check_bug1_gradcheck_reentrancy(use_tape_grads, read_clone, tape_zero)
        bug2_ok = check_bug2_reused_grad_outputs(use_tape_grads, read_clone, tape_zero)
        print(f"{label:<45}{('PASS' if bug1_ok else 'FAIL'):<32}{'PASS' if bug2_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
