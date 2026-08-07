import torch
import warp as wp
from ..util import *
from .cache import *

class StateAwareWarpFunction(torch.autograd.Function):
    """
    Autograd bridge for SPH operations called with structured state arguments
    (ParticleState, CRKState, etc.).

    The caller supplies a *build_fn* closure that maps a flat list of Warp arrays
    (one per tracked torch.Tensor in the flat_tensors list) to the full argument
    tuple expected by the launcher.  This keeps all struct-assembly logic out of
    the autograd Function while still letting gradients flow through every tensor
    that had requires_grad=True.

    Forward signature (excluding ctx):
        build_fn        - closure: List[wp.array] -> tuple of kernel args
        launcher        - e.g. launch_kernel
        kernel          - the wp.kernel to execute
        output_shape    - scalar int or list[int] for multi-output
        output_dtype    - warp dtype or list[warp dtype] for multi-output
        *flat_tensors   - all torch.Tensors in deterministic order
    """

    # Number of non-tensor leading arguments (build_fn, launcher, kernel,
    # output_shape, output_dtype).  backward() must return this many Nones
    # before the per-tensor gradients.
    _N_NON_TENSOR = 5

    @staticmethod
    def forward(ctx, build_fn, launcher, kernel, output_shape, output_dtype, *flat_tensors):
        # with record_function(f"Warp Function State Aware Forward"):
        ctx.build_fn = build_fn
        ctx.launcher = launcher
        ctx.kernel = kernel
        ctx.output_shape = output_shape
        ctx.output_dtype = output_dtype

        ctx.any_requires_grad = any(
            t.requires_grad for t in flat_tensors if isinstance(t, torch.Tensor)
        )
        if ctx.any_requires_grad:
            print(f"Forward pass with gradients required for {sum(1 for t in flat_tensors if isinstance(t, torch.Tensor) and t.requires_grad)} tensors.")
            ctx.save_for_backward(*flat_tensors)
        else:
            
            # Avoid save_for_backward overhead when gradients are not requested.
            ctx.save_for_backward()

        # Detach → warp, preserving requires_grad so the tape tracks them.
        # Always builds fresh wrapper objects -- see the module-level note
        # above on why a data_ptr-keyed cache used to live here and why it
        # was removed.
        warp_arrays = []
        for t in flat_tensors:
            wa = getCachedWarpArray(t.detach())
            wa.requires_grad = t.requires_grad
            warp_arrays.append(wa)
        ctx.warp_arrays = warp_arrays

        # Reconstruct all kernel args via the caller-supplied closure
        kernel_args = build_fn(warp_arrays)

        if ctx.any_requires_grad:
            tape = wp.Tape()
            with tape:
                output = launcher(kernel, output_shape, output_dtype, *kernel_args)
        else:
            tape = None
            output = launcher(kernel, output_shape, output_dtype, *kernel_args)

        ctx.tape = tape
        ctx.output_warp = output

        if isinstance(output, (list, tuple)):
            return tuple(wp.to_torch(o) for o in output)
        return wp.to_torch(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        N = StateAwareWarpFunction._N_NON_TENSOR
        n_tensors = len(ctx.saved_tensors)

        if not ctx.any_requires_grad:
            return (None,) * N + (None,) * n_tensors

        # See WarpFunctionWrapper.backward's comment above for why this
        # seeds gradients via Tape.backward(grads=...) instead of assigning
        # array.grad directly.
        output_warp = ctx.output_warp
        grads = {}
        if isinstance(output_warp, (list, tuple)):
            for out, grad in zip(output_warp, grad_outputs):
                if grad is not None:
                    grads[out] = castTorchToWarpAsBuiltins(grad.contiguous())
        else:
            if grad_outputs[0] is not None:
                grads[output_warp] = castTorchToWarpAsBuiltins(grad_outputs[0].contiguous())

        ctx.tape.backward(grads=grads)

        input_grads = []
        for wa, t in zip(ctx.warp_arrays, ctx.saved_tensors):
            if t.requires_grad:
                input_grads.append(wp.to_torch(wa.grad).clone())
            else:
                input_grads.append(None)
        ctx.tape.zero()  # Clear any accumulated gradients in the tape to avoid affecting future computations

        return (None,) * N + tuple(input_grads)
    
warpWrapperStateaware = StateAwareWarpFunction.apply
    