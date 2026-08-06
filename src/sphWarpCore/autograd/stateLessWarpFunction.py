import torch
import warp as wp
from ..util import *
from .cache import *

class WarpFunctionWrapper(torch.autograd.Function):
    """
    Custom PyTorch autograd function that bridges Warp's tape-based autodiff with PyTorch's autograd.
    This allows gradients to flow smoothly between PyTorch and Warp operations.
    """
    @staticmethod
    def forward(ctx, function, *args):
        # with record_function(f"Warp Function Foward"):
        """
        Forward pass that executes the given Warp function and saves necessary context for backward.
        
        Args:
            ctx: PyTorch context object for saving information for backward pass.
            function: The Warp function to execute (should be a callable that runs a Warp kernel).
            *args: Arguments to pass to the Warp function (should be PyTorch tensors).
        Returns:
            The output from the Warp function, converted back to PyTorch tensors.
        """
        
        ctx.function = function
        # check if any input requires grad
        ctx.any_requires_grad = any(arg.requires_grad for arg in args if isinstance(arg, torch.Tensor))

        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if ctx.any_requires_grad:
            ctx.save_for_backward(*tensor_args)
        else:
            # Avoid save_for_backward overhead on pure inference/no-grad paths.
            ctx.save_for_backward()

        # Convert PyTorch tensors to Warp arrays. Always builds fresh wrapper
        # objects -- see the module-level note above on why this used to
        # reuse cached wrapper/array objects and why that was removed.
        warp_args = []
        requires_grad_mask = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Detach to break the link between PyTorch and Warp gradient buffers
                detached_arg = arg.detach()
                warp_arg = getCachedWarpArray(detached_arg)
                warp_arg.requires_grad = arg.requires_grad  # Preserve requires_grad information for backward
                warp_args.append(warp_arg)
                requires_grad_mask.append(arg.requires_grad)
            else:
                warp_args.append(arg)  # Non-tensor arguments are passed as-is
                requires_grad_mask.append(False)

        ctx.inputs_warp = warp_args
        ctx.requires_grad_mask = requires_grad_mask
        # print(f'Number of warp inputs: {len(warp_args)} | {len(ctx.requires_grad_mask)}, number of inputs: {len(args)}')
                
        # Execute the Warp function (this should run the kernel and produce outputs as Warp arrays)
        if ctx.any_requires_grad:
            tape = wp.Tape()
            with tape:
                output = function(*warp_args)
        else:
            output = function(*warp_args)
            
        ctx.tape = tape if ctx.any_requires_grad else None
        ctx.outputs_warp = output
        
        if isinstance(output, list) or isinstance(output, tuple):
            return tuple(wp.to_torch(out) for out in output)
        else:
            return wp.to_torch(output)
        
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass that computes gradients using Warp's autodiff and converts them back to PyTorch tensors.
        
        Args:
            ctx: PyTorch context object containing saved information from the forward pass.
            *grad_outputs: Gradients of the outputs with respect to some loss, provided by PyTorch's autograd.
        Returns:
            Gradients with respect to the inputs, converted back to PyTorch tensors.
        """
        if not ctx.any_requires_grad:
            return (None,) * (len(ctx.inputs_warp) + 1)  # +1 for the function argument

        outputs_warp = ctx.outputs_warp
        inputs_warp = ctx.inputs_warp

        # Seed output gradients via Tape.backward(grads=...) rather than
        # assigning array.grad directly. wp.from_torch() is zero-copy, so a
        # direct `out_warp.grad = wp.from_torch(grad_out)` makes the incoming
        # torch tensor itself the live output-adjoint buffer; Warp's reverse
        # pass consumes output adjoints by reading *and zeroing* them, which
        # then mutates that torch tensor in place. Any caller that reuses the
        # same grad_outputs tensor object across separate backward() calls
        # (e.g. a preallocated gradient-seed buffer) would see a correct
        # gradient on the first call and silently zero on every call after,
        # since the tensor was zeroed out from under it. Tape.backward(grads=...)
        # instead copies values into each array's own persistent .grad buffer
        # (array.grad.assign(seed)) rather than aliasing the seed itself, so
        # the caller's tensor is never mutated. Confirmed upstream (warp-lang
        # issue tracker) as the intended pattern; see warpier_core.md.
        grads = {}
        if isinstance(outputs_warp, list):
            for out_warp, grad_out in zip(outputs_warp, grad_outputs):
                if grad_out is not None:
                    grads[out_warp] = castTorchToWarpAsBuiltins(grad_out.contiguous())
        else:
            if grad_outputs[0] is not None:
                grads[outputs_warp] = castTorchToWarpAsBuiltins(grad_outputs[0].contiguous())

        # Use the saved tape to compute gradients with respect to inputs
        ctx.tape.backward(grads=grads)

        # Retrieve gradients for inputs from the tape
        input_grads = []
        for i, (arg, requires_grad) in enumerate(zip(ctx.inputs_warp, ctx.requires_grad_mask)):
            if requires_grad:
                grad_warp = inputs_warp[i].grad
                grad_torch = wp.to_torch(grad_warp)
                input_grads.append(grad_torch.clone())
                # print(f'Input {i:02d} requires grad. Retrieved gradient from Warp and converted to torch tensor with shape {grad_torch.shape}, dtype {grad_torch.dtype}, device {grad_torch.device} -> {grad_warp} | {grad_torch}')
            else:
                input_grads.append(None)
                # print(f'Input {i:02d} did not require grad')
        ctx.tape.zero()  # Clear any accumulated gradients in the tape to avoid affecting future computations
        # ctx.tape.reset()  # Clear the tape to free memory
                
        # print("Backward pass completed. Returning gradients for inputs.")
        # print(f'Number of inputs: {len(ctx.inputs_warp)}, number of gradients: {len(input_grads)}')
        return (None,) + tuple(input_grads)  # None for the function argument
    
warpWrapper = WarpFunctionWrapper.apply
