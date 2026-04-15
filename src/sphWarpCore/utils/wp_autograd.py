import torch
import warp as wp
from typing import Any, Tuple
from .wp_util import castTorchToWarpAsBuiltins, castWarpToTorch


class WarpFunctionWrapper(torch.autograd.Function):
    """
    Custom PyTorch autograd function that bridges Warp's tape-based autodiff with PyTorch's autograd.
    This allows gradients to flow smoothly between PyTorch and Warp operations.
    """
    @staticmethod
    def forward(ctx, function, *args):
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
        ctx.save_for_backward(*[arg for arg in args if isinstance(arg, torch.Tensor)])
        
        # Convert PyTorch tensors to Warp arrays
        # IMPORTANT: Detach tensors to prevent Warp from writing gradients back to PyTorch tensors
        # The wrapper will handle gradient flow through return values only
        warp_args = []
        requires_grad_mask = []
        index = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Detach to break the link between PyTorch and Warp gradient buffers
                detached_arg = arg.detach()
                warp_arg = castTorchToWarpAsBuiltins(detached_arg)
                warp_arg.requires_grad = arg.requires_grad  # Preserve requires_grad information for the wrapper's backward pass
                warp_args.append(warp_arg)
                requires_grad_mask.append(arg.requires_grad)
                # print(f'Converted input {index:02d} from torch [dtype: {arg.dtype}, device {arg.device}, shape: {arg.shape}, requires_grad: {arg.requires_grad}] to warp array [dtype: {warp_arg.dtype}, device: {warp_arg.device}, shape: {warp_arg.shape}, requires_grad: {warp_arg.requires_grad}]')
            else:
                warp_args.append(arg)  # Non-tensor arguments are passed as-is
                requires_grad_mask.append(False)
                # print(f'Input {index:02d} is not a Tensor.')
            index += 1
                
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
        outIndex = 0
        # Set output gradients
        if isinstance(outputs_warp, list):
            for out_warp, grad_out in zip(outputs_warp, grad_outputs):
                if grad_out is not None:
                    out_warp.grad = wp.from_torch(grad_out.contiguous())
                    # print(f'Output Grad [{i:2d}]: {grad_out} [dtype: {grad_out.dtype}, device: {grad_out.device}, shape: {grad_out.shape}]')
                    
        else:
            if grad_outputs[0] is not None:
                outputs_warp.grad = wp.from_torch(grad_outputs[0].contiguous())
                # print(f'Output Grad: {grad_outputs[0]} [dtype: {grad_outputs[0].dtype}, device: {grad_outputs[0].device}, shape: {grad_outputs[0].shape}]')
                
        
        # Use the saved tape to compute gradients with respect to inputs
        ctx.tape.backward()
        
        # Retrieve gradients for inputs from the tape
        input_grads = []
        for i, (arg, requires_grad) in enumerate(zip(ctx.inputs_warp, ctx.requires_grad_mask)):
            if requires_grad:
                grad_warp = inputs_warp[i].grad
                grad_torch = wp.to_torch(grad_warp)
                input_grads.append(grad_torch)
                # print(f'Input {i:02d} requires grad. Retrieved gradient from Warp and converted to torch tensor with shape {grad_torch.shape}, dtype {grad_torch.dtype}, device {grad_torch.device} -> {grad_warp} | {grad_torch}')
            else:
                input_grads.append(None)
                # print(f'Input {i:02d} did not require grad')
                
        # print("Backward pass completed. Returning gradients for inputs.")
        # print(f'Number of inputs: {len(ctx.inputs_warp)}, number of gradients: {len(input_grads)}')
        return (None,) + tuple(input_grads)  # None for the function argument
    
warpWrapper = WarpFunctionWrapper.apply

from torch.profiler import record_function

def launch_kernel(kernel, output_shape, output_dtype, *args):
    with record_function(f"Warp Kernel: {kernel.__name__}"):
        inputs = list(args)
        requires_grad = any(input.requires_grad for input in inputs if hasattr(input, 'requires_grad'))
        
        output = wp.zeros(output_shape, dtype=output_dtype, device=inputs[0].device)
        output.requires_grad = requires_grad
        
        wp.launch(
            kernel,
            dim = output_shape[0] if not isinstance(output_shape, int) else output_shape,
            inputs = list(args) + [output],
            device = inputs[0].device
        )
    
    return output