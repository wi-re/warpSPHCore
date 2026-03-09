import torch
import warp as wp
from typing import Any, Tuple


class WarpFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function that bridges Warp's tape-based autodiff with PyTorch's autograd.
    This allows gradients to flow smoothly between PyTorch and Warp operations.
    """
    
    @staticmethod
    def forward(ctx, warp_kernel, dim, device, output_shape, output_dtype, *inputs):
        """
        Forward pass that executes a Warp kernel under a tape.
        
        Args:
            ctx: PyTorch autograd context for saving data for backward pass
            warp_kernel: The Warp kernel function to execute
            dim: Dimension parameter for wp.launch
            device: Warp device to run on
            output_shape: Shape of the output tensor(s)
            output_dtype: Data type of the output tensor(s)
            *inputs: Variable number of input tensors and parameters
        
        Returns:
            Output tensor(s) from the Warp kernel
        """
        # Save metadata for backward pass
        ctx.warp_kernel = warp_kernel
        ctx.dim = dim
        ctx.device = device
        ctx.output_shape = output_shape
        ctx.output_dtype = output_dtype
        
        # Convert PyTorch tensors to Warp arrays
        warp_inputs = []
        requires_grad_mask = []
        
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inp_warp = wp.from_torch(inp.contiguous(), dtype=None, requires_grad=inp.requires_grad)
                warp_inputs.append(inp_warp)
                requires_grad_mask.append(inp.requires_grad)
            else:
                # Non-tensor inputs (scalars, etc.)
                warp_inputs.append(inp)
                requires_grad_mask.append(False)
        
        ctx.requires_grad_mask = requires_grad_mask
        
        # Create output arrays
        if isinstance(output_shape, tuple) and len(output_shape) > 0 and isinstance(output_shape[0], (tuple, list)):
            # Multiple outputs
            outputs_warp = []
            for shape, dtype in zip(output_shape, output_dtype):
                out = wp.zeros(shape, dtype=dtype, device=device, requires_grad=True)
                outputs_warp.append(out)
                warp_inputs.append(out)
        else:
            # Single output
            outputs_warp = wp.zeros(output_shape, dtype=output_dtype, device=device, requires_grad=True)
            warp_inputs.append(outputs_warp)
        
        # Execute kernel under tape
        tape = wp.Tape()
        with tape:
            wp.launch(warp_kernel, dim=dim, inputs=warp_inputs, device=device)
        
        # Save tape and arrays for backward
        ctx.tape = tape
        ctx.warp_inputs = warp_inputs
        ctx.outputs_warp = outputs_warp
        
        # Convert output(s) back to PyTorch
        if isinstance(outputs_warp, list):
            outputs_torch = [wp.to_torch(out) for out in outputs_warp]
            return tuple(outputs_torch)
        else:
            return wp.to_torch(outputs_warp)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass that propagates gradients through the Warp tape.
        
        Args:
            ctx: PyTorch autograd context
            *grad_outputs: Gradients with respect to the outputs
        
        Returns:
            Gradients with respect to the inputs
        """
        tape = ctx.tape
        outputs_warp = ctx.outputs_warp
        warp_inputs = ctx.warp_inputs
        requires_grad_mask = ctx.requires_grad_mask
        
        # Set output gradients
        if isinstance(outputs_warp, list):
            for out_warp, grad_out in zip(outputs_warp, grad_outputs):
                if grad_out is not None:
                    out_warp.grad = wp.from_torch(grad_out.contiguous())
        else:
            if grad_outputs[0] is not None:
                outputs_warp.grad = wp.from_torch(grad_outputs[0].contiguous())
        
        # Run backward pass on tape
        tape.backward()
        
        # Collect gradients for inputs that require grad
        grad_inputs = [None, None, None, None, None]  # None for kernel, dim, device, output_shape, output_dtype
        
        for i, (inp_warp, requires_grad) in enumerate(zip(warp_inputs, requires_grad_mask)):
            if requires_grad and hasattr(inp_warp, 'grad') and inp_warp.grad is not None:
                grad_inputs.append(wp.to_torch(inp_warp.grad))
            else:
                grad_inputs.append(None)
        
        return tuple(grad_inputs)


def warp_op(kernel, dim, device, output_shape, output_dtype, *inputs):
    """
    Convenience wrapper for calling WarpFunction.
    
    Args:
        kernel: Warp kernel function
        dim: Launch dimension
        device: Warp device
        output_shape: Shape of output(s) - can be tuple of shapes for multiple outputs
        output_dtype: Data type of output(s) - can be tuple of dtypes for multiple outputs
        *inputs: Input tensors and scalars
    
    Returns:
        Output tensor(s) from the Warp kernel with autograd support
    """
    return WarpFunction.apply(kernel, dim, device, output_shape, output_dtype, *inputs)


class WarpLossFunction(torch.autograd.Function):
    """
    Specialized version for Warp operations that compute a scalar loss.
    This is optimized for the common pattern of computing a loss and backpropagating.
    """
    
    @staticmethod
    def forward(ctx, warp_forward_fn, *inputs):
        """
        Forward pass that executes Warp operations under a tape and returns a scalar loss.
        
        Args:
            ctx: PyTorch autograd context
            warp_forward_fn: A function that takes torch tensors, returns (loss_warp, tape, warp_inputs)
            *inputs: Input PyTorch tensors
        
        Returns:
            Scalar loss as a PyTorch tensor
        """
        # Execute the forward function which wraps Warp operations
        loss_warp, tape, warp_inputs, input_indices = warp_forward_fn(*inputs)
        
        # Save for backward
        ctx.tape = tape
        ctx.warp_inputs = warp_inputs
        ctx.input_indices = input_indices
        ctx.loss_warp = loss_warp
        
        # Convert loss to PyTorch
        loss_torch = wp.to_torch(loss_warp)
        
        return loss_torch
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass through the Warp tape.
        
        Args:
            ctx: PyTorch autograd context
            grad_output: Gradient with respect to the output (usually 1.0 for loss)
        
        Returns:
            Gradients for each input
        """
        tape = ctx.tape
        loss_warp = ctx.loss_warp
        warp_inputs = ctx.warp_inputs
        input_indices = ctx.input_indices
        
        # Set gradient of loss
        loss_warp.grad = wp.from_torch(grad_output.contiguous())
        
        # Backpropagate through tape
        tape.backward(loss=loss_warp)
        
        # Collect gradients - first return None for warp_forward_fn
        grad_inputs = [None]
        
        # Then gradients for actual inputs
        for idx in input_indices:
            inp = warp_inputs[idx]
            if hasattr(inp, 'grad') and inp.grad is not None:
                grad_inputs.append(wp.to_torch(inp.grad))
            else:
                grad_inputs.append(None)
        
        return tuple(grad_inputs)


def warp_loss_op(warp_forward_fn, *inputs):
    """
    Convenience wrapper for WarpLossFunction.
    
    Args:
        warp_forward_fn: Function that wraps Warp operations and returns (loss, tape, warp_arrays, input_indices)
        *inputs: Input PyTorch tensors
    
    Returns:
        Scalar loss with autograd support
    """
    return WarpLossFunction.apply(warp_forward_fn, *inputs)
