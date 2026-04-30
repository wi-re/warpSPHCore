import torch
import warp as wp
from typing import Any, List, Tuple
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
                    out_warp.grad = castTorchToWarpAsBuiltins(grad_out.contiguous())
                    # print(f'Output Grad [{i:2d}]: {grad_out} [dtype: {grad_out.dtype}, device: {grad_out.device}, shape: {grad_out.shape}]')
                    
        else:
            if grad_outputs[0] is not None:
                outputs_warp.grad = castTorchToWarpAsBuiltins(grad_outputs[0].contiguous())
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
        build_fn        – closure: List[wp.array] -> tuple of kernel args
        launcher        – e.g. launch_kernel
        kernel          – the wp.kernel to execute
        output_shape    – scalar int or list[int] for multi-output
        output_dtype    – warp dtype or list[warp dtype] for multi-output
        *flat_tensors   – all torch.Tensors in deterministic order
    """

    # Number of non-tensor leading arguments (build_fn, launcher, kernel,
    # output_shape, output_dtype).  backward() must return this many Nones
    # before the per-tensor gradients.
    _N_NON_TENSOR = 5

    @staticmethod
    def forward(ctx, build_fn, launcher, kernel, output_shape, output_dtype, *flat_tensors):
        ctx.build_fn = build_fn
        ctx.launcher = launcher
        ctx.kernel = kernel
        ctx.output_shape = output_shape
        ctx.output_dtype = output_dtype

        ctx.any_requires_grad = any(
            t.requires_grad for t in flat_tensors if isinstance(t, torch.Tensor)
        )
        ctx.save_for_backward(*flat_tensors)

        # Detach → warp, preserving requires_grad so the tape tracks them
        warp_arrays = []
        for t in flat_tensors:
            wa = castTorchToWarpAsBuiltins(t.detach())
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

        output_warp = ctx.output_warp
        if isinstance(output_warp, (list, tuple)):
            for out, grad in zip(output_warp, grad_outputs):
                if grad is not None:
                    out.grad = castTorchToWarpAsBuiltins(grad.contiguous())
        else:
            if grad_outputs[0] is not None:
                output_warp.grad = castTorchToWarpAsBuiltins(grad_outputs[0].contiguous())

        ctx.tape.backward()

        input_grads = []
        for wa, t in zip(ctx.warp_arrays, ctx.saved_tensors):
            if t.requires_grad:
                input_grads.append(wp.to_torch(wa.grad))
            else:
                input_grads.append(None)

        return (None,) * N + tuple(input_grads)

from torch.profiler import record_function

from ..warp_state import *

def launch_kernel(kernel, output_shape, output_dtype, *args):
    with record_function(f"Warp Kernel: {kernel.__name__}"):
        inputs = list(args)
        requires_grad = any(input.requires_grad for input in inputs if hasattr(input, 'requires_grad'))
    
        # use the first tensor input to determine the device for the output tensors
        firstTensorInput = next((input for input in inputs if isinstance(input, wp.array)), None)
        if firstTensorInput is None:
            # one of the arguments could be a domain state that contains the device information
            domainInput = next((input for input in inputs if hasattr(input, 'domainMin')), None)
            if domainInput is not None:
                device = domainInput.domainMin.device
            else:
                raise ValueError("At least one input must be a torch.Tensor or a domainData struct to determine the device for the output.")
            # raise ValueError("At least one input must be a torch.Tensor to determine the device for the output.")
        else:
            device = firstTensorInput.device

        if isinstance(output_dtype, List) or isinstance(output_dtype, Tuple):
            outputs = []
            for i, out_type in enumerate(output_dtype):
                output = wp.zeros(output_shape[i] if isinstance(output_shape, List) else output_shape, dtype=out_type, device=device)
                output.requires_grad = requires_grad
                outputs.append(output)

            kernel_dim = output_shape[0] if isinstance(output_shape, List) else output_shape
            kernel_dim = kernel_dim if isinstance(kernel_dim, int) else kernel_dim[0]

            wp.launch(
                kernel,
                dim = kernel_dim,
                inputs = list(args) + outputs,
                device = device
            )
            return tuple(outputs)
        
        output = wp.zeros(output_shape, dtype=output_dtype, device=device)
        output.requires_grad = requires_grad
        
        wp.launch(
            kernel,
            dim = output_shape[0] if not isinstance(output_shape, int) else output_shape,
            inputs = list(args) + [output],
            device = device
        )
    
    return output