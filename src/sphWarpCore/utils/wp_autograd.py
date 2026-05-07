import torch
import warp as wp
from typing import Any, List, Tuple
from .wp_util import castTorchToWarpAsBuiltins, castWarpToTorch, getCachedWarpArray

# ---------------------------------------------------------------------------
# Struct-bundle cache: maps a tuple of data_ptrs to (warp_arrays, kernel_args)
# so that build_fn and all 36 wp.array casts are skipped on steady-state steps.
# ---------------------------------------------------------------------------
_KERNEL_ARGS_CACHE: dict[tuple, tuple] = {}
_WRAPPER_ARGS_CACHE: dict[tuple, tuple] = {}

# data_ptr-based keys can churn in out-of-place update loops (new storage every step).
# Keep caches bounded to avoid retaining stale tensor-backed wrapper objects forever.
_MAX_KERNEL_ARGS_CACHE_ENTRIES = 8
_MAX_WRAPPER_ARGS_CACHE_ENTRIES = 8


def _cache_set_bounded(cache: dict, key: tuple, value: tuple, max_entries: int) -> None:
    cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def _to_hashable(value):
    if isinstance(value, list):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in value.items()))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def clearKernelArgsCache() -> None:
    """Invalidate the struct-bundle cache.

    Call this whenever particle arrays are reallocated (e.g. particle add/remove)
    or after an adjacency rebuild that reuses the same Python tensor objects.
    """
    _KERNEL_ARGS_CACHE.clear()
    _WRAPPER_ARGS_CACHE.clear()


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

        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if ctx.any_requires_grad:
            ctx.save_for_backward(*tensor_args)
        else:
            # Avoid save_for_backward overhead on pure inference/no-grad paths.
            ctx.save_for_backward()

        # Convert PyTorch tensors to Warp arrays.
        # In no-grad steady-state, reuse the fully packed argument bundle.
        warp_args = None
        requires_grad_mask = None

        if not ctx.any_requires_grad:
            key_parts = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    key_parts.append(("t", arg.data_ptr(), arg.shape, arg.stride(), arg.dtype))
                else:
                    key_parts.append(("s", _to_hashable(arg)))
            cache_key = (function, tuple(key_parts))
            cached = _WRAPPER_ARGS_CACHE.get(cache_key)
            if cached is not None:
                warp_args, requires_grad_mask = cached
                # Reset tensor requires_grad flags each call for correctness
                # if the same wrapped array was used in a grad-enabled context.
                for arg, wa in zip(args, warp_args):
                    if isinstance(arg, torch.Tensor):
                        wa.requires_grad = arg.requires_grad

        if warp_args is None:
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

            if not ctx.any_requires_grad:
                _cache_set_bounded(
                    _WRAPPER_ARGS_CACHE,
                    cache_key,
                    (warp_args, requires_grad_mask),
                    _MAX_WRAPPER_ARGS_CACHE_ENTRIES,
                )
                
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
        if ctx.any_requires_grad:
            print(f"Forward pass with gradients required for {sum(1 for t in flat_tensors if isinstance(t, torch.Tensor) and t.requires_grad)} tensors.")
            ctx.save_for_backward(*flat_tensors)
        else:
            
            # Avoid save_for_backward overhead when gradients are not requested.
            ctx.save_for_backward()

        # ------------------------------------------------------------------
        # Hot-path: reuse cached (warp_arrays, kernel_args) when every tensor
        # has the same underlying storage as the previous call.
        # The cache key is a tuple of (data_ptr, shape, stride, dtype) for
        # each tensor, which changes whenever a tensor is reallocated.
        # We skip the cache when any tensor requires grad so that the Warp
        # tape always gets fresh array objects it can attach grad buffers to.
        # ------------------------------------------------------------------
        cache_key = None
        # if not ctx.any_requires_grad:
        #     cache_key = tuple(
        #         (t.data_ptr(), t.shape, t.stride(), t.dtype) for t in flat_tensors
        #     )
        #     cached = _KERNEL_ARGS_CACHE.get(cache_key)
        #     if cached is not None:
        #         warp_arrays, kernel_args = cached
        #         ctx.warp_arrays = warp_arrays
        #         ctx.kernel_args_cache_hit = True
        #     else:
        #         ctx.kernel_args_cache_hit = False
        # else:
        #     cached = None
        #     ctx.kernel_args_cache_hit = False

        # if not ctx.kernel_args_cache_hit:
        if True:
            # Detach → warp, preserving requires_grad so the tape tracks them
            warp_arrays = []
            for t in flat_tensors:
                wa = getCachedWarpArray(t.detach())
                wa.requires_grad = t.requires_grad
                warp_arrays.append(wa)
            ctx.warp_arrays = warp_arrays

            # Reconstruct all kernel args via the caller-supplied closure
            kernel_args = build_fn(warp_arrays)

            # if cache_key is not None:
            #     _cache_set_bounded(
            #         _KERNEL_ARGS_CACHE,
            #         cache_key,
            #         (warp_arrays, kernel_args),
            #         _MAX_KERNEL_ARGS_CACHE_ENTRIES,
            #     )

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

def launch_kernel(kernel, output_shape, output_dtype, *args, numThreads=None):
    with record_function(f"Warp Kernel: {kernel.__name__}"):
        inputs = list(args)

        # Single-pass scan to discover grad requirement and target device source.
        requires_grad = False
        firstTensorInput = None
        domainInput = None
        for input_arg in inputs:
            if not requires_grad and getattr(input_arg, "requires_grad", False):
                requires_grad = True
            if firstTensorInput is None and isinstance(input_arg, wp.array):
                firstTensorInput = input_arg
            if domainInput is None and hasattr(input_arg, "domainMin"):
                domainInput = input_arg

        # use the first tensor input to determine the device for the output tensors
        if firstTensorInput is None:
            # one of the arguments could be a domain state that contains the device information
            if domainInput is not None:
                device = domainInput.domainMin.device
            else:
                raise ValueError("At least one input must be a torch.Tensor or a domainData struct to determine the device for the output.")
            # raise ValueError("At least one input must be a torch.Tensor to determine the device for the output.")
        else:
            device = firstTensorInput.device

        if isinstance(output_dtype, (list, tuple)):
            outputs = []
            for i, out_type in enumerate(output_dtype):
                # print(f"Allocating output {i} with shape {output_shape[i] if isinstance(output_shape, list) or isinstance(output_shape, tuple) else output_shape} and dtype {out_type} on device {device}")
                output = wp.zeros(output_shape[i] if isinstance(output_shape, list) or isinstance(output_shape, tuple) else output_shape, dtype=out_type, device=device)
                output.requires_grad = requires_grad
                outputs.append(output)

            kernel_dim = output_shape[0] if isinstance(output_shape, list) else output_shape
            kernel_dim = kernel_dim if isinstance(kernel_dim, int) else kernel_dim[0]
            actual_dim = numThreads if numThreads is not None else kernel_dim
            kernel_inputs = inputs + outputs

            wp.launch(
                kernel,
                dim = actual_dim,
                inputs = kernel_inputs,
                device = device
            )
            return tuple(outputs)
        
        output = wp.zeros(output_shape, dtype=output_dtype, device=device)
        output.requires_grad = requires_grad
        kernel_inputs = inputs + [output]
        
        actual_dim = numThreads if numThreads is not None else (output_shape[0] if not isinstance(output_shape, int) else output_shape)
        wp.launch(
            kernel,
            dim = actual_dim,
            inputs = kernel_inputs,
            device = device
        )
    
    return output