import torch
import warp as wp
from ..util import *
from .cache import *


def _dtype_is_float(dtype):
    # dtype may be a plain scalar (wp.float32, wp.int32, ...) or a
    # vector/matrix built from one -- vector/matrix types carry their
    # element type on _wp_scalar_type_, plain scalars don't, so this falls
    # back to the dtype itself in that case. requires_grad can only be set
    # on floating-point Warp arrays; setting it on an int-dtype output (e.g.
    # a per-particle counter alongside a differentiable accumulator in the
    # same multi-output kernel) raises in wp.to_torch.
    scalar = getattr(dtype, "_wp_scalar_type_", dtype)
    return wp.types.type_is_float(scalar)


def _allocate_output(shape, dtype, device, requires_grad):
    # Allocate the output on torch's own caching allocator (rather than
    # wp.zeros, which allocates from Warp's separate memory pool) so a
    # torch-heavy caller doesn't pay for two independently-growing GPU
    # allocators. wp.from_torch() gives a zero-copy Warp view of that same
    # buffer for the kernel launch; the torch tensor itself is what gets
    # returned to callers.
    output_torch, output_warp = allocateTorchWarp(
        shape, dtype, device, requires_grad=requires_grad and _dtype_is_float(dtype)
    )
    # Stashed for StateAwareWarpFunction/WarpFunctionWrapper: Warp's tape
    # records the exact wp.array object passed to wp.launch below, so
    # backward() must seed gradients through that same object -- not a
    # freshly re-wrapped one -- for Tape.backward(grads=...) to find it.
    # This is only a courier across this return boundary: wp.from_torch()
    # gives output_warp a `_tensor` back-reference to output_torch, so if a
    # caller reads `_warp_array` into ctx and leaves it attached, that closes
    # a reference cycle (tensor -> wp.array -> tensor) that only the cyclic
    # GC -- not refcounting -- can break, on its own schedule rather than as
    # memory is freed (this produced a sawtooth allocation pattern on every
    # kernel launch, grad or not). Callers MUST `del tensor._warp_array`
    # immediately after stashing the wp.array elsewhere (e.g. ctx.output_warp).
    output_torch._warp_array = output_warp
    return output_torch, output_warp


def launch_kernel(kernel, output_shape, output_dtype, *args, numThreads=None):
    # with record_function(f"Warp Kernel Launch"):
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
        outputs_torch = []
        outputs_warp = []
        for i, out_type in enumerate(output_dtype):
            shape_i = output_shape[i] if isinstance(output_shape, list) or isinstance(output_shape, tuple) else output_shape
            out_torch, out_warp = _allocate_output(shape_i, out_type, device, requires_grad)
            outputs_torch.append(out_torch)
            outputs_warp.append(out_warp)

        kernel_dim = output_shape[0] if isinstance(output_shape, list) else output_shape
        kernel_dim = kernel_dim if isinstance(kernel_dim, int) else kernel_dim[0]
        actual_dim = numThreads if numThreads is not None else kernel_dim
        kernel_inputs = inputs + outputs_warp

        wp.launch(
            kernel,
            dim = actual_dim,
            inputs = kernel_inputs,
            device = device
        )
        return tuple(outputs_torch)

    output_torch, output_warp = _allocate_output(output_shape, output_dtype, device, requires_grad)
    kernel_inputs = inputs + [output_warp]

    actual_dim = numThreads if numThreads is not None else (output_shape[0] if not isinstance(output_shape, int) else output_shape)
    wp.launch(
        kernel,
        dim = actual_dim,
        inputs = kernel_inputs,
        device = device
    )

    return output_torch