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
        outputs = []
        for i, out_type in enumerate(output_dtype):
            # print(f"Allocating output {i} with shape {output_shape[i] if isinstance(output_shape, list) or isinstance(output_shape, tuple) else output_shape} and dtype {out_type} on device {device}")
            output = wp.zeros(output_shape[i] if isinstance(output_shape, list) or isinstance(output_shape, tuple) else output_shape, dtype=out_type, device=device)
            output.requires_grad = requires_grad and _dtype_is_float(out_type)
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
    output.requires_grad = requires_grad and _dtype_is_float(output_dtype)
    kernel_inputs = inputs + [output]
    
    actual_dim = numThreads if numThreads is not None else (output_shape[0] if not isinstance(output_shape, int) else output_shape)
    wp.launch(
        kernel,
        dim = actual_dim,
        inputs = kernel_inputs,
        device = device
    )
    
    return output