import torch
import warp as wp
from ..util import *
from .cache import *
from typing import Optional, Union, Tuple
from .stateAwareWarpFunction import StateAwareWarpFunction
from .arg_extract import extractStateInfo

def warpWrapper2(
    launcher,
    kernel,
    outputSizes,
    outputDtypes,
    defaultStateArguments: tuple,
    additionalArguments: tuple = (),
    numThreads: Optional[int] = None,
):
    """
    State-aware autograd wrapper for SPH kernels.

    Unlike the flat-tensor ``warpWrapper``, this variant accepts high-level
    structured state objects (``ParticleState``, ``CRKState``, etc.) directly.
    It extracts all torch.Tensors from those structures, routes them through
    ``StateAwareWarpFunction`` so that gradients are properly tracked, and
    defers all Warp struct assembly to the autograd forward pass.

    Args:
        launcher:               Kernel launcher, e.g. ``launch_kernel``.
        kernel:                 The ``wp.kernel`` to execute.
        outputSizes:            Output shape passed to the launcher.
        outputDtypes:           Output Warp dtype(s) passed to the launcher.
        defaultStateArguments:  Tuple in the fixed state-argument order:
                                    (queryParticles, operationProperties, domain,
                                     queryVolumes, referenceVolumes, adjacency,
                                     referenceParticles, crkState,
                                     gradHState, renormalizationState)
        additionalArguments:    Extra per-kernel arguments appended after the
                                standard struct args.  Any ``torch.Tensor`` entries
                                will be tracked for gradients; plain Python scalars
                                and ints are forwarded unchanged.
        numThreads:             Explicit thread count for wp.launch(). If None,
                                defaults to outputSizes. Use this when the number
                                of threads should differ from output size.

    Returns:
        torch.Tensor or tuple of torch.Tensor – kernel output(s).
    """
    with record_function("warpWrapper2 [WW2]"):
        # --- extract state tensors and the struct-building closure ---
        flat_state_tensors, state_build_fn, device, dim = extractStateInfo(
            *defaultStateArguments
        )
        n_state = len(flat_state_tensors)

        # --- split additionalArguments into tensors and non-tensors ---
        add_tensor_pos = []   # (original_index, tensor)
        add_scalar_map = {}   # original_index -> scalar value
        for i, arg in enumerate(additionalArguments):
            if isinstance(arg, torch.Tensor):
                add_tensor_pos.append((i, arg))
            else:
                add_scalar_map[i] = arg

        add_tensors = [t for _, t in add_tensor_pos]
        n_add       = len(additionalArguments)

        # --- unified flat tensor list (state first, then additional tensors) ---
        flat_tensors = flat_state_tensors + add_tensors

        # --- build_fn combines struct args + reconstructed additional args ---
        def build_fn(wa: list) -> tuple:
            struct_args = state_build_fn(wa[:n_state])

            # Reconstruct additional args preserving original order
            reconstructed = [None] * n_add
            for pos, (orig_idx, _) in enumerate(add_tensor_pos):
                reconstructed[orig_idx] = wa[n_state + pos]
            for orig_idx, val in add_scalar_map.items():
                reconstructed[orig_idx] = val

            return struct_args + tuple(reconstructed)

        # Wrap launcher to inject numThreads if provided
        if numThreads is not None:
            original_launcher = launcher
            def launcher_with_threads(kernel, output_shape, output_dtype, *args):
                return original_launcher(kernel, output_shape, output_dtype, *args, numThreads=numThreads)
            launcher = launcher_with_threads

        return StateAwareWarpFunction.apply(
            build_fn, launcher, kernel, outputSizes, outputDtypes,
            *flat_tensors,
        )
