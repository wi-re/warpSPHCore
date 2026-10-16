"""Stream ownership at the Torch/Warp boundary.

Callers supply dependencies through Torch's current stream on each device.
Selecting it in Warp must not introduce waits on Warp's previous stream.
"""

from contextlib import contextmanager, nullcontext
import gc

import torch
import warp as wp


def currentTorchWarpStream(device):
    """Select Torch's stream, reusing Warp's current wrapper when possible."""
    if device is None:
        return None
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            return None
    else:
        device = wp.get_device(device)
        if not device.is_cuda:
            return None
        device = torch.device('cuda', device.ordinal)
    torch_stream = torch.cuda.current_stream(device)
    # Re-wrapping an already-selected external handle can unregister the
    # outer Warp stream when the temporary wrapper is destroyed (Warp 1.17).
    # Ordinary launches still work, but subsequent conditional capture fails
    # with "unknown stream". Reuse the existing scope instead.
    warp_device = wp.device_from_torch(torch_stream.device)
    selected = wp.get_stream(warp_device)
    return selected if selected.cuda_stream == torch_stream.cuda_stream else wp.stream_from_torch(torch_stream)


def torchWarpStream(device):
    stream = currentTorchWarpStream(device)
    if stream is None:
        return nullcontext()
    if wp.get_stream(stream.device) is stream:
        return wp.ScopedDevice(stream.device)
    return wp.ScopedStream(stream, sync_enter=False, sync_exit=False)


@contextmanager
def torchWarpCapture(graph):
    """Capture a warmed current stream without torch.cuda.graph's global sync.

    The caller orders/synchronizes its setup stream first. Registering the
    external capture lets Warp retain reduction resources and child graphs.
    Prevent cyclic GC from destroying older CUDA graphs during this capture.
    """
    collect = gc.isenabled()
    gc.disable()
    try:
        graph.capture_begin()
        try:
            with wp.ScopedCapture(external=True, force_module_load=False,
                                  capture_mode=wp.CaptureMode.GLOBAL) as capture:
                yield capture
        finally:
            graph.capture_end()
    finally:
        if collect:
            gc.enable()
