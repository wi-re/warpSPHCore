import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *


# @torch.jit.script # jit script is deprecated :/
def _minimum_image_delta(
        current: torch.Tensor,
        previous: torch.Tensor,
        periodicity: torch.Tensor,
        domainMin: torch.Tensor,
        domainMax: torch.Tensor):
    """Displacement between two position sets, measured through the nearest
    periodic image on whichever axes are periodic.

    Branch-free on purpose. This used to loop over dimensions and test
    ``bool(periodicity[d].item())``, which is a device->host readback per axis
    per call -- and since the Verlet validity check calls this twice per
    adjacency, it was **the single largest source of host stalls in a real
    solver step: 11.5 of 39.1 readbacks per step** (measured at 19k particles,
    2D deltaSPH; see docs/regression/real_workload_bottleneck_audit.md). Each
    readback drains the CUDA queue, and the value being read is a *domain
    property fixed for the whole run*, so the stall bought nothing.

    Computing the wrapped displacement on every axis and selecting with
    ``torch.where`` keeps `periodicity` on the device and costs one extra
    elementwise pass over a tensor this function already materialises. The
    selected values are identical to the loop's, axis for axis.
    """
    delta = current - previous
    domainSize = domainMax - domainMin
    # Shift into [-L/2, L/2) so boundary-crossing motion is measured correctly.
    # Computed for every axis and discarded on the non-periodic ones: domainSize
    # is strictly positive for any valid DomainDescription (min < max), so this
    # cannot produce a NaN that `where` would then have to hide.
    wrapped = torch.remainder(delta + domainSize / 2, domainSize) - domainSize / 2
    return torch.where(periodicity.to(torch.bool).view(1, -1), wrapped, delta)
