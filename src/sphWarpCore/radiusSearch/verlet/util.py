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
    delta = current - previous
    domainSize = domainMax - domainMin
    for d in range(delta.shape[1]):
        if bool(periodicity[d].item()):
            L = domainSize[d]
            # Shift into [-L/2, L/2) so boundary-crossing motion is measured correctly.
            delta_d = torch.remainder(delta[:, d] + L / 2, L) - L / 2
            delta[:, d] = delta_d
    return delta
