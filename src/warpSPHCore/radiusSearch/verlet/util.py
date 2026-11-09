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
    """Nearest-image motion, without recentering or device-to-host reads."""
    return periodic_difference(current, previous, periodicity, domainMax - domainMin)
