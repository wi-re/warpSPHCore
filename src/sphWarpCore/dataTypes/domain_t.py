from dataclasses import dataclass
from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..type_config import *

# @torch.jit.script # jit script is deprecated :/
@dataclass(slots=True)
class DomainDescription:
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodic: torch.Tensor
    dim: int

    def __ne__(self, other: 'DomainDescription') -> bool:
        return not self.__eq__(other)


@wp.struct
class domainData:
    domainMin: wp.array(dtype = scalar_t) # type: ignore
    domainMax: wp.array(dtype = scalar_t) # type: ignore
    periodicity: wp.array(dtype = wp.bool) # type: ignore
    dim: wp.int32