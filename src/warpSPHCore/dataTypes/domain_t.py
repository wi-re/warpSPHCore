from .._models import DomainDescription
from .._backend import BACKEND

if BACKEND == "warp":
    import warp as wp
    from warp.types import vector
    from ..type_config import *

    @wp.struct
    class domainData:
        domainMin: wp.array(dtype = scalar_t) # type: ignore
        domainMax: wp.array(dtype = scalar_t) # type: ignore
        periodicity: wp.array(dtype = wp.bool) # type: ignore
        dim: wp.int32
