from ..type_config import *
import warp as wp

@wp.func
def kroneckerDelta(a: wp.int32, b: wp.int32):
    return scalar_t(1.0) if a == b else scalar_t(0.0)

