from .type_config import scalar_t, dim_t, get_type_config, to_torch, to_numpy, get_torch_precision
from warp.types import vector, matrix
import warp as wp


vec_t: type = vector(length=dim_t, dtype=scalar_t)
mat_t: type = matrix(shape =(dim_t, dim_t), dtype=scalar_t)
int_t: type = wp.int32

vecArray_t: type = wp.array(dtype=vec_t)
matArray_t: type = wp.array(dtype=mat_t)
intArray_t: type = wp.array(dtype=int_t)
scalarArray_t: type = wp.array(dtype=scalar_t)

@wp.func
def scalar(value: float) -> scalar_t:
    """Convert a Python float to the active scalar type."""
    return scalar_t(value)