import warp as wp
from ..type_config import scalar_t
from warp.types import vector, matrix

@wp.func
def warp_eye(input: vector(dtype=scalar_t, length=1)):
    retVal = matrix(shape = (1, 1), dtype=scalar_t)
    for i in range(1):
        for j in range(1):
            retVal[i][j] = scalar_t(1.0) if i == j else scalar_t(0.0)
    return retVal
@wp.func
def warp_eye(input: vector(dtype=scalar_t, length=2)):
    retVal = matrix(shape = (2, 2), dtype=scalar_t)
    for i in range(2):
        for j in range(2):
            retVal[i][j] = scalar_t(1.0) if i == j else scalar_t(0.0)
    return retVal
@wp.func
def warp_eye(input: vector(dtype=scalar_t, length=3)):
    retVal = matrix(shape = (3, 3), dtype=scalar_t)
    for i in range(3):
        for j in range(3):
            retVal[i][j] = scalar_t(1.0) if i == j else scalar_t(0.0)
    return retVal
