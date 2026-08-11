
from typing import Any
from warp.types import vector, matrix
import warp as wp

class vec1f(vector(length=1, dtype=wp.float32)):
    pass

class mat11f(matrix(shape=(1, 1), dtype=wp.float32)):
    pass

class vec1h(vector(length=1, dtype=wp.float16)):
    pass
class mat11h(matrix(shape=(1, 1), dtype=wp.float16)):
    pass

class vec1d(vector(length=1, dtype=wp.float64)):
    pass
class mat11d(matrix(shape=(1, 1), dtype=wp.float64)):
    pass

class vec1i(vector(length=1, dtype=wp.int32)):
    pass
