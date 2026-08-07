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

@wp.func
def mortonPattern(index: wp.int32) -> wp.uint64:
    # This function addresses a fundamental limitaton in warp. We cannot have 64 bit literal values as python does not support the ull suffix. But, we do know the patterns. We can generate the correct ones by building them as as 32 bit values and shifting them left by 32 bits. This allows us to generate the correct patterns for the splitBy3Bits function, which is the basis for the Morton encoding.
    # However
    # this also would require unsigned literals which python does not support so we need to assemble these values from 16 bit components instead
    bits_00_15 = wp.uint64(0)
    bits_16_31 = wp.uint64(0)
    bits_32_47 = wp.uint64(0)
    bits_48_63 = wp.uint64(0)
    
    if index == 0:
        # generate 0x1fffff:
        # lower = wp.uint64(0x1fffff)
        # upper = wp.uint64(0) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xffff)
        bits_16_31 = wp.uint64(0x1fff)
        bits_32_47 = wp.uint64(0)
        bits_48_63 = wp.uint64(0)
    elif index == 1:
        # generate 0x1f0000 0000ffff:
        # lower = wp.uint64(0x0000ffff)
        # upper = wp.uint64(0x001f0000) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xffff)
        bits_16_31 = wp.uint64(0x0000)
        bits_32_47 = wp.uint64(0x0000)
        bits_48_63 = wp.uint64(0x001f)
    elif index == 2:
        # generate 0x1f0000ff 0000ff:
        # lower = wp.uint64(0xff0000ff)
        # upper = wp.uint64(0x001f0000) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x00ff)
        bits_16_31 = wp.uint64(0xff00)
        bits_32_47 = wp.uint64(0x0000)
        bits_48_63 = wp.uint64(0x001f)
    elif index == 3:
        # generate 0x100f00f0 0f00f00f:
        # lower = wp.uint64(0x0f00f00f)
        # upper = wp.uint64(0x100f00f0) << wp.uint64(32)
        bits_00_15 = wp.uint64(0xf00f)
        bits_16_31 = wp.uint64(0x0f00)
        bits_32_47 = wp.uint64(0x00f0)
        bits_48_63 = wp.uint64(0x100f)
    elif index == 4:
        # genrate 0x10c30c30 c30c30c3
        # lower = wp.uint64(0xc30c30c3)
        # upper = wp.uint64(0x10c30c30) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x30c3)
        bits_16_31 = wp.uint64(0xc30c)
        bits_32_47 = wp.uint64(0x0c30)
        bits_48_63 = wp.uint64(0x10c3)
    elif index == 5:
        # generate 0x12492492 49249249
        # lower = wp.uint64(0x49249249)
        # upper = wp.uint64(0x12492492) << wp.uint64(32)
        bits_00_15 = wp.uint64(0x9249)
        bits_16_31 = wp.uint64(0x4924)
        bits_32_47 = wp.uint64(0x2492)
        bits_48_63 = wp.uint64(0x1249)

    return bits_00_15 | (bits_16_31 << wp.uint64(16)) | (bits_32_47 << wp.uint64(32)) | (bits_48_63 << wp.uint64(48))
    

@wp.func
def splitBy3Bits64(n: wp.uint64) -> wp.uint64:
    n = (n | (n << wp.uint64(32))) & mortonPattern(1)
    n = (n | (n << wp.uint64(16))) & mortonPattern(2)
    n = (n | (n << wp.uint64(8))) & mortonPattern(3)
    n = (n | (n << wp.uint64(4))) & mortonPattern(4)
    n = (n | (n << wp.uint64(2))) & mortonPattern(5)
    return n

@wp.func 
def computeZOrderIndex64(index: wp.vec3i) -> wp.int64:
    # Morton encoding (Z-order curve) for 3D indices into a 64-bit integer
    x = wp.uint64(index.x)
    y = wp.uint64(index.y)
    z = wp.uint64(index.z)
    
    return wp.int64(splitBy3Bits64(x) | (splitBy3Bits64(y) << wp.uint64(1)) | (splitBy3Bits64(z) << wp.uint64(2)))
    
    