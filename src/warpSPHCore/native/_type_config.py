"""Host dtype settings without importing Warp; native kernels use tensor dtype."""
import os
from typing import Any
import torch
import warpSPHCore_config as config


def integer_type(setting,override):
    name = override if override is not None else os.getenv(setting,'int32')
    name = str(name).strip().lower() or 'int32'
    if name not in ('int32','int64'):raise ValueError(f'Invalid {setting}={name!r}; expected int32 or int64')
    return getattr(torch,name)


aliases = dict(half=torch.float16,float16=torch.float16,single=torch.float32,float32=torch.float32,
               double=torch.float64,float64=torch.float64)
precision = getattr(config,'precision',None) if getattr(config,'precision',None) is not None else os.getenv('warpSPHCore_PRECISION','float32')
try:scalar_t = aliases[str(precision).strip().lower() or 'float32']
except KeyError as error:raise ValueError(f'Invalid warpSPHCore_PRECISION={precision!r}') from error
index_t = integer_type('warpSPHCore_INDEX_DTYPE',getattr(config,'index_dtype',None))
offset_t = integer_type('warpSPHCore_OFFSET_DTYPE',getattr(config,'offset_dtype',None))
neighbor_count_t = torch.int32
_dimension = getattr(config,'dim',None) if getattr(config,'dim',None) is not None else os.getenv('warpSPHCore_DIM','Any')
if str(_dimension).strip().lower() in ('','any','dynamic','none','*'):dim_t = Any
else:
    dim_t = int(_dimension)
    if dim_t<1:raise ValueError('warpSPHCore_DIM must be positive or Any')


def get_torch_precision():return scalar_t

def get_torch_index_dtype():return index_t

def get_torch_offset_dtype():return offset_t

def get_precision():return scalar_t

def get_dim():return dim_t

def get_type_config():return dict(scalar_t=scalar_t,dim_t=dim_t,index_t=index_t,offset_t=offset_t,neighbor_count_t=neighbor_count_t)

def to_torch(value):return torch.tensor(value,dtype=scalar_t)

def to_numpy(value):return float(value)

__all__ = ['scalar_t','dim_t','index_t','offset_t','neighbor_count_t','get_torch_precision',
           'get_torch_index_dtype','get_torch_offset_dtype','get_precision','get_dim','get_type_config','to_torch','to_numpy']
