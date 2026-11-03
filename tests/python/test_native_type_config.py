"""Native host dtype policy can load independently of Warp and kernel compilation."""
import importlib.util
from pathlib import Path
import sys
import types
import torch

SOURCE=Path(__file__).resolve().parents[2]/'src/warpSPHCore/native/_type_config.py'


def load():
    spec=importlib.util.spec_from_file_location('_native_dtype_probe',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_native_independent_integer_widths_and_preconfiguration(monkeypatch):
    monkeypatch.setitem(sys.modules,'warpSPHCore_config',types.ModuleType('warpSPHCore_config'))
    for index,offset in (('int32','int64'),('int64','int32')):
        monkeypatch.setenv('warpSPHCore_INDEX_DTYPE',index);monkeypatch.setenv('warpSPHCore_OFFSET_DTYPE',offset)
        monkeypatch.setenv('warpSPHCore_PRECISION','double')
        policy=load()
        assert policy.get_torch_index_dtype()==getattr(torch,index)
        assert policy.get_torch_offset_dtype()==getattr(torch,offset)
        assert policy.get_torch_precision()==torch.float64
    config=types.ModuleType('warpSPHCore_config');config.precision='single';config.dim=3;config.index_dtype='int32'
    monkeypatch.setitem(sys.modules,'warpSPHCore_config',config)
    policy=load()
    assert policy.get_torch_precision()==torch.float32 and policy.get_dim()==3
    assert policy.get_torch_index_dtype()==torch.int32
