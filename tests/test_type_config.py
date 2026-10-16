"""Import-time configuration is isolated from already compiled kernel types."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import warp as wp

SOURCE=Path(__file__).resolve().parents[1]/'src/warpSPHCore/type_config.py'


@pytest.mark.parametrize('index,offset,precision',[
    ('int32','int32','float64'),('int32','int64','float32'),
    ('int64','int32','float32'),('int64','int64','float64')])
def test_import_time_integer_widths_are_independent(monkeypatch,index,offset,precision):
    monkeypatch.setenv('warpSPHCore_INDEX_DTYPE',index)
    monkeypatch.setenv('warpSPHCore_OFFSET_DTYPE',offset)
    monkeypatch.setenv('warpSPHCore_PRECISION',precision)
    monkeypatch.setitem(sys.modules,'warpSPHCore_config',types.ModuleType('warpSPHCore_config'))
    name='_index_config_probe_'+index+'_'+offset+'_'+precision
    spec=importlib.util.spec_from_file_location(name,SOURCE)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    assert module.index_t==getattr(wp,index)
    assert module.offset_t==getattr(wp,offset)
    assert module.scalar_t==getattr(wp,precision)
    assert module.neighbor_count_t==wp.int32


def test_python_preconfiguration_takes_precedence(monkeypatch):
    for setting,value in [('INDEX_DTYPE','int32'),('OFFSET_DTYPE','int32'),('PRECISION','float32')]:
        monkeypatch.setenv('warpSPHCore_'+setting,value)
    config=types.ModuleType('warpSPHCore_config')
    config.index_dtype='int64';config.offset_dtype='int64';config.precision='double'
    monkeypatch.setitem(sys.modules,'warpSPHCore_config',config)
    name='_python_index_config_probe'
    spec=importlib.util.spec_from_file_location(name,SOURCE)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    assert module.index_t==module.offset_t==wp.int64
    assert module.scalar_t==wp.float64


@pytest.mark.parametrize('setting',['INDEX_DTYPE','OFFSET_DTYPE'])
def test_invalid_integer_width_is_rejected_at_import(monkeypatch,setting):
    monkeypatch.setenv('warpSPHCore_'+setting,'uint32')
    spec=importlib.util.spec_from_file_location('_invalid_index_config_probe',SOURCE)
    module=importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError,match=setting):spec.loader.exec_module(module)
