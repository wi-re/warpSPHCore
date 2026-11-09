"""The tensor compact-grid record is shared without a Warp type dependency."""
import os
import subprocess
import sys


def test_native_compact_grid_descriptor_without_warp():
    script = '''
import importlib.abc, sys
class NoWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'warp':
            raise AssertionError(name)
sys.meta_path.insert(0, NoWarp())
import torch
from dataclasses import fields
from warpSPHCore import CompactHashMap
from warpSPHCore.dataTypes import CompactHashMap as PublicRecord
from warpSPHCore.dataTypes.hashMap_t import CompactHashMap as SharedRecord
assert CompactHashMap is PublicRecord is SharedRecord
names = ('sortedPositions', 'sortedSupports', 'sortIndex', 'hashTable',
         'sortedCellTable', 'qMin', 'qMax', 'hCell', 'numCells', 'mode_uint',
         'D', 'searchRadius', 'numOffsets', 'cellOffsets')
assert tuple(f.name for f in fields(CompactHashMap)) == names
values = [torch.empty(0) for _ in names]
for name, value in dict(hCell=.5, mode_uint=0, D=2, searchRadius=1, numOffsets=9).items():
    values[names.index(name)] = value
record = CompactHashMap(*values)
assert all(getattr(record, name) is value for name, value in zip(names, values))
assert 'warp' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], text=True,
                            capture_output=True,
                            env=os.environ | {'WARPSPHCORE_BACKEND': 'native',
                                              'WARPSPHCORE_OPERATORS': 'native',
                                              'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
