#!/usr/bin/env python3
"""Run existing CSR consumer regressions with compact-hash collection forced to split.

The override is local to this diagnostic process. It changes no installed defaults
and leaves direct-grid operator paths unchanged. Run once per import-time precision.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
TESTS = ['tests/radiusSearch/test_adjacencyCapacity.py', 'tests/radiusSearch/test_hashMapSlots.py',
         'tests/radiusSearch/test_verletValidity.py', 'tests/operations/test_operations_core.py',
         'tests/operations/test_operations_crk_analytic.py', 'tests/operations/test_tiled_curl_laplacian.py',
         'tests/operations/test_forward_mode_geometry_jvp_density.py',
         'tests/operations/test_forward_mode_geometry_jvp_gradient.py',
         'tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py',
         'tests/operations/test_polyfit.py', 'tests/operations/test_mfm_warp.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('Accessible CUDA required; CPU-only skips do not qualify this gate.')
    from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype
    search = importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
    wrapper = importlib.import_module('warpSPHCore.radiusSearch.compactHash.wrapper')
    original = search.radiusSearchOnCompactHashMap
    calls = 0

    def split(*args, **kwargs):
        nonlocal calls
        kwargs['collection'] = 'split'
        calls += 1
        return original(*args, **kwargs)

    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py')) + [Path(__file__)] + [ROOT / p for p in TESTS]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    search.radiusSearchOnCompactHashMap = wrapper.radiusSearchOnCompactHashMap = split
    try:
        with tempfile.TemporaryDirectory(prefix='split-consumers-') as directory:
            xml = Path(directory) / 'results.xml'
            result = pytest.main(['-q', '-x', *TESTS, f'--junitxml={xml}'])
            stats = [dict(s.attrib) for s in ET.parse(xml).getroot().iter('testsuite')]
    finally:
        search.radiusSearchOnCompactHashMap = wrapper.radiusSearchOnCompactHashMap = original
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    report = dict(complete=result == 0 and calls > 0, returncode=int(result), split_search_calls=calls,
                  dtype=str(get_torch_precision()), index_dtype=str(get_torch_index_dtype()),
                  offset_dtype=str(get_torch_offset_dtype()), gpu=torch.cuda.get_device_name(),
                  tests=stats, selected=TESTS, source_sha256=hashes, scope=__doc__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(int(result) or int(calls == 0))


if __name__ == '__main__':
    main()
