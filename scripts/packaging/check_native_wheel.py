"""Verify a binary native wheel with Warp and compiler/JIT imports blocked."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--tests', action='store_true', help='Run the native correctness suite against the installed wheel')
    args = parser.parse_args()
    example = Path(__file__).resolve().parents[1] / 'examples/native_core.py'
    with tempfile.TemporaryDirectory(prefix='warpsph-prebuilt-check-') as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            names = archive.namelist()
            assert 'warpSPHCore/native/_lib/build.json' in names
            assert 'warpSPHCore/native/_lib/libwarpsph_native.so' in names
            metadata = json.loads(archive.read('warpSPHCore/native/_lib/build.json'))
            archive.extractall(directory)
        env = os.environ.copy()
        if metadata['backend'] == 'cpu':
            if args.device != 'cpu':
                parser.error('A CPU wheel can only be checked with --device cpu')
            env['CUDA_VISIBLE_DEVICES'] = ''
        code = '''
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
sys.path.insert(0, sys.argv[1])
os.environ['WARPSPHCORE_BACKEND'] = 'native'
os.environ['WARPSPHCORE_NATIVE_LOAD'] = 'prebuilt'
os.environ['WARPSPHCORE_NATIVE_BUILD'] = 'auto'
class NoCompilerOrWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'warp' or fullname.startswith('warp.') or fullname == 'torch.utils.cpp_extension':
            raise ImportError('Compiler/Warp import forbidden in prebuilt-wheel check: ' + fullname)
sys.meta_path.insert(0, NoCompilerOrWarp())
import warpSPHCore as sph
assert Path(sph.__file__).is_relative_to(sys.argv[1])
metadata = json.loads((Path(sph.__file__).parent / 'native/_lib/build.json').read_text())
assert metadata['stable_abi'] == '2.10'
os.environ['WARPSPHCORE_NATIVE_BUILD'] = 'cpu' if metadata['backend'] == 'cpu' else 'gpu'
import warpSPHCore.native._build as builder
def forbidden(*args, **kwargs):
    raise AssertionError('JIT must never run for a compatible prebuilt wheel')
builder.load_extension = forbidden
tests = sys.argv[4]
device = sys.argv[3]
sys.argv = [sys.argv[2], '--device', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
assert 'torch.utils.cpp_extension' not in sys.modules
print('Prebuilt wheel: compiler/JIT and Warp imports blocked; passed')
if tests:
    import pytest
    raise SystemExit(pytest.main([str(Path(tests) / 'native/torch'), '--suite', 'torch',
                                 '--device', device, '-k', 'not imports_with_warp_blocked', '-q', '-x']))
'''
        subprocess.run([sys.executable, '-c', code, directory, str(example), args.device,
                        str(Path(__file__).resolve().parents[2] / 'tests') if args.tests else ''],
                       cwd=directory, env=env, check=True)


if __name__ == '__main__':
    main()
