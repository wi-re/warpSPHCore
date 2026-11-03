"""Reject direct ATen/c10/CPython symbol dependencies in a native wheel (Linux)."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='warpsph-abi-audit-') as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            filename = 'warpSPHCore/native/_lib/libwarpsph_native.so'
            archive.extract(filename, directory)
        library = Path(directory) / filename
        symbols = subprocess.check_output(['nm', '-D', '--undefined-only', str(library)], text=True)
        decoded = subprocess.check_output(['c++filt'], input=symbols, text=True)
        forbidden = [line for line in decoded.splitlines()
                     if re.search(r'\b(?:at|c10|torch)::|\b_?Py[A-Za-z_]\w*', line)]
        if forbidden:
            raise RuntimeError('Unstable Torch/Python symbol dependencies:\n' + '\n'.join(forbidden))
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        for line in dynamic.splitlines():
            if '(RPATH)' in line or '(RUNPATH)' in line:
                paths = re.search(r'\[(.*)\]', line).group(1).split(':')
                if any(path.startswith('/') for path in paths):
                    raise RuntimeError('Absolute runtime library path in wheel: ' + line)
        print('Stable ABI symbol audit passed; no direct ATen/c10/CPython symbols or absolute runtime paths')


if __name__ == '__main__':
    main()
