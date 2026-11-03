"""Build one CPU/CUDA/ROCm native wheel against the active Torch installation.

Examples (run on the CPU architecture that will execute the wheel):
  python scripts/packaging/build_native_wheel.py --backend cuda --gpu-arch '80;90;120'
  python scripts/packaging/build_native_wheel.py --backend hip --gpu-arch 'gfx942;gfx1100'
  # On a GH200 build node, targeting Grace CPU + H100:
  python scripts/packaging/build_native_wheel.py --backend cuda --gpu-arch 90 --cpu-arch native
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'hip'], required=True)
    parser.add_argument('--gpu-arch', help='Explicit semicolon/comma-separated CUDA or AMD targets')
    parser.add_argument('--cpu-arch', default='portable', help='portable, native, or a host-compiler target')
    parser.add_argument('--torch-version', help='Require this exact installed Torch version')
    parser.add_argument('--output', type=Path, default=Path('dist/native'))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--cmake-arg', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    import torch
    from wheel.wheelfile import WheelFile
    if args.torch_version and torch.__version__ != args.torch_version:
        parser.error(f'Expected Torch {args.torch_version}, installed {torch.__version__}')
    if args.backend != 'cpu' and not args.gpu_arch:
        parser.error('--gpu-arch is required for GPU wheels; no GPU detection is used')
    if args.backend == 'cpu' and args.gpu_arch:
        parser.error('--gpu-arch cannot be used for a CPU wheel')
    if args.backend == 'cuda' and (not torch.version.cuda or torch.version.hip):
        parser.error('Use CUDA Torch to build a CUDA wheel')
    if args.backend == 'hip' and not torch.version.hip:
        parser.error('Use ROCm Torch to build a ROCm wheel')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Do not inherit the Python build's embedded compiler arguments (notably
    # Conda's compiler_compat linker). Honour explicitly selected site compilers.
    env.setdefault('CXX', shutil.which('c++') or 'c++')
    env.setdefault('CC', shutil.which('cc') or 'cc')
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(args.jobs)
    settings = ['-Cwheel.platlib=true', '-Ccmake.define.WARPSPHCORE_BUILD_NATIVE=ON',
                f'-Ccmake.define.WARPSPHCORE_NATIVE_BACKEND={args.backend}',
                f'-Ccmake.define.WARPSPHCORE_CPU_ARCH={args.cpu_arch}']
    if args.gpu_arch:
        key = 'CMAKE_CUDA_ARCHITECTURES' if args.backend == 'cuda' else 'CMAKE_HIP_ARCHITECTURES'
        settings.append(f'-Ccmake.define.{key}={args.gpu_arch.replace(",", ";")}')
    if args.backend == 'cuda':
        nvcc = env.get('CUDACXX') or shutil.which('nvcc') or str(Path(sys.prefix) / 'bin/nvcc')
        if not Path(nvcc).is_file():
            parser.error('nvcc is required; set CUDACXX or add it to PATH')
        env['CUDACXX'] = nvcc
    settings.extend('-Ccmake.define.' + arg for arg in args.cmake_arg)
    with tempfile.TemporaryDirectory(prefix='warpsph-wheel-') as temporary:
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                        '--outdir', temporary, *settings], cwd=root, env=env, check=True)
        source, = Path(temporary).glob('*.whl')
        with zipfile.ZipFile(source) as archive:
            manifest = __import__('json').loads(archive.read('warpSPHCore/native/_lib/build.json'))
            assert manifest['torch_version'] == torch.__version__
            runtime = manifest['runtime'] or ''
            tag = '1' + args.backend + re.sub('[^a-zA-Z0-9]', '', runtime)
            parts = source.name.removesuffix('.whl').split('-')
            parts.insert(2, tag)
            destination = output / ('-'.join(parts) + '.whl')
            with WheelFile(destination, 'w') as wheel:
                for item in archive.infolist():
                    if item.filename.endswith('/RECORD'):
                        continue
                    data = archive.read(item.filename)
                    if item.filename.endswith('/METADATA'):
                        headers, separator, body = data.decode().partition('\n\n')
                        headers = re.sub(r'^Requires-Dist: torch(?:[ \<>=!~;(].*)?$',
                                         'Requires-Dist: torch==' + torch.__version__, headers,
                                         flags=re.MULTILINE)
                        data = (headers + separator + body).encode()
                    elif item.filename.endswith('/WHEEL'):
                        data = data.rstrip(b'\n') + f'\nBuild: {tag}\n\n'.encode()
                    wheel.writestr(item.filename, data)
        print(f'Native {args.backend} wheel: {destination}')
        print(f'Qualified Torch: {torch.__version__}; CPU: {args.cpu_arch}; GPU: {args.gpu_arch or "none"}')


if __name__ == '__main__':
    main()
