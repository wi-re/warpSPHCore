"""Generate CMake source lists and wheel provenance without importing warpSPHCore."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'src/warpSPHCore'
spec = importlib.util.spec_from_file_location('_native_sources', PACKAGE / 'native/_sources.py')
sources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


def generate(backend, cpu_arch, architectures, output, metadata):
    if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 10):
        raise RuntimeError('Native code requires Torch >= 2.10')
    if backend == 'cuda' and (not torch.version.cuda or torch.version.hip):
        raise RuntimeError('CUDA wheels must be built with CUDA Torch')
    if backend == 'hip' and not torch.version.hip:
        raise RuntimeError('ROCm wheels must be built with ROCm Torch')
    cpu, gpu = sources.discover_sources(PACKAGE)
    common = sorted(str(p) for p in (PACKAGE / 'native/csrc').rglob('*.cpp'))
    includes = [str(PACKAGE / 'native/csrc'), str(PACKAGE), str(Path(torch.__file__).parent / 'include')]
    if backend == 'hip':
        all_cpu, gpu, staged_includes = sources.prepare_hip_sources(common + cpu, gpu, includes[:2],
                                                                   output.parent / 'hipified')
        common, cpu = all_cpu, []
        includes = staged_includes + includes[2:]
    compiler_flags = []
    if cpu_arch != 'portable':
        prefix = '-mcpu=' if platform.machine().lower() in ('aarch64', 'arm64') else '-march='
        compiler_flags.append(prefix + cpu_arch)
    values = dict(NATIVE_CPU_SOURCES=common + cpu, NATIVE_GPU_SOURCES=gpu,
                  NATIVE_INCLUDE_DIRS=includes, NATIVE_TORCH_LIB_DIR=[str(Path(torch.__file__).parent / 'lib')],
                  NATIVE_CPU_FLAGS=compiler_flags)
    output.write_text('\n'.join('set(' + key + ' ' + ' '.join('"' + str(v).replace('\\', '/') + '"' for v in vals) + ')'
                                for key, vals in values.items()) + '\n')
    metadata.write_text(json.dumps(dict(schema=1, stable_abi='2.10', library='libwarpsph_native.so',
        torch_version=torch.__version__, backend=backend,
        runtime=(torch.version.hip if backend == 'hip' else torch.version.cuda) if backend != 'cpu' else None,
        machine=platform.machine(), cpu_arch=cpu_arch,
        cpu_features=sources.cpu_flags() if cpu_arch == 'native' else [],
        gpu_architectures=architectures.split(';') if architectures else [],
        source_digest=sources.source_digest(PACKAGE)), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'hip'], required=True)
    parser.add_argument('--cpu-arch', default='portable')
    parser.add_argument('--architectures', default='')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    generate(args.backend, args.cpu_arch, args.architectures, args.output, args.metadata)
