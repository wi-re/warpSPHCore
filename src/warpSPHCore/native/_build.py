"""Shared source discovery and deferred builds for module-owned kernels."""
import os
from pathlib import Path
import shutil
import threading
import platform
from ._sources import discover_sources, prepare_hip_sources, jit_source_paths

_extension = None
_lock = threading.Lock()



def get_include():
    """Return the installed shared C++ header root without compiling anything."""
    return str(Path(__file__).resolve().parent / 'csrc')


def load_extension(name, cpu_sources, gpu_sources=(), *, include_paths=()):
    """Compile source sets using the installed Torch build's CPU/CUDA/HIP tools.

    Frontend modules may use this helper with their own extension name, sources,
    namespace registration and include roots. Shared core headers are included
    automatically. No core operator is hardcoded here.
    """
    import torch
    mode = os.environ.get('WARPSPHCORE_NATIVE_BUILD', 'auto').lower()
    if mode not in ('auto', 'cpu', 'gpu'):
        raise ValueError('WARPSPHCORE_NATIVE_BUILD must be auto, cpu or gpu')
    gpu = mode == 'gpu' or (mode == 'auto' and torch.cuda.is_available())
    if gpu and not torch.version.hip and 'CUDA_HOME' not in os.environ:
        import sys
        nvcc = shutil.which('nvcc') or str(Path(sys.prefix) / 'bin/nvcc')
        if Path(nvcc).is_file():
            os.environ['CUDA_HOME'] = str(Path(nvcc).parent.parent)
    from torch.utils.cpp_extension import load, get_default_build_root
    if gpu and torch.version.hip:
        workspace = Path(os.environ.get('TORCH_EXTENSIONS_DIR', get_default_build_root())) / (name + '_hip_sources')
        cpu_sources, gpu_sources, include_paths = prepare_hip_sources(
            list(cpu_sources), list(gpu_sources), [get_include(), *include_paths], workspace)
    workspace = Path(os.environ.get('TORCH_EXTENSIONS_DIR', get_default_build_root())) / (name + '_sources')
    cpu_sources, gpu_sources = jit_source_paths(cpu_sources, gpu_sources, workspace)
    version = tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
    if version < (2, 10):
        raise RuntimeError('The native backend requires Torch >= 2.10 (stable LibTorch ABI)')
    cpu_arch = os.environ.get('WARPSPHCORE_NATIVE_CPU_ARCH', 'native')
    cpu_flags = [] if cpu_arch == 'portable' else [
        ('-mcpu=' if platform.machine().lower() in ('aarch64', 'arm64') else '-march=') + cpu_arch]
    flags = ['-O2', '-std=c++17', '-DTORCH_TARGET_VERSION=0x020a000000000000'] + cpu_flags + (['-DWITH_CUDA', '-DUSE_CUDA'] if gpu else [])
    return load(name=f'{name}_stable_{"gpu" if gpu else "cpu"}',
                sources=list(cpu_sources) + (list(gpu_sources) if gpu else []),
                extra_include_paths=list(dict.fromkeys([*([] if gpu and torch.version.hip else [get_include()]),
                                                       *(str(p) for p in include_paths)])),
                extra_cflags=flags, extra_cuda_cflags=['-O2', '-std=c++17', '-DTORCH_TARGET_VERSION=0x020a000000000000', '-DUSE_CUDA'],
                with_cuda=gpu, is_python_module=False,
                verbose=os.environ.get('WARPSPHCORE_NATIVE_VERBOSE', '0') == '1')


def build():
    """Build the core's shared infrastructure and module-local source manifests."""
    global _extension
    if _extension is not None:
        return _extension
    with _lock:
        if _extension is not None:
            return _extension
        import torch
        package = Path(__file__).resolve().parents[1]
        from ._loader import load_prebuilt
        if load_prebuilt(package):
            _extension = torch.ops.warpsph_native
            return _extension
        shared = package / 'native/csrc'
        cpu, gpu = discover_sources(package)
        common = [str(path) for path in sorted([*shared.glob('*.cpp'), *(shared / 'torch').rglob('*.cpp')])]
        common_gpu = [str(path) for path in sorted((shared / 'execution').glob('*.cu'))]
        load_extension('warpsph_native', common + cpu, common_gpu + gpu,
                       include_paths=(package,))
        _extension = torch.ops.warpsph_native
    return _extension
