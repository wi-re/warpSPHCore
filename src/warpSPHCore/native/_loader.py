"""Choose packaged native code before entering any compiler/JIT machinery."""
import json
import os
from pathlib import Path
import platform
import re
from ._sources import cpu_flags, source_digest


def incompatibility(metadata, package, torch):
    if metadata.get('schema') != 1 or metadata.get('stable_abi') != '2.10':
        return 'unrecognised native ABI metadata'
    if metadata.get('machine') != platform.machine():
        return 'CPU architecture differs from this machine'
    if metadata.get('torch_version') != torch.__version__:
        return f"wheel was qualified with Torch {metadata.get('torch_version')}, installed {torch.__version__}"
    backend = metadata.get('backend')
    if backend not in ('cpu', 'cuda', 'hip'):
        return 'unknown native backend'
    mode = os.environ.get('WARPSPHCORE_NATIVE_BUILD', 'auto').lower()
    if mode not in ('auto', 'cpu', 'gpu'):
        raise ValueError('WARPSPHCORE_NATIVE_BUILD must be auto, cpu or gpu')
    if mode == 'cpu' and backend != 'cpu':
        return 'CPU-only build requested'
    if (mode == 'gpu' or (mode == 'auto' and torch.cuda.is_available())) and backend == 'cpu':
        return 'GPU support requested but wheel contains CPU code only'
    runtime = torch.version.hip if backend == 'hip' else torch.version.cuda
    if backend != 'cpu' and metadata.get('runtime') != runtime:
        return f'{backend} runtime does not match this Torch build'
    if backend == 'cuda' and torch.version.hip:
        return 'CUDA library cannot be loaded into ROCm Torch'
    if backend != 'cpu' and torch.cuda.is_available():
        targets = metadata.get('gpu_architectures', [])
        for index in range(torch.cuda.device_count()):
            if backend == 'cuda':
                major, minor = torch.cuda.get_device_capability(index)
                sm = major * 10 + minor
                supported = False
                for target in targets:
                    match = re.fullmatch(r'(\d+)([af]?)(?:-(real|virtual))?', target)
                    if not match:
                        continue
                    architecture, suffix, kind = match.groups()
                    architecture = int(architecture)
                    # Special a/f instruction-set targets require their own
                    # compatibility rules; accept the exact architecture only.
                    supported |= ((sm == architecture and kind != 'virtual') or
                                  (not suffix and kind != 'real' and sm >= architecture))
                if not supported:
                    return f'CUDA device sm_{sm} is not covered by the wheel'
            else:
                architecture = torch.cuda.get_device_properties(index).gcnArchName
                if not any(architecture == target or (':' not in target and
                           architecture.split(':')[0] == target) for target in targets):
                    return f'ROCm device {architecture} is not covered by the wheel'
    if metadata.get('cpu_arch') == 'native':
        required = set(metadata.get('cpu_features', []))
        if not required or not required.issubset(cpu_flags()):
            return 'host-tuned wheel CPU features are not available here'
    if metadata.get('source_digest') != source_digest(package):
        return 'packaged source/header revision differs from the binary'
    return None


def load_prebuilt(package):
    mode = os.environ.get('WARPSPHCORE_NATIVE_LOAD', 'auto').lower()
    if mode not in ('auto', 'prebuilt', 'jit'):
        raise ValueError('WARPSPHCORE_NATIVE_LOAD must be auto, prebuilt or jit')
    if mode == 'jit':
        return False
    import torch
    manifest = Path(package) / 'native/_lib/build.json'
    reason = 'no packaged native library'
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        reason = incompatibility(metadata, package, torch)
        filename = metadata.get('library', '')
        library = manifest.parent / filename
        if not filename or Path(filename).name != filename or not library.is_file():
            reason = 'packaged library file is missing or invalid'
        if reason is None:
            # A dlopen/registration failure is an installation error, not a reason
            # to load a second copy of the operator namespace through JIT.
            torch.ops.load_library(str(library))
            return True
    if mode == 'prebuilt':
        raise RuntimeError(f'No compatible prebuilt native backend: {reason}. '
                           'Install a matching wheel or select WARPSPHCORE_NATIVE_LOAD=jit.')
    return False
