"""Build-time source inventory; usable without importing Torch or the package."""
import hashlib
import json
from pathlib import Path


def discover_sources(package_root):
    """Collect native/sources.json manifests without importing Python modules.

    The same convention works for core and frontend packages, including nested
    modules such as modules/crk/native. Each module owns its source manifest,
    with translation units grouped under csrc_<function>/.
    """
    root = Path(package_root).resolve()
    cpu, gpu = [], []
    for manifest in sorted(root.rglob('native/sources.json')):
        definition = json.loads(manifest.read_text())
        for target, destination in (('cpu', cpu), ('gpu', gpu)):
            for filename in definition.get(target, []):
                source = (manifest.parent / filename).resolve()
                if not source.is_relative_to(manifest.parent) or not source.is_file():
                    raise ValueError(f'Invalid {target} source in {manifest}: {filename}')
                destination.append(str(source))
    return cpu, gpu


def source_digest(package_root):
    """Hash module manifests and compiled sources/headers using relative paths."""
    root = Path(package_root).resolve()
    files = set((root / 'native/csrc').rglob('*'))
    for manifest in root.rglob('native/sources.json'):
        files.add(manifest)
        files.update(manifest.parent.rglob('*'))
    digest = hashlib.sha256()
    for path in sorted(p for p in files if p.is_file() and
                       (p.name == 'sources.json' or p.suffix in ('.h', '.cuh', '.cpp', '.cu'))):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cpu_flags():
    """Conservative feature intersection for a host-tuned wheel's CPU guard."""
    info = Path('/proc/cpuinfo')
    if not info.is_file():
        return []
    features = [set(line.split(':', 1)[1].split()) for line in info.read_text().splitlines()
                if line.split(':', 1)[0].strip().lower() in ('flags', 'features')]
    return sorted(set.intersection(*features)) if features else []


def prepare_hip_sources(cpu, gpu, include_paths, workspace):
    """Hipify a private copy, never writing into an installed/source package."""
    import shutil
    from torch import __version__ as torch_version
    from torch.utils.file_baton import FileBaton
    from torch.utils.hipify import hipify_python
    includes = [Path(p).resolve() for p in include_paths]
    roots = sorted(set(includes), key=lambda p: len(p.parts))
    roots = [p for p in roots if not any(p != other and p.is_relative_to(other) for other in roots)]
    for filename in cpu + gpu:
        parent = Path(filename).resolve().parent
        if not any(parent.is_relative_to(p) for p in roots):
            roots.append(parent)
    digest = hashlib.sha256()
    digest.update(json.dumps([cpu, gpu, [str(p) for p in includes], torch_version]).encode())
    for folder in roots:
        digest.update(str(folder).encode())
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ('.h', '.hpp', '.cuh', '.cpp', '.cu'):
                digest.update(str(path.relative_to(folder)).encode())
                digest.update(path.read_bytes())
    stage = Path(workspace) / digest.hexdigest()[:20]
    stage.mkdir(parents=True, exist_ok=True)
    marker = stage / 'sources.json'
    baton = FileBaton(str(stage / 'lock'))
    if baton.try_acquire():
        try:
            if not marker.is_file():
                destinations = [stage / f'root{i}' for i in range(len(roots))]
                for folder, destination in zip(roots, destinations):
                    shutil.copytree(folder, destination, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('__pycache__', '_lib', '.git'))
                def copied(filename):
                    path = Path(filename).resolve()
                    for folder, destination in zip(roots, destinations):
                        if path.is_relative_to(folder):
                            return str(destination / path.relative_to(folder))
                    raise ValueError(f'No staged include root for {filename}')
                staged_cpu, staged_gpu = ([copied(p) for p in group] for group in (cpu, gpu))
                staged_includes = [copied(p) for p in includes]
                result = hipify_python.hipify(project_directory=str(stage), output_directory=str(stage),
                    header_include_dirs=staged_includes, extra_files=staged_cpu + staged_gpu,
                    show_detailed=False, show_progress=False, is_pytorch_extension=True)
                def converted(filename):
                    item = result.get(filename)
                    return item.hipified_path if item and item.hipified_path else filename
                data = ([converted(p) for p in staged_cpu], [converted(p) for p in staged_gpu], staged_includes)
                marker.write_text(json.dumps(data))
        finally:
            baton.release()
    else:
        baton.wait()
    if not marker.is_file():
        raise RuntimeError('Concurrent HIP preprocessing failed; retry the native build')
    return json.loads(marker.read_text())


def jit_source_paths(cpu, gpu, workspace):
    """Disambiguate Torch JIT's basename-derived object names across modules.

    A wrapper includes the original translation unit so its relative header
    lookup and compiler dependency tracking retain the original directory.
    Only colliding names need wrappers. Source bytes enter the wrapper's hash
    comment as well, so Torch's in-process versioner sees source changes.
    """
    import tempfile
    groups = [list(cpu), list(gpu)]
    result = []
    for sources in groups:
        stems = [Path(filename).stem for filename in sources]
        repeated = {stem for stem in stems if stems.count(stem) > 1}
        rewritten = []
        for filename in sources:
            path = Path(filename).resolve()
            if path.stem not in repeated:
                rewritten.append(str(path))
                continue
            if any(character in str(path) for character in ('"', '\n', '\r')):
                raise ValueError('JIT source paths cannot contain quotes or newlines')
            digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            destination = Path(workspace) / f'{path.stem}_{digest}{path.suffix}'
            content = ('// Original source SHA256: ' + hashlib.sha256(path.read_bytes()).hexdigest()
                       + '\n#include "' + path.as_posix() + '"\n')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.read_text() != content:
                # Atomic publication also handles concurrent builders staging
                # the same source before Torch acquires its compilation lock.
                with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent,
                                                 delete=False) as temporary:
                    temporary.write(content)
                Path(temporary.name).replace(destination)
            rewritten.append(str(destination))
        result.append(rewritten)
    return tuple(result)
