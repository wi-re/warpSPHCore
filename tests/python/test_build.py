"""Selection and provenance checks that do not compile code or need a GPU."""
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import pytest
import torch
from warpSPHCore.native import _loader
from warpSPHCore.native._sources import source_digest
import warpSPHCore as sph


def fake_torch(gpu=False, capability=(9, 0)):
    return SimpleNamespace(__version__='2.13.0+cu130',
        version=SimpleNamespace(cuda='13.0', hip=None),
        cuda=SimpleNamespace(is_available=lambda: gpu, device_count=lambda: 1,
                             get_device_capability=lambda i: capability))


def metadata(package, backend='cpu'):
    return dict(schema=1, stable_abi='2.10', machine=platform.machine(),
                torch_version='2.13.0+cu130', backend=backend,
                runtime='13.0' if backend == 'cuda' else None,
                cpu_arch='portable', gpu_architectures=['90-real'],
                source_digest=source_digest(package))


def test_prebuilt_compatibility(monkeypatch):
    monkeypatch.setenv('WARPSPHCORE_NATIVE_BUILD', 'auto')
    package = Path(sph.__file__).parent
    good = metadata(package)
    assert _loader.incompatibility(good, package, fake_torch()) is None
    for key, value, match in [('torch_version', '2.12.0', 'Torch'),
                              ('machine', 'different-cpu', 'architecture'),
                              ('source_digest', 'stale', 'revision')]:
        bad = dict(good, **{key: value})
        assert match in _loader.incompatibility(bad, package, fake_torch())
    cuda = metadata(package, 'cuda')
    assert _loader.incompatibility(cuda, package, fake_torch(True)) is None
    assert 'not covered' in _loader.incompatibility(cuda, package, fake_torch(True, (12, 0)))
    cuda['gpu_architectures'] = ['90-virtual']
    assert _loader.incompatibility(cuda, package, fake_torch(True, (12, 0))) is None
    monkeypatch.setenv('WARPSPHCORE_NATIVE_BUILD', 'gpu')
    assert 'CPU code only' in _loader.incompatibility(good, package, fake_torch())


def test_strict_prebuilt_never_enters_jit(tmp_path, monkeypatch):
    monkeypatch.setenv('WARPSPHCORE_NATIVE_LOAD', 'prebuilt')
    with pytest.raises(RuntimeError, match='no packaged native library'):
        _loader.load_prebuilt(tmp_path)
    monkeypatch.setenv('WARPSPHCORE_NATIVE_LOAD', 'auto')
    assert not _loader.load_prebuilt(tmp_path)
    monkeypatch.setenv('WARPSPHCORE_NATIVE_LOAD', 'jit')
    assert not _loader.load_prebuilt(tmp_path)


def test_matching_prebuilt_loads_without_compilation(tmp_path, monkeypatch):
    directory = tmp_path / 'native/_lib'
    directory.mkdir(parents=True)
    (directory / 'library.so').touch()
    (directory / 'build.json').write_text(json.dumps({'library': 'library.so'}))
    monkeypatch.setenv('WARPSPHCORE_NATIVE_LOAD', 'prebuilt')
    monkeypatch.setattr(_loader, 'incompatibility', lambda *args: None)
    loaded = []
    monkeypatch.setattr(torch.ops, 'load_library', loaded.append)
    assert _loader.load_prebuilt(tmp_path)
    assert loaded == [str(directory / 'library.so')]


def test_host_tuned_cpu_guard(monkeypatch):
    monkeypatch.setenv('WARPSPHCORE_NATIVE_BUILD', 'auto')
    package = Path(sph.__file__).parent
    native = dict(metadata(package), cpu_arch='native', cpu_features=['test-feature'])
    monkeypatch.setattr(_loader, 'cpu_flags', lambda: [])
    assert 'CPU features' in _loader.incompatibility(native, package, fake_torch())
    monkeypatch.setattr(_loader, 'cpu_flags', lambda: ['test-feature', 'extra'])
    assert _loader.incompatibility(native, package, fake_torch()) is None


def test_hip_preprocessing_uses_private_tree(tmp_path):
    from warpSPHCore.native._sources import prepare_hip_sources
    package = tmp_path / 'package'
    package.mkdir()
    header = package / 'kernel.h'
    header.write_text('#pragma once\n#include <cuda_runtime.h>\n')
    cpu = package / 'binding.cpp'
    cpu.write_text('// CPU registration\n')
    gpu = package / 'kernel.cu'
    gpu.write_text('#include "kernel.h"\n__global__ void kernel() {}\n')
    before = {p.name: p.read_text() for p in package.iterdir()}
    args = ([str(cpu)], [str(gpu)], [str(package)], tmp_path / 'cache')
    converted = prepare_hip_sources(*args)
    assert all(Path(p).is_relative_to(tmp_path / 'cache') for group in converted for p in group)
    assert {p.name: p.read_text() for p in package.iterdir()} == before
    assert prepare_hip_sources(*args) == converted
