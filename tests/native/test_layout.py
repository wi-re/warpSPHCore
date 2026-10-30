"""Module ownership, lazy imports and source discovery contracts."""
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import warpSPHCore as sph
from warpSPHCore.native._build import discover_sources, get_include


def test_source_manifests_are_module_owned():
    package = Path(sph.__file__).parent
    cpu, gpu = discover_sources(package)
    assert any('/coreOperations/native/csrc_density/' in path for path in cpu)
    assert any('/radiusSearch/native/csrc_radius/' in path for path in gpu)
    assert all(not Path(path).is_relative_to(package / 'native/csrc') for path in cpu + gpu)
    # Higher-level native code has no registry of operator source filenames.
    assert not (package / 'native/csrc/kernels.h').exists()
    assert Path(get_include()) == package / 'native/csrc'
    assert (Path(get_include()) / 'execution/traversal.h').is_file()


def test_nested_frontend_manifest_discovery_without_import(tmp_path):
    module = tmp_path / 'modules/crk/native'
    module.mkdir(parents=True)
    (module / '__init__.py').write_text('raise RuntimeError("must not import for discovery")')
    function = module / 'csrc_correction'
    function.mkdir(parents=True)
    (function / 'crk.cpp').write_text('// module-owned CPU implementation')
    (function / 'crk.cu').write_text('// module-owned GPU implementation')
    (module / 'sources.json').write_text(json.dumps({'cpu': ['csrc_correction/crk.cpp'], 'gpu': ['csrc_correction/crk.cu']}))
    cpu, gpu = discover_sources(tmp_path)
    assert cpu == [str(function / 'crk.cpp')]
    assert gpu == [str(function / 'crk.cu')]


@pytest.mark.parametrize('filename', ['../outside.cpp', 'missing.cpp'])
def test_invalid_manifest_sources_are_rejected(tmp_path, filename):
    module = tmp_path / 'crk/native'
    module.mkdir(parents=True)
    (module.parent / 'outside.cpp').write_text('// outside the owning native directory')
    (module / 'sources.json').write_text(json.dumps({'cpu': [filename]}))
    with pytest.raises(ValueError, match='Invalid cpu source'):
        discover_sources(tmp_path)


def test_common_wrapper_owns_high_level_warp_dispatch():
    if sph.backend != 'warp': pytest.skip('Warp process only')
    from warpSPHCore.coreOperations.native import computeDensity, computeGradient
    from warpSPHCore import coreOperations, operations
    assert operations._computeSPHDensity_stateBackend is computeDensity
    assert operations._computeSPHGradient_stateBackend is computeGradient
    assert coreOperations._computeSPHDensity_stateBackend is computeDensity


def test_old_warp_paths_alias_colocated_modules(monkeypatch):
    if sph.backend != 'warp': pytest.skip('Warp process only')
    for name, spec in [('wp_density', '_DENSITY_SPEC'), ('wp_gradient', '_GRADIENT_SPEC')]:
        old = importlib.import_module(f'warpSPHCore.coreOperations.{name}')
        new = importlib.import_module(f'warpSPHCore.coreOperations.native.{name}')
        assert old is new and getattr(old, spec) is getattr(new, spec)
        sentinel = object()
        monkeypatch.setattr(old, spec, sentinel)
        assert getattr(new, spec) is sentinel


def test_module_local_native_wrappers_without_warp():
    if sph.backend != 'native': pytest.skip('native process only')
    from warpSPHCore.coreOperations.native import computeDensity, computeGradient
    from warpSPHCore.radiusSearch.native import radiusSearch
    from warpSPHCore._backend import resolve_backend
    x = torch.tensor([[-.2], [.2]])
    p = sph.ParticleState(x, torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.int32))
    d = sph.DomainDescription(torch.tensor([-1.]), torch.tensor([1.]), torch.tensor([True]), 1)
    a = radiusSearch(p, d)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    p.densities = computeDensity(p, p, d, prop, a)
    torch.testing.assert_close(p.densities, sph.warpOperation(p, prop, d, adjacency=a), rtol=0, atol=0)
    grad_prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
                                         gradientMode=sph.GradientScheme.Difference)
    value = torch.ones(2)
    torch.testing.assert_close(computeGradient(p, p, d, grad_prop, a, value, value), torch.zeros_like(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match='before importing'):
        resolve_backend('warp')
    assert 'warp' not in sys.modules
