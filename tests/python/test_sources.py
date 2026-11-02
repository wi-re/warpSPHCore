"""Module ownership, lazy imports and source discovery contracts."""
import json
from pathlib import Path

import pytest
import warpSPHCore as sph
from warpSPHCore.native._build import discover_sources, get_include


def test_source_manifests_are_module_owned():
    package = Path(sph.__file__).parent
    cpu, gpu = discover_sources(package)
    assert any('/coreOperations/native/csrc_density/' in path for path in cpu)
    assert any('/radiusSearch/native/csrc_radius/' in path for path in gpu)
    assert any('/radiusSearch/native/csrc_compact/' in path for path in cpu + gpu)
    assert all(not Path(path).is_relative_to(package / 'native/csrc') for path in cpu + gpu)
    # Higher-level native code has no registry of operator source filenames.
    assert not (package / 'native/csrc/kernels.h').exists()
    assert Path(get_include()) == package / 'native/csrc'
    assert (Path(get_include()) / 'execution/traversal.h').is_file()
    assert (Path(get_include()) / 'execution/compactTraversal.h').is_file()
    assert (Path(get_include()) / 'execution/scan_gpu.cu').is_file()


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


def test_header_only_module_participates_in_build_digest(tmp_path):
    from warpSPHCore.native._sources import source_digest
    module = tmp_path / 'crk/native'
    folder = module / 'csrc_correction'
    folder.mkdir(parents=True)
    (module / 'sources.json').write_text('{"cpu": [], "gpu": []}')
    header = folder / 'nt_correction.h'
    header.write_text('// initial coefficient formula')
    before = source_digest(tmp_path)
    header.write_text('// changed coefficient formula')
    assert discover_sources(tmp_path) == ([], [])
    assert source_digest(tmp_path) != before


def test_jit_module_basename_collisions_preserve_relative_headers(tmp_path):
    from warpSPHCore.native._sources import jit_source_paths
    sources = []
    for module in ('core', 'crk'):
        folder = tmp_path / module
        folder.mkdir()
        (folder / 'local.h').write_text(f'// {module} header')
        path = folder / 'nt_density_forward.cpp'
        path.write_text('#include "local.h"\n// ' + module)
        sources.append(str(path))
    unique = tmp_path / 'unique.cpp'
    unique.write_text('// no collision')
    cpu, gpu = jit_source_paths([*sources, str(unique)], [], tmp_path / 'staged')
    assert not gpu
    assert len({Path(path).stem for path in cpu}) == 3
    assert cpu[-1] == str(unique)
    for staged, original in zip(cpu[:2], sources):
        # Including the original unit, rather than copying its body, retains
        # the correct local.h directory and the compiler's header dependencies.
        assert f'#include "{original}"' in Path(staged).read_text()
    before = Path(cpu[0]).read_text()
    timestamp = Path(cpu[0]).stat().st_mtime_ns
    assert jit_source_paths([*sources, str(unique)], [], tmp_path / 'staged')[0] == cpu
    assert Path(cpu[0]).stat().st_mtime_ns == timestamp
    Path(sources[0]).write_text('#include "local.h"\n// updated source')
    assert jit_source_paths([*sources, str(unique)], [], tmp_path / 'staged')[0] == cpu
    assert Path(cpu[0]).read_text() != before
