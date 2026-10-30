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
