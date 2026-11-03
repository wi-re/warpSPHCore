"""Changes to a backend must reach its tests without launching unrelated suites."""
import importlib.util
from pathlib import Path
import pytest

source = Path(__file__).resolve().parents[2] / 'scripts/testing/select_suites.py'
spec = importlib.util.spec_from_file_location('suite_selection', source)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def test_native_header_runs_both_native_consumers_without_warp_suite():
    result = selection.select(['src/warpSPHCore/native/csrc/execution/scan.h'])
    assert result['native'] and result['torch']
    assert not result['warp'] and not result['torch_full']


def test_python_bridge_changes_get_full_autograd_coverage():
    result = selection.select(['src/warpSPHCore/native/_autograd.py'])
    assert result['python'] and result['torch_full'] and result['torch']
    assert not result['native'] and not result['warp']


def test_shared_models_reach_all_backends():
    assert all(selection.select(['src/warpSPHCore/_models.py']).values())


@pytest.mark.parametrize('path', [
    'radiusSearch/chunked.py',
    'crk/crk_terms.py',
    'linalg/krylov/_execution.py',
])
def test_shared_numerical_controllers_reach_both_backends(path):
    result = selection.select(['src/warpSPHCore/' + path])
    assert result['warp'] and result['torch'] and result['torch_full'] and result['parity']


def test_docs_only_changes_do_not_run_numerics():
    assert not any(selection.select(['README.md', 'docs/docs/native-backend.md']).values())


def test_standalone_runtime_does_not_recompile_torch_or_warp():
    result = selection.select(['src/warpSPHCore/native/csrc/runtime/api.cpp'])
    assert result['native']
    assert sum(result.values()) == 1


def test_colocated_warp_kernel_does_not_run_native_or_adapter_matrix():
    result = selection.select(['src/warpSPHCore/coreOperations/native/wp_density.py'])
    assert result['warp'] and result['parity']
    assert not result['native'] and not result['torch'] and not result['torch_full']


def test_shared_device_scan_reaches_both_native_consumers():
    result = selection.select(['src/warpSPHCore/native/csrc/execution/scan_gpu.cu'])
    assert result['native'] and result['torch'] and not result['warp']


def test_shared_grid_math_runs_native_search_parity():
    result = selection.select(['src/warpSPHCore/native/csrc/math/grid.h'])
    assert result['native'] and result['torch'] and result['parity']
    assert not result['warp']
