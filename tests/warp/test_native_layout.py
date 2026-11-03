import importlib
import pytest
import warpSPHCore as sph

def test_common_wrapper_owns_high_level_warp_dispatch():
    if sph.backend != 'warp': pytest.skip('Warp process only')
    from warpSPHCore.coreOperations.native import computeDensity, computeGradient
    from warpSPHCore import coreOperations, operations
    assert operations._computeSPHDensity_stateBackend is computeDensity
    assert operations._computeSPHGradient_stateBackend is computeGradient
    assert coreOperations._computeSPHDensity_stateBackend is computeDensity


def test_old_warp_paths_alias_colocated_modules(monkeypatch):
    if sph.backend != 'warp': pytest.skip('Warp process only')
    for name, spec in [('wp_density', '_DENSITY_SPEC'), ('wp_gradient', '_GRADIENT_SPEC'),
                       ('wp_interpolate', '_INTERPOLATE_SPEC'), ('wp_divergence', '_DIVERGENCE_SPEC'),
                       ('wp_curl', '_CURL_SPEC'), ('wp_laplacian', '_LAPLACIAN_SPEC'),
                       ('wp_covariance', '_COVARIANCE_SPEC')]:
        old = importlib.import_module(f'warpSPHCore.coreOperations.{name}')
        new = importlib.import_module(f'warpSPHCore.coreOperations.native.{name}')
        assert old is new and getattr(old, spec) is getattr(new, spec)
        sentinel = object()
        monkeypatch.setattr(old, spec, sentinel)
        assert getattr(new, spec) is sentinel
