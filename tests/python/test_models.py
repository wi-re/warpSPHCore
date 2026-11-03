import sys
import warpSPHCore as sph

def test_shared_models_import_without_warp():
    from warpSPHCore.dataTypes import ParticleState
    from warpSPHCore.dataTypes.particleData import ParticleState as OldParticleState
    from warpSPHCore.dataTypes.domain_t import DomainDescription
    from warpSPHCore.dataTypes.properties_t import OperationProperties
    from warpSPHCore.dataTypes.adjacency_t import AdjacencyList
    assert ParticleState is OldParticleState is sph.ParticleState
    assert DomainDescription is sph.DomainDescription
    assert OperationProperties is sph.OperationProperties
    assert AdjacencyList is sph.AdjacencyList
    from warpSPHCore.dataTypes import CRKState, GradHState, RenormalizationState
    from warpSPHCore.dataTypes.corrections_t import (CRKState as LegacyCRKState,
        GradHState as LegacyGradHState, RenormalizationState as LegacyRenormalizationState)
    assert CRKState is LegacyCRKState is sph.CRKState
    assert GradHState is LegacyGradHState is sph.GradHState
    assert RenormalizationState is LegacyRenormalizationState is sph.RenormalizationState
    assert 'warp' not in sys.modules


def test_operator_selection_is_explicit_and_preserves_native_only_imports(monkeypatch):
    import runpy
    from pathlib import Path
    import pytest
    source = Path(sph.__file__).with_name('_backend.py')
    for backend in ('warp', 'native'):
        monkeypatch.setenv('WARPSPHCORE_BACKEND', backend)
        monkeypatch.delenv('WARPSPHCORE_OPERATORS', raising=False)
        assert runpy.run_path(str(source))['OPERATORS'] == backend
    monkeypatch.setenv('WARPSPHCORE_BACKEND', 'warp')
    monkeypatch.setenv('WARPSPHCORE_OPERATORS', 'native')
    assert runpy.run_path(str(source))['OPERATORS'] == 'native'
    monkeypatch.setenv('WARPSPHCORE_BACKEND', 'native')
    monkeypatch.setenv('WARPSPHCORE_OPERATORS', 'warp')
    with pytest.raises(ValueError, match='native-only'):
        runpy.run_path(str(source))
    monkeypatch.setenv('WARPSPHCORE_OPERATORS', 'automatic')
    with pytest.raises(ValueError, match='OPERATORS'):
        runpy.run_path(str(source))
