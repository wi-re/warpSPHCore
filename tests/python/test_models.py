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
