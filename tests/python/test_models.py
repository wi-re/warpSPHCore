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
    assert 'warp' not in sys.modules
