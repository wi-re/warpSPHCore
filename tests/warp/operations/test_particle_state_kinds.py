import pytest
import torch

from warpSPHCore import ParticleState


def test_particle_state_requires_kinds():
    """kinds is a required ParticleState field (warpier_fields.md Step A) --
    omitting it must raise a clear error at construction time rather than
    silently defaulting to None and letting AllToAll read out of bounds
    later (the previously-open item this closes)."""
    positions = torch.zeros((4, 2))
    supports = torch.ones(4)
    masses = torch.ones(4)

    with pytest.raises(TypeError):
        ParticleState(positions=positions, supports=supports, masses=masses)


def test_particle_state_accepts_kinds():
    positions = torch.zeros((4, 2))
    supports = torch.ones(4)
    masses = torch.ones(4)
    kinds = torch.zeros(4, dtype=torch.int32)

    state = ParticleState(positions=positions, supports=supports, masses=masses, kinds=kinds)
    assert state.kinds is kinds
