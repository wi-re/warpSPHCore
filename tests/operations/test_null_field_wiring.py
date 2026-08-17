"""Step C: null fields replace dummy tensors in extractStateInfo's flat list
(warpier_fields.md Step C). Pins two things the gradcheck/operation-matrix
runs don't check directly: that a disabled correction path is filled with a
``Field`` from the null-field registry rather than a plain torch.Tensor
dummy, and that the exact same Field object comes back on a second call --
the whole point of the registry being "built once, never converted again".
"""

from warpSPHCore import OperationProperties
from warpSPHCore.autograd.arg_extract import extractStateInfo
from warpSPHCore.dataTypes import AdjacencyList, Field
from warpSPHCore.enumTypes import (
    KernelFunctions,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)

# Indices into extractStateInfo's flat list (see its docstring) that are
# null-registry slots when no renormalization/gradH/volumes/CRK state is
# supplied: renormMat, qOmega, rOmega, qVol, rVol, and the eight CRK entries.
_NULL_CORRECTION_INDICES = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def _extract(particle_case):
    particles = particle_case["particles"]
    return extractStateInfo(
        particles,
        OperationProperties(
            kernel=KernelFunctions.Wendland2,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        particle_case["domain"],
        adjacency=particle_case["adjacency"],
    )


def test_disabled_corrections_are_null_fields(particle_case):
    flat_tensors, _, _, _ = _extract(particle_case)
    for i in _NULL_CORRECTION_INDICES:
        assert isinstance(flat_tensors[i], Field), f"index {i} expected a null Field"


def test_null_field_slot_is_stable_across_calls(particle_case):
    flat_a, _, _, _ = _extract(particle_case)
    flat_b, _, _, _ = _extract(particle_case)
    for i in _NULL_CORRECTION_INDICES:
        assert flat_a[i] is flat_b[i], (
            f"index {i} should be the same registry Field on both calls, not rebuilt"
        )


def test_unused_traversal_side_is_a_null_field(particle_case):
    # particle_case's adjacency is an AdjacencyList (neighbor-list traversal),
    # so the grid side (26-32) is the disabled one and the neighbor-list side
    # (23-25) carries the real adjacency tensors.
    adjacency = particle_case["adjacency"]
    assert isinstance(adjacency, AdjacencyList)

    flat_tensors, _, _, _ = _extract(particle_case)
    for i in (23, 24, 25):
        assert not isinstance(flat_tensors[i], Field), f"index {i} expected real adjacency data"
    for i in (26, 29, 30, 31, 32):
        assert isinstance(flat_tensors[i], Field), f"index {i} expected a null Field"


def test_real_tensors_still_flow_through_unwrapped(particle_case):
    flat_tensors, _, _, _ = _extract(particle_case)
    particles = particle_case["particles"]
    assert flat_tensors[0] is particles.positions
    assert flat_tensors[6] is particles.densities
    assert not isinstance(flat_tensors[0], Field)
    assert not isinstance(flat_tensors[6], Field)
