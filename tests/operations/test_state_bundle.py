"""Step F: StateBundle (warpier_fields.md Step F). Two concerns:

1. The no-grad path actually reuses a bundle (the entire point of the
   step).
2. The correctness property that makes gating necessary in the first
   place: `wp.Tape` holds a live reference to whatever struct object a
   launch was given and re-reads its fields lazily at `tape.backward()`
   time -- it does not snapshot values at launch time (verified directly
   against warp 1.16.0; see `stateBundle.py`'s module docstring for the
   minimal repro). Sharing a mutable struct across grad-requiring calls
   would silently corrupt an earlier call's gradient the moment its
   backward is deferred past a later call that refreshes the same bundle --
   completely ordinary PyTorch usage (build a graph across several ops,
   call `.backward()` once), not an edge case. Test 2 below pins exactly
   that scenario: two independent grad-requiring forward calls, both
   sharing the same dim (so they *would* share a bundle if the no-grad-only
   gate in `arg_extract.py`'s `build_fn` / `StateAwareWarpFunction.forward`
   were ever weakened), with neither backward run until both forwards have
   completed.
"""

import pytest
import torch

from warpSPHCore import DomainDescription, OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
from warpSPHCore.util.stateBundle import clearStateBundleCache, getStateBundle
from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme, WarpOperation


def _particles(support: float, n: int = 5, requires_grad: bool = False):
    device = torch.device("cpu")
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(-1).contiguous()
    positions = x.clone().requires_grad_(requires_grad)
    supports = torch.full((n,), support, requires_grad=requires_grad)
    masses = torch.full((n,), 1.0, requires_grad=requires_grad)
    kinds = torch.zeros(n, dtype=torch.int32, device=device)
    domain = DomainDescription(
        min=torch.tensor([-10.0]), max=torch.tensor([10.0]), periodic=torch.tensor([False]), dim=1
    )
    particles = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
    return particles, domain, adjacency


def _density_props():
    return OperationProperties(
        kernel=KernelFunctions.Wendland2,
        operation=WarpOperation.Density,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )


def test_no_grad_path_reuses_the_bundle():
    clearStateBundleCache()
    particles, domain, adjacency = _particles(0.0)

    dim = particles.positions.shape[1]
    warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    bundle1 = getStateBundle(dim)
    warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    bundle2 = getStateBundle(dim)

    assert bundle1 is bundle2, "the same (dim,) key should return the same persistent StateBundle"
    assert bundle1._last_wa is not None


def test_deferred_backward_across_two_grad_calls_not_corrupted():
    """The scenario Step F's no-grad-only gate exists to prevent: two
    independent grad-requiring forward calls sharing dim=1, neither
    backward run until both forwards are done. Correctness reference is a
    fully sequential ordering (forward+backward for A, then for B, on fresh
    tensors) -- an ordering that cannot suffer this class of bug since each
    call's backward completes before the next call's forward starts."""
    clearStateBundleCache()

    pA, dA, aA = _particles(0.6, requires_grad=True)
    pB, dB, aB = _particles(0.9, requires_grad=True)

    outA = warpOperation(pA, _density_props(), dA, adjacency=aA)
    outB = warpOperation(pB, _density_props(), dB, adjacency=aB)  # A's backward not yet run

    outA.sum().backward()
    outB.sum().backward()

    grad_A_deferred = pA.positions.grad.clone()
    grad_B_deferred = pB.positions.grad.clone()

    clearStateBundleCache()
    pA2, dA2, aA2 = _particles(0.6, requires_grad=True)
    outA2 = warpOperation(pA2, _density_props(), dA2, adjacency=aA2)
    outA2.sum().backward()

    pB2, dB2, aB2 = _particles(0.9, requires_grad=True)
    outB2 = warpOperation(pB2, _density_props(), dB2, adjacency=aB2)
    outB2.sum().backward()

    assert torch.allclose(grad_A_deferred, pA2.positions.grad), (
        "deferred-backward gradient for A must match the sequential reference -- "
        "a mismatch here means a later call's struct refresh corrupted an earlier, "
        "still-pending tape"
    )
    assert torch.allclose(grad_B_deferred, pB2.positions.grad)
    # Sanity: A and B use different support radii, so their gradients should
    # differ from each other (a false pass where both silently ended up
    # identical -- e.g. a rigid translation, which this operator is
    # invariant to -- would also hide this bug).
    assert not torch.allclose(grad_A_deferred, grad_B_deferred)


def test_forward_mode_bundle_matches_reverse_regardless_of_cache_warmth():
    """Step G, audit item 1 originally required FORWARD to be rejected with a
    clear error at *every* entry, not only on a cold cache -- because
    forward mode had no implementation at all yet. `warpier_forward_mode_plan.md`
    Phase 2 changes that premise: Tier 1 forward mode needs no struct shape
    different from REVERSE's, so `getStateBundle(dim, FORWARD)` now hands back
    the same (dim-keyed, mode-independent) bundle REVERSE uses, on both a cold
    and a warm cache -- consistent behavior either way, not a regression.
    """
    from warpSPHCore.dataTypes import ExecutionMode

    clearStateBundleCache()
    cold = getStateBundle(2, ExecutionMode.FORWARD)  # cold cache
    assert cold is not None

    warm = getStateBundle(2, ExecutionMode.REVERSE)
    assert warm is getStateBundle(2, ExecutionMode.REVERSE)
    assert warm is getStateBundle(2, ExecutionMode.FORWARD)  # warm cache -- consistent with cold
    clearStateBundleCache()
