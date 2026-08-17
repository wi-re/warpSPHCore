"""`computeRenormalizationMatrices` must not write its own internal choice of
operation into the caller's `OperationProperties`.

Covariance is renorm's internal step, not a caller-selected parameter, so it is
applied to a copy (see the comment at that call site in `renorm.py`). Before
that fix the function did `operationProperties.operation =
WarpOperation.Covariance` on the object the caller still owns, so a caller that
reused its properties -- compute L, then run the Gradient the correction is for
-- silently launched Covariance the second time and got a plausible-looking
(N, D, D) tensor back instead of the gradient it asked for.

No call site in either repo hit this, because they all construct a fresh
`OperationProperties` inline for the renorm call. That is exactly why it needs a
test rather than just a fix: warpier_fields.md Section 3.5 wants to hoist those
constructions out of the hot path so a reusable, hashable properties object can
key the StateBundle, which would have *introduced* the bug instead of finding
it. Found by scripts/spike_forward_mode_tier1.py (Step G), which reuses one
properties object across both calls.
"""

import torch

from warpSPHCore import OperationProperties, warpOperation
from warpSPHCore.enumTypes import (
    GradientScheme,
    KernelFunctions,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from warpSPHCore.renorm import computeRenormalizationMatrices


def _properties():
    return OperationProperties(
        kernel=KernelFunctions.Wendland2,
        operation=WarpOperation.Gradient,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Difference,
    )


def test_caller_properties_object_is_not_mutated(particle_case):
    properties = _properties()
    before = repr(properties)

    computeRenormalizationMatrices(
        particle_case["particles"],
        properties,
        particle_case["domain"],
        adjacency=particle_case["adjacency"],
    )

    assert properties.operation is WarpOperation.Gradient
    assert repr(properties) == before


def test_reused_properties_still_runs_the_requested_operation(particle_case):
    """The consequence the mutation actually had: reuse the same properties
    object for the Gradient the correction was computed for, and check the
    result is a gradient (shape (N, D), value-dependent) rather than the
    (N, D, D) Covariance output the mutated properties would have produced."""
    particles = particle_case["particles"]
    domain = particle_case["domain"]
    adjacency = particle_case["adjacency"]
    n, dim = particles.positions.shape

    properties = _properties()
    _, _, renormState = computeRenormalizationMatrices(
        particles, properties, domain, adjacency=adjacency
    )

    values = torch.randn(n, dtype=particles.positions.dtype, device=particles.positions.device)
    out = warpOperation(
        particles,
        properties,  # deliberately the *same* object the renorm call just saw
        domain,
        queryValues=values,
        referenceValues=values,
        adjacency=adjacency,
        renormalizationState=renormState,
    )

    assert out.shape == (n, dim)

    # A Covariance launch ignores queryValues entirely, so a value-dependence
    # check catches the wrong-operation case even if a shape ever coincided.
    other = warpOperation(
        particles, properties, domain,
        queryValues=values * 2.0, referenceValues=values * 2.0,
        adjacency=adjacency, renormalizationState=renormState,
    )
    assert not torch.allclose(out, other)
