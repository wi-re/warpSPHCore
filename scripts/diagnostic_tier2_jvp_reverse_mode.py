"""Step-0 diagnostic for `warpier_tier2_operators_plan.md` (Lookout 2): does
`computeSPHDensityGeometryJVP`'s output carry a gradient back to its own
inputs under ordinary reverse-mode torch autograd?

The suspicion (reasoned from code, not yet measured): the pair-indexed
kernel launch inside `wp_densityJVP.py` is a bare `wp.launch`, deliberately
bypassing `OperatorSpec`/`launchOperator` -- and `launchOperator`'s
reverse-mode support comes precisely from wrapping launches in a
`torch.autograd.Function` that records a `wp.Tape`. A bare `wp.launch`
outside that wrapper registers no autograd node, so the warp-kernel-derived
part of the output (`W_t`/`dW_t`, and therefore anything routed through
them) should carry **no** gradient back to `positions`/`tangentQueryPositions`
-- only the plain-torch mass-tangent indexing path should.

Run directly: `python scripts/diagnostic_tier2_jvp_reverse_mode.py`.
"""

import torch
import warp as wp

wp.init()

from warpSPHCore import AdjacencyList, DomainDescription, ParticleState, ParticleTangentState, radiusSearchCompactHashMap
from warpSPHCore.enumTypes import SupportScheme, KernelFunctions
from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP

DEVICE = torch.device("cpu")
DTYPE = torch.float32
KERNEL = KernelFunctions.Wendland2


def main():
    n = 7
    positions = torch.linspace(-1.0, 1.0, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    supports = torch.full((n,), 0.5, dtype=DTYPE, device=DEVICE) * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)

    domain = DomainDescription(
        min=torch.tensor([-10.0], dtype=DTYPE, device=DEVICE),
        max=torch.tensor([10.0], dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False], device=DEVICE),
        dim=1,
    )

    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)

    positions = positions.clone().requires_grad_(True)
    tangentQueryPositions = torch.randn(n, 1, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    tangentReferenceMasses = torch.randn(n, dtype=DTYPE, device=DEVICE).requires_grad_(True)

    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)

    dDensity = computeSPHDensityGeometryJVP(
        p, domain, KERNEL, SupportScheme.Gather, adjacency,
        queryTangentState=ParticleTangentState(positions=tangentQueryPositions, supports=None, masses=None),
        referenceTangentState=ParticleTangentState(positions=None, supports=None, masses=tangentReferenceMasses),
    )

    # allow_unused=True is the unambiguous check: it returns None per-tensor
    # for anything the autograd graph never actually reaches, rather than
    # relying on .grad's more ambiguous None-vs-never-visited semantics.
    gPos, gTanPos, gTanMass = torch.autograd.grad(
        dDensity.sum(), [positions, tangentQueryPositions, tangentReferenceMasses], allow_unused=True,
    )
    print("positions grad reaches autograd graph:", gPos is not None)
    print("tangentQueryPositions grad reaches autograd graph:", gTanPos is not None)
    print("tangentReferenceMasses grad reaches autograd graph:", gTanMass is not None,
          "(magnitude" if gTanMass is not None else "", "" if gTanMass is None else f"{gTanMass.abs().sum().item():.4g})")


if __name__ == "__main__":
    main()
