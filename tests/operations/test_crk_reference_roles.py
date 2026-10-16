"""CRK reference volumes follow reference IDs, independently of query IDs."""

from dataclasses import replace

import pytest
import torch
import warp as wp

from warpSPHCore import (
    DomainDescription, KernelFunctions, ParticleState, ParticleTangentState,
    radiusSearchCompactHashMap, SupportScheme,
)
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.type_config import scalar_t


@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('reference_count', [5, 9])
def test_distinct_reference_permutation_preserves_corrections_and_jvp(device, traversal, reference_count):
    dtype = wp.dtype_to_torch(scalar_t)
    xy = torch.tensor([[-.4, -.3], [0., -.4], [.35, -.25], [-.3, .2],
                       [.1, .1], [.4, .35], [-.05, .45], [.2, -.1], [-.2, .05]], dtype=dtype, device=device)
    h = torch.linspace(1.2, 1.7, 9, dtype=dtype, device=device)
    kinds = torch.zeros(9, dtype=torch.int32, device=device)
    query = ParticleState(positions=xy[:7], supports=h[:7], masses=torch.ones_like(h[:7]),
                          kinds=kinds[:7], densities=torch.ones_like(h[:7]))
    reference = replace(query, positions=xy[:reference_count] * .95 + .03,
                        supports=h[:reference_count].flip(0), masses=h[:reference_count],
                        densities=torch.ones_like(h[:reference_count]), kinds=kinds[:reference_count])
    domain = DomainDescription(min=xy.new_full((2,), -2), max=xy.new_full((2,), 2),
                               periodic=torch.zeros(2, dtype=torch.bool, device=device), dim=2)
    tq = ParticleTangentState(positions=query.positions.sin() * .1, supports=query.supports * .02, masses=None)
    tr = ParticleTangentState(positions=reference.positions.cos() * .1,
                              supports=reference.supports * -.03, masses=None)
    permutation = torch.arange(reference_count, device=device).roll(2).flip(0)

    def run(ref, tangent):
        adjacency = radiusSearchCompactHashMap(query, domain, referenceParticles=ref,
                                              mode=SupportScheme.SuperSymmetric)
        adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
        primal = computeCRKFactors(query, domain, KernelFunctions.Wendland2,
                                   adjacency=adj, referenceState=ref)
        jvp = computeCRKFactorsJVP(query, domain, KernelFunctions.Wendland2, tq,
                                   adjacency=adj, referenceState=ref, referenceTangentState=tangent)
        return (*primal[:2], *vars(primal[2]).values(), *jvp[:2],
                *vars(jvp[2]).values(), *vars(jvp[3]).values())

    ref_permuted = replace(reference, **{name: value[permutation] for name, value in vars(reference).items()})
    tr_permuted = replace(tr, positions=tr.positions[permutation], supports=tr.supports[permutation])
    expected = run(reference, tr)
    actual = run(ref_permuted, tr_permuted)
    tol = 3e-4 if dtype == torch.float32 else 2e-10
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)

    # Both roles, through volume estimation, moment construction and CRK terms.
    # This also qualifies omitted-adjacency correction JVP construction.
    inputs = tuple(v.detach().requires_grad_() for v in
                   (query.positions, query.supports, reference.positions, reference.supports))
    tangents = (tq.positions, tq.supports, tr.positions, tr.supports)

    def chain(qp, qh, rp, rh):
        volume, _, state = computeCRKFactors(
            replace(query, positions=qp, supports=qh), domain, KernelFunctions.Wendland2,
            referenceState=replace(reference, positions=rp, supports=rh))
        return torch.cat([v.reshape(-1) for v in (volume, *vars(state).values())])

    assembled = torch.cat([v.reshape(-1) for v in (expected[7], *expected[12:])])
    output = chain(*inputs)
    cotangent = torch.arange(output.numel(), dtype=dtype, device=device).cos()
    vjp = torch.autograd.grad((output * cotangent).sum(), inputs)
    torch.testing.assert_close((assembled * cotangent).sum(),
                               sum((a * b).sum() for a, b in zip(vjp, tangents)), rtol=tol, atol=tol)
    # FP32 moment contractions lose several digits under cancellation; use a
    # displacement resolvable at that precision rather than a float64 step.
    eps = .02 if dtype == torch.float32 else 1e-5
    plus = chain(*(v + eps * dv for v, dv in zip(inputs, tangents)))
    minus = chain(*(v - eps * dv for v, dv in zip(inputs, tangents)))
    torch.testing.assert_close(assembled, (plus - minus) / (2 * eps),
                               rtol=3e-3 if dtype == torch.float32 else 2e-7,
                               atol=1e-4 if dtype == torch.float32 else 2e-8)
