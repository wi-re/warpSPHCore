"""Differentiate a complete native CRK preparation and corrected operator.

Run with WARPSPHCORE_BACKEND=native and --device cpu or cuda. A fixed CSR covers
small perturbations used by the numerical check. All pair passes and derivatives
run in native code; the small CRK factor solve and loss algebra run in Torch.
"""
import argparse
from dataclasses import replace
import json
import sys
import torch
import warpSPHCore as sph
from warpSPHCore.native import CSRTopology


def run(device):
    dtype = torch.float64
    axis = torch.tensor([-.4, 0., .4], device=device, dtype=dtype)
    reference_positions = torch.cartesian_prod(axis, axis)
    query_positions = axis.new_tensor([[-.11, .07], [.13, -.06]])

    def particles(positions):
        return sph.ParticleState(positions=positions,
            supports=positions.new_full((len(positions),), 1.3),
            masses=positions.new_full((len(positions),), .2),
            kinds=torch.zeros(len(positions), device=device, dtype=torch.int32))

    query, reference = particles(query_positions), particles(reference_positions)
    domain = sph.DomainDescription(axis.new_full((2,), -1), axis.new_ones(2),
                                    torch.zeros(2, device=device, dtype=torch.bool), 2)
    graph = CSRTopology(sph.radiusSearch(query, domain, reference, sph.SupportScheme.SuperSymmetric))
    reference_graph = CSRTopology(sph.radiusSearch(reference, domain))
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Gradient, supportMode=sph.SupportScheme.Scatter,
        operationMode=sph.OperationDirection.TrueAllToToAll)

    def loss(qx, rx, qh, rh, mass):
        left = replace(query, positions=qx, supports=qh)
        right = replace(reference, positions=rx, supports=rh, masses=mass)
        preparation = sph.prepareCRK(left, domain, properties.kernel,
            properties.operationMode, graph, right, referenceAdjacency=reference_graph)
        # A nonlinear field makes both the geometry and prepared correction
        # derivatives matter. The consistency diagnostic also trains masses.
        fi = left.positions.square().sum(1)
        fj = right.positions.square().sum(1)
        gradient = sph.warpOperation(left, properties, domain, fi, fj,
            referenceParticles=right, adjacency=graph,
            referenceVolumes=preparation.referenceVolumes, crkState=preparation.crkState)
        return .5 * gradient.square().sum() + .01 * preparation.consistencyDensity.square().sum()

    values = [x.detach().clone().requires_grad_() for x in (query.positions, reference.positions,
              query.supports, reference.supports, reference.masses)]
    objective = loss(*values)
    derivatives = torch.autograd.grad(objective, values)
    directions = [(x.detach() * 1.3 + .2).sin().contiguous() for x in values]
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(x.detach(), t) for x, t in zip(values, directions)]
        jvp = torch.autograd.forward_ad.unpack_dual(loss(*duals)).tangent
    pullback_dot = sum((gradient * direction).sum() for gradient, direction in zip(derivatives, directions))
    with torch.no_grad():
        eps = 2e-6
        plus = loss(*[x + eps*t for x, t in zip(values, directions)])
        minus = loss(*[x - eps*t for x, t in zip(values, directions)])
        finite_difference = (plus - minus) / (2 * eps)
    torch.testing.assert_close(jvp, pullback_dot, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(jvp, finite_difference, rtol=2e-5, atol=1e-7)
    assert all(torch.isfinite(gradient).all() for gradient in derivatives)
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)
    return dict(device=device, loss=float(objective.detach()), directional_derivative=float(jvp),
        transpose_identity_error=float((jvp - pullback_dot).abs()),
        finite_difference_error=float((jvp - finite_difference).abs()),
        query_csc_cached=graph.transpose() is graph.transpose(),
        reference_csc_cached=reference_graph.transpose() is reference_graph.transpose())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sph.backend != 'native':
        raise RuntimeError('Select WARPSPHCORE_BACKEND=native before importing')
    print(json.dumps(run(args.device), indent=2))


if __name__ == '__main__':
    main()
