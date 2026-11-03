"""Warp-free core example: neighbors, density, corrected gradient and a solve.

Run with WARPSPHCORE_BACKEND=native before package import. No timing is collected.
"""
import argparse
import sys
import torch
import warpSPHCore as sph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sph.backend != 'native':
        raise RuntimeError('Set WARPSPHCORE_BACKEND=native before import')
    x = torch.tensor([[-.3, -.2], [0., 0.], [.3, .2]], dtype=torch.float64, device=args.device)
    p = sph.ParticleState(x, torch.ones_like(x[:, 0]), torch.ones_like(x[:, 0]),
                          torch.zeros(x.shape[0], dtype=torch.int32, device=x.device))
    domain = sph.DomainDescription(x.new_tensor([-1., -1.]), x.new_tensor([1., 1.]),
                                    torch.tensor([True, True], device=x.device), 2)
    # Keep a padded candidate list while geometry stays within its skin, then
    # filter the current physical neighborhood without rebuilding the index.
    candidates = sph.buildVerletList(p, domain, verletScale=1.4, storeI=False)
    assert sph.buildVerletList(p, domain, 1.4, priorNeighborhood=candidates, storeI=False) is candidates
    adjacency = sph.filterVerletList(p, domain, candidates, storeI=False)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density,
                                         calibrateNormalization=True, n_h=4.)
    p.densities = sph.warpOperation(p, properties, domain, adjacency=adjacency)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
                                         gradientMode=sph.GradientScheme.Difference)
    # Build the correction through the native covariance and dense inverse.
    # This tiny cloud deliberately exercises the low-neighbor identity rule.
    correction = sph.computeRenormalizationMatrices(
        p, properties, domain, adjacency=adjacency, returnEigVals=False)
    gradient = sph.warpOperation(p, properties, domain, torch.ones_like(p.masses), adjacency=adjacency,
                                 renormalizationState=correction)
    torch.testing.assert_close(gradient, torch.zeros_like(x), atol=0, rtol=0)
    assert torch.isfinite(p.densities).all() and (p.densities > 0).all()
    # Real solvers supply an SPH matrix action here. A diagonal system keeps
    # the example readable while demonstrating reusable native CPU/GPU scratch.
    diagonal = x.new_tensor([1., 2., 3.])
    rhs = torch.ones_like(diagonal)
    workspace = sph.CGWorkspace(rhs, maxiter=16, history=False)
    solution = sph.cg(lambda value: diagonal*value, rhs,
                      workspace=workspace, maxiter=16, rtol=1e-10)
    torch.testing.assert_close(diagonal*solution.x, rhs)
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)
    print(f'Native {args.device}: {len(x)} particles, {adjacency.numEdges} CSR entries; density, corrected gradient and CG passed; Warp not imported')


if __name__ == '__main__':
    main()
