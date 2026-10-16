#!/usr/bin/env python3
"""Check a built wheel in an isolated interpreter without installing it."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='device for the public examples (CUDA requires a real GPU)')
    args = parser.parse_args()
    examples = Path(__file__).resolve().parents[1] / 'examples' / 'run_all.py'
    with tempfile.TemporaryDirectory(prefix='warpsphcore-wheel-') as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(directory)
        code = '''
import importlib.metadata
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import torch
import warpSPHCore as core
from warpSPHCore.linalg import (BiCGStabWorkspace, CGWorkspace, LinearOperator, DiagonalPreconditioner,
                                BlockJacobiPreconditioner, RestrictedSystem, NullspaceSystem, bicgstab, cg, gmres, minres, richardson)
assert Path(core.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert core.__version__ == importlib.metadata.version('warpSPHCore')
b = torch.ones(3)
result = cg(lambda v: 2 * v, b, rtol=0, atol=0)
assert result.statusInt() >= 0 and result.trueResidual == 0
torch.testing.assert_close(result.x, b / 2)
operator = LinearOperator((3, 3), b.dtype, b.device, lambda v: 2 * v,
                          adjoint=lambda v: 2 * v)
general = gmres(operator, b, rtol=1e-6)
assert general.statusInt() >= 0 and general.trueResidual <= 1e-6 * b.norm()
torch.testing.assert_close(general.x, b / 2)
torch.testing.assert_close(operator.applyAdjoint(b), 2 * b)
restricted = RestrictedSystem(operator, torch.tensor([1]))
prescribed = torch.tensor([1., 0., 3.])
rhs = restricted.liftRhs(b, prescribed)
constrained = cg(restricted, rhs, rtol=0)
assert constrained.statusInt() >= 0
torch.testing.assert_close(restricted.reconstruct(constrained.x, prescribed), torch.tensor([1., .5, 3.]))
singular_A = torch.diag(torch.tensor([2., 0.]))
singular_op = LinearOperator((2, 2), b.dtype, b.device, lambda v: singular_A @ v,
                             adjoint=lambda v: singular_A.T @ v)
basis = torch.tensor([[0.], [1.]])
singular_system = NullspaceSystem(singular_op, basis, basis)
selected = singular_system.prepareRhs(torch.ones(2), policy='project')
singular_result = cg(singular_system, selected.rhs, rtol=0)
certificate = singular_system.verify(singular_system.gauge(singular_result.x), selected, rtol=0)
assert singular_result.statusInt() >= 0 and certificate.solvedConverged
assert not certificate.requestedConverged and certificate.removedNorm == 1
indefinite_A = torch.diag(torch.tensor([-2., 3.]))
indefinite_rhs = torch.tensor([1., -1.])
minimum_residual = minres(lambda v: indefinite_A @ v, indefinite_rhs, symmetric=True, rtol=1e-5)
assert minimum_residual.statusInt() >= 0
torch.testing.assert_close(minimum_residual.x, torch.linalg.solve(indefinite_A, indefinite_rhs))
fallback = richardson(lambda v: 2*v, b, omega=.5, rtol=0)
assert fallback.statusInt() >= 0 and fallback.trueResidual == 0
torch.testing.assert_close(fallback.x, b/2)
diagonal = DiagonalPreconditioner(torch.full_like(b, 2), requireSPD=True)
torch.testing.assert_close(diagonal(b), b / 2)
block = BlockJacobiPreconditioner((2 * torch.eye(3))[None], requireSPD=True)
torch.testing.assert_close(block(b), b / 2)
block.update((4 * torch.eye(3))[None])
torch.testing.assert_close(block.applyAdjoint(b), b / 4)
stabilized = BiCGStabWorkspace(b, 8, history=False)
stable = bicgstab(lambda v: 2*v, b, workspace=stabilized, maxiter=8, rtol=0)
assert stable.statusInt() >= 0 and stable.trueResidual == 0
bicgstab(lambda v: 2*v, b*2, workspace=stabilized, maxiter=8, rtol=0)
torch.testing.assert_close(stable.x, b/2)
workspace = CGWorkspace(b, 8, history=False)
native = cg(lambda v: 2 * v, b, workspace=workspace, maxiter=8, rtol=0)
assert native.statusInt() >= 0 and native.trueResidual == 0
assert native.norms.numel() == 0
cg(lambda v: 2 * v, b * 2, workspace=workspace, maxiter=8, rtol=0)
torch.testing.assert_close(native.x, b / 2)
import warp as wp
from warpSPHCore.polyfit import PolyFit, Reconstructor
wp.init()
x = torch.linspace(-.3, .3, 5, dtype=core.get_torch_precision())
particles = core.ParticleState(positions=x[:, None], supports=torch.ones_like(x),
    masses=torch.ones_like(x), densities=torch.ones_like(x), kinds=torch.zeros(5, dtype=torch.int32))
domain = core.DomainDescription(x.new_tensor([-2.]), x.new_tensor([2.]), torch.tensor([False]), 1)
field = 1+x.square()
fit = PolyFit.build(particles, domain, core.KernelFunctions.Wendland2, 2)
assert not fit.deficient.any()
torch.testing.assert_close(fit.gradient(field)[:, 0], 2*x, rtol=1e-3, atol=1e-3)
rec = Reconstructor.weno(particles, domain, core.KernelFunctions.Wendland2, cond_max=1.)
i, j, delta = rec.pairs()
left, right = rec.interfaceStates(field, i, j, delta)
assert rec.last_fallback.all() and torch.equal(left, field[i]) and torch.equal(right, field[j])
from warpSPHCore.mfm import MeshlessGeometry, MeshlessWarp, mfmRates
mesh = MeshlessGeometry.build(particles, domain, core.KernelFunctions.Wendland2, closure='none')
mesh_warp = MeshlessWarp.build(particles.positions, particles.supports, domain,
                              core.KernelFunctions.Wendland2, closure='none')
rho, velocity, pressure = torch.ones_like(x), x[:, None] * .1, 1 + .1*x
reference_rate, _ = mfmRates(mesh, rho, velocity, pressure, 1.4, mode='MFV', dt=.001)
warp_rate = mesh_warp.rates(rho, velocity, pressure, 1.4, mode='MFV', dt=.001)
torch.testing.assert_close(warp_rate, reference_rate, rtol=1e-3, atol=1e-3)
assert torch.isfinite(warp_rate).all() and warp_rate.sum(0).abs().max() < 1e-4
assert mesh_warp.rates(rho, velocity, pressure, 1.4, mode='MFM')[:, 0].count_nonzero() == 0
if torch.cuda.is_available():
    source = b.cuda()
    graph_workspace = CGWorkspace(source, 8, capture=True)
    action = lambda v: 2 * v
    captured = cg(action, source, workspace=graph_workspace, maxiter=8, rtol=0)
    assert captured.statusInt() >= 0 and captured.trueResidual == 0
    builds = graph_workspace.captureBuilds
    cg(action, source * 2, workspace=graph_workspace, maxiter=8, rtol=0)
    assert graph_workspace.captureBuilds == builds
    torch.testing.assert_close(captured.x, source / 2)
    from warpSPHCore.util import generateNeighborTestData
    positions, supports, _, domain, dx = generateNeighborTestData(8, 25, 2, True, source.device)
    count = positions.shape[0]
    values = positions.sin().contiguous()
    particles = core.ParticleState(positions=positions, supports=supports,
        masses=torch.full_like(supports, dx**2),
        densities=torch.ones_like(supports),
        kinds=torch.zeros(count, dtype=torch.int32, device=source.device))
    adjacency = core.radiusSearchCompactHashMap(particles, domain)
    try:
        for operation in (core.WarpOperation.Curl, core.WarpOperation.Laplacian):
            field = values if operation == core.WarpOperation.Curl else values[:, 0].contiguous()
            props = core.OperationProperties(operation=operation, kernel=core.KernelFunctions.Wendland2,
                gradientMode=core.GradientScheme.Difference,
                laplacianMode=core.LaplacianScheme.Brookshaw)
            core.setNeighborLanes(1)
            serial = core.warpOperation(particles, props, domain, adjacency=adjacency, queryValues=field)
            core.setNeighborLanes(32)
            tiled = core.warpOperation(particles, props, domain, adjacency=adjacency, queryValues=field)
            torch.testing.assert_close(tiled, serial, rtol=3e-5, atol=3e-5)
    finally:
        core.setNeighborLanes(None)
print('Isolated wheel import/solvers/reconstruction/MFM and available CUDA tiled checks passed:', core.__version__)
'''
        subprocess.run([sys.executable, '-I', '-c', code, directory], cwd=directory, check=True)
        # -I ignores PYTHONPATH and editable-checkout entries. Only the unpacked
        # wheel is prepended; example files themselves remain external clients.
        example_code = '''
from pathlib import Path
import runpy
import sys
sys.path.insert(0, sys.argv[1])
import warpSPHCore as core
assert Path(core.__file__).resolve().is_relative_to(Path(sys.argv[1]))
runpy.run_path(sys.argv[2])['run'](sys.argv[3], sys.argv[4])
'''
        for precision in ('float32', 'float64'):
            env = dict(os.environ, warpSPHCore_PRECISION=precision)
            subprocess.run([sys.executable, '-I', '-c', example_code, directory,
                            str(examples), args.device, precision], cwd=directory, env=env, check=True)


if __name__ == '__main__':
    main()
