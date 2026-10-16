"""Solve two disconnected path Laplacians with explicit compatibility/gauges.

Use --project-incompatible to change the requested source explicitly and show
why successful convergence then certifies only the selected source.
"""

import argparse
import json

import torch

from warpSPHCore.linalg import DiagonalPreconditioner, LinearOperator, NullspaceSystem, cg


def solve(*, device='cpu', dtype=torch.float64, projectIncompatible=False):
    sizes = (8, 9)
    n = sum(sizes)
    Q = torch.zeros((n, len(sizes)), device=device, dtype=dtype)
    degree = torch.full((n,), 2., device=device, dtype=dtype)
    start = 0
    for j, size in enumerate(sizes):
        Q[start:start + size, j] = size**-.5
        degree[start], degree[start + size - 1] = 1., 1.
        start += size

    def laplacian(v):
        parts, start = [], 0
        for size in sizes:
            u = v[start:start + size]
            d = u[1:] - u[:-1]
            parts.append(torch.cat((-d[:1], d[:-1] - d[1:], d[-1:])))
            start += size
        return torch.cat(parts)

    action = LinearOperator((n, n), dtype, Q.device, laplacian, adjoint=laplacian)
    system = NullspaceSystem(action, Q, Q)
    exact = torch.arange(n, device=device, dtype=dtype).sin()
    b = action(exact)
    if projectIncompatible:
        b = b + Q @ Q.new_tensor([2., -1.])
        try:
            system.prepareRhs(b)  # Default policy rejects the incompatible source.
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible source was accepted')
    prepared = system.prepareRhs(b, policy='project' if projectIncompatible else 'error')
    inverse = DiagonalPreconditioner(degree, requireSPD=True)
    precond = system.projectPreconditioner(inverse.asLinearOperator())
    rtol = 4e-5 if dtype == torch.float32 else 1e-11
    result = cg(system, prepared.rhs, precond=precond, rtol=rtol, maxiter=4 * n)
    if result.statusInt() < 0:
        raise RuntimeError(f'singular solve failed with status {result.statusInt()}')
    x = system.gauge(result.x)
    reference = torch.cat((exact[:sizes[0]] - exact[:sizes[0]].mean(),
                           exact[sizes[0]:] - exact[sizes[0]:].mean()))
    accuracy = 1e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(x, reference, rtol=accuracy, atol=accuracy)
    report = system.verify(x, prepared, rtol=rtol * 2)
    if not bool(report.solvedConverged & report.projectedConverged):
        raise AssertionError('selected source failed original-action verification')
    if bool(report.requestedConverged) == projectIncompatible:
        raise AssertionError('requested-source certificate is incorrect')
    # Weighted gauges change only the final solution, not the Krylov projector.
    weights = Q.abs() * torch.arange(1., n + 1, device=device, dtype=dtype)[:, None]
    means = Q.new_tensor([1., -2.])
    weighted = system.weightedGauge(x, weights, means)
    torch.testing.assert_close((weights / weights.sum(dim=0)).T @ weighted, means,
                               rtol=accuracy, atol=accuracy)
    if not bool(system.verify(weighted, prepared, rtol=rtol * 3).solvedConverged):
        raise AssertionError('weighted gauge changed the selected physical equation')
    return dict(device=str(Q.device), dtype=str(dtype), nodes=n, modes=system.modes,
                status=result.statusInt(), iterations=int(result.iterations),
                requested_compatible=prepared.compatible,
                removed_norm=float(report.removedNorm),
                requested_residual=float(report.requestedResidual),
                solved_residual=float(report.solvedResidual),
                projected_residual=float(report.projectedResidual),
                requested_converged=bool(report.requestedConverged),
                solved_converged=bool(report.solvedConverged),
                max_error=float((x - reference).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--project-incompatible', action='store_true')
    args = parser.parse_args()
    print(json.dumps(solve(device=args.device, dtype=getattr(torch, args.dtype),
                           projectIncompatible=args.project_incompatible), indent=2))


if __name__ == '__main__':
    main()
