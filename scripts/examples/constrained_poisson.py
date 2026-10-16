"""Solve -u'' = 2 on [0, 1], u(0) = 1, u(1) = 3, with Dirichlet lifting.

Run from an installed checkout, for example:
    python scripts/examples/constrained_poisson.py --device cpu --dtype float64
"""

import argparse
import json

import torch

from warpSPHCore.linalg import DiagonalPreconditioner, LinearOperator, RestrictedSystem, cg


def solve(*, nodes=33, device='cpu', dtype=torch.float64):
    if nodes < 3:
        raise ValueError('nodes must be at least 3')
    x = torch.linspace(0., 1., nodes, device=device, dtype=dtype)
    h = 1 / (nodes - 1)

    def laplacian(v):
        # Symmetric full tridiagonal action. Boundary rows carry reactions;
        # only the interior equations are imposed by the reduced system.
        left = torch.cat((v.new_zeros(1), v[:-1]))
        right = torch.cat((v[1:], v.new_zeros(1)))
        return (2 * v - left - right) / h**2

    A = LinearOperator((nodes, nodes), dtype, x.device, laplacian, adjoint=laplacian)
    free = torch.arange(1, nodes - 1, device=device)
    system = RestrictedSystem(A, free)
    prescribed = torch.zeros_like(x)
    prescribed[0], prescribed[-1] = 1., 3.
    b = torch.full_like(x, 2.)
    b[0], b[-1] = 0., 0.
    rhs = system.liftRhs(b, prescribed)
    diagonal = torch.full_like(rhs, 2 / h**2)
    precond = DiagonalPreconditioner(diagonal, requireSPD=True)
    rtol = 2e-5 if dtype == torch.float32 else 1e-11
    result = cg(system, rhs, precond=precond, rtol=rtol, maxiter=4 * nodes)
    if result.statusInt() < 0:
        raise RuntimeError(f'constrained solve failed with status {result.statusInt()}')
    solution = system.reconstruct(result.x, prescribed)
    freeResidual = system.restrict(b - A(solution)).norm()
    if freeResidual > 2 * rtol * rhs.norm():
        raise AssertionError('reconstructed free-row residual exceeds tolerance')
    # A quadratic is reproduced exactly by the centered second difference.
    exact = 1 + 2 * x + x * (1 - x)
    accuracy = 1e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(solution, exact, rtol=accuracy, atol=accuracy)
    torch.testing.assert_close(solution[[0, -1]], prescribed[[0, -1]], rtol=0, atol=0)
    return dict(device=str(x.device), dtype=str(dtype), nodes=nodes,
                status=result.statusInt(), iterations=int(result.iterations),
                reduced_residual=float(result.trueResidual),
                reconstructed_free_residual=float(freeResidual),
                max_error=float((solution - exact).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=33)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    args = parser.parse_args()
    print(json.dumps(solve(nodes=args.nodes, device=args.device,
                           dtype=getattr(torch, args.dtype)), indent=2))


if __name__ == '__main__':
    main()
