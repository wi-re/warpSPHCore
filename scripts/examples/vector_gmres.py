"""Solve a nonsymmetric two-component periodic transport/reaction problem.

The local 2x2 blocks couple components; unequal left/right coefficients make
the spatial action nonsymmetric. Block Jacobi supplies an inverse action.
"""

import argparse
import json

import torch

from warpSPHCore.linalg import BlockJacobiPreconditioner, LinearOperator, gmres


def solve(*, device='cpu', dtype=torch.float64, side='right'):
    n = 16
    x = torch.arange(n, device=device, dtype=dtype) / n
    blocks = x.new_tensor([[3., .4], [-.2, 2.]])[None].repeat(n, 1, 1)
    blocks[:, 0, 0] += x

    def action(flat):
        v = flat.reshape(n, 2)
        return (torch.einsum('nij,nj->ni', blocks, v)
                - .6*v.roll(1, 0) - .2*v.roll(-1, 0)).reshape(-1)

    op = LinearOperator((2*n, 2*n), dtype, x.device, action)
    inverse = BlockJacobiPreconditioner(blocks)  # General nonsymmetric blocks.
    exact = torch.stack(((2*torch.pi*x).sin(), (2*torch.pi*x).cos()), -1).reshape(-1)
    rhs = op(exact)
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    result = gmres(op, rhs, precond=inverse, preconditionerSide=side,
                   restart=8, maxiter=128, checkEvery=1, rtol=tol)
    residual = (rhs-op(result.x)).norm()
    if result.statusInt() < 0 or residual > tol*rhs.norm():
        raise AssertionError('GMRES failed original-action verification')
    torch.testing.assert_close(result.x, exact, rtol=tol*5, atol=tol*5)
    return dict(device=str(x.device), dtype=str(dtype), components=2, nodes=n,
                preconditioner_side=side, status=result.statusInt(),
                iterations=int(result.iterations), original_residual=float(residual),
                max_error=float((result.x-exact).abs().max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--side', choices=['left', 'right'], default='right')
    args = parser.parse_args()
    print(json.dumps(solve(device=args.device, dtype=getattr(torch, args.dtype),
                           side=args.side), indent=2))
