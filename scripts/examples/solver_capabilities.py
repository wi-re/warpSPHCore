"""MINRES on a singular saddle-point system and Richardson as a warm fallback."""

import argparse
import json

import torch

from warpSPHCore.linalg import (
    DiagonalPreconditioner, LinearOperator, NullspaceSystem, minres, richardson,
)


def saddlePoint(*, device='cpu', dtype=torch.float64):
    n = 8
    eye = torch.eye(n, device=device, dtype=dtype)
    divergence = eye - eye.roll(1, dims=1)
    h = torch.arange(2., 3., 1/n, device=device, dtype=dtype)
    A = torch.cat((torch.cat((torch.diag(h), divergence.T), dim=1),
                   torch.cat((divergence, torch.zeros_like(eye)), dim=1)), dim=0)
    op = LinearOperator(A.shape, dtype, A.device, lambda v: A@v, adjoint=lambda v: A.T@v)
    Q = torch.cat((torch.zeros(n, device=device, dtype=dtype),
                   torch.ones(n, device=device, dtype=dtype) / n**.5))[:, None]
    system = NullspaceSystem(op, Q, Q)
    exact = torch.arange(2*n, device=device, dtype=dtype).sin()
    exact = torch.cat((exact[:n], exact[n:] - exact[n:].mean()))
    prepared = system.prepareRhs(A@exact)
    inverse = DiagonalPreconditioner(torch.cat((h, torch.full_like(h, 2.))), requireSPD=True)
    projected = system.projectPreconditioner(inverse.asLinearOperator())
    tolerance = 5e-5 if dtype == torch.float32 else 1e-11
    result = minres(system, prepared.rhs, symmetric=True, precond=projected,
                    preconditionerSPD=True, rtol=tolerance, maxiter=128)
    solution = system.gauge(result.x)
    report = system.verify(solution, prepared, rtol=tolerance*2)
    if result.statusInt() < 0 or not bool(report.requestedConverged):
        raise AssertionError(f'MINRES saddle-point solve failed: {result.statusInt()}')
    accuracy = 1e-3 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(solution, exact, rtol=accuracy, atol=accuracy)
    return dict(solver='minres', device=str(A.device), dtype=str(dtype), unknowns=2*n,
                status=result.statusInt(), iterations=int(result.iterations),
                original_residual=float(report.requestedResidual),
                max_error=float((solution-exact).abs().max()), matvecs=result.matvecs)


def shiftingFallback(*, device='cpu', dtype=torch.float64):
    n = 8
    eye = torch.eye(n, device=device, dtype=dtype)
    differences = eye - eye.roll(1, dims=1)
    A = differences.T @ torch.diag(torch.linspace(.7, 1.3, n, device=device, dtype=dtype)) @ differences
    exact = torch.arange(n, device=device, dtype=dtype).sin()
    exact = exact - exact.mean()
    b = A@exact
    # Translation-nullspace Hessian model, warm-started from a finite iterate
    # supplied by an earlier Krylov attempt. No inverse diagonal is applied.
    warm = .8*exact + 2
    warmResidual = (b-A@warm).norm()
    tolerance = 4e-5 if dtype == torch.float32 else 1e-11
    result = richardson(lambda v: A@v, b, warm, rtol=tolerance, maxiter=400)
    actual = (b-A@result.x).norm()
    if result.statusInt() < 0 or not bool(actual <= tolerance*b.norm()):
        raise AssertionError(f'Richardson fallback failed: {result.statusInt()}')
    if not bool(actual < warmResidual):
        raise AssertionError('fallback did not improve the supplied residual')
    accuracy = 2e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(result.x-result.x.mean(), exact, rtol=accuracy, atol=accuracy)
    torch.testing.assert_close(result.stampedResidual, actual, rtol=0, atol=0)
    return dict(solver='richardson', device=str(A.device), dtype=str(dtype), unknowns=n,
                status=result.statusInt(), iterations=int(result.iterations),
                original_residual=float(actual), warm_residual=float(warmResidual),
                max_error=float((result.x-result.x.mean()-exact).abs().max()),
                trials=result.trialSteps, backtracks=result.backtracks, matvecs=result.matvecs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--solver', choices=['minres', 'richardson', 'both'], default='both')
    args = parser.parse_args()
    functions = (saddlePoint, shiftingFallback) if args.solver == 'both' else (saddlePoint if args.solver == 'minres' else shiftingFallback,)
    for run in functions:
        print(json.dumps(run(device=args.device, dtype=getattr(torch, args.dtype))))


if __name__ == '__main__':
    main()
