"""Matrix-free Newton/GMRES and implicit adjoint using production field products.

Run WARPSPHCORE_BACKEND=native python scripts/examples/native_field_implicit.py
--device cpu (or cuda). The geometry is fixed for one solve. No Newton/GMRES
trajectory is recorded by autograd; only the converged state enters the adjoint.
"""
import argparse
import json
import sys
import torch
import warpSPHCore as sph
from warpSPHCore.native import CSRTopology, linearizeFields


def gmres(action, rhs, tolerance=2e-12, maximum=48):
    """Small, unrestarted GMRES with reorthogonalization for this example."""
    beta = rhs.norm()
    if float(beta) == 0:
        return torch.zeros_like(rhs), 0
    basis = [rhs / beta]
    hessenberg = rhs.new_zeros((maximum + 1, maximum))
    target = rhs.new_zeros(maximum + 1)
    target[0] = beta
    for column in range(maximum):
        candidate = action(basis[column])
        for _ in range(2):
            for row, vector in enumerate(basis):
                coefficient = torch.dot(vector, candidate)
                hessenberg[row, column] += coefficient
                candidate -= coefficient * vector
        length = candidate.norm()
        hessenberg[column + 1, column] = length
        # CUDA lstsq uses a full-rank driver; the converged square/rectangular
        # Krylov problem here is full rank before a possible happy breakdown.
        coefficients = torch.linalg.lstsq(hessenberg[:column + 2, :column + 1],
                                         target[:column + 2]).solution
        solution = sum(c * v for c, v in zip(coefficients, basis))
        if float((action(solution) - rhs).norm()) <= tolerance * max(1., float(beta)):
            return solution, column + 1
        if float(length) < 1e-15:
            break
        basis.append(candidate / length)
    raise RuntimeError('Example GMRES did not converge')


@torch.no_grad()
def run(device):
    n, step, parameter = 48, .15, .2
    positions = (torch.arange(n, device=device, dtype=torch.float64) + .5)[:, None] / n
    particles = sph.ParticleState(positions=positions,
        supports=positions.new_full((n,), 4 / n), masses=positions.new_full((n,), 1 / n),
        densities=positions.new_ones(n), kinds=torch.zeros(n, device=device, dtype=torch.int32))
    domain = sph.DomainDescription(positions.new_zeros(1), positions.new_ones(1),
                                  torch.ones(1, device=device, dtype=torch.bool), 1)
    graph = sph.radiusSearch(particles, domain)
    topology = CSRTopology(graph)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)
    # Normalization is an external diagonal finalize: the native interpolation
    # map itself matches warpOperation exactly, with no hidden normalization.
    linear = linearizeFields(particles, properties, domain, positions.new_ones(n), adjacency=topology)
    row_sum = linear.value
    source = torch.sin(2 * torch.pi * positions[:, 0])

    def interpolate(value):
        return linear.apply(value) / row_sum

    def residual(value):
        return value + step * (value**3 - interpolate(value)) - parameter * source

    value = torch.zeros_like(source)
    iterations = []
    for _ in range(8):
        current = residual(value)
        if float(current.norm()) < 2e-12:
            break
        def jvp(direction):
            return direction + step * (3 * value.square() * direction -
                                       linear.jvp(reference=direction) / row_sum)
        update, count = gmres(jvp, -current)
        value += update
        iterations.append(count)
    else:
        raise RuntimeError('Example Newton solve did not converge')

    # Solve R_u^T lambda = dL/du for L = ||u||^2/2. The normalization's
    # transpose divides the incoming cotangent before the native CSC gather.
    def transpose_product(direction):
        pullback = linear.vjp((direction / row_sum).contiguous(), query=False).reference
        return direction + step * (3 * value.square() * direction - pullback)

    multiplier, transpose_iterations = gmres(transpose_product, value)
    parameter_gradient = torch.dot(multiplier, source)
    cached = topology.transpose()
    transpose_product(source)
    assert topology.transpose() is cached

    # Verification only: form the independent dense 1D Wendland-2 operator
    # after both matrix-free solves. No dense matrix enters the solver path.
    distance = (positions[:, None, 0] - positions[None, :, 0] + .5).remainder(1) - .5
    ratio = distance.abs() / particles.supports[:, None]
    weights = torch.where(ratio < 1, 1.25 * (1 - ratio)**3 * (1 + 3 * ratio) /
                          particles.supports[:, None] / n, 0.)
    normalized = weights / weights.sum(1, keepdim=True)
    jacobian = torch.eye(n, device=device, dtype=value.dtype) + step * (
        torch.diag(3 * value.square()) - normalized)
    expected = torch.linalg.solve(jacobian.T, value)
    probe = source.cos()
    errors = dict(residual=float(residual(value).norm()),
        jvp=float((jvp(probe) - jacobian @ probe).norm()),
        vjp=float((transpose_product(probe) - jacobian.T @ probe).norm()),
        adjoint=float((multiplier - expected).norm()),
        parameter_gradient=float((parameter_gradient - torch.dot(expected, source)).abs()))
    assert max(errors.values()) < 2e-10, errors
    assert not value.requires_grad and not multiplier.requires_grad
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)
    return dict(device=device, newton_iterations=len(iterations), gmres_iterations=iterations,
                transpose_gmres_iterations=transpose_iterations,
                parameter_gradient=float(parameter_gradient), errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sph.backend != 'native':
        raise RuntimeError('Select WARPSPHCORE_BACKEND=native before importing')
    print(json.dumps(run(args.device), indent=2))


if __name__ == '__main__':
    main()
