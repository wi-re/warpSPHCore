"""Differentiate density -> CRK/renormalization -> gradient on fixed neighbors.

Rebuild every geometry-dependent correction inside evaluate. The explicit
JVP includes density, volume, correction and field tangents; compare it with
reverse AD and a centered finite difference. No differentiation of membership.
"""

import argparse
from dataclasses import replace
import json

import torch
import warp as wp

from warpSPHCore import (
    DomainDescription, GradientScheme, KernelFunctions, OperationProperties,
    ParticleState, ParticleTangentState, SupportScheme, WarpOperation,
    get_torch_precision, radiusSearchCompactHashMap, warpOperation, warpOperationJVP,
)
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP


def run(*, device='cpu', crk=True):
    wp.init()
    axis = torch.linspace(-.3, .3, 3, dtype=get_torch_precision(), device=device)
    positions = torch.cartesian_prod(axis, axis)
    h = torch.linspace(1.1, 1.3, len(positions), dtype=axis.dtype, device=device)
    p = ParticleState(positions, h, torch.full_like(h, .09),
                      torch.zeros(len(h), dtype=torch.int32, device=device))
    domain = DomainDescription(axis.new_full((2,), -2), axis.new_full((2,), 2),
                               torch.zeros(2, dtype=torch.bool, device=device), 2)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.SuperSymmetric)
    density_props = OperationProperties(operation=WarpOperation.Density, kernel=KernelFunctions.Wendland2)
    props = OperationProperties(operation=WarpOperation.Gradient, kernel=KernelFunctions.Wendland2,
                                gradientMode=GradientScheme.Difference)
    dp, dh = positions.sin()*.1+.03, h.cos()*.02
    tangent = ParticleTangentState(dp, dh, None)

    def evaluate(pos, support, jvp=False):
        state = replace(p, positions=pos, supports=support)
        density = warpOperation(state, density_props, domain, adjacency=adjacency)
        state = replace(state, densities=density)
        if jvp:
            drho = warpOperationJVP(state, density_props, domain, adjacency=adjacency,
                queryTangentState=tangent, referenceTangentState=tangent)
            state_tangent = replace(tangent, densities=drho)
        volume = dvolume = correction = dcorrection = None
        if crk:
            if jvp:
                volume, dvolume, correction, dcorrection = computeCRKFactorsJVP(
                    state, domain, KernelFunctions.Wendland2, tangent,
                    referenceTangentState=tangent, adjacency=adjacency)
            else:
                volume, _, correction = computeCRKFactors(
                    state, domain, KernelFunctions.Wendland2, adjacency=adjacency)
        if jvp:
            renorm, drenorm = computeRenormalizationMatricesJVP(
                state, props, domain, state_tangent, referenceTangentState=state_tangent,
                referenceVolumes=volume, tangentReferenceVolumes=dvolume,
                adjacency=adjacency, returnEigVals=False)
        else:
            renorm = computeRenormalizationMatrices(state, props, domain,
                referenceVolumes=volume, adjacency=adjacency, returnEigVals=False)
        field = 1+pos[:, 0].square()+.3*pos[:, 1].sin()
        kwargs = dict(adjacency=adjacency, queryValues=field, referenceValues=field,
                      referenceVolumes=volume, crkState=correction, renormalizationState=renorm)
        if not jvp:
            return warpOperation(state, props, domain, **kwargs)
        dfield = 2*pos[:, 0]*dp[:, 0]+.3*pos[:, 1].cos()*dp[:, 1]
        return warpOperationJVP(state, props, domain, **kwargs,
            queryTangentState=state_tangent, referenceTangentState=state_tangent,
            tangentQueryValues=dfield, tangentReferenceValues=dfield,
            tangentReferenceVolumes=dvolume, crkTangentState=dcorrection,
            renormalizationTangentState=drenorm)

    pos, support = positions.detach().requires_grad_(), h.detach().requires_grad_()
    output, derivative = evaluate(pos, support), evaluate(pos, support, jvp=True)
    cotangent = torch.arange(output.numel(), dtype=axis.dtype, device=device).reshape_as(output).sin()
    gpos, gh = torch.autograd.grad((output*cotangent).sum(), (pos, support))
    forward_dot, reverse_dot = (derivative*cotangent).sum(), (gpos*dp).sum()+(gh*dh).sum()
    f32 = axis.dtype == torch.float32
    tol = 5e-4 if f32 else 2e-9
    torch.testing.assert_close(forward_dot, reverse_dot, rtol=tol, atol=tol)
    eps = .002 if f32 else 1e-5
    with torch.no_grad():
        fd = (evaluate(pos+eps*dp, support+eps*dh)-evaluate(pos-eps*dp, support-eps*dh))/(2*eps)
    torch.testing.assert_close(derivative, fd, rtol=4e-3 if f32 else 2e-7,
                               atol=3e-4 if f32 else 2e-8)
    return dict(device=str(axis.device), dtype=str(axis.dtype), crk=crk,
                adjoint_dot_error=float((forward_dot-reverse_dot).abs().detach()),
                finite_difference_max_error=float((derivative-fd).abs().max().detach()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--no-crk', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(device=args.device, crk=not args.no_crk), indent=2))
