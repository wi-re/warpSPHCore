"""Raw moments, their explicit products and an independent displacement oracle."""
from dataclasses import replace
import math
import torch
import warpSPHCore as sph
from .density_contract import case as geometry_case
from .operators import normalization
from .kernels import shape


def case(device, dimension=2, dtype=torch.float64, kernel=None, support=None, graph_types=None,
         explicit=False, geometry=True, volume_mode='reference'):
    q, r, domain, properties, graph, values, states, _ = geometry_case(
        device, dimension, dtype, kernel=kernel, support=support or sph.SupportScheme.Scatter,
        graph_types=graph_types)
    values = list(values)
    if explicit:
        properties = replace(properties, supportMode=sph.SupportScheme.Scatter)
        values[5] = values[5].detach(); values[6] = values[6].detach()
    volumes = torch.linspace(.2, .7, len(r.positions), device=device, dtype=dtype).requires_grad_()
    query_volumes = torch.full((len(q.positions),), .6, device=device, dtype=dtype).requires_grad_()
    values = (*values, volumes, query_volumes)
    if explicit:
        values = (*values, *((values[k].detach().cos() * .03).requires_grad_() for k in (0, 1, 2, 3, 10)))

    def functions(backend, props=properties, adjacency=graph):
        def run(*x):
            qi, rj, d = states(x[:10])
            if explicit:
                if backend == 'native':
                    from warpSPHCore.crk.native.nt_jvp import computeCRKMomentsGeometryJVP as compute
                else:
                    from warpSPHCore.crk.crk_moments_jvp import computeCRKMomentsGeometryJVP as compute
                qt = sph.ParticleTangentState(x[12] if geometry else (None if backend == 'native' else torch.zeros_like(x[0])), x[14] if geometry else None, None)
                rt = sph.ParticleTangentState(x[13] if geometry else None, x[15] if geometry else None, None)
                return compute(qi, d, props.kernel, adjacency, x[10] if volume_mode != 'none' else None,
                               x[16], qt, rj, rt, props.operationMode)
            if backend == 'native':
                from warpSPHCore.crk.native.nt_moments import computeCRKMoments as compute
            else:
                from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend as compute
            qv = x[10] if volume_mode == 'query' else x[11] if volume_mode == 'reference' else None
            rv = x[10] if volume_mode == 'reference' else None
            return compute(qi, props, d, qv, rv, adjacency, rj)
        return run

    return q, r, domain, properties, graph, values, states, functions


def equation(values, domain, properties, graph):
    """Differentiate geometric moments using a virtual pair displacement.

    This derives all three spatial-derivative tensors from the primal moments,
    rather than copying either production moment-gradient implementation.
    Smooth fixtures use the ordinary radial derivative; singular contracts use
    direct Warp comparisons instead.
    """
    if properties.operationMode != sph.OperationDirection.TrueAllToToAll:
        raise ValueError('The independent displacement oracle uses unfiltered fixtures')
    n, m, d = len(values[0]), len(values[1]), domain.dim
    delta = values[0].new_zeros((n, m, d), requires_grad=True)
    dx = values[0][:, None] + delta - values[1][None]
    length = values[6] - values[5]
    dx = torch.where(domain.periodic, (dx + length / 2).remainder(length) - length / 2, dx)
    radius = dx.norm(dim=-1)
    hi, hj = values[2][:, None], values[3][None]

    def kernel(h):
        ratio = radius / h
        active = (h > 0) & (ratio <= 1)
        if properties.kernel == sph.KernelFunctions.AdhesionKernel:
            active = active & (ratio > .5)
            ratio = torch.where(active, ratio, ratio.new_tensor(.75))
        return torch.where(active, normalization(properties.kernel, d) * shape(ratio, properties.kernel, d) / h**d, 0.)

    mode = properties.supportMode.value
    w = ((kernel(hi) + kernel(hj)) / 2 if mode in (14, 15) else
         kernel(hi if mode == 11 else hj if mode == 12 else (hi + hj) / 2 if mode == 13 else torch.maximum(hi, hj)))
    selected = torch.zeros((n, m), device=w.device, dtype=w.dtype)
    # Duplicates are true repeated contributions; sparse prefixes/gaps do not
    # enter the independently constructed pair weights.
    for i in range(n):
        ids = graph.paddedJ[int(graph.edgeOffsets[i]):int(graph.edgeOffsets[i]) + int(graph.numNeighbors[i])].long()
        selected[i].index_add_(0, ids, torch.ones_like(ids, dtype=w.dtype))
    weighted = w * values[10][None] * selected
    m0 = weighted.sum(1)
    m1 = (weighted[..., None] * dx).sum(1)
    m2 = torch.einsum('ij,ija,ijb->iab', weighted, dx, dx)

    def derivative(value):
        return torch.autograd.grad(value.sum(), delta, create_graph=True, retain_graph=True)[0].sum(1)

    dm0 = derivative(m0)
    dm1 = torch.stack([derivative(m1[:, a]) for a in range(d)], 1)
    dm2 = torch.stack([derivative(m2[:, a, b]) for a in range(d) for b in range(d)], -1).reshape(n, d, d, d)
    return m0, m1, m2, dm0, dm1, dm2


def product_equation(values, domain, properties, graph, geometry=True):
    primals = values[:12]
    slots = (0, 1, 2, 3, 10)
    directions = dict(zip(slots, values[12:]))
    if not geometry:
        directions = {10: directions[10]}
    def primal(*x):
        x = list(x); x[5] = x[5].detach(); x[6] = x[6].detach()
        return equation(x, domain, properties, graph)
    return torch.autograd.functional.jvp(primal, primals,
        tuple(directions.get(k, torch.zeros_like(x)) for k, x in enumerate(primals)), create_graph=True)[1]


def flat(outputs):
    return torch.cat([x.reshape(len(x), math.prod(x.shape[1:])) for x in outputs], 1)
