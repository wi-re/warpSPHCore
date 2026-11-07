"""Full-rank factor pipelines with a physical boundary-only volume stencil."""
import torch
import warpSPHCore as sph


def case(device, dimension=2, dtype=torch.float64, kernel=None, *, consistency=False):
    # Every fluid reference estimates its volume against the boundary anchor.
    # This excludes self-pair singularities without modifying a kernel or its
    # reference formula. The moment cloud remains full rank in 1D-3D, including
    # adhesion's annular support, and exercises the active coefficient solve.
    eye = torch.eye(dimension, device=device, dtype=dtype) * .7
    rx = torch.cat((eye, -eye, torch.zeros((1, dimension), device=device, dtype=dtype)))
    qx = torch.tensor([[.015, -.01, .005], [-.01, .005, -.015]], device=device, dtype=dtype)[:, :dimension].clone()
    qx[:, 0] += 8; rx[:, 0] -= 4
    qh = torch.full((2,), 1.1, device=device, dtype=dtype)
    rh = torch.full((len(rx),), 1.3, device=device, dtype=dtype)
    values = tuple(x.detach().requires_grad_() for x in
        (qx, rx, qh, rh, qx.cos()*.03, rx.cos()*.02, qh.cos()*.03, rh.cos()*.02))
    qkind = torch.ones(2, device=device, dtype=torch.int32)
    rkind = torch.ones(len(rx), device=device, dtype=torch.int32); rkind[-1] = 0
    domain = sph.DomainDescription(qx.new_full((dimension,), -2), qx.new_full((dimension,), 2),
        torch.tensor([True] + [False]*(dimension-1), device=device), dimension)
    graph = sph.AdjacencyList(None, torch.arange(len(rx), device=device, dtype=sph.get_torch_index_dtype()).repeat(2),
        torch.full((2,), len(rx), device=device, dtype=torch.int32),
        torch.arange(2, device=device, dtype=sph.get_torch_offset_dtype())*len(rx), 2, len(rx))
    reference_graph = sph.AdjacencyList(None,
        torch.arange(len(rx), device=device, dtype=sph.get_torch_index_dtype()).repeat(len(rx)),
        torch.full((len(rx),), len(rx), device=device, dtype=torch.int32),
        torch.arange(len(rx), device=device, dtype=sph.get_torch_offset_dtype())*len(rx), len(rx), len(rx))
    kernel = kernel or sph.KernelFunctions.Wendland2

    def functions(backend):
        def run(*x):
            q = sph.ParticleState(x[0], x[2], torch.ones_like(x[2]), qkind, torch.ones_like(x[2]))
            r = sph.ParticleState(x[1], x[3], torch.ones_like(x[3]), rkind, torch.ones_like(x[3]))
            qt, rt = sph.ParticleTangentState(x[4], x[6], None), sph.ParticleTangentState(x[5], x[7], None)
            if consistency:
                if backend == 'native':
                    from warpSPHCore.crk.native.nt_factors import prepareCRK
                    prepared=prepareCRK(q,domain,kernel,sph.OperationDirection.TrueAllToToAll,graph,r,
                        referenceVolumeMode=sph.OperationDirection.FluidToBoundary,referenceAdjacency=reference_graph)
                    area,density,state=prepared.queryVolumes,prepared.consistencyDensity,prepared.crkState
                else:
                    from warpSPHCore.crk.crk_wrapper import computeCRKFactors
                    area,density,state=computeCRKFactors(q,domain,kernel,sph.OperationDirection.TrueAllToToAll,graph,r,
                        referenceVolumeMode=sph.OperationDirection.FluidToBoundary)
                return area,density,state.A,state.B,state.gradA,state.gradB
            if backend == 'native':
                from warpSPHCore.crk.native.nt_jvp import computeCRKFactorsJVP as compute
            else:
                from warpSPHCore.crk.crk_wrapper import computeCRKFactorsJVP as compute
            area, darea, primal, product = compute(q, domain, kernel, qt,
                operationMode=sph.OperationDirection.TrueAllToToAll, adjacency=graph,
                referenceState=r, referenceTangentState=rt,
                referenceVolumeMode=sph.OperationDirection.FluidToBoundary,
                **({'referenceAdjacency': reference_graph} if backend == 'native' else {}))
            return area, darea, primal.A, primal.B, primal.gradA, primal.gradB, product.A, product.B, product.gradA, product.gradB
        return run
    return values, functions
