"""Bounded native contracts: dense rank policies, correction AD and solver reuse."""
from dataclasses import replace
import pytest
import torch
from torch.autograd import forward_ad
import warpSPHCore as sph


def test_dense_inverse_rank_cutoffs_and_fused_application(device):
    for dim, dtype in ((1, torch.float32), (2, torch.float64), (3, torch.float32)):
        seed = torch.arange(9*dim*dim, dtype=dtype, device=device).reshape(9, dim, dim).sin()
        matrix = seed @ seed.mT + torch.eye(dim, device=device, dtype=dtype)*.4
        matrix[0] = 0
        matrix[1] = -torch.eye(dim, device=device, dtype=dtype)
        values = torch.arange(9*4*dim, dtype=dtype, device=device).reshape(9,4,dim).cos()
        expected = torch.linalg.pinv(matrix, rtol=1e-5, hermitian=True)
        tolerance = 3e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(sph.pinvSymmetric(matrix, rtol=1e-5), expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(sph.applyPinv(values, matrix, rtol=1e-5), values @ expected, rtol=tolerance, atol=tolerance)
        assert sph.pinvSymmetric(matrix[:0]).shape == matrix[:0].shape


def test_dense_selected_branch_derivatives_and_legacy_policy(device):
    matrix = torch.diag(torch.tensor([2., .1], device=device, dtype=torch.float64))[None].requires_grad_()
    values = matrix.new_tensor([[[.3, -.7]]]).requires_grad_()
    function = lambda m, v: sph.applyPinv(v, m, rtol=.1)
    assert torch.autograd.gradcheck(function, (matrix, values))
    assert torch.autograd.gradgradcheck(function, (matrix, values))

    # Renormalization has its own absolute/relative cutoffs and low-count rule.
    matrix = torch.stack((torch.eye(2, device=device, dtype=torch.float64),
                          matrix.detach()[0]*2, matrix.detach()[0],
                          torch.diag(matrix.new_tensor([2., 1e-7])))).requires_grad_()
    counts = torch.tensor([3, 4, 7, 8], device=device, dtype=torch.int32)
    inverse, eigenvalues = sph.pinv2x2_warpBackend(matrix, counts)
    expected = torch.linalg.pinv(matrix.detach(), rtol=1e-6, hermitian=True)
    expected[0] = torch.eye(2, device=device, dtype=matrix.dtype)
    torch.testing.assert_close(inverse, expected)
    torch.testing.assert_close(eigenvalues[0], torch.ones(2, device=device, dtype=matrix.dtype))
    # Isotropy has a smooth inverse but nonsmooth diagnostic eigenvectors.
    assert torch.autograd.gradcheck(lambda m: sph.pinv2x2_warpBackend(m, counts)[0], (matrix,))
    tangent = torch.arange(matrix.numel(), device=device, dtype=matrix.dtype).reshape_as(matrix).cos()*.1
    with forward_ad.dual_level():
        dual_inverse, _ = sph.pinv2x2_warpBackend(forward_ad.make_dual(matrix, tangent), counts)
        jvp = forward_ad.unpack_dual(dual_inverse).tangent
    cotangent = tangent.sin()
    pullback, = torch.autograd.grad((inverse*cotangent).sum(), matrix)
    torch.testing.assert_close((jvp*cotangent).sum(), (pullback*tangent).sum())


def test_renormalization_rotating_subspace_and_reference_roles(device):
    for dim in (1, 2, 3):
        axis = torch.linspace(-.3, .3, 7, device=device, dtype=torch.float64)
        positions = torch.zeros((7,dim), device=device, dtype=axis.dtype)
        positions[:,0] = axis
        if dim == 2: positions[:,1] = axis.square()*1e-3
        supports = torch.ones_like(axis)*1.2
        particles = sph.ParticleState(positions, supports, torch.ones_like(axis),
                                     torch.zeros(7,device=device,dtype=torch.int32), torch.ones_like(axis))
        domain = sph.DomainDescription(axis.new_full((dim,),-2),axis.new_full((dim,),2),
                                       torch.zeros(dim,device=device,dtype=torch.bool),dim)
        properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient)
        adjacency = sph.radiusSearch(particles,domain)
        if dim == 1: rotation = axis.new_tensor([[.1]])
        else:
            rotation = axis.new_zeros((dim,dim)); rotation[0,-1]=-1; rotation[-1,0]=1
        direction = positions @ rotation.T
        tangent = sph.ParticleTangentState(positions=direction,supports=None,masses=None)
        _, _, state, derivative = sph.computeRenormalizationMatricesJVP(
            particles,properties,domain,tangent,referenceTangentState=tangent,adjacency=adjacency)
        if dim > 1:
            inverse = state.renormalizationMatrices
            torch.testing.assert_close(derivative.renormalizationMatrices,rotation@inverse-inverse@rotation,rtol=2e-9,atol=2e-9)
        # Query-only explicit tangent must keep the reference cloud fixed.
        _, _, _, query_derivative = sph.computeRenormalizationMatricesJVP(
            particles,properties,domain,tangent,adjacency=adjacency)
        moving = positions.detach().requires_grad_()
        output = sph.computeRenormalizationMatrices(replace(particles,positions=moving),properties,domain,
            referenceParticles=particles,adjacency=adjacency,returnEigVals=False).renormalizationMatrices
        cotangent = torch.arange(output.numel(),device=device,dtype=axis.dtype).reshape_as(output).sin()
        pullback, = torch.autograd.grad((output*cotangent).sum(),moving)
        torch.testing.assert_close((query_derivative.renormalizationMatrices*cotangent).sum(),(pullback*direction).sum(),rtol=2e-9,atol=2e-9)
        assert properties.operation == sph.WarpOperation.Gradient


def test_native_workspace_reuse_block_adjoint_and_verified_residual(device):
    for solver, workspace_type, dtype in ((sph.cg,sph.CGWorkspace,torch.float32),
                                           (sph.bicgstab,sph.BiCGStabWorkspace,torch.float64)):
        matrix = torch.diag(torch.linspace(1,4,18,device=device,dtype=dtype))
        if solver is sph.bicgstab: matrix[0,1]=.2
        blocks = torch.stack([matrix[i:i+3,i:i+3] for i in range(0,18,3)])
        preconditioner = sph.BlockJacobiPreconditioner(blocks,application='native')
        rhs = torch.arange(36,device=device,dtype=dtype).cos()[::2]
        output = torch.empty_like(rhs.repeat_interleave(2))[::2]
        preconditioner.applyAdjoint(rhs,output)
        torch.testing.assert_close(output.reshape(-1,3),torch.linalg.solve(blocks.mT,rhs.reshape(-1,3)))
        workspace = workspace_type(rhs,64,history=True,capture=device!='cpu')
        pointers = {key:value.data_ptr() for key,value in workspace._t.items()}
        action = lambda v:matrix@v
        into = lambda v,out:torch.mv(matrix,v,out=out)
        for scale in (1.,-.5):
            result = solver(action,rhs*scale,workspace=workspace,matvecOut=into,
                precond=preconditioner,maxiter=64,
                rtol=1e-5 if dtype==torch.float32 else 1e-11)
            torch.testing.assert_close(matrix@result.x,rhs*scale,
                rtol=2e-5 if dtype==torch.float32 else 1e-10,atol=2e-6 if dtype==torch.float32 else 1e-11)
            assert pointers == {key:value.data_ptr() for key,value in workspace._t.items()}
        if device != 'cpu': assert workspace.captureBuilds > 0
        # A diagonal view is caller-owned and may be updated at the same address.
        diagonal = matrix.diag().reciprocal().repeat_interleave(2)[::2]
        for factor in (1., .5):
            diagonal.mul_(factor)
            result = solver(action,rhs,workspace=workspace,matvecOut=into,
                precond=diagonal,maxiter=64,rtol=1e-5 if dtype==torch.float32 else 1e-11)
            torch.testing.assert_close(matrix@result.x,rhs,
                rtol=2e-5 if dtype==torch.float32 else 1e-10,atol=2e-6 if dtype==torch.float32 else 1e-11)


def test_dense_current_stream_and_graph_replay(device):
    if device == 'cpu': pytest.skip('GPU stream/capture contract')
    matrix = torch.eye(3,device=device)[None]*2
    values = torch.ones((1,2,3),device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sph.pinvSymmetric(matrix); sph.applyPinv(values,matrix)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            inverse = sph.pinvSymmetric(matrix)
            output = sph.applyPinv(values,matrix)
        matrix.mul_(2)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(inverse,torch.eye(3,device=device)[None]/4)
    torch.testing.assert_close(output,values/4)
