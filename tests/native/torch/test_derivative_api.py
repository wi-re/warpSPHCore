"""Bounded public product contracts; dimensions/schemes rotate within each check."""
from dataclasses import replace
import pytest
import torch
from torch.autograd import forward_ad
import warpSPHCore as sph


def problem(device,dimension=2):
    def particles(count,phase):
        t=torch.arange(count,dtype=torch.float64,device=device)
        x=torch.stack([.4*torch.sin(t*(.4+d*.3)+phase+d) for d in range(dimension)],1)
        return sph.ParticleState(x,1.1+.03*t,.8+.02*t,(t.long()%3).int(),1.2+.04*t)
    q,r=particles(5,.1),particles(7,.3)
    domain=sph.DomainDescription(q.positions.new_full((dimension,),-1),q.positions.new_full((dimension,),1),
                               torch.ones(dimension,dtype=torch.bool,device=device),dimension)
    return q,r,domain


def tangent(state):
    return sph.ParticleTangentState(*(getattr(state,n).cos()*.03 for n in ('positions','supports','masses','densities')))


def shifted(state,direction,epsilon):
    return replace(state,**{name:getattr(state,name)+epsilon*getattr(direction,name)
        for name in ('positions','supports','masses','densities')})


def test_public_combined_jvp_implicit_duals_and_reverse_identity(device):
    for k,operation in enumerate(sph.WarpOperation):
        if operation==sph.WarpOperation.Custom:continue
        dimension=1+k%3;q,r,domain=problem(device,dimension)
        tq,tr=tangent(q),tangent(r)
        properties=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=operation,
            supportMode=list(sph.SupportScheme)[k%6],gradientMode=list(sph.GradientScheme)[k%4],
            laplacianMode=sph.LaplacianScheme.Brookshaw)
        graph=sph.radiusSearch(q,domain,r,sph.SupportScheme.SuperSymmetric)
        qv,rv=q.positions.sin(),r.positions.cos()
        vq,vr=qv.cos()*.04,rv.sin()*.04
        arguments=dict(referenceParticles=r,adjacency=graph,queryValues=qv,referenceValues=rv,
            queryTangentState=tq,referenceTangentState=tr)
        if operation not in (sph.WarpOperation.Density,sph.WarpOperation.Covariance):
            arguments.update(tangentQueryValues=vq,tangentReferenceValues=vr)
        actual=sph.warpOperationJVP(q,properties,domain,**arguments)
        epsilon=1e-5
        def function(sign):
            return sph.warpOperation(shifted(q,tq,sign*epsilon),properties,domain,qv+sign*epsilon*vq,
                rv+sign*epsilon*vr,adjacency=graph,referenceParticles=shifted(r,tr,sign*epsilon))
        finite=(function(1)-function(-1))/(2*epsilon)
        torch.testing.assert_close(actual,finite,rtol=3e-6,atol=2e-7)
        # The explicit helper is level-independent; the ordinary call supports
        # implicit dual propagation with the same independent population roles.
        with forward_ad.dual_level():
            def dual_state(state,direction):
                return replace(state,**{n:forward_ad.make_dual(getattr(state,n),getattr(direction,n))
                    for n in ('positions','supports','masses','densities')})
            output=sph.warpOperation(dual_state(q,tq),properties,domain,
                forward_ad.make_dual(qv,vq),forward_ad.make_dual(rv,vr),
                adjacency=graph,referenceParticles=dual_state(r,tr))
            torch.testing.assert_close(forward_ad.unpack_dual(output).tangent,actual)
            torch.testing.assert_close(sph.warpOperationJVP(q,properties,domain,**arguments),actual)
        values=[getattr(q,n).clone().requires_grad_() for n in ('positions','supports','masses','densities')]
        values += [getattr(r,n).clone().requires_grad_() for n in ('positions','supports','masses','densities')]
        values += [qv.clone().requires_grad_(),rv.clone().requires_grad_()]
        query=replace(q,**dict(zip(('positions','supports','masses','densities'),values[:4])))
        reference=replace(r,**dict(zip(('positions','supports','masses','densities'),values[4:8])))
        output=sph.warpOperation(query,properties,domain,*values[8:],adjacency=graph,referenceParticles=reference)
        bar=output.cos()
        bars=torch.autograd.grad(output,values,bar,allow_unused=True)
        directions=[getattr(t,n) for t in (tq,tr) for n in ('positions','supports','masses','densities')]+[vq,vr]
        dot=sum((b*d).sum() for b,d in zip(bars,directions) if b is not None)
        torch.testing.assert_close((actual*bar).sum(),dot,rtol=3e-8,atol=2e-8)


def test_density_position_block_finite_difference_transpose_and_self_policy(device):
    for dimension,support in ((1,sph.SupportScheme.Gather),(2,sph.SupportScheme.MeanSymmetric),
                              (3,sph.SupportScheme.SuperSymmetric)):
        q,r,domain=problem(device,dimension)
        graph=sph.radiusSearch(q,domain,r,sph.SupportScheme.SuperSymmetric)
        # The established helper excludes equal global IDs even for separate
        # clouds; remove those edges from the independent gradient oracle.
        rows=[];counts=[]
        for i in range(len(q.positions)):
            start=int(graph.edgeOffsets[i]);n=int(graph.numNeighbors[i]);ids=graph.paddedJ[start:start+n]
            rows.append(ids[ids!=i]);counts.append(len(rows[-1]))
        counts=torch.tensor(counts,dtype=torch.int32,device=device)
        offsets=counts.long().cumsum(0)-counts
        oracle_graph=sph.AdjacencyList(None,torch.cat(rows),counts,offsets,len(q.positions),len(r.positions))
        vq,vr=q.positions.cos()*.07,r.positions.sin()*.09
        properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Density,supportMode=support)
        epsilon=1e-5
        def gradient(sign):
            query=replace(q,positions=(q.positions+sign*epsilon*vq).requires_grad_())
            reference=replace(r,positions=r.positions+sign*epsilon*vr)
            output=sph.warpOperation(query,properties,domain,referenceParticles=reference,adjacency=oracle_graph)
            return torch.autograd.grad(output.sum(),query.positions)[0]
        expected=(gradient(1)-gradient(-1))/(2*epsilon)
        actual=sph.computeSPHDensityPositionHVP(q,domain,properties.kernel,support,graph,vq,r,vr)
        torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-7)
        mass=r.masses.clone().requires_grad_();left=vq.clone().requires_grad_();right=vr.clone().requires_grad_()
        def block(m,a,b):
            return sph.computeSPHDensityPositionHVP(q,domain,properties.kernel,support,graph,a,replace(r,masses=m),b)
        assert torch.autograd.gradcheck(block,(mass,left,right),fast_mode=True,atol=2e-6)
        directions=tuple(x.detach().cos()*.02 for x in (mass,left,right))
        with forward_ad.dual_level():
            output=block(*(forward_ad.make_dual(x.detach(),t) for x,t in zip((mass,left,right),directions)))
            product=forward_ad.unpack_dual(output).tangent
        finite=(block(*(x.detach()+epsilon*t for x,t in zip((mass,left,right),directions)))-
                block(*(x.detach()-epsilon*t for x,t in zip((mass,left,right),directions))))/(2*epsilon)
        torch.testing.assert_close(product,finite,rtol=3e-7,atol=3e-7)
        # A single self ID contributes zero even when the reference vector is
        # omitted. Distinct coincident IDs retain the finite origin Hessian.
        one=replace(q,positions=q.positions[:1],supports=q.supports[:1],masses=q.masses[:1],
                    kinds=q.kinds[:1],densities=q.densities[:1])
        single=sph.radiusSearch(one,domain)
        assert not sph.computeSPHDensityPositionHVP(one,domain,properties.kernel,support,single,
                                                    torch.ones_like(one.positions)).any()
        with pytest.raises(NotImplementedError,match='forward-only'):
            sph.computeSPHDensityPositionHVP(replace(q,positions=q.positions.requires_grad_()),domain,
                properties.kernel,support,graph,vq,r,vr)


def test_explicit_jvp_roles_and_first_order_boundary(device):
    q,_,domain=problem(device);graph=sph.radiusSearch(q,domain)
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Density)
    direction=sph.ParticleTangentState(torch.ones_like(q.positions),None,None,None)
    query_only=sph.warpOperationJVP(q,properties,domain,queryTangentState=direction,adjacency=graph)
    both=sph.warpOperationJVP(q,properties,domain,queryTangentState=direction,
                             referenceTangentState=direction,adjacency=graph)
    assert query_only.abs().max()>0
    torch.testing.assert_close(both,torch.zeros_like(both),atol=1e-12,rtol=0)
    moving=replace(q,positions=q.positions.clone().requires_grad_())
    product=sph.warpOperationJVP(moving,properties,domain,queryTangentState=direction,adjacency=graph)
    with pytest.raises(NotImplementedError,match='first-order'):torch.autograd.grad(product.sum(),moving.positions)
    with forward_ad.dual_level():
        dual=replace(q,positions=forward_ad.make_dual(q.positions,torch.ones_like(q.positions)))
        with pytest.raises(NotImplementedError,match='first-order'):
            sph.warpOperationJVP(dual,properties,domain,queryTangentState=direction,adjacency=graph)
    cusp=replace(properties,kernel=sph.KernelFunctions.Spiky)
    with pytest.raises(NotImplementedError,match='cusp'):sph.warpOperationJVP(q,cusp,domain,queryTangentState=direction,adjacency=graph)


def test_explicit_crk_chain_and_public_module_imports(device):
    from warpSPHCore.operations import warpOperationJVP,warpOperationHVP,sphOperation_warp
    from warpSPHCore.coreOperations import computeSPHGradientGeometryJVP
    from warpSPHCore.math import getNextPrime
    assert getNextPrime(12)==13
    q,r,domain=problem(device)
    # A full stencil is small enough to exercise the factor algebra directly.
    q=replace(q,kinds=torch.zeros_like(q.kinds));r=replace(r,kinds=torch.zeros_like(r.kinds))
    tq,tr=tangent(q),tangent(r)
    graph=sph.radiusSearch(q,domain,r,sph.SupportScheme.SuperSymmetric)
    area,darea,state,dstate=sph.computeCRKFactorsJVP(q,domain,sph.KernelFunctions.Wendland2,tq,
        adjacency=graph,referenceState=r,referenceTangentState=tr)
    with forward_ad.dual_level():
        sph.computeCRKFactorsJVP(q,domain,sph.KernelFunctions.Wendland2,tq,
            adjacency=graph,referenceState=r,referenceTangentState=tr)
    epsilon=1e-5
    plus=sph.prepareCRK(shifted(q,tq,epsilon),domain,sph.KernelFunctions.Wendland2,
                       adjacency=graph,referenceState=shifted(r,tr,epsilon))
    minus=sph.prepareCRK(shifted(q,tq,-epsilon),domain,sph.KernelFunctions.Wendland2,
                        adjacency=graph,referenceState=shifted(r,tr,-epsilon))
    torch.testing.assert_close(darea,(plus.queryVolumes-minus.queryVolumes)/(2*epsilon),rtol=2e-5,atol=1e-6)
    for name in ('A','B','gradA','gradB'):
        torch.testing.assert_close(getattr(dstate,name),
            (getattr(plus.crkState,name)-getattr(minus.crkState,name))/(2*epsilon),rtol=2e-4,atol=2e-5)
    # Explicit corrections and their products compose in the same dispatcher.
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient,
                                       gradientMode=sph.GradientScheme.Difference)
    reference_volumes=r.masses/r.densities
    volume_tangent=tr.masses/r.densities-r.masses*tr.densities/r.densities.square()
    matrix=torch.eye(2,dtype=q.positions.dtype,device=device).repeat(len(q.positions),1,1)
    matrix_tangent=matrix*.02
    actual=warpOperationJVP(q,properties,domain,queryTangentState=tq,referenceTangentState=tr,
        queryValues=q.positions,referenceValues=r.positions,tangentQueryValues=tq.positions,
        tangentReferenceValues=tr.positions,referenceParticles=r,adjacency=graph,
        referenceVolumes=reference_volumes,tangentReferenceVolumes=volume_tangent,
        crkState=state,crkTangentState=dstate,renormalizationState=sph.RenormalizationState(matrix),
        renormalizationTangentState=sph.RenormalizationTangentState(matrix_tangent))
    def gradient(sign,preparation):
        query=shifted(q,tq,sign*epsilon);reference=shifted(r,tr,sign*epsilon)
        return sph.warpOperation(query,properties,domain,query.positions,reference.positions,
            referenceParticles=reference,adjacency=graph,crkState=preparation.crkState,
            referenceVolumes=reference.masses/reference.densities,
            renormalizationState=sph.RenormalizationState(matrix+sign*epsilon*matrix_tangent))
    torch.testing.assert_close(actual,(gradient(1,plus)-gradient(-1,minus))/(2*epsilon),rtol=3e-4,atol=3e-5)
    assert callable(computeSPHGradientGeometryJVP) and callable(warpOperationHVP) and callable(sphOperation_warp)
