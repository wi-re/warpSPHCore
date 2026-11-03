"""Remaining standard operator options against independent equations and AD."""
from dataclasses import replace
import itertools
import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture, dense, OPS, normalization, differentiable_case
from _support.kernels import shape


def test_normalization_calibration_all_ops_and_covariance_counts(device):
    # Compute the lattice sum directly over integer lattice points, independently
    # of the production shell-count/cached host evaluator.
    q,r,domain,qf,rf=fixture(device,2,1)
    n_h=2.57
    nodes=torch.tensor(list(itertools.product(range(-3,4),repeat=2)),dtype=qf.dtype,device=device)
    ratio=nodes.norm(dim=-1)/n_h
    lattice=(normalization(sph.KernelFunctions.Wendland4,2)/n_h**2*
             torch.where(ratio<=1,shape(ratio,sph.KernelFunctions.Wendland4,2),0.)).sum()
    for op in OPS:
        props=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,
            gradientMode=sph.GradientScheme.Difference,calibrateNormalization=True,n_h=n_h)
        actual=sph.warpOperation(q,props,domain,qf,rf,referenceParticles=r)
        expected=dense(q,r,domain,props,qf,rf)/lattice
        torch.testing.assert_close(actual,expected,rtol=3e-9,atol=3e-9)
    covariance=replace(props,operation=sph.WarpOperation.Covariance)
    value,counts=sph.warpOperation(q,covariance,domain,qf,rf,referenceParticles=r,covarianceReturnNumNeighbors=True)
    plain,plain_counts=sph.warpOperation(q,replace(covariance,calibrateNormalization=False),domain,
        qf,rf,referenceParticles=r,covarianceReturnNumNeighbors=True)
    torch.testing.assert_close(value,plain/lattice)
    torch.testing.assert_close(counts,plain_counts)


def test_calibrated_field_map_transpose_and_backwards(device):
    q,r,domain,qf,rf=fixture(device,2,1)
    props=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient,
        gradientMode=sph.GradientScheme.Difference,calibrateNormalization=True,n_h=4.)
    operator=sph.linearizeFields(q,props,domain,qf,rf,referenceParticles=r)
    actual=sph.warpOperation(q,props,domain,qf,rf,referenceParticles=r)
    torch.testing.assert_close(operator.value,actual)
    dq,dr=torch.sin(qf),torch.cos(rf)
    tangent=operator.jvp(query=dq,reference=dr)
    bar=torch.cos(actual)
    qbar,rbar=operator.vjp(bar)
    torch.testing.assert_close((tangent*bar).sum(),(dq*qbar).sum()+(dr*rbar).sum())
    qf,rf=qf.requires_grad_(),rf.requires_grad_()
    value=sph.warpOperation(q,props,domain,qf,rf,referenceParticles=r)
    gradients=torch.autograd.grad(value,(qf,rf),bar)
    for got,want in zip(gradients,(qbar,rbar)):torch.testing.assert_close(got,want)


def test_positive_divergence_primals_pullbacks_and_jvps(device):
    for dim,rank,variant in ((1,1,sph.LaplacianScheme.Naive),(2,1,sph.LaplacianScheme.Brookshaw),
                             (3,1,sph.LaplacianScheme.Dot),(2,2,sph.LaplacianScheme.Default)):
        q,r,domain,qf,rf,dependencies=differentiable_case(device,dim,rank)
        props=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=sph.WarpOperation.Laplacian,
            gradientMode=(sph.GradientScheme.Symmetric if variant==sph.LaplacianScheme.Dot else sph.GradientScheme.Difference),
            laplacianMode=variant,positiveDivergence=True)
        actual=sph.warpOperation(q,props,domain,qf,rf,referenceParticles=r)
        expected=dense(q,r,domain,props,qf,rf,differentiable=True)
        torch.testing.assert_close(actual,expected,rtol=3e-9,atol=3e-9)
        bar=torch.cos(actual.detach())
        got=torch.autograd.grad(actual,dependencies,bar,allow_unused=True)
        want=torch.autograd.grad(expected,dependencies,bar,allow_unused=True,retain_graph=True)
        for a,b,original in zip(got,want,dependencies):
            a=torch.zeros_like(original) if a is None else a
            b=torch.zeros_like(original) if b is None else b
            torch.testing.assert_close(a,b,rtol=3e-8,atol=3e-8)
        with torch.autograd.forward_ad.dual_level():
            dualq=torch.autograd.forward_ad.make_dual(qf,torch.sin(qf.detach()))
            dualr=torch.autograd.forward_ad.make_dual(rf,torch.cos(rf.detach()))
            result=sph.warpOperation(q,props,domain,dualq,dualr,referenceParticles=r)
            _,tangent=torch.autograd.forward_ad.unpack_dual(result)
        _,reference_tangent=torch.autograd.functional.jvp(
            lambda a,b:dense(q,r,domain,props,a,b,differentiable=True),(qf,rf),
            (torch.sin(qf.detach()),torch.cos(rf.detach())))
        torch.testing.assert_close(tangent,reference_tangent,rtol=3e-9,atol=3e-9)
        with pytest.raises(ValueError,match='nonlinear'):
            sph.linearizeFields(q,props,domain,qf,rf,referenceParticles=r)
