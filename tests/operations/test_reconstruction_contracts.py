"""Production fits on incomplete stencils, independent references and frozen masks."""

from dataclasses import replace
import math

import pytest
import torch

from warpSPHCore import (DomainDescription, ParticleState, KernelFunctions,
    SupportScheme, OperationDirection, radiusSearchCompactHashMap)
from warpSPHCore.polyfit import PolyFit, Reconstructor, monomialExponents
from warpSPHCore.type_config import get_torch_precision


KERNEL = KernelFunctions.Wendland2


def state(positions):
    scalar = positions.new_ones(len(positions))
    return ParticleState(positions=positions, supports=scalar*1.4, masses=scalar*.04,
        densities=scalar, kinds=torch.zeros(len(positions), dtype=torch.int32, device=positions.device))


def domain(positions):
    dim = positions.shape[1]
    return DomainDescription(positions.new_full((dim,), -2), positions.new_full((dim,), 2),
                             torch.zeros(dim, device=positions.device, dtype=torch.bool), dim)


def cloud(device, dim=2, n=5):
    axis = torch.linspace(-.3, .3, n, dtype=get_torch_precision(), device=device)
    positions = torch.stack(torch.meshgrid(*([axis]*dim), indexing='ij'), -1).reshape(-1, dim)
    positions = positions + torch.arange(positions.numel(), dtype=axis.dtype, device=device).reshape_as(positions).sin()*.002
    return state(positions)


def traversal_for(p, dom, traversal, reference=None):
    if traversal == 'implicit':
        return None
    adjacency = radiusSearchCompactHashMap(p, dom, referenceParticles=reference, mode=SupportScheme.SuperSymmetric)
    return adjacency if traversal == 'csr' else adjacency.hashMap


def polynomial(positions, order, derivative=None):
    dim = positions.shape[1]
    derivative = (0,)*dim if derivative is None else derivative
    result = positions.new_zeros(len(positions))
    for k, exponent in enumerate(monomialExponents(dim, order)):
        term = positions.new_full((len(positions),), math.cos(k*.4)/ (1+k))
        for axis, (power, diff) in enumerate(zip(exponent, derivative)):
            if power < diff:
                term = term * 0
                break
            term = term * (math.factorial(power)/math.factorial(power-diff)) * positions[:,axis]**(power-diff)
        result = result + term
    return result


@pytest.mark.parametrize('dim', [1,2,3])
@pytest.mark.parametrize('order', [1,2,3])
@pytest.mark.parametrize('constant', [True,False])
@pytest.mark.parametrize('traversal', ['csr','grid','implicit'])
def test_production_polynomial_and_diffuse_derivatives(device, dim, order, constant, traversal):
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D SPH qualification runs on CUDA')
    p = cloud(device, dim, 7 if dim == 1 else 5)
    dom = domain(p.positions)
    fit = PolyFit.build(p, dom, KERNEL, order, constant=constant,
                       adjacency=traversal_for(p, dom, traversal))
    assert torch.isfinite(fit.Minv).all()
    ok = ~fit.deficient
    assert bool(ok.any())
    f = polynomial(p.positions, order)
    tol = .006 if f.dtype == torch.float32 else 2e-8
    torch.testing.assert_close(fit.values(f)[ok], f[ok], rtol=tol, atol=tol)
    grad = torch.stack([polynomial(p.positions, order, tuple(int(k==axis) for k in range(dim)))
                        for axis in range(dim)], -1)
    torch.testing.assert_close(fit.gradient(f)[ok], grad[ok], rtol=tol*2, atol=tol*2)
    alpha = (order,)+(0,)*(dim-1)
    torch.testing.assert_close(fit.derivative(f, alpha)[ok], polynomial(p.positions, order, alpha)[ok], rtol=tol*4, atol=tol*4)
    if order >= 2:
        exact = torch.zeros(len(f), dim, dim, dtype=f.dtype, device=device)
        for a in range(dim):
            for b in range(dim):
                exact[:,a,b] = polynomial(p.positions, order, tuple(int(k==a)+int(k==b) for k in range(dim)))
        torch.testing.assert_close(fit.hessian(f)[ok], exact[ok], rtol=tol*3, atol=tol*3)
        torch.testing.assert_close(fit.laplacian(f)[ok], exact.diagonal(dim1=-2,dim2=-1).sum(-1)[ok], rtol=tol*3, atol=tol*3)
    offsets = torch.full_like(p.positions, .01)
    indices = torch.arange(len(f), device=device)
    torch.testing.assert_close(fit.evaluate(f, indices, offsets)[ok], polynomial(p.positions+offsets,order)[ok], rtol=tol, atol=tol)


@pytest.mark.parametrize('geometry', ['empty','isolated','coincident','line'])
@pytest.mark.parametrize('constant', [True,False])
def test_degenerate_fits_are_finite_and_flagged(device, geometry, constant):
    n = {'empty':0,'isolated':1,'coincident':4,'line':9}[geometry]
    p = state(torch.zeros(n,2,dtype=get_torch_precision(),device=device))
    if geometry == 'line':
        p.positions[:,0] = torch.linspace(-.3,.3,n,device=device,dtype=p.positions.dtype)
    fit = PolyFit.build(p, domain(p.positions), KERNEL, 2, constant=constant)
    assert bool(fit.deficient.all()) and torch.isfinite(fit.Minv).all()
    field = p.positions[:,0].cos().requires_grad_()
    result = fit.gradient(field)
    assert torch.isfinite(result).all()
    g, = torch.autograd.grad(result.sum(), (field,))
    assert torch.isfinite(g).all()


@pytest.mark.parametrize('kind', ['teno','weno'])
@pytest.mark.parametrize('bundle', [False,True])
@pytest.mark.parametrize('unusable', ['condition','ghost'])
def test_no_usable_reconstruction_is_first_order_with_finite_ad(device, kind, bundle, unusable):
    from torch.autograd import forward_ad
    p = cloud(device, n=3)
    p.supports.fill_(.4)
    if unusable == 'ghost':
        p.kinds.fill_(2)
    rec = getattr(Reconstructor,kind)(p, domain(p.positions), KERNEL,
                                    cond_max=1. if unusable=='condition' else 1e10)
    i,j,d = rec.pairs()
    assert len(i) > 0
    f = torch.arange(len(p.positions),device=device,dtype=p.positions.dtype).sin()+2
    f = torch.stack((f,2*f),-1) if bundle else f
    leaf = f.detach().requires_grad_()
    left,right = rec.interfaceStates(leaf,i,j,d)
    assert rec.last_fallback.all()
    torch.testing.assert_close(left,leaf[i],rtol=0,atol=0)
    torch.testing.assert_close(right,leaf[j],rtol=0,atol=0)
    g, = torch.autograd.grad(left.sum()+right.sum(),leaf)
    count = torch.bincount(torch.cat((i,j)),minlength=len(f)).to(f.dtype)
    expected = count[:,None].expand_as(f) if bundle else count
    torch.testing.assert_close(g,expected,rtol=0,atol=0)
    tangent = f.cos()*.03
    with forward_ad.dual_level():
        l,r = rec.interfaceStates(forward_ad.make_dual(f,tangent),i,j,d)
        torch.testing.assert_close(forward_ad.unpack_dual(l).tangent,tangent[i],rtol=0,atol=0)
        torch.testing.assert_close(forward_ad.unpack_dual(r).tangent,tangent[j],rtol=0,atol=0)


@pytest.mark.parametrize('sizes', [(2,9),(13,9),(3,0),(0,9),(0,0)])
@pytest.mark.parametrize('explicit_volume', [False,True])
@pytest.mark.parametrize('traversal', ['csr','grid','implicit'])
def test_separate_mls_reference_field_shapes_and_duals(device,sizes,explicit_volume,traversal):
    from torch.autograd import forward_ad
    nq,nr = sizes
    r = cloud(device,n=3)
    r = replace(r, **{name:getattr(r,name)[:nr] for name in ('positions','supports','masses','densities','kinds')})
    qp = torch.arange(nq*2,device=device,dtype=r.positions.dtype).reshape(nq,2).sin()*.1
    q = state(qp)
    dom = domain(qp)
    volumes = dict(queryVolumes=q.masses/q.densities, referenceVolumes=r.masses/r.densities) if explicit_volume else {}
    fit = PolyFit.build(q,dom,KERNEL,2,referenceParticles=r,
                       adjacency=traversal_for(q,dom,traversal,r),**volumes)
    field = polynomial(r.positions,2).requires_grad_()
    value = fit.values(field)
    assert value.shape == (nq,) and fit.hessian(field).shape == (nq,2,2)
    assert torch.isfinite(value).all()
    if nr:
        torch.testing.assert_close(value,polynomial(qp,2),rtol=.001 if field.dtype==torch.float32 else 1e-9,atol=.001 if field.dtype==torch.float32 else 1e-9)
    else:
        assert torch.count_nonzero(value)==0
    with forward_ad.dual_level():
        result = fit.values(forward_ad.make_dual(field.detach(),torch.ones_like(field)))
        tangent = forward_ad.unpack_dual(result).tangent
        expected = fit.values(torch.ones_like(field))
        torch.testing.assert_close(tangent,expected,rtol=0,atol=0)
    grad, = torch.autograd.grad(value.sum(),field)
    assert grad.shape == field.shape and torch.isfinite(grad).all()


@pytest.mark.parametrize('direction', list(OperationDirection))
@pytest.mark.parametrize('traversal', ['csr','grid','implicit'])
def test_frozen_kind_geometry_and_field_duals_match_primal(device,direction,traversal):
    from torch.autograd import forward_ad
    p = cloud(device,n=3)
    offsets = p.positions.new_tensor([[-.012,-.008],[.012,-.008],[-.012,.008],[.012,.008]])
    p = state((p.positions[:,None,:]+offsets[None,:,:]).reshape(-1,2))
    p.supports.fill_(.5)
    p.kinds.copy_(torch.arange(len(p.positions),device=device,dtype=torch.int32)%4)
    # A centered periodic stencil keeps the float32 inverse well resolved.
    # Open/corner reproduction and deficiency have separate tests above.
    dom = DomainDescription(p.positions.new_full((2,),-.45),p.positions.new_full((2,),.45),
                            torch.ones(2,dtype=torch.bool,device=device),2)
    adj = traversal_for(p,dom,traversal)
    names = ('positions','supports','masses','densities')
    primals = [getattr(p,name).detach().requires_grad_() for name in names]
    scales = (.03,.03,.05,.04)
    tangents = [torch.arange(v.numel(),device=device,dtype=v.dtype).reshape_as(v).mul(1.7).sin()
                *scale*(1 if index==0 else v.detach())
                for index,(v,scale) in enumerate(zip(primals,scales))]
    field = polynomial(p.positions,2).requires_grad_()
    df = (11*p.positions[:,0]+4*p.positions[:,1]).sin()*.1
    def f(values,field):
        q = replace(p,**dict(zip(names,values)))
        fit = PolyFit.build(q,dom,KERNEL,2,adjacency=adj,operationMode=direction)
        assert bool((~fit.deficient[fit.num_nbrs>0]).all())
        return fit.gradient(field)
    out = f(primals,field)
    cot = torch.arange(out.numel(),device=device,dtype=out.dtype).reshape_as(out).sin()
    grads = torch.autograd.grad((out*cot).sum(),(*primals,field))
    with forward_ad.dual_level():
        duals = [forward_ad.make_dual(v.detach(),dv) for v,dv in zip(primals,tangents)]
        result = f(duals,forward_ad.make_dual(field.detach(),df))
        tangent = forward_ad.unpack_dual(result).tangent
    reverse = sum((g*dv).sum() for g,dv in zip(grads,(*tangents,df)))
    tol = .002 if out.dtype==torch.float32 else 2e-8
    torch.testing.assert_close((tangent*cot).sum(),reverse,rtol=tol,atol=tol)
    eps = .02 if out.dtype==torch.float32 else 1e-5
    plus = f([v+eps*dv for v,dv in zip(primals,tangents)],field+eps*df)
    minus = f([v-eps*dv for v,dv in zip(primals,tangents)],field-eps*df)
    torch.testing.assert_close(tangent,(plus-minus)/(2*eps),rtol=.02 if out.dtype==torch.float32 else 2e-6,atol=.002 if out.dtype==torch.float32 else 2e-7)


@pytest.mark.parametrize('configuration', ['teno-O4','teno-O5','teno-O6','weno-2','weno-3'])
def test_reconstruction_orders_preserve_constants_and_resolved_quadratics(device, configuration):
    p = cloud(device,n=11)
    p.supports.fill_(.12)
    dom = domain(p.positions)
    kind,order = configuration.split('-')
    rec = Reconstructor.teno(p,dom,KERNEL,order) if kind=='teno' else Reconstructor.weno(p,dom,KERNEL,degree=int(order))
    i,j,d = rec.pairs()
    constant = p.masses.new_full((len(p.masses),),1.7)
    left,right = rec.interfaceStates(constant,i,j,d)
    tol = .003 if constant.dtype==torch.float32 else 2e-7
    for result in (left,right):
        torch.testing.assert_close(result,torch.full_like(result,1.7),rtol=tol,atol=tol)
    field = polynomial(p.positions,2)
    left,right = rec.interfaceStates(field,i,j,d)
    point = p.positions[i] + d*.5
    for index,result in ((i,left),(j,right)):
        ok = ~rec.last_fallback[index]
        assert bool(ok.any())
        torch.testing.assert_close(result[ok],polynomial(point,2)[ok],rtol=tol*3,atol=tol*3)


def test_selected_rank_is_reported_even_with_many_neighbors(device):
    p = cloud(device)
    fit = PolyFit.build(p,domain(p.positions),KERNEL,2,rtol=.5,cond_max=float('inf'))
    assert bool((fit.num_nbrs >= fit.n_basis).all())
    assert bool(fit.deficient.all())


@pytest.mark.parametrize('kind',['teno','weno'])
def test_empty_reconstruction_shapes_and_3d_rejection(device,kind):
    p = state(torch.zeros(0,2,dtype=get_torch_precision(),device=device))
    rec = getattr(Reconstructor,kind)(p,domain(p.positions),KERNEL)
    field = p.masses.clone().requires_grad_()
    left,right = rec.interfaceStates(field,*rec.pairs())
    assert left.shape == right.shape == (0,) and rec.last_fallback.shape == (0,)
    g, = torch.autograd.grad(left.sum()+right.sum(),field)
    assert g.shape == (0,)
    p3 = state(torch.zeros(0,3,dtype=get_torch_precision(),device=device))
    with pytest.raises(NotImplementedError,match='sector reconstruction'):
        getattr(Reconstructor,kind)(p3,domain(p3.positions),KERNEL)


@pytest.mark.parametrize('unsupported', ['reference','volume','labfm','field_shape'])
def test_unsupported_role_tangents_and_fields_fail_before_launch(device,unsupported):
    from torch.autograd import forward_ad
    p = cloud(device,n=3)
    dom = domain(p.positions)
    if unsupported == 'labfm':
        with pytest.raises(NotImplementedError,match='LABFM'):
            PolyFit.build(p,dom,KERNEL,2,constant=False,referenceParticles=replace(p))
    elif unsupported == 'field_shape':
        fit = PolyFit.build(p,dom,KERNEL,2)
        with pytest.raises(ValueError,match='N_reference'):
            fit.coefficients(p.masses[:-1])
    else:
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(p.positions,torch.ones_like(p.positions))
            kwargs = {'referenceParticles':replace(p,positions=dual)} if unsupported=='reference' else {
                'referenceVolumes':forward_ad.make_dual(p.masses,torch.ones_like(p.masses))}
            with pytest.raises(NotImplementedError,match='geometry tangents'):
                PolyFit.build(p,dom,KERNEL,2,**kwargs)
