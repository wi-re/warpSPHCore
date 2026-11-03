"""Focused derivatives across independent coefficient, geometry and preparation paths.

Cases rotate kernel/layout/support choices rather than taking their Cartesian
product. Primitive-kernel qualification owns the exhaustive radial matrix.
"""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture
from _support.corrected import pair_terms
from test_corrected_primal import correction_fixture
from test_crk_preparation import full_csr

OPS = [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient, sph.WarpOperation.Divergence,
       sph.WarpOperation.Curl, sph.WarpOperation.Laplacian, sph.WarpOperation.Covariance]


def field_equation(q, r, domain, prop, fi, fj, inputs):
    volume, A, B, gradA, gradB, qo, ro, R, mass, rhoi, rhoj = [v.double() for v in inputs]
    fi, fj = fi.double(), fj.double()
    dx, distance, h, w, g, lap, allowed = pair_terms(q, r, domain, prop)
    d = domain.dim
    naive = prop.operation == sph.WarpOperation.Laplacian and prop.laplacianMode == sph.LaplacianScheme.Naive
    if not naive:
        polynomial = 1 + (B[:, None] * dx).sum(-1)
        cg = (A[:, None, None] * B[:, None] * w[..., None] +
              (A[:, None] * polynomial)[..., None] * g +
              (polynomial * w)[..., None] * gradA[:, None] +
              (A[:, None] * w)[..., None] * torch.einsum('idc,ijc->ijd', gradB, dx))
        g = torch.einsum('idc,ijc->ijd', R, cg)
        w = A[:, None] * polynomial * w
    g = g * allowed[..., None]
    if prop.operation == sph.WarpOperation.Covariance:
        return -torch.einsum('j,ija,ijb->iab', volume, dx, g)
    fshape = fi.shape
    fi, fj = fi.reshape(len(fi), -1), fj.reshape(len(fj), -1)
    if prop.operation == sph.WarpOperation.Interpolate:
        return torch.einsum('ij,j,jf->if', w * allowed, volume, fj).reshape(len(fi), *fshape[1:])
    fj = fj / ro[:, None]
    if prop.operation != sph.WarpOperation.Gradient: fi = fi / qo[:, None]
    mode = prop.gradientMode
    if prop.operation == sph.WarpOperation.Laplacian:
        scale = mass[None] * rhoi[:, None] / rhoj[None]**2 if mode == sph.GradientScheme.Symmetric else volume
        z = (fj[None] - fi[:, None]) * scale[..., None]
        if naive: out = torch.einsum('ijf,ij->if', z, lap * allowed)
        else:
            normal = dx / (distance + 1e-8 * h)[..., None]
            product = (g * normal).sum(-1)
            if prop.laplacianMode == sph.LaplacianScheme.Dot:
                blocks = z.reshape(*z.shape[:2], -1, d)
                projection = (blocks * normal[:, :, None]).sum(-1)
                out = ((blocks - (d + 2) * projection[..., None] * normal[:, :, None]) *
                       (product / (distance + 1e-8 * h))[:, :, None, None]).sum(1).flatten(1)
            else:
                denominator = distance + (1e-12 if prop.laplacianMode == sph.LaplacianScheme.Default else 1e-8) * h
                out = torch.einsum('ijf,ij->if', z, -2 * product / denominator)
        return out.reshape(fshape)
    if mode == sph.GradientScheme.Symmetric:
        z = mass[None, :, None] * (fi[:, None] / rhoi[:, None, None] +
                                   rhoi[:, None, None] * fj[None] / rhoj[None, :, None]**2)
    else:
        z = (fj[None] if mode == sph.GradientScheme.Naive else fj[None] - fi[:, None]
             if mode == sph.GradientScheme.Difference else fj[None] + fi[:, None]) * volume[None, :, None]
        if prop.operation == sph.WarpOperation.Divergence: z = z * rhoj[None, :, None] / rhoi[:, None, None]
    jacobian = torch.einsum('ijf,ijd->ifd', z, g)
    if prop.operation == sph.WarpOperation.Gradient: return jacobian.reshape(*fshape, d)
    if prop.operation == sph.WarpOperation.Divergence:
        if len(fshape) < 3: return jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)
        return jacobian.reshape(-1,d,d,d).diagonal(dim1=1 if prop.divergenceDotMode else 2,dim2=3).sum(-1)
    if d == 1: return jacobian.new_zeros(len(fi))
    if d == 2: return jacobian[:,d:,0] - jacobian[:,:d,1] if len(fshape) == 3 else (jacobian[:,1,0] - jacobian[:,0,1])[:,None]
    tensor = jacobian.reshape(len(fi),d,-1,d)
    return torch.stack((tensor[:,2,:,1]-tensor[:,1,:,2], tensor[:,0,:,2]-tensor[:,2,:,0],
                        tensor[:,1,:,0]-tensor[:,0,:,1]),1).reshape(fshape)


def coefficient_case(device, case):
    operation = OPS[case % len(OPS)]
    variation = case // len(OPS)
    dimension = 1 + variation % 3
    rank = variation % 3
    if operation in (sph.WarpOperation.Divergence, sph.WarpOperation.Curl): rank = max(1, rank)
    if operation == sph.WarpOperation.Laplacian: rank = max(1, rank)
    dtype = torch.float32 if variation % 2 else torch.float64
    q, r, domain, fi, fj = fixture(device, dimension, rank)
    def cast(state): return replace(state, **{name:getattr(state,name).to(dtype) for name in
        ('positions','supports','masses','densities')})
    q, r = cast(q), cast(r)
    domain = replace(domain,min=domain.min.to(dtype),max=domain.max.to(dtype))
    fi, fj = fi.to(dtype), fj.to(dtype)
    state, (_, volume), omega, renorm = correction_fixture(q, r)
    values = [volume,state.A,state.B,state.gradA,state.gradB,omega.queryOmegas,
              omega.referenceOmegas,renorm.renormalizationMatrices,r.masses,q.densities,r.densities]
    values = [x.detach().requires_grad_() for x in values]
    prop = sph.OperationProperties(list(sph.KernelFunctions)[case % 15], operation=operation,
        gradientMode=list(sph.GradientScheme)[variation % 4],
        laplacianMode=list(sph.LaplacianScheme)[variation % 4],
        divergenceDotMode=bool(variation % 2), supportMode=list(sph.SupportScheme)[:5][variation % 5],
        operationMode=sph.OperationDirection.TrueAllToToAll)
    graph = full_csr(q,r,torch.int32 if variation % 2 else torch.int64,
                     torch.int32 if variation % 3 else torch.int64)
    def native(*v):
        volume,A,B,ga,gb,qo,ro,R,mass,qr,rr=v
        options = dict(referenceVolumes=volume, crkState=sph.CRKState(A,B,ga,gb),
                       referenceParticles=replace(r,masses=mass,densities=rr), adjacency=graph)
        if operation != sph.WarpOperation.Interpolate:
            options['renormalizationState']=R
        if operation not in (sph.WarpOperation.Interpolate,sph.WarpOperation.Covariance):
            options['gradHState']=sph.GradHState(qo,ro)
        if operation == sph.WarpOperation.Divergence: options['consistentDivergence']=True
        return sph.warpOperation(replace(q,densities=qr),prop,domain,fi,fj,**options)
    return native, lambda *v: field_equation(q,r,domain,prop,fi,fj,v), values, dtype


@pytest.mark.parametrize('case', range(60))
def test_coefficient_products(device, case):
    native, oracle, values, dtype = coefficient_case(device,case)
    actual, expected = native(*values), oracle(*values)
    tolerance = 3e-3 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.double(),expected.double(),rtol=tolerance,atol=tolerance)
    cotangent = actual.detach().cos().contiguous()
    gradients = torch.autograd.grad(actual,values,cotangent,allow_unused=True)
    if expected.requires_grad:
        wanted = torch.autograd.grad(expected,values,cotangent.double(),allow_unused=True)
    else: wanted = [None]*len(values)
    for x,y,v in zip(gradients,wanted,values):
        x = torch.zeros_like(v) if x is None else x
        y = torch.zeros_like(v).double() if y is None else y
        torch.testing.assert_close(x.double(),y.double(),rtol=tolerance,atol=tolerance)
    directions = [v.detach().cos().contiguous() for v in values]
    with torch.autograd.forward_ad.dual_level():
        dual = [torch.autograd.forward_ad.make_dual(v.detach(),t) for v,t in zip(values,directions)]
        product = torch.autograd.forward_ad.unpack_dual(native(*dual)).tangent
    _, expected_jvp = torch.autograd.functional.jvp(oracle,tuple(values),tuple(directions))
    torch.testing.assert_close(product.double(),expected_jvp.double(),rtol=tolerance,atol=tolerance)
    right = sum((g*t).sum() for g,t in zip(gradients,directions) if g is not None)
    torch.testing.assert_close((product*cotangent).sum(),torch.as_tensor(right).to(product),rtol=tolerance,atol=tolerance)


def geometry_case(device, case, dtype=torch.float64):
    operation = OPS[case % 6]
    variation = case // 6
    dimension = 1 + variation % 3
    q, r, domain, fi, fj = fixture(device, dimension, 1 + variation % 2)
    if dtype != torch.float64:
        def cast(state): return replace(state, **{key:getattr(state,key).to(dtype) for key in
            ('positions','supports','masses','densities')})
        q, r = cast(q), cast(r)
        domain = replace(domain,min=domain.min.to(dtype),max=domain.max.to(dtype))
        fi, fj = fi.to(dtype), fj.to(dtype)
    state, (_, volume), omega, R = correction_fixture(q, r)
    # Include unwrapped positions so bound derivatives carry nonzero winding.
    q.positions[:, 0] += 4
    r.positions[:, 0] -= 6
    from warpSPHCore.native._autograd import REVERSE_KERNELS
    smooth = [kernel for kernel in sph.KernelFunctions if kernel.value in REVERSE_KERNELS]
    prop = sph.OperationProperties(smooth[case % len(smooth)], operation=operation,
        gradientMode=list(sph.GradientScheme)[variation % 4],
        laplacianMode=list(sph.LaplacianScheme)[variation % 4],
        supportMode=list(sph.SupportScheme)[:5][variation % 5],
        divergenceDotMode=bool(variation % 2), operationMode=sph.OperationDirection.TrueAllToToAll)
    graph = full_csr(q, r)
    def function(qx,rx,qh,rh,lower,upper):
        options = dict(referenceParticles=replace(r,positions=rx,supports=rh),
                       adjacency=graph,referenceVolumes=volume,crkState=state)
        if operation != sph.WarpOperation.Interpolate: options['renormalizationState'] = R
        if operation not in (sph.WarpOperation.Interpolate,sph.WarpOperation.Covariance): options['gradHState'] = omega
        if operation == sph.WarpOperation.Divergence: options['consistentDivergence'] = True
        return sph.warpOperation(replace(q,positions=qx,supports=qh),prop,
            replace(domain,min=lower,max=upper),fi,fj,**options)
    values = [x.detach().clone().requires_grad_() for x in
              (q.positions,r.positions,q.supports,r.supports,domain.min,domain.max)]
    return function, values


@pytest.mark.parametrize('case', [*range(36), 52, 88])
def test_corrected_geometry_products(device, case):
    function, values = geometry_case(device,case)
    directions = [(x.detach()*1.13+.4).cos().contiguous() for x in values]
    result = function(*values)
    cotangent = result.detach().sin().contiguous()
    gradients = torch.autograd.grad(result,values,cotangent,allow_unused=True)
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(values,directions)]
        product = torch.autograd.forward_ad.unpack_dual(function(*duals)).tangent
    # Differentiate the existing primal implementation, independent of the new
    # analytic derivative algebra, both jointly and one input role at a time.
    eps = 2e-6
    with torch.no_grad():
        plus = function(*[x+eps*t for x,t in zip(values,directions)])
        minus = function(*[x-eps*t for x,t in zip(values,directions)])
        torch.testing.assert_close(product,(plus-minus)/(2*eps),rtol=2e-5,atol=3e-6)
        for slot,(gradient,direction) in enumerate(zip(gradients,directions)):
            if gradient is None: continue
            forward, reverse = list(values), list(values)
            forward[slot] = values[slot] + eps*direction
            reverse[slot] = values[slot] - eps*direction
            expected = ((function(*forward)-function(*reverse))*cotangent).sum()/(2*eps)
            torch.testing.assert_close((gradient*direction).sum(),expected,rtol=2e-5,atol=4e-6)
    right = sum((g*t).sum() for g,t in zip(gradients,directions) if g is not None)
    torch.testing.assert_close((product*cotangent).sum(),right,rtol=3e-10,atol=3e-9)


def test_corrected_geometry_auto_search_and_counts(device):
    q, r, domain, fi, fj = fixture(device,2,1)
    state, (_, volume), _, _ = correction_fixture(q,r)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Covariance)
    q = replace(q,positions=q.positions.requires_grad_())
    value, counts = sph.warpOperation(q,prop,domain,referenceParticles=r,
        referenceVolumes=volume,crkState=state,covarianceReturnNumNeighbors=True)
    assert counts.dtype == torch.int32 and not counts.requires_grad
    assert torch.isfinite(torch.autograd.grad(value.sum(),q.positions)[0]).all()
    with torch.autograd.forward_ad.dual_level():
        query = replace(q,positions=torch.autograd.forward_ad.make_dual(q.positions.detach(),torch.ones_like(q.positions)))
        value, counts = sph.warpOperation(query,prop,domain,referenceParticles=r,
            referenceVolumes=volume,crkState=state,covarianceReturnNumNeighbors=True)
        assert torch.isfinite(torch.autograd.forward_ad.unpack_dual(value).tangent).all()
        assert torch.autograd.forward_ad.unpack_dual(counts).tangent is None


def preparation_case(device, dimension, pass_name, aliased=False, dtype=torch.float64):
    from test_crk_preparation import cloud
    q, r, domain = cloud(device,dimension,dtype)
    if aliased: q = r
    graph, reference_graph = full_csr(q,r), full_csr(r,r)
    state, (_, volume), _, _ = correction_fixture(q,r)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        supportMode=sph.SupportScheme.Scatter,operationMode=sph.OperationDirection.TrueAllToToAll)
    # Nonconstant directions distinguish query/reference and support ownership.
    values = [q.positions,r.positions,q.supports,r.supports,r.masses,volume,state.A,state.B]
    values = [x.detach().clone().requires_grad_() for x in values]
    def function(qx,rx,qh,rh,mass,vol,A,B):
        left, right = replace(q,positions=qx,supports=qh),replace(r,positions=rx,supports=rh,masses=mass)
        if pass_name == 'volume': return sph.computeCRKVolume(left,properties,domain,graph,right)
        if pass_name == 'moments':
            moments = sph.computeCRKMoments(left,properties,domain,None,vol,graph,right)
            return torch.cat([x.reshape(len(qx),-1) for x in moments[:6]],1)
        if pass_name == 'density':
            return sph.computeCRKDensity(left,properties,domain,sph.CRKState(A,B),None,vol,graph,right)
        prepared = sph.prepareCRK(left,domain,properties.kernel,properties.operationMode,graph,right,
                                  referenceAdjacency=reference_graph)
        if pass_name == 'chain':
            field_i = (left.positions**2).sum(-1)
            field_j = (right.positions**2).sum(-1)
            return sph.warpOperation(left,replace(properties,operation=sph.WarpOperation.Gradient),domain,
                field_i,field_j,referenceParticles=right,adjacency=graph,
                referenceVolumes=prepared.referenceVolumes,crkState=prepared.crkState)
        outputs = [prepared.queryVolumes,prepared.referenceVolumes,prepared.consistencyDensity,
                   prepared.crkState.A,prepared.crkState.B,prepared.crkState.gradA,prepared.crkState.gradB]
        return torch.cat([x.reshape(-1) for x in outputs])
    return function,values


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('pass_name',['volume','moments','density','complete','chain'])
def test_preparation_products(device,dimension,pass_name):
    function, values = preparation_case(device,dimension,pass_name)
    result = function(*values)
    cotangent = (result.detach()*.17+.3).cos().contiguous()
    gradients = torch.autograd.grad(result,values,cotangent,allow_unused=True)
    directions = [(x.detach()*1.7+.31).sin().contiguous() for x in values]
    with torch.autograd.forward_ad.dual_level():
        dual = [torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(values,directions)]
        product = torch.autograd.forward_ad.unpack_dual(function(*dual)).tangent
    eps = 2e-6
    with torch.no_grad():
        forward = function(*[x+eps*t for x,t in zip(values,directions)])
        backward = function(*[x-eps*t for x,t in zip(values,directions)])
        finite = (forward-backward)/(2*eps)
        torch.testing.assert_close(product,finite,rtol=5e-5,atol=5e-7)
        for slot,(gradient,tangent) in enumerate(zip(gradients,directions)):
            if gradient is None: continue
            plus, minus = list(values),list(values)
            plus[slot],minus[slot] = values[slot]+eps*tangent,values[slot]-eps*tangent
            fd = ((function(*plus)-function(*minus))*cotangent).sum()/(2*eps)
            torch.testing.assert_close((gradient*tangent).sum(),fd,rtol=5e-5,atol=1e-6)
    right = sum((g*t).sum() for g,t in zip(gradients,directions) if g is not None)
    torch.testing.assert_close((product*cotangent).sum(),right,rtol=1e-9,atol=1e-9)


def test_preparation_limiter_empty_and_aliases(device):
    from test_crk_preparation import cloud
    q,r,domain = cloud(device,2,torch.float64)
    x = r.positions.detach().clone().requires_grad_()
    particles = replace(r,positions=x)
    result = sph.prepareCRK(particles,domain,sph.KernelFunctions.Wendland2,
                            sph.OperationDirection.TrueAllToToAll,full_csr(particles,particles))
    assert result.queryVolumes is result.referenceVolumes
    loss = result.crkState.A.sum()+result.crkState.B.square().sum()+result.consistencyDensity.sum()
    gradient, = torch.autograd.grad(loss,x)
    assert torch.isfinite(gradient).all()
    # Fewer than two samples selects identity correction with zero derivative.
    single = replace(q,positions=x[:1],supports=r.supports[:1],masses=r.masses[:1],kinds=r.kinds[:1])
    result = sph.prepareCRK(single,domain,sph.KernelFunctions.Wendland2,
                            sph.OperationDirection.TrueAllToToAll,full_csr(single,single))
    torch.testing.assert_close(result.crkState.A,torch.ones_like(result.crkState.A))
    loss = result.crkState.A.sum()+result.crkState.gradB.sum()
    assert torch.count_nonzero(torch.autograd.grad(loss,x)[0]) == 0


@pytest.mark.parametrize('empty_side',['query','reference'])
def test_empty_derivative_populations(device,empty_side):
    q,r,domain,fi,fj = fixture(device,2,1)
    state,(_,volume),_,_ = correction_fixture(q,r)
    if empty_side == 'query':
        q = replace(q,**{key:getattr(q,key)[:0] for key in ('positions','supports','masses','densities','kinds')})
        fi = fi[:0]
        state = sph.CRKState(*(value[:0] for value in (state.A,state.B,state.gradA,state.gradB)))
    else:
        r = replace(r,**{key:getattr(r,key)[:0] for key in ('positions','supports','masses','densities','kinds')})
        fj,volume = fj[:0],volume[:0]
    x,y = q.positions.requires_grad_(),r.positions.requires_grad_()
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient)
    output = sph.warpOperation(q,prop,domain,fi,fj,referenceParticles=r,
        adjacency=full_csr(q,r),referenceVolumes=volume,crkState=state)
    for bar in torch.autograd.grad(output.sum(),(x,y)):
        assert bar.shape[1] == 2 and torch.count_nonzero(bar) == 0
    preparation = sph.prepareCRK(q,domain,prop.kernel,referenceState=r,
        adjacency=full_csr(q,r),referenceAdjacency=full_csr(r,r))
    loss = preparation.crkState.A.sum()+preparation.consistencyDensity.sum()
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss,(x,y)))


def test_fixed_origin_force_coefficients_and_geometry_policy(device):
    q,r,domain,fi,fj = fixture(device,2,1)
    r = replace(q,positions=q.positions.clone())
    state,(_,volume),_,_ = correction_fixture(q,r)
    # A single self edge per row has a singular geometric Hessian for Spiky,
    # but fixed-origin coefficient and field derivatives remain well defined.
    graph = sph.AdjacencyList(None,torch.arange(len(fi),device=device,dtype=torch.int32),
        torch.ones(len(fi),device=device,dtype=torch.int32),
        torch.arange(len(fi),device=device,dtype=torch.int64),len(fi),len(fi))
    prop = sph.OperationProperties(sph.KernelFunctions.Spiky,operation=sph.WarpOperation.Gradient,
                                   operationMode=sph.OperationDirection.TrueAllToToAll)
    state.A.requires_grad_()
    def action(A,query=q):
        return sph.warpOperation(query,prop,domain,fi,fi+1,referenceParticles=r,
            adjacency=graph,referenceVolumes=volume,crkState=replace(state,A=A))
    result = action(state.A)
    assert torch.isfinite(torch.autograd.grad(result.sum(),state.A)[0]).all()
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(state.A.detach(),torch.ones_like(state.A))
        assert torch.isfinite(torch.autograd.forward_ad.unpack_dual(action(dual)).tangent).all()
    with pytest.raises(NotImplementedError,match='cusp/singular'):
        action(state.A.detach(),replace(q,positions=q.positions.requires_grad_()))


def test_owned_corrected_determinism_and_mutation(device):
    from warpSPHCore.native import CSRTopology
    q,r,domain,fi,fj = fixture(device,2,1)
    x, y = q.positions.requires_grad_(), r.positions.requires_grad_()
    state,(_,volume),_,_ = correction_fixture(q,r)
    topology = CSRTopology(full_csr(q,r))
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient)
    output = sph.warpOperation(q,prop,domain,fi,fj,referenceParticles=r,
        adjacency=topology,referenceVolumes=volume,crkState=state)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        a = torch.autograd.grad(output.sum(),(x,y),retain_graph=True)
        b = torch.autograd.grad(output.sum(),(x,y),retain_graph=True)
        for left,right in zip(a,b): torch.testing.assert_close(left,right,rtol=0,atol=0)
    finally: torch.use_deterministic_algorithms(previous)
    topology.adjacency.paddedJ.add_(0)
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):
        torch.autograd.grad(output.sum(),x)


@pytest.mark.parametrize('case',[1,10,14,52])
def test_geometry_float32(device,case):
    function,values = geometry_case(device,case,torch.float32)
    directions = [(x.detach()*.71+.2).cos().contiguous() for x in values]
    output = function(*values)
    cotangent = output.detach().cos().contiguous()
    bars = torch.autograd.grad(output,values,cotangent)
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(values,directions)]
        tangent = torch.autograd.forward_ad.unpack_dual(function(*duals)).tangent
    with torch.no_grad():
        eps = 1e-3
        forward = function(*[x+eps*t for x,t in zip(values,directions)])
        reverse = function(*[x-eps*t for x,t in zip(values,directions)])
        torch.testing.assert_close(tangent,(forward-reverse)/(2*eps),rtol=2e-2,atol=8e-3)
    torch.testing.assert_close((tangent*cotangent).sum(),sum((g*t).sum() for g,t in zip(bars,directions)),
                               rtol=4e-4,atol=1e-3)


def test_affine_crk_derivative_cancellation(device):
    from test_crk_preparation import cloud
    q,r,domain = cloud(device,2,torch.float64)
    q = replace(q,positions=q.positions.requires_grad_(),supports=q.supports.requires_grad_())
    r = replace(r,positions=r.positions.requires_grad_(),supports=r.supports.requires_grad_())
    graph = full_csr(q,r)
    prepared = sph.prepareCRK(q,domain,sph.KernelFunctions.Wendland2,
        sph.OperationDirection.TrueAllToToAll,graph,r,referenceAdjacency=full_csr(r,r))
    slope = q.positions.new_tensor([2.,-3.])
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient,
        supportMode=sph.SupportScheme.Scatter,operationMode=sph.OperationDirection.TrueAllToToAll)
    gradient = sph.warpOperation(q,prop,domain,1+q.positions@slope,1+r.positions@slope,
        referenceParticles=r,adjacency=graph,referenceVolumes=prepared.referenceVolumes,crkState=prepared.crkState)
    torch.testing.assert_close(gradient,slope.expand_as(gradient),rtol=1e-11,atol=1e-11)
    derivatives = torch.autograd.grad(gradient.sum(),(q.positions,r.positions,q.supports,r.supports),allow_unused=True)
    # Linear reproduction holds throughout this full-rank branch. Its gradient
    # therefore stays constant as geometry, supports and volumes vary together.
    for derivative in derivatives:
        if derivative is not None: torch.testing.assert_close(derivative,torch.zeros_like(derivative),rtol=0,atol=2e-10)


def test_reverse_over_preparation_jvp_is_rejected(device):
    from test_crk_preparation import cloud
    q,r,domain = cloud(device,2,torch.float64)
    x = q.positions.requires_grad_()
    with torch.autograd.forward_ad.dual_level():
        query = replace(q,positions=torch.autograd.forward_ad.make_dual(x,torch.ones_like(x)))
        result = sph.prepareCRK(query,domain,sph.KernelFunctions.Wendland2,
            sph.OperationDirection.TrueAllToToAll,full_csr(q,r),r,referenceAdjacency=full_csr(r,r))
        tangent = torch.autograd.forward_ad.unpack_dual(result.crkState.A).tangent
        with pytest.raises(NotImplementedError,match='first-order'):
            torch.autograd.grad(tangent.sum(),x)
    with torch.autograd.forward_ad.dual_level():
        query = replace(q,positions=torch.autograd.forward_ad.make_dual(x,torch.ones_like(x)))
        result = sph.prepareCRK(query,domain,sph.KernelFunctions.Wendland2,
            sph.OperationDirection.TrueAllToToAll,full_csr(q,r),r,referenceAdjacency=full_csr(r,r))
        with pytest.raises(NotImplementedError,match='first-order'):
            torch.autograd.grad(result.crkState.A.sum(),x)


def test_preparation_float32_chain(device):
    function,values = preparation_case(device,2,'chain',dtype=torch.float32)
    output = function(*values)
    cotangent = output.detach().cos().contiguous()
    gradients = torch.autograd.grad(output,values,cotangent,allow_unused=True)
    directions = [(x.detach()*1.2+.31).sin().contiguous() for x in values]
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(values,directions)]
        product = torch.autograd.forward_ad.unpack_dual(function(*duals)).tangent
    right = sum((g*t).sum() for g,t in zip(gradients,directions) if g is not None)
    torch.testing.assert_close((product*cotangent).sum(),right,rtol=5e-4,atol=1e-3)
    with torch.no_grad():
        eps = 1e-3
        forward = function(*[x+eps*t for x,t in zip(values,directions)])
        reverse = function(*[x-eps*t for x,t in zip(values,directions)])
        torch.testing.assert_close(product,(forward-reverse)/(2*eps),rtol=2e-2,atol=8e-3)
