"""Native meshless geometry/gradient products retain continuous dependencies."""
from dataclasses import replace
import pytest
import torch
from torch.autograd import forward_ad
import warpSPHCore as sph
from warpSPHCore.mfm import MeshlessGeometry
from warpSPHCore.mfm.backend import Meshless as MeshlessNative


def fixture(device, dim, dtype, kernel=sph.KernelFunctions.Wendland2):
    axis=torch.arange(3,device=device,dtype=dtype)*.5+.2
    x=axis[:,None] if dim==1 else torch.cartesian_prod(*([axis]*dim))
    x=x+.013*torch.sin(torch.arange(x.numel(),device=device,dtype=dtype).reshape_as(x))
    n=len(x);h=x.new_full((n,),.84)
    domain=sph.DomainDescription(x.new_zeros(dim),x.new_full((dim,),1.5),torch.ones(dim,device=device,dtype=torch.bool),dim)
    p=sph.ParticleState(x,h,torch.ones_like(h),torch.zeros(n,device=device,dtype=torch.int32),torch.ones_like(h))
    displacement=x[None,:,:]-x[:,None,:]
    displacement=displacement-torch.floor(displacement/1.5+.5)*1.5
    membership=displacement.norm(dim=-1)<=torch.maximum(h[:,None],h[None,:])
    rows,columns=torch.nonzero(membership,as_tuple=True)
    counts=membership.sum(-1).to(torch.int32)
    offsets=(counts.cumsum(0)-counts).to(sph.get_torch_offset_dtype())
    adjacency=sph.AdjacencyList(rows.to(sph.get_torch_index_dtype()),columns.to(sph.get_torch_index_dtype()),counts,offsets,n,n,numEdges=len(columns))

    rho=1+.1*torch.sin(torch.arange(n,device=device,dtype=dtype))
    velocity=.2*torch.cos(3*x);pressure=1+.1*torch.sin(4*x[:,0])
    values=tuple(t.detach().clone().requires_grad_() for t in (x,h,domain.min,domain.max,rho,velocity,pressure))
    def run(v,reference=False,search=False):
        x,h,lo,hi,rho,vel,pres=v;d=replace(domain,min=lo,max=hi)
        if reference:
            geom=MeshlessGeometry.build(replace(p,positions=x,supports=h),d,kernel,adjacency=adjacency,closure='none')
            gradients=(geom.gradient(rho),geom.gradient(vel),geom.gradient(pres))
        else:
            geom=MeshlessNative.build(x,h,d,kernel,adjacency=None if search else adjacency,closure='none',backend='native')
            gradients=geom.gradients(rho,vel,pres)
        return geom.volume,geom.Einv,geom.cond,*gradients
    return values,run,adjacency


def products(output,values,seed):
    g=torch.autograd.grad(output,values,seed,allow_unused=True)
    return tuple(torch.zeros_like(v) if x is None else x for v,x in zip(values,g))


@pytest.mark.parametrize('kernel',[sph.KernelFunctions.Wendland2,sph.KernelFunctions.Wendland4,sph.KernelFunctions.Wendland6,sph.KernelFunctions.CubicSpline,sph.KernelFunctions.QuinticSpline,sph.KernelFunctions.B7])
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_public_geometry_and_gradients_independent_vjp_jvp(device,dim,dtype,kernel):
    values,run,_=fixture(device,dim,dtype,kernel)
    tol=4e-5 if dtype==torch.float32 else 5e-10
    a,b=run(values),run(values,reference=True)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=tol,rtol=tol)
    seed=tuple(torch.linspace(-.2,.8,x.numel(),device=device,dtype=dtype).reshape_as(x) for x in a)
    ga,gb=products(a,values,seed),products(b,values,seed)
    for k,(x,y) in enumerate(zip(ga,gb)):
        torch.testing.assert_close(x,y,atol=tol,rtol=tol,msg=lambda m:f'role {k}: {m}')
    directions=tuple(v.detach().cos()*.001 for v in values)
    with forward_ad.dual_level():
        result=run(tuple(forward_ad.make_dual(v,t) for v,t in zip(values,directions)))
        tangents=tuple(forward_ad.unpack_dual(x).tangent for x in result)
    torch.testing.assert_close(sum((t*s).sum() for t,s in zip(tangents,seed)),sum((g*t).sum() for g,t in zip(gb,directions)),atol=tol,rtol=tol)


def test_geometry_numerical_reverse_forward_gradcheck(device):
    values,run,_=fixture(device,2,torch.float64)
    assert torch.autograd.gradcheck(lambda *v:run(v),values,fast_mode=True,check_forward_ad=True,atol=4e-5,rtol=1e-3)


def test_discrete_search_keeps_position_and_support_products(device):
    values,run,_=fixture(device,1,torch.float64)
    a,b=run(values,search=True),run(values)
    seed=tuple(torch.ones_like(x) for x in a)
    for x,y in zip(products(a,values,seed),products(b,values,seed)):
        torch.testing.assert_close(x,y,atol=5e-10,rtol=5e-10)


def test_input_and_topology_lifetimes(device):
    values,run,adj=fixture(device,1,torch.float64)
    out=run(values)
    adj.paddedJ=adj.paddedJ.clone()
    with pytest.raises(RuntimeError,match='topology changed'):
        products(out,values,tuple(torch.ones_like(x) for x in out))
    out=run(values)
    with torch.no_grad():values[0].add_(.001)
    with pytest.raises(RuntimeError,match='inplace|in-place'):
        products(out,values,tuple(torch.ones_like(x) for x in out))


def limiter_fixture(device,dim,dtype):
    base,_,adj=fixture(device,dim,dtype)
    x,h,lo,hi,rho,vel,pres=base
    n=len(x)
    gradients=tuple((.3+torch.sin(torch.arange(n*k,device=device,dtype=dtype)*.73)).reshape(n,*shape).requires_grad_() for k,shape in ((dim,(dim,)),(dim*dim,(dim,dim)),(dim,(dim,))))
    beta=(rho.detach()*.3).requires_grad_()
    values=(*base,*gradients,beta)
    def run(v,reference=False):
        x,h,lo,hi,rho,vel,pres,gr,gv,gp,beta=v
        if not reference:
            from warpSPHCore.mfm.native.nt_mfm import mfmLimiterNative
            domain=sph.DomainDescription(lo,hi,torch.ones(dim,device=device,dtype=torch.bool),dim)
            state=sph.ParticleState(x,h,torch.ones_like(h),torch.zeros(n,device=device,dtype=torch.int32),torch.ones_like(h))
            props=sph.OperationProperties(kernel=sph.KernelFunctions.Wendland2)
            return mfmLimiterNative(state,props,domain,adj,rho,vel,pres,gr,gv,gp,beta)
        fields=torch.cat((rho[:,None],vel,pres[:,None]),1)
        grad=torch.cat((gr[:,None],gv,gp[:,None]),1)
        answer=[]
        choose=lambda a,b,high:torch.where(a>b if high else a<b,a,b)
        for i in range(n):
            high=low=mhigh=mlow=fields[i]
            for j in range(n):
                if i==j:continue
                dx=x[j]-x[i];length=hi-lo
                dx=dx-torch.floor((dx/length).detach()+.5)*length
                if bool((dx.norm()>torch.maximum(h[i],h[j])).detach()):continue
                mid=fields[i]+(grad[i]*(h[i]/(h[i]+h[j])*dx)).sum(-1)
                high=choose(high,fields[j],True);low=choose(low,fields[j],False)
                mhigh=choose(mhigh,mid,True);mlow=choose(mlow,mid,False)
            up=mhigh-fields[i];down=fields[i]-mlow
            upper=torch.stack([(high[c]-fields[i,c])/up[c] if bool((up[c]>1e-30).detach()) else up[c]*0+1e30 for c in range(dim+2)])
            lower=torch.stack([(fields[i,c]-low[c])/down[c] if bool((down[c]>1e-30).detach()) else down[c]*0+1e30 for c in range(dim+2)])
            answer.append(choose(beta[i]*choose(upper,lower,False),torch.ones_like(upper),False))
        result=torch.stack(answer)
        return result[:,0],result[:,1:-1],result[:,-1]
    return values,run


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_limiter_selected_extrema_geometry_products(device,dim,dtype):
    values,run=limiter_fixture(device,dim,dtype)
    a,b=run(values),run(values,True)
    tol=4e-5 if dtype==torch.float32 else 5e-10
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=tol,atol=tol)
    seed=tuple(torch.linspace(-.2,.8,x.numel(),device=device,dtype=dtype).reshape_as(x) for x in a)
    ga,gb=products(a,values,seed),products(b,values,seed)
    for k,(x,y) in enumerate(zip(ga,gb)):torch.testing.assert_close(x,y,rtol=tol,atol=tol,msg=lambda m:f'role {k}: {m}')
    directions=tuple(x.detach().cos()*.001 for x in values)
    with forward_ad.dual_level():
        tangents=tuple(forward_ad.unpack_dual(x).tangent for x in run(tuple(forward_ad.make_dual(x,t) for x,t in zip(values,directions))))
    torch.testing.assert_close(sum((t*s).sum() for t,s in zip(tangents,seed)),sum((g*t).sum() for g,t in zip(gb,directions)),rtol=tol,atol=tol)


def test_limiter_numerical_products(device):
    values,run=limiter_fixture(device,2,torch.float64)
    assert torch.autograd.gradcheck(lambda *v:run(v),values,fast_mode=True,check_forward_ad=True,atol=4e-5,rtol=1e-3)


def rates_fixture(device,dim,dtype,mode,order,kernel=sph.KernelFunctions.Wendland2,geometry_options=None,rate_options=None):
    values,_,adj=fixture(device,dim,dtype,kernel)
    geometry_options=geometry_options or {}
    rate_options=rate_options or {}
    def run(v,reference=False):
        x,h,lo,hi,rho,vel,pres=v
        domain=sph.DomainDescription(lo,hi,torch.ones(dim,device=device,dtype=torch.bool),dim)
        if reference:
            from warpSPHCore.mfm import mfmRates
            state=sph.ParticleState(x,h,torch.ones_like(h),torch.zeros(len(x),device=device,dtype=torch.int32),torch.ones_like(h))
            geom=MeshlessGeometry.build(state,domain,kernel,adjacency=adj,closure='none',**geometry_options)
            return mfmRates(geom,rho,vel,pres,1.4,dt=.01,mode=mode,order=order,**rate_options)[0]
        geom=MeshlessNative.build(x,h,domain,kernel,adjacency=adj,closure='none',backend='native',**geometry_options)
        return geom.rates(rho,vel,pres,1.4,dt=.01,mode=mode,order=order,**rate_options)
    return values,run


@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_rates_independent_conserved_vjp_jvp(device,mode,order,dim,dtype):
    values,run=rates_fixture(device,dim,dtype,mode,order)
    assert_rate_products(values,run,device,dtype,mode)


def assert_rate_products(values,run,device,dtype,mode):
    a,b=run(values),run(values,True)
    tol=4e-5 if dtype==torch.float32 else 5e-10
    torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    torch.testing.assert_close(a.sum(0),torch.zeros_like(a[0]),rtol=tol,atol=tol)
    if mode=='MFM':assert torch.equal(a[:,0],torch.zeros_like(a[:,0]))
    seed=torch.linspace(-.2,.8,a.numel(),device=device,dtype=dtype).reshape_as(a)
    ga,gb=products(a,values,seed),products(b,values,seed)
    for k,(x,y) in enumerate(zip(ga,gb)):torch.testing.assert_close(x,y,rtol=tol,atol=tol,msg=lambda m:f'role {k}: {m}')
    directions=tuple(x.detach().cos()*.001 for x in values)
    with forward_ad.dual_level():
        tangent=forward_ad.unpack_dual(run(tuple(forward_ad.make_dual(x,t) for x,t in zip(values,directions)))).tangent
    torch.testing.assert_close((tangent*seed).sum(),sum((g*t).sum() for g,t in zip(gb,directions)),rtol=tol,atol=tol)
    torch.testing.assert_close(tangent.sum(0),torch.zeros_like(tangent[0]),rtol=tol,atol=tol)


@pytest.mark.parametrize('mode',['MFM','MFV'])
def test_actual_rates_numerical_products(device,mode):
    values,run=rates_fixture(device,1,torch.float64,mode,2)
    assert torch.autograd.gradcheck(lambda *v:run(v),values,fast_mode=True,check_forward_ad=True,atol=4e-5,rtol=1e-3)


def test_explicit_unsupported_geometry_and_timestep_scope(device):
    values,_,adj=fixture(device,2,torch.float64)
    x,h,lo,hi,*_=values
    domain=sph.DomainDescription(lo,hi,torch.ones(2,device=device,dtype=torch.bool),2)
    with pytest.raises(NotImplementedError,match='forward-only'):
        MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,adjacency=adj,closure='project',backend='native')
    with pytest.raises(NotImplementedError,match='not qualified'):
        MeshlessNative.build(x,h,domain,sph.KernelFunctions.Gaussian,adjacency=adj,closure='none',backend='native')
    rank_one=torch.stack((x[:,0],torch.zeros_like(x[:,0])),1)
    with pytest.raises(NotImplementedError,match='full-rank'):
        MeshlessNative.build(rank_one,h,domain,sph.KernelFunctions.Wendland2,adjacency=adj,closure='none',backend='native')
    geom=MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,adjacency=adj,closure='none',backend='native')
    with pytest.raises(NotImplementedError,match='forward-only'):
        geom.timestep(values[4],values[5],values[6],1.4)


@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fixed_rank_prepared_geometry_and_rates_changed_input_graph(device,mode,dtype):
    if str(device)=='cpu':pytest.skip('CUDA graph execution contract')
    from warpSPHCore.mfm.native import nt_mfm as operators
    from warpSPHCore.autograd import SPHContext
    from warpSPHCore.mfm import conditionBeta
    values,_,adj=fixture(device,2,dtype)
    n=len(values[0]);dim=2
    def run(v):
        x,h,lo,hi,rho,vel,pres=v
        domain=sph.DomainDescription(lo,hi,torch.ones(dim,device=device,dtype=torch.bool),dim)
        state=sph.ParticleState(x,h,torch.ones_like(h),torch.zeros(n,device=device,dtype=torch.int32),torch.ones_like(h))
        launcher=operators.PreparedLauncher(SPHContext(state,sph.OperationProperties(kernel=sph.KernelFunctions.Wendland2),domain,adj))
        omega,moment,_=launcher.launch(operators._MOMENTS_SPEC,rowStart=0)
        scale=(omega*h*h)[:,None,None];scaled=moment/scale
        # Rank/conditioning is a precondition of this prepared graph, verified
        # by the public builder outside capture. No host convergence/rank read.
        inverse,_=torch.linalg.inv_ex(scaled,check_errors=False)
        cond=torch.linalg.matrix_norm(scaled)*torch.linalg.matrix_norm(inverse)/dim
        Einv=inverse/scale;bad=torch.zeros(n,device=device,dtype=torch.int32)
        gr,gv,gp=launcher.launch(operators._GRADIENTS_SPEC,rho=rho,vel=vel,pres=pres,Einv=Einv,deficient=bad,rowStart=0)
        ar,av,ap=launcher.launch(operators._LIMITER_SPEC,rho=rho,vel=vel,pres=pres,gRho=gr,gVel=gv,gP=gp,beta=conditionBeta(cond),rowStart=0)
        dm,dp,de=launcher.launch(operators._FLUX_SPEC,rho=rho,vel=vel,pres=pres,gRho=gr,gVel=gv,gP=gp,aRho=ar,aVel=av,aP=ap,
            vol=1/omega,Einv=Einv,lam=torch.zeros_like(vel),condBad=bad,gamma=1.4,dt=.01,mode=int(mode=='MFV'),rowStart=0)
        return torch.cat((dm[:,None],dp,de[:,None]),1)
    def public(v):
        x,h,lo,hi,rho,vel,pres=v
        domain=sph.DomainDescription(lo,hi,torch.ones(dim,device=device,dtype=torch.bool),dim)
        geom=MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,adjacency=adj,closure='none',backend='native')
        return geom.rates(rho,vel,pres,1.4,dt=.01,mode=mode)
    tol=4e-5 if dtype==torch.float32 else 5e-10
    torch.testing.assert_close(run(values),public(values),rtol=tol,atol=tol)
    directions=tuple(v.detach().cos()*.001 for v in values)
    def outputs():
        y=run(values);g=products(y,values,torch.ones_like(y))
        with forward_ad.dual_level():tangent=forward_ad.unpack_dual(run(tuple(forward_ad.make_dual(v,t) for v,t in zip(values,directions)))).tangent
        return y,*g,tangent
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):outputs()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=outputs()
    with torch.no_grad():values[0].add_(.001);values[1].mul_(1.01);values[4].add_(.01)
    # The public rebuild verifies the fixed-rank precondition after mutation.
    torch.testing.assert_close(run(values),public(values),rtol=tol,atol=tol)
    graph.replay();torch.cuda.synchronize()
    for a,b in zip(captured,outputs()):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


@pytest.mark.parametrize('kernel',[sph.KernelFunctions.Wendland2,sph.KernelFunctions.Wendland4,sph.KernelFunctions.Wendland6,sph.KernelFunctions.CubicSpline,sph.KernelFunctions.QuinticSpline,sph.KernelFunctions.B7])
@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_rates_smooth_kernel_composition(device,kernel,mode,dtype):
    values,run=rates_fixture(device,2,dtype,mode,2,kernel)
    assert_rate_products(values,run,device,dtype,mode)


@pytest.mark.parametrize('configuration',['fallback','area_cap','uncentred','mass_cap','no_retry','uncentred_frame'])
@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_rates_guard_and_configuration_products(device,configuration,mode,dtype):
    geometry_options={'face_cond_max':.1} if configuration=='fallback' else {'area_cap':True} if configuration=='area_cap' else {'centred_weights':False} if configuration=='uncentred' else {}
    rate_options={'massFluxLimit':1e-5} if configuration=='mass_cap' else {'retry':False} if configuration=='no_retry' else {'timeCentredFrame':False} if configuration=='uncentred_frame' else {}
    values,run=rates_fixture(device,2,dtype,mode,2,geometry_options=geometry_options,rate_options=rate_options)
    assert_rate_products(values,run,device,dtype,mode)
    if configuration=='mass_cap' and mode=='MFV':
        _,uncapped=rates_fixture(device,2,dtype,mode,2,rate_options={'massFluxLimit':0.})
        assert (run(values)[:,0]-uncapped(values)[:,0]).abs().max()>1e-4


@pytest.mark.parametrize('branch',['retry','pressure_floor'])
@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_rates_retry_and_pressure_floor_products(device,branch,mode,dtype):
    values,run=rates_fixture(device,1,dtype,mode,1)
    with torch.no_grad():
        values[4].fill_(1);values[6].fill_(1)
        if branch=='retry':
            values[4].copy_(1+.1*values[0][:,0]);values[6].copy_(1+.1*torch.sin(4*values[0][:,0]));values[5].copy_(4*(values[0]-.7))
        else:values[5].zero_();values[6].fill_(-.01)
    assert_rate_products(values,run,device,dtype,mode)
    if branch=='retry':
        _,without_retry=rates_fixture(device,1,dtype,mode,1,rate_options={'retry':False})
        assert (run(values)-without_retry(values)).abs().max()>1e-3
    if branch=='pressure_floor':
        y=run(values);g=products(y,values,torch.ones_like(y))
        assert all(torch.isfinite(x).all() for x in g)
        assert torch.count_nonzero(g[6])==0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_riemann_equal_sound_speed_uses_declared_second_operand(device,dtype):
    x=torch.tensor([[.1],[.7]],device=device,dtype=dtype);h=x.new_ones(2)
    domain=sph.DomainDescription(x.new_zeros(1),x.new_ones(1),torch.zeros(1,device=device,dtype=torch.bool),1)
    geom=MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,closure='none',backend='native')
    rho=x.new_tensor([1.,2.]).requires_grad_();pres=x.new_tensor([1.,2.]).requires_grad_();vel=torch.zeros_like(x)
    rate=geom.rates(rho,vel,pres,1.4,dt=0.,mode='MFM',order=1)
    seed=torch.zeros_like(rate);seed[0,-1]=1
    actual=products(rate,(rho,pres),seed)
    # Both sound speeds are equal. Native maximum(aL,aR) selects aR;
    # differentiating an averaged tie would be a different convention.
    right_sound=(1.4*pres[1]/rho[1]).sqrt()
    contact=(pres[1]-pres[0])/(-right_sound*(rho[0]+rho[1]))
    pressure=pres[0]-rho[0]*right_sound*contact
    area=(geom.volume[0]+geom.volume[1])/(x[1,0]-x[0,0])
    expected=products(-area*pressure*contact,(rho,pres),torch.ones((),device=device,dtype=dtype))
    tol=4e-5 if dtype==torch.float32 else 5e-10
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_constant_field_gradient_accumulators_do_not_seed_positions(device,dim,dtype):
    values,run,_=fixture(device,dim,dtype)
    values=(*values[:4],*(torch.ones_like(t).requires_grad_() for t in values[4:]))
    for output in (3,4,5):
        value=run(values)[output]
        assert value.count_nonzero()==0
        seed=torch.linspace(-.2,.8,value.numel(),device=device,dtype=dtype).reshape_as(value)
        derivative=torch.autograd.grad(value,values[:4],seed,allow_unused=True)
        for tensor in derivative:
            if tensor is not None:torch.testing.assert_close(tensor,torch.zeros_like(tensor),rtol=0,atol=0)


@pytest.mark.parametrize('mode',['MFM','MFV'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_positions_velocity_and_density_pressure_true_input_aliases(device,mode,dtype):
    values,run=rates_fixture(device,2,dtype,mode,2)
    unique=values[:5]
    # Three position roles (query/reference/velocity) and two thermodynamic
    # roles share actual Tensor objects, not just equal numerical values.
    def alias(v,reference=False):return run((*v,v[0],v[4]),reference)
    a,b=alias(unique),alias(unique,True)
    tol=4e-5 if dtype==torch.float32 else 5e-10
    torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    seed=torch.linspace(-.2,.8,a.numel(),device=device,dtype=dtype).reshape_as(a)
    ga,gb=products(a,unique,seed),products(b,unique,seed)
    for g,e in zip(ga,gb):torch.testing.assert_close(g,e,atol=tol,rtol=tol)
    directions=tuple(.001*t.detach().cos() for t in unique)
    with forward_ad.dual_level():
        tangent=forward_ad.unpack_dual(alias(tuple(forward_ad.make_dual(t,d) for t,d in zip(unique,directions)))).tangent
    torch.testing.assert_close((tangent*seed).sum(),sum((g*d).sum() for g,d in zip(gb,directions)),atol=tol,rtol=tol)
