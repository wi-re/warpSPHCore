"""Maximum-support values and inherited product branches on real core rows."""
from importlib import import_module
import pytest,torch,warp as wp
import warpSPHCore as sph
from parity.test_periodic_branch_contract import close

OWNERS=('density','interpolate','gradient','divergence','curl','laplacian','covariance','polyfit','crk_volume','crk_moments')
BRANCHES=('right_nan','left_nan','both_nan','tie','right_nan_outside')


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():wp.init()


def fixture(device,owner,branch):
    dimension=1+OWNERS.index(owner)%3
    if owner=='curl':dimension=3
    data=import_module('_support.'+owner+'_contract').case(device,dimension,sph.get_torch_precision(),support=sph.SupportScheme.PartialSymmetric)
    q,r,d,p,a,*rest=data
    if owner=='polyfit':e,v,functions=rest
    else:v,states,functions=rest
    with torch.no_grad():
        d.periodic.zero_();v[0].fill_(.2);v[1].fill_(.1);v[2].fill_(.8);v[3].fill_(.8)
        a.numNeighbors.zero_();a.numNeighbors[0]=1;a.paddedJ[0]=1
        if branch in ('right_nan','both_nan','right_nan_outside'):v[3][1]=float('nan')
        if branch in ('left_nan','both_nan'):v[2][0]=float('nan')
        if branch=='right_nan_outside':v[0][0].fill_(1.5)
    return v,functions


def outputs(y):return y if isinstance(y,tuple) else (y,)


def bars(y,v,weighted):
    live=[x for x in outputs(y) if x.requires_grad]
    seeds=[torch.linspace(.2,.9,x.numel(),device=x.device,dtype=x.dtype).reshape_as(x) if weighted else torch.ones_like(x) for x in live]
    return tuple(torch.zeros_like(x) if g is None else g for x,g in zip(v,torch.autograd.grad(live,v,seeds,allow_unused=True)))


@pytest.mark.parametrize('owner',OWNERS)
@pytest.mark.parametrize('branch',BRANCHES)
def test_maximum_support_value_and_products(device,owner,branch):
    v,functions=fixture(device,owner,branch)
    for weighted in (False,True):
        y,z=functions('native')(*v),functions('warp')(*v)
        for x,w in zip(outputs(y),outputs(z)):close(x,w)
        ga,gb=bars(y,v,weighted),bars(z,v,weighted)
        for k,(x,w) in enumerate(zip(ga,gb)):
            try:close(x,w)
            except AssertionError as error:raise AssertionError(f'role {k}: {error}') from error
        # Isolate the support direction. The pair is always query 0 -> ref 1,
        # so summed taped support bars give the selected kernel's total dh.
        # Warp's explicit support JVP uses >= (including its NaN fallback),
        # separately from adj_max's selected finite input and right-side ties.
        if all(torch.isfinite(x).all() for x in (*outputs(y),gb[2],gb[3])):
            reference_direction=.003 if owner=='polyfit' else .007
            t=tuple(torch.full_like(x,.003 if k==2 else reference_direction) if k in (2,3) else torch.zeros_like(x) for k,x in enumerate(v))
            with torch.autograd.forward_ad.dual_level():
                dual=tuple(torch.autograd.forward_ad.make_dual(x,d) for x,d in zip(v,t))
                product=[torch.autograd.forward_ad.unpack_dual(x).tangent for x in outputs(functions('native')(*dual))]
            seed=[torch.linspace(.2,.9,x.numel(),device=x.device,dtype=x.dtype).reshape_as(x) if weighted else torch.ones_like(x) for x in outputs(y)]
            # Polynomial monomials also normalize by query support. An equal
            # direction on both supports includes that independent derivative.
            selected=.003 if v[2][0]>=v[3][1] else reference_direction
            close(sum((j*b).sum() for j,b in zip(product,seed) if j is not None),(gb[2].sum()+gb[3].sum())*selected)


@pytest.mark.parametrize('owner',['density','polyfit'])
def test_smooth_maximum_actual_reverse_forward_gradcheck(device,owner):
    v,functions=fixture(device,owner,'tie')
    if v[0].dtype!=torch.float64:pytest.skip('FP64 numerical derivative contract')
    with torch.no_grad():v[3].fill_(1.)
    assert torch.autograd.gradcheck(lambda *x:functions('native')(*x),v,fast_mode=True,check_forward_ad=True)


@pytest.mark.parametrize('owner',['density','interpolate','gradient','laplacian'])
@pytest.mark.parametrize('branch',['right_nan','left_nan','tie'])
def test_explicit_maximum_product_reverse_composed_warp(device,owner,branch):
    from dataclasses import replace
    module=import_module('_support.'+owner+'_explicit_contract')
    q,r,d,p,a,v,states,functions=module.case(device,2,sph.get_torch_precision(),support=sph.SupportScheme.PartialSymmetric)
    query_direction={'density':12,'interpolate':16,'gradient':22,'laplacian':22}[owner]
    reference_direction=query_direction+1
    with torch.no_grad():
        d.periodic.zero_();v[0].fill_(.2);v[1].fill_(.1);v[2].fill_(.8);v[3].fill_(.8)
        a.numNeighbors.zero_();a.numNeighbors[0]=1;a.paddedJ[0]=1
        if branch=='right_nan':v[3][1]=float('nan')
        elif branch=='left_nan':v[2][0]=float('nan')
    # Public Warp explicit products reject PartialSymmetric. For this single
    # fixed pair, compose its two support primitives into supported Gather:
    # max value/taped dependency and explicit >= direction stay independent.
    left,right=v[2],v[3][1].expand_as(v[2])
    choose=(~left.isnan())&(right.isnan()|(left>right))
    inputs=tuple(x for x in v if x.requires_grad)
    for weighted in (False,True):
        control=list(v);control[2]=torch.where(choose,left,right);control[query_direction]=torch.where(left>=right,v[query_direction],v[reference_direction][1].expand_as(left))
        y=functions('native')(*v)
        z=functions('warp',properties=replace(p,supportMode=sph.SupportScheme.Gather))(*control)
        close(y,z)
        seed=torch.linspace(.2,.9,y.numel(),device=y.device,dtype=y.dtype).reshape_as(y) if weighted else torch.ones_like(y)
        ga=torch.autograd.grad(y,inputs,seed,allow_unused=True)
        gb=torch.autograd.grad(z,inputs,seed,allow_unused=True)
        for k,(x,w,original) in enumerate(zip(ga,gb,inputs)):
            try:close(torch.zeros_like(original) if x is None else x,torch.zeros_like(original) if w is None else w)
            except AssertionError as error:raise AssertionError(f'active role {k}: {error}') from error


@pytest.mark.parametrize('branch',BRANCHES)
def test_fixed_geometry_density_position_block(device,branch):
    from _support.density_block_contract import case
    q,r,d,p,a,v,functions=case(device,2,sph.get_torch_precision(),support=sph.SupportScheme.PartialSymmetric)
    with torch.no_grad():
        d.periodic.zero_();q.positions.fill_(.2);r.positions.fill_(.1);q.supports.fill_(.8);r.supports.fill_(.8)
        a.numNeighbors.zero_();a.numNeighbors[0]=1;a.paddedJ[0]=1;a.numEdges=1
        if branch in ('right_nan','both_nan','right_nan_outside'):r.supports[1]=float('nan')
        if branch in ('left_nan','both_nan'):q.supports[0]=float('nan')
        if branch=='right_nan_outside':q.positions[0].fill_(1.5)
    for weighted in (False,True):
        y,z=functions('native')(*v),functions('warp')(*v);close(y,z)
        ga,gb=bars(y,v,weighted),bars(z,v,weighted)
        for x,w in zip(ga,gb):close(x,w)
        t=tuple(x.detach().cos()*.003 for x in v)
        with torch.autograd.forward_ad.dual_level():
            dual=tuple(torch.autograd.forward_ad.make_dual(x,d) for x,d in zip(v,t))
            j=torch.autograd.forward_ad.unpack_dual(functions('native')(*dual)).tangent
        if all(torch.isfinite(x).all() for x in (*outputs(y),*ga)):
            seed=torch.linspace(.2,.9,y.numel(),device=y.device,dtype=y.dtype).reshape_as(y) if weighted else torch.ones_like(y)
            close((j*seed).sum(),sum((g*d).sum() for g,d in zip(ga,t)))
