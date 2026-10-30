"""Cross-backend standard-primal parity, including declared tensor-axis layout."""
import os
import pytest
import torch
if os.environ.get('WARPSPHCORE_NATIVE_PARITY')!='1':pytest.skip('opt-in Warp reference process',allow_module_level=True)
import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import sphOperation as native, radiusSearch

OPS=[sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient,sph.WarpOperation.Divergence,
     sph.WarpOperation.Curl,sph.WarpOperation.Laplacian,sph.WarpOperation.Covariance]

@pytest.fixture(scope='module',autouse=True)
def primal_reference_modules():
    # This process compares primal values only. Avoid compiling Warp's large
    # reverse programs for every tensor rank; native VJPs have separate AD oracles.
    modules=[f'warpSPHCore.coreOperations.native.wp_{op.name.lower()}' for op in OPS]
    previous=[wp.get_module_options(module=name) for name in modules]
    for name in modules:wp.set_module_options({'enable_backward':False},module=name)
    yield
    for name,options in zip(modules,previous):wp.set_module_options(options,module=name)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('op',OPS)
def test_standard_primal_matches_warp(device,dim,rank,scheme,op):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('requires GPU')
    if op==sph.WarpOperation.Divergence and rank==0:pytest.skip('vector field required')
    if op==sph.WarpOperation.Curl and rank==0 and dim>1:pytest.skip('vector field required')
    wp.init();dtype=torch.float64 if os.environ.get('warpSPHCore_PRECISION')=='float64' else torch.float32
    gen=torch.Generator().manual_seed(771)
    def rand(*size):return torch.rand(size,generator=gen,dtype=dtype).to(device)
    def state(n):return sph.ParticleState((rand(n,dim)-.5)*1.2,rand(n)*.4+.7,rand(n)+.3,
                                            (torch.arange(n,device=device)%4).int(),rand(n)+.7)
    q,r=state(7),state(5)
    d=sph.DomainDescription(q.positions.new_full((dim,),-1),q.positions.new_full((dim,),1),torch.tensor([True]*dim,device=device),dim)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=scheme,
        supportMode=sph.SupportScheme.MeanSymmetric,operationMode=sph.OperationDirection.TrueAllToToAll)
    qf,rf=rand(7,*([dim]*rank)),rand(5,*([dim]*rank))
    a=radiusSearch(q,d,r,prop.supportMode)
    a.paddedJ=a.paddedJ.to(sph.get_torch_index_dtype());a.edgeOffsets=a.edgeOffsets.to(sph.get_torch_offset_dtype())
    expected=sph.warpOperation(q,prop,d,qf,rf,adjacency=a,referenceParticles=r)
    actual=native(q,prop,d,qf,rf,adjacency=a,referenceParticles=r)
    tol=2e-4 if dtype==torch.float32 else 3e-10
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
