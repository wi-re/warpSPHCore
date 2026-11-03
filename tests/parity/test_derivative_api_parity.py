"""Public native products against the readable Warp first-order/position blocks."""
from dataclasses import replace
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import warpOperationJVP as native_jvp
from warpSPHCore.coreOperations.native.nt_density_hvp import warpOperationHVP as native_hvp
from warpSPHCore.type_config import get_torch_precision


def test_public_products_and_legacy_density_block_parity(device):
    wp.init()
    dtype=get_torch_precision()
    for dimension in (1,2,3):
        t=torch.arange(9,dtype=dtype,device=device)
        x=torch.stack([.3*torch.sin(t*(d+.4)+d) for d in range(dimension)],1)
        h=.9+t*.01
        particles=sph.ParticleState(x,h,h*.7,(t.long()%3).int(),h+1)
        reference=replace(particles,positions=x*.93+.02,supports=h*1.1)
        domain=sph.DomainDescription(x.new_full((dimension,),-1),x.new_full((dimension,),1),
            torch.ones(dimension,dtype=torch.bool,device=device),dimension)
        graph=sph.buildVerletList(particles,domain,verletScale=1.,referenceParticles=reference,
                                  supportMode=sph.SupportScheme.SuperSymmetric)
        query_tangent=sph.ParticleTangentState(x.cos()*.03,h*.02,None,None)
        reference_tangent=sph.ParticleTangentState(x.sin()*.02,h*.03,h*.01,None)
        tolerance=4e-4 if dtype==torch.float32 else 3e-9
        for support in sph.SupportScheme:
            properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Density,
                supportMode=support,operationMode=sph.OperationDirection.FluidToAll)
            options=dict(adjacency=graph,referenceParticles=reference)
            expected=sph.warpOperationHVP(particles,properties,domain,x.cos(),tangentReferencePositions=x.sin(),**options)
            actual=native_hvp(particles,properties,domain,x.cos(),tangentReferencePositions=x.sin(),**options)
            torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
            # Warp's geometry JVP currently rejects PartialSymmetric while its
            # primal and explicit Hessian use maximum support for this enum.
            if support==sph.SupportScheme.PartialSymmetric:continue
            options.update(queryTangentState=query_tangent,referenceTangentState=reference_tangent)
            torch.testing.assert_close(native_jvp(particles,properties,domain,**options),
                sph.warpOperationJVP(particles,properties,domain,**options),rtol=tolerance,atol=tolerance)
