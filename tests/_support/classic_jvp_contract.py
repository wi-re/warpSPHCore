"""Exercise ordinary forward AD and reverse its tangent, through public owners."""
import importlib
from dataclasses import replace
import torch
import warpSPHCore as sph

VARIANTS=['density','interpolate','gradient','divergence','curl',*list(sph.LaplacianScheme),'covariance']
FIELD_VARIANTS=[v for v in VARIANTS if v not in ('density','covariance')]


def case(device,variant,dimension=2,dtype=torch.float64,kernel=None,correction='plain',field_only=False,packed=False,side='both',counts=False,gradh=False,zero_bounds=False,positive=False,consistent=False,dot=False):
    family='laplacian' if isinstance(variant,sph.LaplacianScheme) else variant
    module=importlib.import_module('_support.'+family+'_contract')
    options=dict(dimension=dimension,dtype=dtype,kernel=kernel)
    if family not in ('density','covariance'):
        options['field_shape']=((2,3,dimension) if family in ('divergence','laplacian') else (dimension,2,3) if family=='curl' else (2,3)) if packed else (dimension,) if family in ('divergence','curl') or variant==sph.LaplacianScheme.Dot or (positive and family=='laplacian') else ()
    if family=='laplacian':options['mode']=variant
    q,r,d,p,g,v,states,_=module.case(device,**options)
    p=replace(p,positiveDivergence=positive,divergenceDotMode=dot)
    v=list(v)
    if not field_only:v[5]=v[5].detach();v[6]=v[6].detach()
    v=tuple(v)
    slots=(10,) if family=='interpolate' and field_only else (10,14) if field_only else (0,1,2,3,4) if family=='density' else (0,1,2,3,4,9,10,11,12,13) if family=='interpolate' else (0,1,2,3,4,8,9,10,11,12,13,14,15,16,18)
    if field_only and side!='both':slots=tuple(k for k in slots if k==(14 if side=='query' else 10))
    if zero_bounds:slots=(*slots,5,6)
    directions=tuple((torch.zeros_like(v[k]) if zero_bounds and k in (5,6) else v[k].detach().cos()*.03).requires_grad_() for k in slots)
    values=(*v,*directions);number=len(v)
    def functions(backend):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        def run(*all_values):
            primal=list(all_values[:number]);tangent=dict(zip(slots,all_values[number:]))
            with torch.autograd.forward_ad.dual_level():
                for k,t in (() if backend=='warp_explicit' else tangent.items()):primal[k]=torch.autograd.forward_ad.make_dual(primal[k],t)
                qi,rj,domain=states(primal[:10]);arguments=dict(referenceParticles=rj,adjacency=g)
                if family not in ('density','covariance'):arguments['referenceValues']=primal[10]
                if family not in ('density','covariance','interpolate'):arguments['queryValues']=primal[14]
                if family=='divergence':arguments['consistentDivergence']=consistent
                if family!='density' and correction in ('volume','combined'):arguments['referenceVolumes']=primal[13]
                if family!='density' and correction=='combined':
                    arguments['crkState']=sph.CRKState(primal[11],primal[12],None if family=='interpolate' else primal[15],None if family=='interpolate' else primal[16])
                    if family!='interpolate':arguments['renormalizationState']=sph.RenormalizationState(primal[18])
                if gradh:arguments['gradHState']=sph.GradHState(primal[19],primal[17])
                if backend=='warp_explicit':
                    arguments.update(tangentQueryValues=tangent.get(14),tangentReferenceValues=tangent.get(10))
                    return sph.warpOperationJVP(qi,p,domain,**arguments)
                if family=='covariance':arguments['covarianceReturnNumNeighbors']=counts
                output=sphOperation(qi,p,domain,backend=backend,**arguments)
                if counts:
                    matrix,discrete=output;assert torch.autograd.forward_ad.unpack_dual(discrete).tangent is None;output=matrix
                primal_output,result=torch.autograd.forward_ad.unpack_dual(output)
                if result is None:result=torch.zeros_like(primal_output)
            return result
        return run
    return q,r,d,p,g,values,states,functions
