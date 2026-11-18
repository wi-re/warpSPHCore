"""Full interpolation products and independent fixed-CSR JVP equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .interpolate_contract import case as interpolate_case,oracle as interpolate_equation


def case(*args,volume=False,crk=False,**kwargs):
    q,r,d,p,g,v,states,_=interpolate_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    directions=tuple((v[i].detach().cos()*.03).requires_grad_() for i in (0,1,2,3,4,9,10,13,11,12))
    values=(*v,*directions)
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_interpolateJVP import computeSPHInterpolateGeometryJVP as compute
        def run(*x):
            qi,rj,domain=states(x[:10]);qt=sph.ParticleTangentState(x[14],x[16],None,None)
            rt=sph.ParticleTangentState(x[15],x[17],x[18],x[19]);options=dict(referenceParticles=rj,
                referenceValues=x[10],tangentReferenceValues=x[20],referenceVolumes=x[13] if volume else None,
                tangentReferenceVolumes=x[21] if volume else None)
            if backend=='native':
                if crk:options.update(crkState=sph.CRKState(x[11],x[12],None,None),crkTangentState=sph.CRKTangentState(x[22],x[23],None,None))
                return compute(qi,properties,domain,queryTangentState=qt,referenceTangentState=rt,adjacency=adjacency,**options)
            if crk:return composed_crk_reference(qi,rj,domain,properties,adjacency,x,volume)
            options['referenceValues']=x[10].reshape(len(rj.positions),-1)
            options['tangentReferenceValues']=x[20].reshape(len(rj.positions),-1)
            result=compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,referenceTangentState=rt,operationMode=properties.operationMode,**options)
            return result.reshape(len(qi.positions),*x[10].shape[1:])
        return run
    return q,r,d,p,g,values,states,functions


def oracle(values,domain,properties,graph,volume=False,crk=False):
    v=values[:14];t=[torch.zeros_like(x) for x in v]
    for slot,direction in zip((0,1,2,3,4,9,10,13,11,12),values[14:]):t[slot]=direction
    def primal(*x):
        x=list(x)
        if crk and not volume:x[13]=x[4]/x[9]
        if volume and not crk:x[11]=torch.ones_like(x[11]);x[12]=torch.zeros_like(x[12])
        return interpolate_equation(x,domain,properties,graph,volume or crk)
    return torch.autograd.functional.jvp(primal,v,tuple(t),create_graph=True)[1]


def composed_crk_reference(q,r,d,p,g,x,volume):
    """CRK extension oracle from supported raw Warp JVPs and Torch polynomials."""
    from warpSPHCore.coreOperations.wp_interpolateJVP import computeSPHInterpolateGeometryJVP
    output=[];domain=replace(d,min=d.min.detach(),max=d.max.detach());length=domain.max-domain.min
    for i in range(len(q.positions)):
        dx=q.positions[i:i+1]-r.positions
        dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
        dt=x[14][i:i+1]-x[15]
        poly=1+(x[12][i:i+1]*dx).sum(-1)
        dpoly=(x[23][i:i+1]*dx+x[12][i:i+1]*dt).sum(-1)
        C=x[11][i]*poly;dC=x[22][i]*poly+x[11][i]*dpoly
        shape=(len(r.positions),*([1]*(x[10].ndim-1)))
        f=x[10]*C.reshape(shape);df=x[20]*C.reshape(shape)+x[10]*dC.reshape(shape)
        qi=replace(q,positions=q.positions[i:i+1],supports=q.supports[i:i+1],masses=q.masses[i:i+1],kinds=q.kinds[i:i+1],densities=q.densities[i:i+1])
        gi=sph.AdjacencyList(None,g.paddedJ,g.numNeighbors[i:i+1],g.edgeOffsets[i:i+1],1,len(r.positions))
        value=computeSPHInterpolateGeometryJVP(qi,domain,p.kernel,p.supportMode,gi,
            sph.ParticleTangentState(x[14][i:i+1],x[16][i:i+1],None,None),r,
            sph.ParticleTangentState(x[15],x[17],x[18],x[19]),referenceValues=f.reshape(len(r.positions),-1),tangentReferenceValues=df.reshape(len(r.positions),-1),
            referenceVolumes=x[13] if volume else None,tangentReferenceVolumes=x[21] if volume else None,operationMode=p.operationMode)
        output.append(value.reshape(1,*x[10].shape[1:]))
    return torch.cat(output,0)


def boundary_case(device,dimension,dtype,kernel,support,delta,volume=False,crk=False,role='joint'):
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=support,volume=volume,crk=crk)
    d.periodic.zero_();q.kinds.zero_();r.kinds.zero_()
    g.paddedJ=torch.tensor([0,1],device=device,dtype=sph.get_torch_index_dtype())
    g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype())
    v=list(v);v[0]=torch.zeros_like(v[0]);v[0][:,0]=delta;v[1]=torch.zeros_like(v[1]);v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
    v[14]=torch.full_like(v[14],.03);v[15]=torch.full_like(v[15],-.02);v[16]=v[2]*.03;v[17]=v[3]*-.02
    if role!='joint':
        active=(16,17) if role=='support' else (18,19) if role=='mass_density' else (20,) if role=='field' else (21,)
        for k in range(14,24):
            if k not in active:v[k]=torch.zeros_like(v[k])
    v=tuple(x.detach().clone().requires_grad_(k not in (5,6)) for k,x in enumerate(v))
    return v,functions


def viscosity_crk_support_oracle(values, seed, volume):
    """Query-support bar and roundoff bound for the one-edge 1D Gather fixture."""
    from .operators import normalization

    n = len(values[0])
    v = [x.detach()[:n].double() for x in values]
    x, dt = (v[0] - v[1]).flatten(), (v[14] - v[15]).flatten()
    h, dh = v[2], v[16]
    q = x.abs() / h
    norm = float(values[0].new_tensor(normalization(sph.KernelFunctions.ViscosityKernel, 1)))
    eps = float(values[0].new_tensor(1e-7 if values[0].dtype == torch.float32 else 1e-15))
    poly = 1 + v[12].flatten() * x
    C = v[11] * poly
    dC = v[22] * poly + v[11] * (v[23].flatten() * x + v[12].flatten() * dt)
    f, df = v[10].reshape(n, -1), v[20].reshape(n, -1)
    cot = seed.detach().double().reshape(n, -1)
    V = v[13] if volume else v[4] / v[9]
    dV = v[21] if volume else v[18] / v[9] - v[4] * v[19] / v[9].square()
    dV_bound = dV.abs() if volume else v[18].abs() / v[9] + v[4].abs() * v[19].abs() / v[9].square()
    F = C[:, None] * f
    dF = C[:, None] * df + dC[:, None] * f
    bw = ((V[:, None] * dF + dV[:, None] * F) * cot).sum(-1)
    bdw = V * (F * cot).sum(-1)
    # The 1/(2r) term of W is independent of h. Cancel it analytically first.
    wh = norm / h.square() * (2 * q**3 - 3 * q.square() + 1)
    whh = norm / h**3 * (-10 * q**3 + 12 * q.square() - 2)
    gh = -x / x.abs().clamp_min(eps) * norm / h**3 * (6 * q * (1 - q))
    expected = bw * wh + bdw * (gh * dt + whh * dh)
    # The implemented reverse evaluates the singular terms separately.
    # Bound their positive paths before cancellation.
    shape = q**3 / 2 + q.square() + .5 / q + 1
    first = 1.5 * q.square() + .5 / q.square() + 2 * q
    second = 1 / q**3 + 3 * q + 2
    wh_bound = norm / h.square() * (shape + q * first)
    whh_bound = norm / h**3 * (2 * shape + 4 * q * first + q.square() * second)
    dF_bound = C.abs()[:, None] * df.abs() + dC.abs()[:, None] * f.abs()
    bw_bound = ((V.abs()[:, None] * dF_bound + dV_bound[:, None] * F.abs()) * cot.abs()).sum(-1)
    bdw_bound = V.abs() * (F.abs() * cot.abs()).sum(-1)
    unit = torch.finfo(values[0].dtype).eps
    gamma = 256 * unit / (1 - 256 * unit)
    return expected, gamma * (bw_bound * wh_bound + bdw_bound * ((gh * dt).abs() + whh_bound * dh.abs()))
