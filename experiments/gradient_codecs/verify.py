#!/usr/bin/env python3
"""Dense primal oracles, nontrivial CRK contraction and exact codec controls."""
import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('WARPSPHCORE_BACKEND','warp')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
import warpSPHCore as sph
from operator_runtime import Library, encode, request_for, packing_pitch, output_shape, NAMES, RAW32, RAW64, DELTA, BAND, NIBBLE
from common import properties,error


def dense(state,values,domain,volumes,correction,work):
    x=state.positions.double();n,d=x.shape
    dx=x[:,None,:]-x[None,:,:]
    dx-=torch.floor(dx+.5)*domain.periodic
    r=dx.norm(dim=-1);h=state.supports.double()
    q=r/h[:,None].clamp_min(1e-100);t=(1-q).clamp_min(0)
    c=(1.25,2.228169203286535,3.3422538049298023)[d-1]
    w=c*t.pow(3 if d==1 else 4)*(1+(3 if d==1 else 4)*q)/h[:,None].clamp_min(1e-100).pow(d)
    g=-(12 if d==1 else 20)*c*t.pow(2 if d==1 else 3)/h[:,None].clamp_min(1e-100).pow(d+2)
    w[h<=0]=0;g[h<=0]=0
    gradient=g[:,:,None]*dx
    if correction is not None:
        A=correction.A.double()[:,None,None];B=correction.B.double()[:,None,:]
        dot=1+(B*dx).sum(-1,keepdim=True)
        product=torch.einsum('ngc,njc->njg',correction.gradB.double(),dx)
        gradient=A*w[:,:,None]*B+A*dot*gradient+dot*w[:,:,None]*correction.gradA.double()[:,None,:]+A*w[:,:,None]*product
    volume=volumes.double()[None,:]
    if work==3:
        expected=-torch.einsum('njb,nj,njd->nbd',dx,volume.expand(n,-1),gradient)
    else:
        field=values.double().reshape(n,-1)
        weighted=(field[None,:,:]-field[:,None,:])*volume[:,:,None]
        if work==2:expected=(weighted*gradient).sum((1,2))
        else:
            expected=torch.einsum('njc,njd->ncd',weighted,gradient)
            if work==0:expected=expected[:,0,:]
    return expected,r<=h[:,None]


def fixture(n,d,dtype,periodic):
    gen=torch.Generator().manual_seed(261101+n+d)
    x=torch.rand((n,d),generator=gen,dtype=torch.float64).to(device='cuda',dtype=dtype)
    # Mixed periodicity and variable/zero support probe branches independently
    # from the uniform-support performance fixture.
    h=x.new_full((n,),.3);h[::61]=0
    state=sph.ParticleState(x,h,x.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),x.new_ones(n))
    flags=torch.full((d,),periodic,device='cuda',dtype=torch.bool)
    if d>1:flags[-1]=not periodic
    domain=sph.DomainDescription(x.new_zeros(d),x.new_ones(d),flags,d)
    vector=torch.randn((n,d),generator=gen,dtype=torch.float64).to(x)
    scalar=vector.sum(-1)
    # Deliberately nonsymmetric gradB catches the transpose/layout error even
    # when physical particle moments would make some matrix terms symmetric.
    correction=sph.CRKState((.8+.3*torch.rand(n,generator=gen,dtype=torch.float64)).to(x),
        (.2*torch.randn((n,d),generator=gen,dtype=torch.float64)).to(x),
        (.3*torch.randn((n,d),generator=gen,dtype=torch.float64)).to(x),
        (.4*torch.randn((n,d,d),generator=gen,dtype=torch.float64)).to(x))
    volumes=(.5+torch.rand(n,generator=gen,dtype=torch.float64)).to(x)/n
    return state,scalar,vector,domain,correction,volumes


def check(library,n,d,dtype,periodic,quick,sanitizer=False):
    state,scalar,vector,domain,correction,volumes=fixture(n,d,dtype,periodic)
    configurations=[(16,2,1,128)] if sanitizer else list(itertools.product((8,16,32),(1,2),(1,2),(128,) if quick else (128,256)))
    count=0
    for work,corrected in itertools.product(range(4),(False,True)):
        values=scalar if work==0 else vector
        selected_volumes=volumes if corrected else state.masses/state.densities
        expected,mask=dense(state,values,domain,selected_volumes,correction if corrected else None,work)
        counts=mask.sum(-1).int();offsets=torch.cat((counts.new_zeros(1,dtype=torch.int64),counts.cumsum(0)))
        indices=mask.nonzero()[:,1].contiguous();indices32=indices.int()
        output=values.new_empty(output_shape(work,n,d))
        common={name:getattr(state,name) for name in ('positions','supports','masses','densities')}
        common.update(values=values,lower=domain.min,upper=domain.max,periodic=domain.periodic,
                      packed=values.new_empty((n,packing_pitch(work,d))),output=output)
        plans={(encoding,chunk):encode(library,indices,counts,offsets,chunk,encoding)
               for encoding,chunk in itertools.product((DELTA,BAND,NIBBLE),(8,16,32))}
        for chunk,items,batch,threads in configurations:
            baseline=values.new_empty(output.shape)
            raw=request_for(dict(common,output=baseline),dict(counts=counts,row_offsets=offsets,indices=indices32),n,d,int(dtype==torch.float64),volumes,correction)
            library.run(raw,RAW32,work,corrected,chunk,items,batch,threads)
            torch.testing.assert_close(baseline.double(),expected,rtol=1e-4 if dtype==torch.float32 else 2e-10,
                                      atol=1e-4 if dtype==torch.float32 else 2e-10)
            for encoding in (RAW64,DELTA,BAND,NIBBLE):
                topology=dict(counts=counts,row_offsets=offsets,indices=indices) if encoding==RAW64 else plans[encoding,chunk].buffers
                request=request_for(common,topology,n,d,int(dtype==torch.float64),volumes,correction)
                if encoding!=RAW64:assert not request.pairs.indices and not request.pairs.row_offsets
                library.run(request,encoding,work,corrected,chunk,items,batch,threads)
                torch.testing.assert_close(output,baseline,rtol=0,atol=0)
                count+=1
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--sanitizer',action='store_true',help='Exercise every workload and codec with one representative schedule')
    args=parser.parse_args();library=Library(args.library);torch.set_num_threads(8)
    count=0
    for n,d,dtype,periodic in itertools.product((33,) if args.quick else (1,33,129),
            (3,) if args.quick else (1,2,3),(torch.float32,torch.float64),(False,True)):
        count+=check(library,n,d,dtype,periodic,args.quick,args.sanitizer)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):count+=check(library,17,2,torch.float32,True,True,args.sanitizer)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    args.output.write_text(json.dumps(dict(complete=True,configurations=count,nondefault_stream=True,
        coverage='Independent dense FP64 oracle; nonsymmetric CRK gradB; zero support, mixed periodicity; bit-exact matched CSR32/64/byte/two-bit/nibble controls.'),indent=2)+'\n')
    print('Passed',count,'configurations',flush=True)

if __name__=='__main__':main()
