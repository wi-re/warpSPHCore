"""Independent dense Torch oracle for primal/JVP/VJP and traversal contracts."""
import argparse
import json
from pathlib import Path
import torch

from template_runtime import Library,Plan,SLOTS,field_count,transpose,differentiable


def displacement(data):
    dx=data['qx'][:,None,:]-data['rx'][None,:,:]
    length=data['upper']-data['lower']
    wrapped=dx-length*torch.floor((dx+length/2)/length)
    return torch.where(data['periodic'],wrapped,dx)


def allowed(data):
    q=len(data['qx']);r=len(data['rx']);device=data['qx'].device
    qkind=data.get('qkind');rkind=data.get('rkind');direction=data.get('direction',0)
    if direction==0:return torch.ones((q,r),device=device,dtype=torch.bool)
    if direction==1:
        return ((qkind if qkind is not None else torch.zeros(q,device=device))==0)[:,None] & \
               ((rkind if rkind is not None else torch.zeros(r,device=device))==0)[None,:]
    raise NotImplementedError(direction)


def geometry(data):
    dx=displacement(data);r=torch.linalg.vector_norm(dx,dim=-1)
    h=data['h'][:,None];safe_h=torch.where(h>0,h,torch.ones_like(h))
    dim=dx.shape[-1];C=(1.25,2.228169203286535,3.3422538049298023)[dim-1]
    q=r/safe_h;t=1-q
    inside=(h>0)&(r<h)&allowed(data)
    w=C/safe_h**dim*t**(3 if dim==1 else 4)*(1+(3 if dim==1 else 4)*q)
    g=-(12 if dim==1 else 20)*C/safe_h**(dim+2)*t**(2 if dim==1 else 3)
    w=torch.where(inside,w,torch.zeros_like(w))
    grad=torch.where(inside[:,:,None],g[:,:,None]*dx,torch.zeros_like(dx))
    return dx,w,grad,inside


def oracle(data,work):
    dx,w,grad,inside=geometry(data);d=dx.shape[-1]
    volume=data.get('volume')
    if volume is None:volume=data['mass']/data['density']
    if work==0:return (w*data['mass']).sum(1,keepdim=True)
    if work in (1,5,8):
        values=data['rvalue'].reshape(len(data['rx']),-1)
        weighted=w*volume
        numerator=torch.einsum('ij,jf->if',weighted,values)
        if work==5:
            denominator=weighted.sum(1,keepdim=True)
            safe=torch.where(denominator!=0,denominator,torch.ones_like(denominator))
            numerator=torch.where(denominator!=0,numerator/safe,torch.zeros_like(numerator))
        return numerator
    if work in (2,3):
        qv=data['qvalue'].reshape(len(data['qx']),-1)
        rv=data['rvalue'].reshape(len(data['rx']),-1)
        return torch.einsum('ijf,ijd,j->ifd',rv[None]-qv[:,None],grad,volume).flatten(1)
    if work==4:return torch.einsum('ija,ijb,j->iab',-dx,grad,volume).flatten(1)
    if work==6:
        vw=w*volume
        m0=vw.sum(1,keepdim=True)
        m1=torch.einsum('ij,ija->ia',vw,dx)
        m2=torch.einsum('ij,ija,ijb->iab',vw,dx,dx).flatten(1)
        dm0=torch.einsum('ijb,j->ib',grad,volume)
        eye=torch.eye(d,device=dx.device,dtype=dx.dtype)
        dm1=(torch.einsum('ija,ijb,j->iab',dx,grad,volume)+vw.sum(1)[:,None,None]*eye).flatten(1)
        dm2=torch.einsum('ijc,ija,ijb,j->icab',grad,dx,dx,volume)
        dm2+=torch.einsum('ij,ac,ijb->icab',vw,eye,dx)
        dm2+=torch.einsum('ij,bc,ija->icab',vw,eye,dx)
        return torch.cat([m0,m1,m2,dm0,dm1,dm2.flatten(1)],1)
    if work==7:
        candidates=torch.where(inside,data['rvalue'].reshape(1,-1),-torch.inf)
        values=candidates.max(1).values
        return torch.where(inside.any(1),values,torch.zeros_like(values))[:,None]
    raise ValueError(work)


def fixture(dim,dtype,device,work,masked=False):
    gen=torch.Generator().manual_seed(1234+dim+work*101)
    q,r=23,31;f=field_count(work,dim)
    def random(shape):return torch.rand(shape,generator=gen,dtype=dtype).to(device)
    qx=random((q,dim))*1.4-.2;rx=random((r,dim))*1.4-.2
    rx[0]=qx[1]
    h=.12+.4*random((q,));h[0]=0
    data=dict(qx=qx,rx=rx,h=h,mass=.1+random((r,)),density=.6+random((r,)),
        qvalue=random((q,f)) if f else None,rvalue=random((r,f)) if f else None,
        lower=torch.zeros(dim,device=device,dtype=dtype),upper=torch.ones(dim,device=device,dtype=dtype),
        periodic=torch.tensor([d%2==0 for d in range(dim)],device=device,dtype=torch.bool))
    if work==7:data['rvalue'][:2]=.875
    if masked:
        data.update(qkind=torch.arange(q,device=device,dtype=torch.int32)%3,
                    rkind=torch.arange(r,device=device,dtype=torch.int32)%3,direction=1)
    inside=geometry(data)[-1]
    # Keep direction predicates in the operation, rather than prefiltering them
    # in topology. This explicitly tests source/query masking in both passes.
    distance=torch.linalg.vector_norm(displacement(data),dim=-1)
    membership=(data['h'][:,None]>0)&(distance<data['h'][:,None])
    rows,cols=torch.where(membership)
    counts=membership.sum(1).to(torch.int32)
    offsets=torch.cat([counts.new_zeros(1,dtype=torch.int64),counts.long().cumsum(0)])[:-1]
    csr=Plan(dict(indices=cols.int(),counts=counts,offsets=offsets))
    ranges=Plan(dict(offsets=torch.arange(q,device=device,dtype=torch.int64),
        counts=torch.ones(q,device=device,dtype=torch.int32),
        begins=torch.zeros(q,device=device,dtype=torch.int32),ends=torch.full((q,),r,device=device,dtype=torch.int32)),1)
    begin=torch.arange(0,q,8,device=device,dtype=torch.int32);tasks=len(begin)
    tiles=Plan(dict(offsets=torch.arange(tasks,device=device,dtype=torch.int64),
        counts=torch.ones(tasks,device=device,dtype=torch.int32),
        begins=torch.zeros(tasks,device=device,dtype=torch.int32),ends=torch.full((tasks,),r,device=device,dtype=torch.int32),
        query_begin=begin,query_end=torch.minimum(begin+8,torch.tensor(q,device=device,dtype=torch.int32))),2,tasks)
    return data,csr,ranges,tiles


def metric(actual,expected):
    error=(actual.double()-expected.double()).norm()
    scale=expected.double().norm().clamp_min(1)
    return float((error/scale).detach())


def check(actual,expected,tolerance,context):
    error=metric(actual,expected)
    assert bool(torch.isfinite(actual).all()) and error<tolerance,(context,error)
    return error


def verify(library,devices,quick=False):
    result=dict(primal=[],derivatives=[],autograd=[],edge_cases=[])
    for device in devices:
        for dtype in (torch.float32,torch.float64):
            tolerance=6e-5 if dtype==torch.float32 else 2e-11
            for dim in (1,2,3):
                for work in range(9):
                    for masked in ((False,) if quick else (False,True)):
                        data,csr,ranges,tiles=fixture(dim,dtype,device,work,masked)
                        expected=oracle(data,work)
                        for plan_name,plan in [('csr32',csr),('csr64',Plan(dict(csr.buffers,indices=csr.buffers['indices'].long()))),('ranges',ranges),('shared',tiles)]:
                            widths=(16,) if device=='cpu' or quick else (8,16,32)
                            for width in widths:
                                value=library.forward(data,plan,work,width=width,save=True,auxiliary=True)
                                error=check(value.values,expected,tolerance,(device,dtype,dim,work,masked,plan_name,width))
                                if value.counts is not None:assert torch.equal(value.counts,geometry(data)[-1].sum(1).int())
                                if value.winners is not None:
                                    inside=geometry(data)[-1]
                                    candidates=torch.where(inside,data['rvalue'].reshape(1,-1),-torch.inf)
                                    winner=candidates.argmax(1).int();winner[~inside.any(1)]=-1
                                    assert torch.equal(winner,value.winners)
                                result['primal'].append(dict(device=device,precision=str(dtype),dimension=dim,work=work,masked=masked,path=plan_name,width=width,error=error))
                        # A caller can request cache/diagnostics without an output
                        # tensor. Absence is a real null pointer, not a dummy array.
                        if work in (5,6,7):
                            optional=library.allocate(data,work,save=True,auxiliary=True)
                            optional.values=None
                            library.forward(data,csr,work,buffers=optional)
                            if optional.counts is not None:
                                assert torch.equal(optional.counts,geometry(data)[-1].sum(1).int())
                            if optional.cache is not None:
                                ordinary=library.forward(data,csr,work,save=True)
                                check(optional.cache,ordinary.cache,tolerance,'null output with cache')
                        # Optional apparent volume must replace m/rho, not add to it.
                        if work in (1,2,3,4,5,6,8):
                            alternate=dict(data,volume=data['mass']*.37)
                            value=library.forward(alternate,csr,work)
                            check(value.values,oracle(alternate,work),tolerance,'optional volume')
                        if work not in (0,1,2,3,5,8):continue
                        reverse=transpose(csr,len(data['qx']),len(data['rx']))
                        differentiable_inputs=tuple(value.detach().requires_grad_() if value is not None else None for value in (data[name] for name in SLOTS))
                        differentiable_data=dict(data,**dict(zip(SLOTS,differentiable_inputs)))
                        oracle_value=oracle(differentiable_data,work)
                        cot=torch.cos(torch.arange(oracle_value.numel(),device=device,dtype=dtype)).reshape_as(oracle_value)
                        active=tuple(v for v in differentiable_inputs if v is not None)
                        expected_gradients=torch.autograd.grad(oracle_value,active,cot,allow_unused=True)
                        normalized=library.forward(data,csr,work,save=True)
                        tangents=tuple(torch.sin(torch.arange(v.numel(),device=device,dtype=dtype)).reshape_as(v)*.03 if v is not None else None for v in differentiable_inputs)
                        oracle_tangents=tuple(v for v in tangents if v is not None)
                        def function(*values):
                            iterator=iter(values)
                            changed={name:(next(iterator) if data[name] is not None else None) for name in SLOTS}
                            return oracle(dict(data,**changed),work)
                        expected_jvp=torch.autograd.functional.jvp(function,active,oracle_tangents)[1]
                        actual_jvp=library.derivative(data,csr,None,work,tangents=tangents)
                        jvp_error=check(actual_jvp,expected_jvp,tolerance*3,'JVP')
                        for atomic in (False,True):
                            actual=library.derivative(data,csr,reverse,work,cotangent=cot,cache=normalized.cache,atomic=atomic)
                            errors=[];expected_index=0
                            for tensor,gradient in zip(differentiable_inputs,actual):
                                if tensor is None:continue
                                expected_gradient=expected_gradients[expected_index];expected_index+=1
                                if expected_gradient is None:expected_gradient=torch.zeros_like(tensor)
                                errors.append(check(gradient,expected_gradient,tolerance*3,('VJP',device,dtype,dim,work,atomic)))
                            lhs=(cot*actual_jvp).sum()
                            rhs=sum((g*t).sum() for g,t in zip(actual,tangents) if g is not None)
                            check(lhs,rhs,tolerance*3,'adjoint identity')
                            result['derivatives'].append(dict(device=device,precision=str(dtype),dimension=dim,work=work,masked=masked,atomic=atomic,vjp_error=max(errors),jvp_error=jvp_error))
                        if field_count(work,dim):
                            expected_by_slot={}
                            cursor=0
                            for name,value in zip(SLOTS,differentiable_inputs):
                                if value is not None:
                                    expected_by_slot[name]=expected_gradients[cursor];cursor+=1
                            for atomic in (False,True):
                                requested=(None,)*5+(None,torch.empty_like(data['rvalue']))
                                subset=library.derivative(data,csr,reverse,work,cotangent=cot,
                                    cache=normalized.cache,gradients=requested,atomic=atomic)
                                check(subset[6],expected_by_slot['rvalue'],tolerance*3,'field-only VJP')
                            field_tangents=(None,)*5+(tangents[5],tangents[6])
                            active_tangents=tuple(t if slot>=5 else torch.zeros_like(v)
                                for slot,(v,t) in enumerate(zip(differentiable_inputs,tangents)) if v is not None)
                            field_expected=torch.autograd.functional.jvp(function,active,active_tangents)[1]
                            field_actual=library.derivative(data,csr,None,work,tangents=field_tangents)
                            check(field_actual,field_expected,tolerance*3,'field-only JVP')
                        custom=differentiable(library,differentiable_data,csr,reverse,work)
                        actual=torch.autograd.grad(custom,active,cot,allow_unused=True)
                        for a,e,t in zip(actual,expected_gradients,active):
                            check(a if a is not None else torch.zeros_like(t),e if e is not None else torch.zeros_like(t),tolerance*3,'Torch backward')
                        result['autograd'].append(dict(device=device,precision=str(dtype),dimension=dim,work=work))
    for device in devices:
        data,csr,_,_=fixture(3,torch.float64,device,1)
        # Empty rows and empty particle sets must not dereference absent fields.
        for empty_query,empty_reference in [(True,False),(False,True),(True,True)]:
            changed=dict(data)
            if empty_query:
                for name in ('qx','h','qvalue'):changed[name]=changed[name][:0]
            if empty_reference:
                for name in ('rx','mass','density','rvalue'):changed[name]=changed[name][:0]
            q=len(changed['qx']);r=len(changed['rx'])
            empty=Plan(dict(offsets=torch.zeros(q,device=device,dtype=torch.int64),
                counts=torch.zeros(q,device=device,dtype=torch.int32),indices=torch.empty(0,device=device,dtype=torch.int32)))
            for work in (0,1,2,4,5,6,7):
                value=library.forward(changed,empty,work,save=True,auxiliary=True)
                assert torch.equal(value.values,torch.zeros_like(value.values))
            result['edge_cases'].append(dict(device=device,empty_query=empty_query,empty_reference=empty_reference))
        data,csr,_,_=fixture(2,torch.float64,device,1)
        # Both coordinate arguments can refer to one simulation tensor.
        data['rx']=data['qx'];n=len(data['qx'])
        data['mass']=data['mass'][:n];data['density']=data['density'][:n]
        data['rvalue']=data['qvalue']
        data['qx']=data['qx'].detach().requires_grad_();data['rx']=data['qx']
        data['qvalue']=data['qvalue'].detach().requires_grad_();data['rvalue']=data['qvalue']
        distance=torch.linalg.vector_norm(displacement(data),dim=-1)
        membership=(data['h'][:,None]>0)&(distance<data['h'][:,None])
        _,ids=torch.where(membership);counts=membership.sum(1).int()
        csr=Plan(dict(indices=ids.int(),counts=counts,offsets=torch.cat([counts.new_zeros(1,dtype=torch.int64),counts.long().cumsum(0)])[:-1]))
        reverse=transpose(csr,n,n)
        expected=oracle(data,1).sum()
        expected_grad=torch.autograd.grad(expected,(data['qx'],data['qvalue']))
        actual=differentiable(library,data,csr,reverse,1).sum()
        actual_grad=torch.autograd.grad(actual,(data['qx'],data['qvalue']))
        for a,e in zip(actual_grad,expected_grad):check(a,e,2e-11,'aliased simulation tensors')
        # Replacement of a Plan buffer after forward must use saved topology.
        actual=differentiable(library,data,csr,reverse,1).sum()
        original=csr.buffers['indices'];csr.buffers['indices']=torch.zeros_like(original)
        replacement_grad=torch.autograd.grad(actual,(data['qx'],data['qvalue']))
        for a,e in zip(replacement_grad,expected_grad):check(a,e,2e-11,'saved topology replacement')
        csr.buffers['indices']=original
        actual=differentiable(library,data,csr,reverse,1).sum()
        csr.buffers['indices'].add_(0)
        try:
            torch.autograd.grad(actual,(data['qx'],data['qvalue']))
            raise AssertionError('Mutated topology was accepted')
        except RuntimeError as error:assert 'modified by an inplace operation' in str(error)
        result['edge_cases'].append(dict(device=device,alias=True,topology_replacement=True,topology_version_guard=True))
        if device=='cuda':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                value=library.forward(data,csr,1)
            stream.synchronize();check(value.values,oracle(data,1),2e-11,'nondefault stream')
            result['edge_cases'].append(dict(device=device,nondefault_stream=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True);parser.add_argument('--gpu',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();torch.set_num_threads(24)
    results=verify(Library(args.cpu,args.gpu),['cpu','cuda'] if args.gpu else ['cpu'],args.quick)
    if args.gpu:torch.cuda.synchronize()
    results['complete']=True
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print({key:len(value) for key,value in results.items() if isinstance(value,list)})
