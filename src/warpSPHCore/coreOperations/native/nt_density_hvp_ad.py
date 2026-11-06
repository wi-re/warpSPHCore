"""Higher and legacy batched products over fixed native position-block matrices.

Warp's block uses Torch composition after assembling constant pair Hessians.
Keep that same AD contract here; all pair geometry is evaluated by native code.
Ordinary first-order products retain the owned native CSR/CSC kernels.
"""
import torch


def legacy_batched(values):
    return any(isinstance(x,torch.Tensor) and torch._C._functorch.is_legacy_batchedtensor(x) for x in values)


def matrices(inputs,options,row_start):
    from ...native._build import build
    geometry=list(inputs)
    for slot in (4,13,14,15,16):geometry[slot]=None
    H,encoded=build().density_position_hvp_matrices(geometry,*options,row_start)
    rows=(encoded-1).clamp_min(0)
    ids=inputs[12].long().clamp(0,max(len(inputs[1])-1,0))
    return H,rows,ids,encoded>0


def zero(example,shape,values):
    output=example.new_zeros(shape)
    # Empty slices connect zero products without evaluating unused Inf values.
    for value in values:
        if value is not None:output=output+value.reshape(-1)[:0].sum()
    return output


def edge_vectors(inputs,rows,ids,live):
    left=inputs[13][rows]
    right=inputs[14][ids] if inputs[14] is not None else torch.zeros_like(left)
    return torch.where(live[:,None],left-right,0.)


def primal(inputs,options,row_start,prepared=None):
    mass,left,right=inputs[4],inputs[13],inputs[14]
    if not len(inputs[0]) or not len(inputs[1]):return zero(left,left.shape,(mass,left,right))
    H,rows,ids,live=matrices(inputs,options,row_start) if prepared is None else prepared
    direction=edge_vectors(inputs,rows,ids,live)
    weight=torch.where(live,mass[ids],0.)
    contribution=weight[:,None]*torch.bmm(H,direction.unsqueeze(-1)).squeeze(-1)
    return left.new_zeros(left.shape).index_add(0,rows,contribution)


def vjp(inputs,options,row_start,cotangent):
    mass,left,right=inputs[4],inputs[13],inputs[14]
    nq,nr,dim=len(inputs[0]),len(inputs[1]),inputs[0].shape[1]
    if not nq or not nr:
        dependencies=(mass,left,right,cotangent)
        return (zero(left,(nq,dim),dependencies),zero(left,(nr,dim),dependencies),zero(mass,mass.shape,dependencies))
    H,rows,ids,live=matrices(inputs,options,row_start)
    direction=edge_vectors(inputs,rows,ids,live)
    seed=torch.where(live[:,None],cotangent[rows],0.)
    weight=torch.where(live,mass[ids],0.)
    vector_bar=torch.bmm(H.transpose(1,2),(seed*weight[:,None]).unsqueeze(-1)).squeeze(-1)
    mass_bar=(seed*torch.bmm(H,direction.unsqueeze(-1)).squeeze(-1)).sum(-1)
    return (left.new_zeros((nq,dim)).index_add(0,rows,vector_bar),
            left.new_zeros((nr,dim)).index_add(0,ids,-vector_bar),
            mass.new_zeros(mass.shape).index_add(0,ids,mass_bar))


def jvp(inputs,options,row_start,directions):
    prepared=None if not len(inputs[0]) or not len(inputs[1]) else matrices(inputs,options,row_start)
    result=zero(inputs[13],inputs[13].shape,(*inputs[4:5],inputs[13],inputs[14],*directions))
    if directions[4] is not None:
        values=list(inputs);values[4]=directions[4]
        result=result+primal(values,options,row_start,prepared)
    if directions[13] is not None or directions[14] is not None:
        values=list(inputs)
        values[13]=directions[13] if directions[13] is not None else torch.zeros_like(inputs[13])
        values[14]=directions[14]
        result=result+primal(values,options,row_start,prepared)
    return result
