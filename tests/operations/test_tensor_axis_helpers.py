"""Flattened SPH tensor helpers preserve the documented field-axis order."""
import pytest
import torch
import warp as wp
from warpSPHCore.type_config import scalar_t, get_torch_precision
from warpSPHCore.math.wp_outerTensorProduct import outerTensorProduct
from warpSPHCore.math.wp_divdot import divergenceProduct


def make_kernel(dim, rank):
    field_t=wp.types.vector(length=dim**rank,dtype=scalar_t)
    axis_t=wp.types.vector(length=dim,dtype=scalar_t)
    gradient_t=wp.types.vector(length=dim**(rank+1),dtype=scalar_t)
    result_t=wp.types.vector(length=dim**max(rank-1,0),dtype=scalar_t)
    input_elements=dim**rank
    outer_elements=dim**(rank+1)
    result_elements=dim**max(rank-1,0)
    @wp.kernel
    def evaluate(field:wp.array(dtype=field_t),axis:wp.array(dtype=axis_t),
                 outer:wp.array(dtype=gradient_t),first:wp.array(dtype=result_t),last:wp.array(dtype=result_t)):
        outer[0]=outerTensorProduct(field[0],axis[0],gradient_t(),dim,input_elements,outer_elements)
        first[0]=divergenceProduct(field[0],axis[0],result_t(),result_elements,True)
        last[0]=divergenceProduct(field[0],axis[0],result_t(),result_elements,False)
    return evaluate,field_t,axis_t,gradient_t,result_t


@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('rank',[1,2])
def test_field_axis_contractions(dim,rank):
    wp.init()
    kernel,field_t,axis_t,gradient_t,result_t=make_kernel(dim,rank)
    field=torch.arange(1,dim**rank+1,dtype=get_torch_precision()).reshape([dim]*rank)
    axis=torch.arange(2,dim+2,dtype=field.dtype)
    outputs=[wp.empty(1,dtype=t,device='cpu') for t in (gradient_t,result_t,result_t)]
    wp.launch(kernel,1,[wp.from_torch(field.reshape(1,-1),dtype=field_t),
                       wp.from_torch(axis.reshape(1,-1),dtype=axis_t),*outputs],device='cpu')
    expected=[field[...,None]*axis,torch.tensordot(axis,field,dims=([0],[0])),torch.tensordot(field,axis,dims=([-1],[0]))]
    for actual,want in zip(outputs,expected):
        torch.testing.assert_close(wp.to_torch(actual).reshape(want.shape),want,rtol=0,atol=0)
