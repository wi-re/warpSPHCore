"""Native small-matrix primals with the core's existing Torch spectral AD rule."""
import math
from numbers import Real
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ..symmetric import selectedSymmetricInverse, _symmetric_pinv_jvp


def _check(matrix):
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2] or not 1 <= matrix.shape[1] <= 3:
        raise ValueError('Expected (n,d,d) matrices, d=1,2,3')
    if matrix.dtype not in (torch.float32, torch.float64) or matrix.device.type not in ('cpu','cuda'):
        raise ValueError('Expected CPU/CUDA float32/float64 matrices')


def _rtol(value, matrix):
    if value is None: return matrix.shape[1]*torch.finfo(matrix.dtype).eps
    if not isinstance(value,Real) or not math.isfinite(value) or value<0:
        raise ValueError('rtol must be a finite nonnegative scalar')
    return float(value)


def _ad(value):
    return value.requires_grad or forward_ad.unpack_dual(value).tangent is not None


def pinvSymmetric(matrix, rtol=None):
    _check(matrix)
    cutoff=_rtol(rtol,matrix)
    if _ad(matrix) or not len(matrix): return selectedSymmetricInverse(matrix,cutoff)
    from ...native._build import build
    return build().small_inverse(matrix.contiguous(),None,None,0,cutoff)[0]


def applyPinv(values,matrix,rtol=None):
    _check(matrix)
    if not isinstance(values,torch.Tensor) or values.ndim!=3 or values.shape[0]!=len(matrix) or values.shape[2]!=matrix.shape[1]:
        raise ValueError('Values must have shape (n,p,d) for (n,d,d) matrices')
    if values.device!=matrix.device or values.dtype!=matrix.dtype:
        raise ValueError('Values and matrices must share dtype/device')
    cutoff=_rtol(rtol,matrix)
    if _ad(values) or _ad(matrix) or not len(matrix): return values @ selectedSymmetricInverse(matrix,cutoff)
    from ...native._build import build
    return build().small_inverse(matrix.contiguous(),values.contiguous(),None,2,cutoff)[0]


def _eigen2_differential(matrix,perturbation):
    a,d=matrix[:,0,0],matrix[:,1,1]
    b=(matrix[:,0,1]+matrix[:,1,0])*.5
    discriminant=((a-d).square()+4*b.square()).sqrt()
    inverse=torch.where(discriminant>0,1/torch.where(discriminant>0,discriminant,1.),0.)
    swap=((a+d-discriminant)*.5).abs()>((a+d+discriminant)*.5).abs()
    da,dd=perturbation[:,0,0],perturbation[:,1,1]
    db=(perturbation[:,0,1]+perturbation[:,1,0])*.5
    tangent=((a-d)*(da-dd)+4*b*db)*inverse
    plus,minus=(da+dd+tangent)*.5,(da+dd-tangent)*.5
    return torch.stack((torch.where(swap,minus,plus),torch.where(swap,plus,minus)),-1)


def _eigen2_pullback(matrix,cotangent):
    # The two output channels are diagnostic signed eigenvalues, ordered by
    # magnitude. Repeated-value and cutoff branches follow the Warp rule.
    basis=torch.eye(4,dtype=matrix.dtype,device=matrix.device).reshape(4,2,2)
    columns=[_eigen2_differential(matrix,value.expand_as(matrix)) for value in basis]
    return torch.stack([(column*cotangent).sum(-1) for column in columns],-1).reshape_as(matrix)


class _Renormalization2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,matrix,counts):
        from ...native._build import build
        inverse,eigenvalues=build().small_inverse(matrix.contiguous(),None,counts.contiguous(),1,1e-6)
        ctx.save_for_backward(matrix,inverse,counts)
        ctx.save_for_forward(matrix,inverse,counts)
        return inverse,eigenvalues

    @staticmethod
    @once_differentiable
    def backward(ctx,inverse_bar,eigenvalue_bar):
        matrix,inverse,counts=ctx.saved_tensors
        result=torch.zeros_like(matrix)
        if inverse_bar is not None: result=_symmetric_pinv_jvp(matrix,inverse,inverse_bar)
        if eigenvalue_bar is not None: result=result+_eigen2_pullback(matrix,eigenvalue_bar)
        return torch.where((counts>=4)[:,None,None],result,0.),None

    @staticmethod
    def jvp(ctx,tangent,_counts):
        matrix,inverse,counts=ctx.saved_tensors
        if tangent is None: tangent=torch.zeros_like(matrix)
        active=(counts>=4)
        return (torch.where(active[:,None,None],_symmetric_pinv_jvp(matrix,inverse,tangent),0.),
                torch.where(active[:,None],_eigen2_differential(matrix,tangent),0.))


def pinv1x1(matrix):
    _check(matrix)
    if matrix.shape[1]!=1: raise ValueError('pinv1x1 requires 1x1 matrices')
    value=matrix[:,0,0]
    inverse=torch.where(value>1e-10,1/torch.where(value>1e-10,value,1.),0.)
    return inverse.reshape_as(matrix),value[:,None]


def pinv2x2_warpBackend(matrix,counts):
    _check(matrix)
    if matrix.shape[1]!=2: raise ValueError('2D renormalization requires 2x2 matrices')
    if not isinstance(counts,torch.Tensor) or counts.shape!=(len(matrix),) or counts.dtype!=torch.int32 or counts.device!=matrix.device:
        raise ValueError('Neighbor counts must be int32 with one entry per matrix on its device')
    return _Renormalization2.apply(matrix,counts)


def pinv_warp(matrix,counts,*,symmetric=False):
    _check(matrix)
    if matrix.shape[1]==1:return pinv1x1(matrix)
    if matrix.shape[1]==2:return pinv2x2_warpBackend(matrix,counts)
    values=torch.linalg.eigvals(matrix).real
    values=values.gather(-1,values.abs().argsort(-1,descending=True))
    inverse=selectedSymmetricInverse(matrix) if symmetric else torch.linalg.pinv(matrix,rtol=1e-6)
    return inverse,values


def symmetricPinv3(matrix,rtol):
    return pinvSymmetric(matrix.unsqueeze(0),rtol)[0]


def loadPadded3x3(values,index,dimension):
    result=values.new_zeros((3,3))
    result[:dimension,:dimension]=values[index].reshape(dimension,dimension)
    return result
