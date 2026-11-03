"""Independent corrected-pair equations for the remaining native primals."""
import torch
import warpSPHCore as sph
from .operators import normalization
from .kernels import shape


def pair_terms(q, r, domain, prop, state=None, renormalization=None, precision=torch.float64):
    d = domain.dim
    qx, rx = q.positions.to(precision), r.positions.to(precision)
    delta = torch.zeros((len(qx), len(rx), d), device=qx.device,
                        dtype=precision, requires_grad=True)
    dx = qx[:, None] + delta - rx[None]
    length = (domain.max - domain.min).to(precision)
    dx = torch.where(domain.periodic, (dx + length / 2).remainder(length) - length / 2, dx)
    distance = dx.norm(dim=-1)
    hi, hj = q.supports.to(precision)[:, None], r.supports.to(precision)[None]
    support = prop.supportMode.value
    h = {11: hi, 12: hj, 13: (hi + hj) / 2,
         14: torch.maximum(hi, hj), 15: torch.maximum(hi, hj)}[support]

    def kernel(radius):
        ratio = (distance / radius).detach().requires_grad_()
        active = (radius > 0) & (ratio < 1)
        if prop.kernel == sph.KernelFunctions.AdhesionKernel:
            active = active & (ratio > .5)
            argument = torch.where(active, ratio, .75)
        else:
            argument = ratio
        radial = torch.where(active, shape(argument, prop.kernel, d), 0.)
        first = torch.autograd.grad(radial.sum(), ratio, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), ratio)[0]
        C = normalization(prop.kernel, d)
        # The legacy raw Laplacian is regularized radially. It is not the
        # Hessian trace of the corrected kernel (including in Naive mode).
        eps = 1e-7 if q.positions.dtype == torch.float32 else 1e-15
        re = distance + eps * radius
        lap = (distance.square() / re.square() * C * second / radius**(d + 2) +
               (-distance.square() / re**3 + d / re) * C * first / radius**(d + 1))
        lap = torch.where((distance / radius >= eps) & active, lap, 0.)
        argument = distance / radius
        if prop.kernel == sph.KernelFunctions.AdhesionKernel:
            argument = torch.where(active, argument, .75)
        value = torch.where(active, C * shape(argument, prop.kernel, d) / radius**d, 0.)
        return value, lap.detach()

    wi, li = kernel(hi)
    wj, lj = kernel(hj)
    w, lap = kernel(h)
    if support in (14, 15): w = (wi + wj) / 2
    if support == 15: lap = (li + lj) / 2
    if state is not None:
        A = state.A.to(precision)[:, None] + (state.gradA.to(precision)[:, None] * delta).sum(-1)
        B = state.B.to(precision)[:, None] + torch.einsum('ngc,nmg->nmc', state.gradB.to(precision), delta)
        w = A * (1 + (B * dx).sum(-1)) * w
    gradient = torch.autograd.grad(w.sum(), delta)[0]
    if renormalization is not None:
        gradient = torch.einsum('nab,nmb->nma', renormalization.to(precision), gradient)
    allowed = torch.ones_like(w, dtype=torch.bool)
    if prop.operationMode == sph.OperationDirection.FluidToBoundary:
        allowed = (q.kinds == 1)[:, None] & (r.kinds == 0)[None]
    elif prop.operationMode == sph.OperationDirection.AllToAll:
        allowed &= (r.kinds != 2)[None]
    elif prop.operationMode != sph.OperationDirection.TrueAllToToAll:
        raise AssertionError('oracle direction not implemented')
    return dx.detach(), distance.detach(), h.detach(), w.detach(), gradient.detach(), lap, allowed


def remaining_oracle(q, r, domain, prop, qf, rf, *, referenceVolumes=None, crkState=None,
                     gradHState=None, renormalizationState=None, consistentDivergence=False,
                     covarianceReturnNumNeighbors=False):
    naive_laplacian = prop.operation == sph.WarpOperation.Laplacian and prop.laplacianMode == sph.LaplacianScheme.Naive
    state = crkState[0] if isinstance(crkState, tuple) else crkState
    R = renormalizationState.renormalizationMatrices if isinstance(renormalizationState, sph.RenormalizationState) else renormalizationState
    dx, distance, h, w, gradient, lap, allowed = pair_terms(q, r, domain, prop,
        None if naive_laplacian else state, None if naive_laplacian else R)
    gradient = gradient * allowed[..., None]
    volume = referenceVolumes.double() if referenceVolumes is not None else r.masses.double() / r.densities.double()
    d = domain.dim
    if prop.operation == sph.WarpOperation.Covariance:
        value = -torch.einsum('ij,ija,ijb->iab', allowed * volume, dx, gradient)
        # Positive-weight counts are discrete precision-dependent metadata.
        # In particular, a Gaussian tail can be positive in FP64 and exactly
        # zero in FP32. Evaluate the predicate in the requested precision.
        count_w = pair_terms(q, r, domain, prop, state, None, q.positions.dtype)[3]
        count = ((count_w > 0) & allowed).sum(1).int()
        return (value, count) if covarianceReturnNumNeighbors else value
    fi, fj = qf.double().flatten(1) if qf.ndim > 1 else qf.double()[:, None], rf.double().flatten(1) if rf.ndim > 1 else rf.double()[:, None]
    if gradHState is not None:
        omega = gradHState.referenceOmegas if isinstance(gradHState, sph.GradHState) else gradHState
        fj = fj / omega.double()[:, None]
        query_omega = gradHState.queryOmegas if isinstance(gradHState, sph.GradHState) else gradHState
        fi = fi / query_omega.double()[:, None]
    mode = prop.gradientMode
    if prop.operation == sph.WarpOperation.Laplacian:
        scale = r.masses.double()[None] * q.densities.double()[:, None] / r.densities.double()[None]**2 if mode == sph.GradientScheme.Symmetric else volume
        weighted = (fj[None] - fi[:, None]) * scale[..., None]
        if prop.laplacianMode == sph.LaplacianScheme.Naive:
            value = torch.einsum('ijf,ij->if', weighted, lap * allowed)
        else:
            normal = dx / (distance + 1e-8 * h)[..., None]
            product = (gradient * normal).sum(-1)
            if prop.laplacianMode == sph.LaplacianScheme.Dot:
                blocks = weighted.reshape(*weighted.shape[:2], -1, d)
                projection = (blocks * normal[:, :, None]).sum(-1)
                value = ((blocks - (d + 2) * projection[..., None] * normal[:, :, None]) *
                         (product / (distance + 1e-8 * h))[:, :, None, None]).sum(1).flatten(1)
            else:
                denominator = distance + (1e-12 if prop.laplacianMode == sph.LaplacianScheme.Default else 1e-8) * h
                value = torch.einsum('ijf,ij->if', weighted, -2 * product / denominator)
        return value.reshape(qf.shape)
    if mode == sph.GradientScheme.Symmetric:
        rhoi, rhoj = q.densities.double(), r.densities.double()
        weighted = r.masses.double()[None, :, None] * rhoi[:, None, None] * (
            fi[:, None] / rhoi[:, None, None]**2 + fj[None] / rhoj[None, :, None]**2)
    else:
        values = fj[None] if mode == sph.GradientScheme.Naive else fj[None] - fi[:, None] if mode == sph.GradientScheme.Difference else fj[None] + fi[:, None]
        scale = volume * r.densities.double()[None] / q.densities.double()[:, None] if consistentDivergence else volume
        weighted = values * scale[..., None]
    jacobian = torch.einsum('ijf,ijd->ifd', weighted, gradient)
    if prop.operation == sph.WarpOperation.Divergence:
        if rf.ndim <= 2: return jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)
        tensor = jacobian.reshape(-1, d, d, d)
        return tensor.diagonal(dim1=1 if prop.divergenceDotMode else 2, dim2=3).sum(-1)
    if d == 1: return gradient.new_zeros(len(q.positions))
    if d == 2:
        return (jacobian[:, d:, 0] - jacobian[:, :d, 1]) if rf.ndim == 3 else (jacobian[:, 1, 0] - jacobian[:, 0, 1])[:, None]
    tensor = jacobian.reshape(len(q.positions), d, rf[0].numel() // d, d)
    value = torch.stack((tensor[:, 2, :, 1] - tensor[:, 1, :, 2],
                         tensor[:, 0, :, 2] - tensor[:, 2, :, 0],
                         tensor[:, 1, :, 0] - tensor[:, 0, :, 1]), 1)
    return value.reshape(qf.shape)
