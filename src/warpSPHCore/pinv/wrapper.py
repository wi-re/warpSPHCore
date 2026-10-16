import torch
from .wp_pinv1x1 import pinv1x1
from .wp_pinv2x2 import pinv2x2_warpBackend
from .symmetric import selectedSymmetricInverse
# from .wp_pinv3x3 import pinv3x3_warpBackend

def pinv_warp(C: torch.Tensor, numNbrs: torch.Tensor, *, symmetric: bool = False) -> torch.Tensor:
    if C.shape[1] == 1 and C.shape[2] == 1:
        return pinv1x1(C)
    elif C.shape[1] == 2 and C.shape[2] == 2:
        return pinv2x2_warpBackend(C, numNbrs)
    else:
        eigVals = torch.linalg.eigvals(C).real

        # Flipping a whole 3D row to swap two entries can leave the middle
        # eigenvalue out of order (e.g. [1,3,2] became [2,3,1]). Keep each
        # diagnostic eigenvalue and its sign, sorting only their magnitudes.
        order = eigVals.abs().argsort(dim=-1, descending=True)
        eigVals = eigVals.gather(-1, order)
        inverse = selectedSymmetricInverse(C) if symmetric else torch.linalg.pinv(C, rtol=1e-6)
        return inverse, eigVals
