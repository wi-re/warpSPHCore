import torch
from .wp_pinv1x1 import pinv1x1
from .wp_pinv2x2 import pinv2x2_warpBackend
# from .wp_pinv3x3 import pinv3x3_warpBackend

def pinv_warp(C: torch.Tensor, numNbrs: torch.Tensor) -> torch.Tensor:
    if C.shape[1] == 1 and C.shape[2] == 1:
        return pinv1x1(C)
    elif C.shape[1] == 2 and C.shape[2] == 2:
        return pinv2x2_warpBackend(C, numNbrs)
    else:
        eigVals = torch.linalg.eigvals(C).real

        if C.shape[1] == 3:
            eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])
        elif C.shape[1] == 2:
            eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
        return torch.linalg.pinv(C, rtol=1e-6), eigVals