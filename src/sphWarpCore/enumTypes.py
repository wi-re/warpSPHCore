from enum import Enum

# For gradients there are many different variants for the kernel alone. Given a standard difference formulation mj/rhoj (fi - fj) \nabla W_ij
# We can compute:
# 1: Gather: mj/rhoj (fi - fj) \nabla W(x_ij, h_i)
# 2: Scatter: mj/rhoj (fi - fj) \nabla W(x_ij, h_j)
# 3: (Mean) Symmetric: mj/rhoj (fi - fj) \nabla W(x_ij, (h_i + h_j)/2)
# 4: Kernel Mean Symmetric: mj/rhoj (fi - fj) 0.5 * (\nabla W(x_ij, h_i) + \nabla W(x_ij, h_j))
# 5: Super Symmetric: mj/rhoj (fi - fj) 0.5 * (\nabla W(x_ij, h_i) - \nabla W(x_ji, h_j)) # CRK SPH uses this formulation
# 6: Partial Symmetric: mj/rhoj [(fi \nabla W(x_ij, h_i) + fj \nabla W(x_ij, h_j))] # PESPH uses this

class SupportScheme(Enum):
    Gather = 11 # uses hi
    Scatter = 12 # uses hj
    MeanSymmetric = 13 # uses (hi + hj) / 2
    KernelMeanSymmetric = 14 # uses 0.5 * (k(q,hi) + k(q,hj))
    SuperSymmetric = 15 # uses 0.5 * (k(q,hi) - k(q,hj))
    PartialSymmetric = 16 # uses fi * k(q,hi) + fj * k(q,hj)

# def supportSchemeTomode(scheme: SupportScheme) -> str:
#     return scheme.name
    
import warp as wp
def supportSchemeToUint(scheme: SupportScheme) -> wp.uint32:
    scheme_map = {
        SupportScheme.Gather: 11,
        SupportScheme.Scatter: 12,
        SupportScheme.MeanSymmetric: 13,
        SupportScheme.KernelMeanSymmetric: 14,
        SupportScheme.SuperSymmetric: 15,
        SupportScheme.PartialSymmetric: 16
    }
    id = wp.uint32(scheme_map.get(scheme, 0))
    if id == 0:
        raise ValueError(f"Unsupported support scheme: {scheme}")
    return id

# @torch.jit.script
class KernelFunctions(Enum):
    Poly6 = 20
    CubicSpline = 30
    QuarticSpline = 31
    QuinticSpline = 32
    B7 = 33
    Wendland2 = 0
    Wendland4 = 1
    Wendland6 = 2
    Spiky = 21
    ViscosityKernel = 30
    CohesionKernel = 31
    AdhesionKernel = 32

class GradientScheme(Enum):
    Naive = 1
    Symmetric = 2
    Difference = 3
    Summation = 4

class LaplacianScheme(Enum):
    Naive = 1
    Brookshaw = 2
    Dot = 3
    Default = 4

class WarpOperation(Enum):
    Interpolate = 1
    Gradient = 2
    Divergence = 3
    Curl = 4
    Laplacian = 5
    Density = 6
    Covariance = 7
    Custom = 8


class HashMapLengthMode(Enum):
    Fixed = 0
    NumberOfParticles = 1
    NextPrime = 2


class ViscosityTerms(Enum):
    Default = 0
    MonaghanGingold1983 = 1
    Cleary1998 = 2
    Monaghan1992 = 3
    Monaghan1997a = 4
    Monaghan1997b = 5
    Dukowicz = 6
    Price2012_98 = 7
    Price2012 = 8
    Price2008 = 9
    Wadsley2008 = 10
    DeltaSPH = 11

class OperationDirection(Enum):
    NoGhost = 0
    FluidToFluid = 1
    FluidToBoundary = 2
    BoundaryToFluid = 3
    BoundaryToBoundary = 4
    FluidToGhost = 5
    GhostToFluid = 6
    BoundaryToGhost = 7
    GhostToBoundary = 8
    AllToGhost = 10
    AllToAll = 9
    AllToFluid = 11
    AllToBoundary = 12
    FluidToAll = 13
    BoundaryToAll = 14
    
    
class ParticleType(Enum):
    Fluid = 0
    Boundary = 1
    Ghost = 2
    Other = 3


__all__ = [
    "SupportScheme",
    "KernelFunctions",
    "GradientScheme",
    "LaplacianScheme",
    "WarpOperation",    
    "HashMapLengthMode",
    "ViscosityTerms",
    "OperationDirection",
    "ParticleType",
    "supportSchemeToUint",
]