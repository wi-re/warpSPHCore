from enum import Enum

class SupportScheme(Enum):
    Gather = 1
    Scatter = 2
    Symmetric = 3
    SuperSymmetric = 4

def supportSchemeTomode(scheme: SupportScheme) -> str:
    if scheme == SupportScheme.Gather:
        return 'gather'
    elif scheme == SupportScheme.Scatter:
        return 'scatter'
    elif scheme == SupportScheme.Symmetric:
        return 'symmetric'
    elif scheme == SupportScheme.SuperSymmetric:
        return 'superSymmetric'
    else:
        raise ValueError(f"Unsupported support scheme: {scheme}")

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
    AllToAll = 0
    FluidToFluid = 1
    FluidToBoundary = 2
    BoundaryToFluid = 3
    BoundaryToBoundary = 4
    FluidToGhost = 5
    GhostToFluid = 6
    BoundaryToGhost = 7
    GhostToBoundary = 8