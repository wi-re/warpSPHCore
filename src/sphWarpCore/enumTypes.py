from enum import Enum

class SupportScheme(Enum):
    Gather = 1
    Scatter = 2
    Symmetric = 3
    SuperSymmetric = 4

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
