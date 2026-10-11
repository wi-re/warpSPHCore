"""Meshless finite-mass (MFM) / finite-volume (MFV) SPH (Hopkins 2015).

Two implementations of one scheme: a differentiable torch reference (the
oracle) and a forward-only warp backend with O(N) memory.

The public API, grouped for the frontend hand-over (COMPRESSIBLE_SPH_PLAN
Track E0). The frontend imports these; it does not copy the modules.

Torch reference (differentiable; the only gradable path)
    MeshlessGeometry, mfmRates, primitives, conserved, signalTimestep,
    starState, faceFlux, robustFaceFlux,
    conditionBeta, slopeLimiter, pairLimit,
    kernelWeight, kernelDerivative, SUPPORTED_KERNELS

Warp backend (forward only)
    MeshlessWarp        stepper: build / residual / gradients / rates / timestep
    mfmMomentsWarp      per-particle moments (omega, Ehat, neighbour count)
    mfmGradientsWarp    matrix least-squares gradients (rho, v, P)
    mfmLimiterWarp      App. B limiter factors (aRho, aVel, aP)
    mfmFluxWarp         face fluxes and the conservative dQ/dt assembly
    mfmTimestepWarp     per-particle CFL step 2 C h / v_sig

Face geometry (@wp.func; a frontend pair loop imports these, never copies)
    faceVector, guardedFace, pairFace
"""

from .kernels import kernelWeight, kernelDerivative, SUPPORTED_KERNELS
from .geometry import MeshlessGeometry
from .limiters import conditionBeta, slopeLimiter, pairLimit
from .riemann import starState, faceFlux, robustFaceFlux
from .scheme import mfmRates, primitives, conserved, signalTimestep
from .warpBackend import MeshlessWarp
from .wp_mfm import (faceVector, guardedFace, pairFace,
                     mfmMomentsWarp, mfmGradientsWarp, mfmLimiterWarp, mfmFluxWarp, mfmTimestepWarp)

__all__ = [
    # torch reference (the oracle)
    "MeshlessGeometry", "mfmRates", "primitives", "conserved", "signalTimestep",
    "starState", "faceFlux", "robustFaceFlux",
    "conditionBeta", "slopeLimiter", "pairLimit",
    "kernelWeight", "kernelDerivative", "SUPPORTED_KERNELS",
    # warp backend
    "MeshlessWarp",
    "mfmMomentsWarp", "mfmGradientsWarp", "mfmLimiterWarp", "mfmFluxWarp", "mfmTimestepWarp",
    # face geometry (@wp.func)
    "faceVector", "guardedFace", "pairFace",
]
