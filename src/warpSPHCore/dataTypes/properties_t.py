from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..enumTypes import *

@dataclass(frozen=True)
class OperationProperties:
    kernel: KernelFunctions
    operation: WarpOperation = WarpOperation.Interpolate
    
    gradientMode: GradientScheme = GradientScheme.Naive
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw

    positiveDivergence: bool = False

    supportMode: SupportScheme = SupportScheme.Gather
    operationMode: OperationDirection = OperationDirection.AllToAll

    divergenceDotMode: bool = False

    #: Lattice-normalisation correction (LATTICE_DENSITY_PLAN.md). `C_d`
    #: normalises the kernel's integral, so a density summed on a lattice of
    #: spacing `s = h / n_h` reads `rho0 * L(n_h)` rather than `rho0`. Setting
    #: `calibrateNormalization` scales every kernel value and derivative by
    #: `1 / L`, resolved from `(kernel, n_h, dim)` in `extractStateInfo` and
    #: cached there. `n_h` is the *nominal* support-to-spacing ratio the case
    #: was configured with, not a measured one -- see the plan's 3.5.
    #:
    #: Meaningful only for a uniform-resolution lattice: with adaptive support
    #: `n_h` is per particle and a single scalar cannot represent it.
    n_h: Optional[float] = None
    calibrateNormalization: bool = False

    def __post_init__(self):
        # Frozen dataclasses still run __post_init__; they only cannot assign.
        # This is the earliest point the invalid combination can be caught, and
        # it covers every backend at once. Deliberately loud: the alternative
        # (silently treating a missing n_h as "off") makes "calibration
        # requested, never computed" indistinguishable from "calibration off".
        if self.calibrateNormalization and not (self.n_h is not None and self.n_h > 0.0):
            raise ValueError(
                'calibrateNormalization=True requires a positive n_h '
                f'(got {self.n_h!r}); the correction 1/L is a function of '
                '(kernel, n_h, dim) and cannot be resolved without it.')
