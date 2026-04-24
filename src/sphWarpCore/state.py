from dataclasses import dataclass
import torch
import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional

from .enumTypes import *

@dataclass
class ParticleState:
    positions: torch.Tensor # shape [N,D] # D is the dimensionality of the simulation, e.g. 2 for 2D, 3 for 3D. This tensor holds the spatial coordinates of each particle in the simulation. For periodic domains these positions can be provided in the original unwrapped form, and the kernel computations will handle the periodicity based on the domain information provided separately.
    supports: torch.Tensor # Effective radius or smoothing length for each particle, which can be used to determine the influence radius for neighbor interactions. This is a crucial property for SPH computations as it defines the spatial extent of each particle's influence. Shape [N]
    masses: torch.Tensor # Shape [N]
    densities: Optional[torch.Tensor] = None # Densities are optional to allow for density computations which do not require pre-computed densities.
    kinds: Optional[torch.Tensor] = None # For storing particle types or categories, which can be useful for operations that depend on particle type, e.g. for directional interactions or multi-material simulations. This is an optional field and can be None if not needed.

@dataclass
class OperationProperties:
    kernel: KernelFunctions
    operation: WarpOperation = WarpOperation.Interpolate
    
    gradientMode: GradientScheme = GradientScheme.Naive
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw

    positiveDivergence: bool = False

    supportMode: SupportScheme = SupportScheme.Gather
    operationMode: OperationDirection = OperationDirection.AllToAll

@dataclass
class CRKState:
    A: torch.Tensor # shape [N]
    B: torch.Tensor # shape [N,D]
    gradA: torch.Tensor # shape [N,D]
    gradB: torch.Tensor # shape [N,D,D]

@dataclass
class GradHState:
    queryOmegas: torch.Tensor # shape [N]
    referenceOmegas : Optional[torch.Tensor] = None # shape [N], if None then queryOmegas are used as referenceOmegas

@dataclass
class RenormalizationState:
    renormalizationMatrices: torch.Tensor # shape [N,D,D]