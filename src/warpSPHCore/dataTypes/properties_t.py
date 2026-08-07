from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..enumTypes import *

@dataclass
class OperationProperties:
    kernel: KernelFunctions
    operation: WarpOperation = WarpOperation.Interpolate
    
    gradientMode: GradientScheme = GradientScheme.Naive
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw

    positiveDivergence: bool = False

    supportMode: SupportScheme = SupportScheme.Gather
    operationMode: OperationDirection = OperationDirection.AllToAll

    divergenceDotMode: bool = False
