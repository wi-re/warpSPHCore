"""Torch-only semantic state shared by execution backends.

Warp array/struct definitions remain in dataTypes; importing these models does
not initialise an execution runtime.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import torch
from .enumTypes import *
from ._backend import BACKEND
if TYPE_CHECKING:
    from .dataTypes.hashMap_t import CompactHashMap

@dataclass#(slots=True)
class PointCloud:
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: torch.Tensor

    def __ne__(self, other: 'PointCloud') -> bool:
        return not self.__eq__(other)


@dataclass
class ParticleState:
    positions: torch.Tensor # shape [N,D] # D is the dimensionality of the simulation, e.g. 2 for 2D, 3 for 3D. This tensor holds the spatial coordinates of each particle in the simulation. For periodic domains these positions can be provided in the original unwrapped form, and the kernel computations will handle the periodicity based on the domain information provided separately.
    supports: torch.Tensor # Effective radius or smoothing length for each particle, which can be used to determine the influence radius for neighbor interactions. This is a crucial property for SPH computations as it defines the spatial extent of each particle's influence. Shape [N]
    masses: torch.Tensor # Shape [N]
    kinds: torch.Tensor # Particle types/categories, used e.g. for directional interactions or multi-material simulations. Shape [N], int32. Required: every operator kernel indexes SoA.kinds[i] unconditionally (see util/stateUtil.py's getParticleData), and OperationDirection.AllToAll used to paper over a missing kinds tensor with an N-sized zero dummy -- see warpier_fields.md Section 2.5.
    densities: Optional[torch.Tensor] = None # Densities are optional to allow for density computations which do not require pre-computed densities.


@dataclass
class ParticleTangentState:
    """Bundled JVP tangent counterpart to `ParticleState`, minus `kinds`
    (categorical, no tangent). Used by `warpOperationJVP`'s geometry-tangent
    path (`_jvpCommon.py`'s `launchGeometryJVP`, every `wp_<op>JVP.py`'s
    `computeSPH<Op>GeometryJVP`) in place of the loose parallel
    `tangentQuery*`/`tangentReference*` tensor kwargs it replaced
    (`warpier_tier2_correction_jvp_plan.md` phase a1)."""
    positions: torch.Tensor # [N,D]
    supports: torch.Tensor # [N]
    masses: torch.Tensor # [N]
    densities: Optional[torch.Tensor] = None # [N]


@dataclass(slots=True)
class DomainDescription:
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodic: torch.Tensor
    dim: int

    def __ne__(self, other: 'DomainDescription') -> bool:
        return not self.__eq__(other)


@dataclass
class CRKState:
    """Query correction coefficients; gradB axes are [derivative, component].

    Native interpolation only needs A and B. Gradient operators also need gradA and
    gradB. This semantic state is shared by Warp and native execution.
    """
    A: torch.Tensor
    B: torch.Tensor
    gradA: Optional[torch.Tensor] = None
    gradB: Optional[torch.Tensor] = None


@dataclass
class CRKTangentState:
    A: torch.Tensor
    B: torch.Tensor
    gradA: torch.Tensor
    gradB: torch.Tensor


@dataclass
class GradHState:
    queryOmegas: torch.Tensor
    referenceOmegas: Optional[torch.Tensor] = None


@dataclass
class RenormalizationState:
    renormalizationMatrices: torch.Tensor


@dataclass
class RenormalizationTangentState:
    renormalizationMatrices: torch.Tensor


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


@dataclass(frozen=True)
class CapacityPolicy:
    """How much spare room the flat pair arrays of a rebuilt `AdjacencyList` get.

    headroom     fractional extra capacity over the real edge count (0.1 = 10 %).
    shrinkBelow  with a prior capacity: keep that capacity (stable shapes) while the real count is within
                 [shrinkBelow * priorCapacity, priorCapacity]; reallocate when it overflows or falls below.
    inPlace      a rebuild that keeps the capacity also keeps the BUFFERS: it writes `paddedI` / `paddedJ` /
                 `numNeighbors` / `edgeOffsets` of the list it replaces (same shapes, same device addresses, so a
                 captured CUDA graph that reads them stays valid) and returns a new `AdjacencyList` object that
                 shares them. Off by default: the list that was replaced is then overwritten, so nothing may keep
                 it (or tensors taken from it) for later; opt in per run (a persistent graph), never for one-off
                 callers. A rebuild that outgrows or underfills the capacity allocates anew (the graph is recaptured).
    """
    headroom: float = 0.1
    shrinkBelow: float = 0.5
    inPlace: bool = False

    def capacityFor(self, numEdges: int, priorCapacity: Optional[int] = None) -> int:
        if priorCapacity is not None and self.shrinkBelow * priorCapacity <= numEdges <= priorCapacity:
            return int(priorCapacity)
        return int(numEdges + max(1, int(numEdges * self.headroom)))


@dataclass(slots=True)
class AdjacencyList:
    paddedI: Optional[torch.Tensor]
    paddedJ: torch.Tensor
    numNeighbors: torch.Tensor
    edgeOffsets: torch.Tensor
    numRows: int
    numCols: int

    # Required for velocity verlet list, which stores not just the required neighbors but also some extra neighbors to avoid rebuilding the neighbor list every step. These positions are the original positions with which the list was built, NOT the current positions. These are used to determine when the list needs to be rebuilt, by checking if any particle has moved more than a certain threshold distance from its original position.
    queryPositions: torch.Tensor = None
    referencePositions: torch.Tensor = None
    querySupports: torch.Tensor = None
    referenceSupports: torch.Tensor = None

    hashMap: CompactHashMap = None

    # Number of real entries at the front of paddedI / paddedJ (the rest is -1 padding). Host int, known at build time.
    numEdges: Optional[int] = None

    # Backend-owned reuse metadata (support policy and domain tensor versions).
    # Signature comparisons use host metadata only; bound domain owners are retained.
    # Restored/foreign lists start unbound and conservatively rebuild.
    searchContext: Optional[tuple] = None

    def cooRows(self, *, padded=True):
        """Explicit temporary COO rows; CSR-only lists never cache this allocation.

        Normal operators use edgeOffsets/numNeighbors/paddedJ. Development,
        differentiation and wall consumers may request rows explicitly.
        """
        n = self.paddedJ.numel() if padded or self.numEdges is None else self.numEdges
        if self.paddedI is not None:
            return self.paddedI[:n]
        if BACKEND == 'native':
            live = torch.repeat_interleave(
                torch.arange(self.numRows, dtype=self.paddedJ.dtype, device=self.paddedJ.device),
                self.numNeighbors.long(), output_size=self.numEdges,
            )
            if n == live.numel():
                return live
            rows = torch.full((n,), -1, dtype=self.paddedJ.dtype, device=self.paddedJ.device)
            rows[:live.numel()] = live
            return rows
        import warp as wp
        from .radiusSearch.compactHash.wp_collectNeighbors import fillCSRRows
        rows = torch.full((n,), -1, dtype=self.paddedJ.dtype, device=self.paddedJ.device)
        if self.numRows and n:
            stream = wp.stream_from_torch(torch.cuda.current_stream(rows.device)) if rows.is_cuda else None
            wp.launch(fillCSRRows, dim=(self.numRows, 32), block_dim=128,
                      inputs=[wp.from_torch(self.numNeighbors), wp.from_torch(self.edgeOffsets), wp.from_torch(rows)],
                      device=str(rows.device), stream=stream)
        return rows

    def livePairs(self):
        """(i, j) views of the real entries only (the padding sliced off): the exact arrays the unpadded list used to
        have. Stored COO rows are views; CSR-only lists explicitly allocate temporary
        rows. Neither caches the reconstruction. Shapes follow rebuilds: eager consumers only."""
        n = self.paddedJ.shape[0] if self.numEdges is None else self.numEdges
        return self.cooRows(padded=False), self.paddedJ[:n]


DEFAULT_CAPACITY = CapacityPolicy()
