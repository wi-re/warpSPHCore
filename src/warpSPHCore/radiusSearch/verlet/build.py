import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *

from ...profiling import record_function
from ..compactHash import *

from .validCheck import _verlet_validity_metrics
from .util import _minimum_image_delta

# @torch.jit.script
def buildVerletList_(
        queryPositions: torch.Tensor, referencePositions: torch.Tensor,
        querySupports: torch.Tensor, referenceSupports: torch.Tensor,
        domain : DomainDescription, verletScale : float = 1.0, 
        mode : SupportScheme = SupportScheme.SuperSymmetric,
        priorNeighborhood : Optional[AdjacencyList] = None, 
        verbose : bool = False):
    with record_function(f"[Verlet] {mode}"):
        if priorNeighborhood is None or (priorNeighborhood is not None and priorNeighborhood.queryPositions is None) or (priorNeighborhood is not None and priorNeighborhood.referencePositions is None):
            if verbose:
                print('Building neighborhood from scratch [no prior neighborhood]')
            with record_function(f"[Verlet] no prior"):
                adjacency = radiusSearchCompactHashMap_(
                    queryPositions, referencePositions,
                    querySupports * verletScale, referenceSupports * verletScale,
                    domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                )
        else:
            if queryPositions.shape[0] != priorNeighborhood.queryPositions.shape[0] or referencePositions.shape[0] != priorNeighborhood.referencePositions.shape[0]:
                if verbose:
                    print('Building neighborhood from scratch [different number of particles]')
                with record_function(f"[Verlet] mismatch"):
                    adjacency = radiusSearchCompactHashMap_(
                        queryPositions, referencePositions,
                        querySupports * verletScale, referenceSupports * verletScale,
                        domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                    )
            else:
                with record_function(f"[Verlet] Checking validity of prior neighborhood"):
                    supports_a = priorNeighborhood.querySupports
                    supports_b = priorNeighborhood.referenceSupports

                    if mode == SupportScheme.Gather:
                        support_case = 0
                    elif mode == SupportScheme.Scatter:
                        support_case = 1
                    else:
                        support_case = 2

                    shouldRebuild_t, maxDistance, supportFactor, minSupport, supportBuffer = _verlet_validity_metrics(
                        queryPositions,
                        referencePositions,
                        priorNeighborhood.queryPositions,
                        priorNeighborhood.referencePositions,
                        querySupports,
                        referenceSupports,
                        supports_a,
                        supports_b,
                        domain.periodic,
                        domain.min,
                        domain.max,
                        verletScale,
                        support_case,
                    )
                    shouldRebuild = bool(shouldRebuild_t.item())

                    if verbose:
                        distance_a = torch.linalg.vector_norm(
                            _minimum_image_delta(
                                queryPositions,
                                priorNeighborhood.queryPositions,
                                domain.periodic,
                                domain.min,
                                domain.max,
                            ),
                            dim=-1,
                        )
                        distance_b = torch.linalg.vector_norm(
                            _minimum_image_delta(
                                referencePositions,
                                priorNeighborhood.referencePositions,
                                domain.periodic,
                                domain.min,
                                domain.max,
                            ),
                            dim=-1,
                        )
                        priorQuerySupports = supports_a / verletScale
                        priorReferenceSupports = supports_b / verletScale
                        print(f'Distance a: min: {distance_a.min()}, max: {distance_a.max()}, avg: {distance_a.mean()}')
                        print(f'Distance b: min: {distance_b.min()}, max: {distance_b.max()}, avg: {distance_b.mean()}')
                        print(f'Support a (prior): min: {priorQuerySupports.min()}, max: {priorQuerySupports.max()}, avg: {priorQuerySupports.mean()}')
                        print(f'Support b (prior): min: {priorReferenceSupports.min()}, max: {priorReferenceSupports.max()}, avg: {priorReferenceSupports.mean()}')
                        print(f'Support a (current): min: {querySupports.min()}, max: {querySupports.max()}, avg: {querySupports.mean()}')
                        print(f'Support b (current): min: {referenceSupports.min()}, max: {referenceSupports.max()}, avg: {referenceSupports.mean()}')
                        print(f'Max |Δh|: {supportFactor}, Support buffer: {supportBuffer}, Minimum support: {minSupport}')
                        print(f'Max Distance: {maxDistance}')
                        print(f'Distance Factor: {maxDistance / minSupport}')
                        print(f'Distance Threshold Ratio: {1 + maxDistance / minSupport}, Verlet Scale: {verletScale}')



                    # print(maxDistance, minSupport * verletScale)
                    if shouldRebuild:
                        if verbose:
                    # if verbose:
                            print('Support Factor: ', supportFactor / minSupport, 'Minimum Support: ', minSupport)
                            print('Distance Factor: ', maxDistance / minSupport)
                            print('Threshold exceeded by: ', torch.maximum(maxDistance, supportFactor) / supportBuffer)
                            print('Building neighborhood from scratch [verlet buffer exceeded]')
                        with record_function(f"[Verlet] Rebuilding Invalid"):
                            adjacency = radiusSearchCompactHashMap_(
                                queryPositions, referencePositions,
                                querySupports * verletScale, referenceSupports * verletScale,
                                domain.periodic, domain, mode, hashMapLength=queryPositions.shape[0]+1,
                            )
                            
                    else:
                        if verbose:
                            print('Reusing neighborhood')
                        adjacency = priorNeighborhood
        return adjacency

    
def buildVerletList(
        queryParticles: ParticleState, 
        domain: DomainDescription, 
        verletScale : float = 1.0,
        supportMode: SupportScheme = SupportScheme.Gather,
        priorNeighborhood : Optional[AdjacencyList] = None,
        verbose : bool = False,
        referenceParticles: Optional[ParticleState] = None
):
    with record_function(f"[warpSPH] - buildVerletList"):
        if referenceParticles is None:
            referenceParticles = queryParticles

        return buildVerletList_(
            queryParticles.positions, referenceParticles.positions,
            queryParticles.supports, referenceParticles.supports,
            domain, verletScale, supportMode, 
            priorNeighborhood, verbose
        )
