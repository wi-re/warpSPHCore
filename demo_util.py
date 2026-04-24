import diffSPH
from diffSPH.sampling import ParticleSet
from diffSPH.schemes.states.common import BasicState
from diffSPH.modules.density import computeDensity
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from diffSPH.kernels import *
from diffSPH.neighborhood import evaluateNeighborhood, SupportScheme, computeNeighborhoodStates
from diffSPH.enums import Operation, SupportScheme, GradientMode, LaplacianMode
from diffSPH.operations import SPHOperation
from sphWarpCore.util import getNextPrime, generateNeighborTestData

    
import os, sys
import torch
import numpy as np

# Warp-based Radius Search Implementation
import warp as wp

# Initialize Warp
wp.init()

from sphWarpCore.math import *
from sphWarpCore.util import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
from sphWarpCore.radiusSearch.wp_radius_small import warp_radius_search_small
from sphWarpCore.radius import *
from warp.types import vector
from sphWarpCore.autograd import warpWrapper, WarpFunctionWrapper
from sphWarpCore.ops import *
import matplotlib.pyplot as plt
from sphWarpCore.ops import sphOperation_warp
from sphWarpCore.enumTypes import *
import diffSPH
from diffSPH.sampling import ParticleSet
from diffSPH.schemes.states.common import BasicState
from diffSPH.modules.density import computeDensity
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from diffSPH.kernels import *
from diffSPH.neighborhood import evaluateNeighborhood, SupportScheme, computeNeighborhoodStates
from diffSPH.enums import Operation, SupportScheme, GradientMode, LaplacianMode
from diffSPH.operations import SPHOperation
from sphWarpCore.ops import sphOperation_warp
from sphWarpCore.enumTypes import *
from sphWarpCore.sph import computeSPHCovariance_warpBackend
from diffSPH.math import pinv2x2
from diffSPH.modules.renorm import computeCovarianceMatrices
from diffSPH.operations import KernelCorrectionScheme
from diffSPH.modules.adaptiveSmoothing import *
import diffSPH
from diffSPH.sampling import ParticleSet
from diffSPH.schemes.states.common import BasicState
from diffSPH.modules.density import computeDensity
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from diffSPH.kernels import *
from diffSPH.neighborhood import evaluateNeighborhood, SupportScheme, computeNeighborhoodStates
from diffSPH.enums import Operation, SupportScheme, GradientMode, LaplacianMode
from diffSPH.operations import SPHOperation

import time
def timeFunction(func, *args, **kwargs):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    cpuBegin = time.time()
    begin.record()
    
    result = func(*args, **kwargs)
    
    end.record()
    torch.cuda.synchronize()
    cpuEnd = time.time()
    gpuTime = begin.elapsed_time(end)
    cpuTime = cpuEnd - cpuBegin
    
    return result, gpuTime, cpuTime * 1000



# wp.config.verify_autograd_array_access = True
# wp.config.verbose = True

from sphWarpCore import *

def prepData(
    nx, targetNumNeighbors, dim, device, periodic = False, warpOnly = False, noiseAmplitude = 0.1
):

    device = torch.device(device)
    numParticles = nx**dim
    hashMapLength = getNextPrime(numParticles)
    periodic = periodic
    x, h, numParticles, domain, dx = generateNeighborTestData(nx, targetNumNeighbors, dim, periodic, device)

    x += torch.randn_like(x) * dx * noiseAmplitude

    pointCloud = PointCloud(x, h)
    queryPositions = pointCloud.positions.contiguous()
    querySupports = pointCloud.supports
    referencePositions = pointCloud.positions.contiguous()
    referenceSupports = pointCloud.supports

    queryMasses = torch.ones(x.shape[0], device = x.device) * dx**dim
    referenceMasses = torch.ones(x.shape[0], device = x.device) * dx**dim

    mode = 'gather'

    particleState = ParticleState(
        positions=queryPositions, 
        supports=querySupports, 
        masses=queryMasses, 
        densities=None,
        kinds = torch.zeros(x.shape[0], dtype = torch.int32, device = x.device),
    )


    adjacency, adjacency_warp_gpu, adjacency_warp_cpu = timeFunction(radiusSearchCompactHashMap,
        particleState, domain, 
        mode = SupportScheme.SuperSymmetric,
        hashMapLengthMode = HashMapLengthMode.Fixed, fixedHashMapLength = hashMapLength
    )

    densities = warpOperation(
        particleState,
        operationProperties = OperationProperties(
            operation=WarpOperation.Density,
            kernel = KernelFunctions.Wendland2, 
            supportMode = SupportScheme.Gather
        ),
        domain = domain,
        adjacency = adjacency
    )
    
    # densities = sphOperation_warp(
    #         queryPositions, referencePositions,
    #         querySupports, referenceSupports,
    #         queryMasses, referenceMasses,
    #         None, None,
    #         None, None,
    #         domain, adjacency,
    #         operation=WarpOperation.Density,
    #         kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather
    # )
    particleState.densities = densities

    measurement = {
        'numParticles': nx**dim,
        'targetNumNeighbors': targetNumNeighbors,
        'dim': dim,
        'device': device,
        'operation': 'Adjacency',
        'backend': 'warp',
        'gpuTime': adjacency_warp_gpu,
        'cpuTime': adjacency_warp_cpu        
    }

    if warpOnly == True:
        # print("Adjacency: ", adjacency)
        return particleState, domain, adjacency, None, None, [measurement]

    particles_l = ParticleSet(positions = referencePositions, supports = referenceSupports, masses = referenceMasses, densities = torch.zeros_like(referenceMasses))

    positions_t = referencePositions.clone()
    simulationState = BasicState(
        positions = positions_t,
        supports = particles_l.supports,
        masses = particles_l.masses,
        densities = particles_l.densities,        
        velocities = torch.zeros_like(referencePositions),
        kinds = torch.zeros(referencePositions.shape[0], dtype = torch.int64, device = referencePositions.device),
        materials = torch.zeros(referencePositions.shape[0], dtype = torch.int64, device = referencePositions.device),
        UIDs = torch.arange(referencePositions.shape[0], device = referencePositions.device)
    )

    # neighborhood, neighbors = evaluateNeighborhood(simulationState, domain,  KernelType.Wendland4, verletScale =1.0, mode = SupportScheme.Gather, priorNeighborhood=None, computeHessian=False, computeDkDh=False, only_j = False)
    (neighborhood, sparseNeighborhood_), neighborhoodDiffSPHTime_gpu, neighborhoodDiffSPHTime_cpu  = timeFunction(buildNeighborhood, simulationState, simulationState, domain, verletScale = 1.0, mode = 'gather', priorNeighborhood=None)

    state, stateTime_gpu, stateTime_cpu = timeFunction(computeNeighborhoodStates, simulationState, sparseNeighborhood_, 'gather', KernelType.Wendland2, KernelType.Wendland2, True, True, False)

    neighborhood = state.get('noghost')

    measurement_diffSPH = {
        'numParticles': nx**dim,
        'targetNumNeighbors': targetNumNeighbors,
        'dim': dim,
        'device': device,
        'operation': 'Adjacency',
        'backend': 'diffSPH',
        'gpuTime': neighborhoodDiffSPHTime_gpu,
        'cpuTime': neighborhoodDiffSPHTime_cpu        
    }

    measurement_diffSPH_state = {
        'numParticles': nx**dim,
        'targetNumNeighbors': targetNumNeighbors,
        'dim': dim,
        'device': device,
        'operation': 'State',
        'backend': 'diffSPH',
        'gpuTime': stateTime_gpu,
        'cpuTime': stateTime_cpu        
    }


    simulationState.densities = densities
    return particleState, domain, adjacency, neighborhood, simulationState, [measurement, measurement_diffSPH, measurement_diffSPH_state]




def warptodiffOperation(operation, gradientMode, laplacianMode):
    diffSPHOperation = None
    if operation == WarpOperation.Density:
        diffSPHOperation = Operation.Density
    elif operation == WarpOperation.Interpolate:
        diffSPHOperation = Operation.Interpolate
    elif operation == WarpOperation.Gradient:
        diffSPHOperation = Operation.Gradient
    elif operation == WarpOperation.Laplacian:
        diffSPHOperation = Operation.Laplacian
    elif operation == WarpOperation.Divergence:
        diffSPHOperation = Operation.Divergence
    elif operation == WarpOperation.Curl:
        diffSPHOperation = Operation.Curl
    else:
        raise ValueError(f"Unsupported operation type: {operation}")
    
    diffSPHGradientMode = None
    if gradientMode == GradientScheme.Naive:
        diffSPHGradientMode = GradientMode.Naive
    elif gradientMode == GradientScheme.Difference:
        diffSPHGradientMode = GradientMode.Difference
    elif gradientMode == GradientScheme.Summation:
        diffSPHGradientMode = GradientMode.Summation
    else:
        raise ValueError(f"Unsupported gradient mode: {gradientMode}")
    
    diffSPHLaplacianMode = None
    if laplacianMode == LaplacianScheme.Naive:
        diffSPHLaplacianMode = LaplacianMode.naive
    elif laplacianMode == LaplacianScheme.Brookshaw:
        diffSPHLaplacianMode = LaplacianMode.Brookshaw
    elif laplacianMode == LaplacianScheme.Dot:
        diffSPHLaplacianMode = LaplacianMode.dot
    elif laplacianMode == LaplacianScheme.Default:
        diffSPHLaplacianMode = LaplacianMode.default
    else:
        raise ValueError(f"Unsupported laplacian mode: {laplacianMode}")
    
    return diffSPHOperation, diffSPHGradientMode, diffSPHLaplacianMode



from typing import NamedTuple

class PlotSet(NamedTuple):
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor
    kinds: torch.Tensor


from diffSPH.plotting import visualizeParticles



def plotToAxis(fig, axis, particleState, quantity, title, cmap, domain, markerSize = 1, gridVisualization = False, gridResolution = 128, mask = None):
    return visualizeParticles(
        fig, axis,
        particleState if mask is None else ParticleState(
            positions = particleState.positions[mask] if particleState.positions is not None else None,
            supports = particleState.supports[mask] if particleState.supports is not None else None,
            masses = particleState.masses[mask] if particleState.masses is not None else None,
            densities = particleState.densities[mask] if particleState.densities is not None else None,
            kinds = particleState.kinds[mask] if particleState.kinds is not None else None
        ),
        quantity = quantity[mask] if mask is not None else quantity,
        kernel = KernelType.Wendland2,
        domain = domain,

        cmap = cmap,
        markerSize = markerSize,
        gridVisualization = gridVisualization,
        gridResolution = gridResolution,

        streamLines = False,
        plotDomain = True,
        title = title,
    )