import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
       
import torch
import numpy as np
import warp as wp

# Initialize Warp
wp.config.verify_autograd_array_access = False
wp.config.verbose = False
wp.init()

from sphWarpCore import radiusSearchCompactHashMap, sphOperation_warp
from sphWarpCore.enumTypes import *
from sphWarpCore.util import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
from sphWarpCore.radiusSearch.wp_radius_small import warp_radius_search_small
from sphWarpCore.util import getNextPrime, generateNeighborTestData

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

    
from dataclasses import dataclass
@torch.jit.script
@dataclass#(slots=True)
class PointCloud:
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: torch.Tensor

    def __ne__(self, other: 'PointCloud') -> bool:
        return not self.__eq__(other)



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

    adjacency, adjacency_warp_gpu, adjacency_warp_cpu = timeFunction(radiusSearchCompactHashMap,
            queryPositions, referencePositions, 
            querySupports, referenceSupports, 
            domain.periodic, domain,
            mode, hashMapLength
    )
    
    densities = sphOperation_warp(
            queryPositions, referencePositions,
            querySupports, referenceSupports,
            queryMasses, referenceMasses,
            None, None,
            None, None,
            domain, adjacency,
            operation=WarpOperation.Density,
            kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather
    )
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

    return queryPositions, referencePositions, querySupports, referenceSupports, queryMasses, referenceMasses, densities, densities, domain, adjacency, None, None, [measurement]


device = torch.device('cpu')
device = torch.device('cuda')
targetNumNeighbors = 50
nx = 128
dim = 2
numParticles = nx**dim

warpOnly = False
periodic = True

queryPositions, referencePositions, \
        querySupports, referenceSupports, \
        queryMasses, referenceMasses, \
        queryDensities, referenceDensities, \
        domain, adjacency, neighborhood, simulationState, measurements = prepData(nx, targetNumNeighbors, dim, device, periodic, warpOnly)

f = torch.randn(numParticles, device=device, dtype=torch.float32)

f_linear = queryPositions[:,0] * 5 + 10
f_grad_x = torch.full_like(f_linear, 5.0)
f_grad_y = torch.zeros_like(f_linear)


linear_gradient_warp = sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    f_linear, f_linear,
    domain = domain, adjacency = adjacency, 
    operation = WarpOperation.Gradient, operationMode = OperationDirection.AllToAll,
    kernel = KernelFunctions.Wendland2, supportMode = SupportScheme.Gather,
    gradientMode = GradientScheme.Difference
)

dx = 2.0 / nx
mask = torch.logical_and(queryPositions[:,0] > -1 + dx * 4, queryPositions[:,0] < 1 - dx * 4)

print("Linear Gradient (WarpSPH): ", linear_gradient_warp[mask])

mean_error_x = torch.mean(torch.abs(linear_gradient_warp[:,0] - f_grad_x)[mask])
mean_error_y = torch.mean(torch.abs(linear_gradient_warp[:,1] - f_grad_y)[mask])

print("Mean Absolute Error in X Gradient: ", mean_error_x.item())
print("Mean Absolute Error in Y Gradient: ", mean_error_y.item())