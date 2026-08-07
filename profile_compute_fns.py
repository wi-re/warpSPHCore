
from warpSPHCore import *
import torch
import math

def compute_crk_factors(particles, adjacency, domain, kernel):
    return computeCRKFactors(
        particles,
        domain,
        kernel,
        operationMode=OperationDirection.AllToAll,
        adjacency=adjacency,
    )

def compute_adjacency(particles, domain):
    return radiusSearchCompactHashMap(
        particles,
        domain,
        mode=SupportScheme.SuperSymmetric,
    )

def compute_renorm_factors(particles, adjacency, domain, kernel):
    return computeRenormalizationMatrices(
        queryParticles=particles,
        operationProperties=OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Gradient,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=GradientScheme.Difference,
        ),
        domain=domain,
        adjacency=adjacency,
        returnEigVals=True,
    )

def compute_density(particles, domain, kernel, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency
    )

def compute_interpolation(particles, domain, kernel, field, gradientMode = GradientScheme.Difference, laplacianScheme = LaplacianScheme.Brookshaw, divergenceDotMode = True, positiveDivergence = False, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency,
        queryValues=field
    )

def compute_gradient(particles, domain, kernel, field, gradientMode = GradientScheme.Difference, laplacianScheme = LaplacianScheme.Brookshaw, divergenceDotMode = True, positiveDivergence = False, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Gradient,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=gradientMode,
        ),
        domain,
        adjacency=adjacency,
        queryValues=field
    )

def compute_divergence(particles, domain, kernel, field, gradientMode = GradientScheme.Difference, laplacianScheme = LaplacianScheme.Brookshaw, divergenceDotMode = True, positiveDivergence = False, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Divergence,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=gradientMode,
            divergenceDotMode=divergenceDotMode,
            positiveDivergence=positiveDivergence
        ),
        domain,
        adjacency=adjacency,
        queryValues=field
    )

def compute_laplacian(particles, domain, kernel, field, gradientMode = GradientScheme.Difference, laplacianScheme = LaplacianScheme.Brookshaw, divergenceDotMode = True, positiveDivergence = False, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Laplacian,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=gradientMode,
            laplacianMode=laplacianScheme
        ),
        domain,
        adjacency=adjacency,
        queryValues=field
    )

def compute_curl(particles, domain, kernel, field, gradientMode = GradientScheme.Difference, laplacianScheme = LaplacianScheme.Brookshaw, divergenceDotMode = True, positiveDivergence = False, adjacency=None):
    return warpOperation(
        particles,
        OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Curl,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=gradientMode
        ),
        domain,
        adjacency=adjacency,
        queryValues=field
    )