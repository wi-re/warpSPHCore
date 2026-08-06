
from sphWarpCore import *
import torch
import math
from sphWarpCore.util import generateNeighborTestData


def make_scalar_field(dim, positions, field_mode="linear", linear_slope_x=1.0, linear_slope_y=1.0, linear_slope_z=1.0):
    if field_mode == "periodic_sinusoidal":
        if dim == 1:
            x = positions[:, 0]
            return torch.sin(math.pi * x)
        elif dim == 2:
            x = positions[:, 0]
            y = positions[:, 1]
            return torch.sin(math.pi * x) * torch.cos(math.pi * y)
        elif dim == 3:
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            return torch.sin(math.pi * x) * torch.cos(math.pi * y) * torch.sin(math.pi * z)
        else:
            raise ValueError("Unsupported dimension for scalar field")
    else:
        if dim == 1:
            x = positions[:, 0]
            return linear_slope_x * x
        elif dim == 2:
            x = positions[:, 0]
            y = positions[:, 1]
            return linear_slope_x * x + linear_slope_y * y
        elif dim == 3:
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            return linear_slope_x * x + linear_slope_y * y + linear_slope_z * z
        else:
            raise ValueError("Unsupported dimension for scalar field")

def make_vector_field(dim, positions, field_mode="linear", linear_slope_x=1.0, linear_slope_y=1.0, linear_slope_z=1.0):
    if field_mode == "periodic_sinusoidal":
        if dim == 1:
            x = positions[:, 0]
            return torch.stack((torch.sin(math.pi * x),), dim=1)
        elif dim == 2:
            x = positions[:, 0]
            y = positions[:, 1]
            return torch.stack((torch.sin(math.pi * x), torch.cos(math.pi * y)), dim=1)
        elif dim == 3:
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            return torch.stack((torch.sin(math.pi * x), torch.cos(math.pi * y), torch.sin(math.pi * z)), dim=1)
    if dim == 1:
        x = positions[:, 0]
        return torch.stack((linear_slope_x * x,), dim=1)
    elif dim == 2:
        x = positions[:, 0]
        y = positions[:, 1]
        return torch.stack((linear_slope_x * x + linear_slope_y * y, -linear_slope_x * x + 0.5 * linear_slope_y * y), dim=1)
    elif dim == 3:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return torch.stack((linear_slope_x * x + linear_slope_y * y + linear_slope_z * z, -linear_slope_x * x + 0.5 * linear_slope_y * y + linear_slope_z * z, linear_slope_x * x + linear_slope_y * y + 0.5 * linear_slope_z * z), dim=1)
    else:
        raise ValueError("Unsupported dimension for vector field")

def make_matrix_field(dim, positions, field_mode="linear", linear_slope_x=1.0, linear_slope_y=1.0, linear_slope_z=1.0):
    if dim == 1:
        x = positions[:, 0]
        return torch.stack((linear_slope_x * x,), dim=1).reshape(-1, 1, 1)
    elif dim == 2:
        x = positions[:, 0]
        y = positions[:, 1]
        return torch.stack((linear_slope_x * x + linear_slope_y * y, -linear_slope_x * x + 0.5 * linear_slope_y * y, linear_slope_x * x + linear_slope_y * y, -linear_slope_x * x + 0.5 * linear_slope_y * y), dim=1).reshape(-1, 2, 2)
    elif dim == 3:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return torch.stack((linear_slope_x * x + linear_slope_y * y + linear_slope_z * z, -linear_slope_x * x + 0.5 * linear_slope_y * y + linear_slope_z * z, linear_slope_x * x + linear_slope_y * y + 0.5 * linear_slope_z * z, -linear_slope_x * x + 0.5 * linear_slope_y * y + linear_slope_z * z, linear_slope_x * x + linear_slope_y * y + 0.5 * linear_slope_z * z, -linear_slope_x * x + 0.5 * linear_slope_y * y + linear_slope_z * z), dim=1).reshape(-1, 3, 3)
    else:
        raise ValueError("Unsupported dimension for matrix field")

def make_jittered_particles(base_positions, base_supports, base_masses, base_kinds, jitter_scale, dx, device):
    jitter = jitter_scale * dx * torch.randn_like(base_positions)
    jittered_positions = (base_positions + jitter).contiguous()
    return ParticleState(
        positions=jittered_positions,
        supports=base_supports.contiguous(),
        masses=base_masses.contiguous(),
        densities=None,
        kinds=base_kinds,
    )


def generateData(nx, targetNumNeighbors, dim, periodic, device, jitter, field_mode, linear_slope_x, linear_slope_y, linear_slope_z):

    positions, supports, _, domain, dx = generateNeighborTestData(
        nx, targetNumNeighbors, dim, periodic, device
    )

    masses = torch.full((positions.shape[0],), dx ** dim, device=device, dtype=positions.dtype)
    kinds = torch.zeros(positions.shape[0], device=device, dtype=torch.int32)

    particles = ParticleState(
        positions=positions.contiguous(),
        supports=supports.contiguous(),
        masses=masses.contiguous(),
        densities=None,
        kinds=kinds,
    )

    adjacency = radiusSearchCompactHashMap(
        particles,
        domain,
        mode=SupportScheme.SuperSymmetric,
    )

    densities = warpOperation(
        particles,
        OperationProperties(
            kernel=KernelFunctions.Wendland2,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency,
    )
    particles.densities = densities

    jittered_particles = make_jittered_particles(
        positions,
        supports,
        masses,
        kinds,
        jitter,
        dx,
        device,
    )

    jittered_adjacency = radiusSearchCompactHashMap(
        jittered_particles,
        domain,
        mode=SupportScheme.SuperSymmetric,
    )

    jittered_densities = warpOperation(
        jittered_particles,
        OperationProperties(
            kernel=KernelFunctions.Wendland2,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=jittered_adjacency,
    )
    jittered_particles.densities = jittered_densities

    scalar_field = make_scalar_field(dim, jittered_particles.positions, field_mode=field_mode, linear_slope_x=linear_slope_x, linear_slope_y=linear_slope_y, linear_slope_z=linear_slope_z)
    vector_field = make_vector_field(dim, jittered_particles.positions, field_mode=field_mode, linear_slope_x=linear_slope_x, linear_slope_y=linear_slope_y, linear_slope_z=linear_slope_z)
    matrix_field = make_matrix_field(dim, jittered_particles.positions, field_mode=field_mode, linear_slope_x=linear_slope_x, linear_slope_y=linear_slope_y, linear_slope_z=linear_slope_z)


    return jittered_particles, jittered_adjacency, domain, dx, scalar_field, vector_field, matrix_field


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

    
def measure_timing(func, n_samples, measure_cuda=True, warm_start = True, *args, **kwargs):
    import time
    if warm_start:
        func(*args, **kwargs)
    times = []
    for _ in range(n_samples):
        if measure_cuda:
            torch.cuda.synchronize()
            start_cuda = torch.cuda.Event(enable_timing=True)
            end_cuda = torch.cuda.Event(enable_timing=True)
            start_cuda.record()
        start_time = time.time()
        result = func(*args, **kwargs)
        if measure_cuda:
            end_cuda.record() 
        end_time = time.time()
        elapsed_time = end_time - start_time
        if measure_cuda:
            torch.cuda.synchronize()
            elapsed_time_gpu = start_cuda.elapsed_time(end_cuda) / 1000.0  # Convert milliseconds to seconds
            times.append((elapsed_time, elapsed_time_gpu))
        else:
            times.append((elapsed_time, None))
    return result, times

from profile_compute_fns import *


def get_default_dict(nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start):
    return {
        'operation': None,
        'adjacency': None,
        'times': None,
        'fieldType': None,
        'nx': nx,
        'dim': dim,
        'targetNumNeighbors': targetNumNeighbors,
        'periodic': periodic,
        'jitter': jitter,
        'num_samples': n_samples,
        'measure_cuda': measure_cuda,
        'warm_start': warm_start
    }


def measure_adjacency(particles, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start):
    *_, adjacency_times = measure_timing(compute_adjacency, n_samples, measure_cuda, warm_start,
        particles,
        domain
    )
    return {
        'operation': 'adjacency',
        'adjacency': False,
        'times': adjacency_times,
        'fieldType': None,
        'nx': nx,
        'dim': dim,
        'targetNumNeighbors': targetNumNeighbors,
        'periodic': periodic,
        'jitter': jitter,
        'num_samples': n_samples,
        'measure_cuda': measure_cuda,
        'warm_start': warm_start
    }

def measure_crk(particles, adjacency, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start):
    *_, crk_factor_times = measure_timing(compute_crk_factors, n_samples, measure_cuda, warm_start,
        particles,
        adjacency,
        domain,
        kernel
    )
    return {
        'operation': 'crk',
        'adjacency': True if adjacency is not None else False,
        'times': crk_factor_times,
        'fieldType': None,
        'nx': nx,
        'dim': dim,
        'targetNumNeighbors': targetNumNeighbors,
        'periodic': periodic,
        'jitter': jitter,
        'num_samples': n_samples,
        'measure_cuda': measure_cuda,
        'warm_start': warm_start
    }

def measure_renorm(particles, adjacency, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start):
    *_, renorm_times = measure_timing(compute_renorm_factors, n_samples, measure_cuda, warm_start,
        particles,
        adjacency,
        domain,
        kernel
    )
    return {
        'operation': 'renorm',
        'adjacency': True if adjacency is not None else False,
        'times': renorm_times,
        'fieldType': None,
        'nx': nx,
        'dim': dim,
        'targetNumNeighbors': targetNumNeighbors,
        'periodic': periodic,
        'jitter': jitter,
        'num_samples': n_samples,
        'measure_cuda': measure_cuda,
        'warm_start': warm_start
    }



def measure_density(particles, domain, kernel, adjacency, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start, eval_grid = True):
    results = []
    *_, density_times_adjacency = measure_timing(compute_density, n_samples, measure_cuda, warm_start, 
                                             particles, domain, kernel, adjacency=adjacency)
    results.append(get_default_dict(nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
    results[-1].update({
        'operation': 'density',
        'adjacency': True,
        'times': density_times_adjacency,
        'fieldType': 'scalar',
    })
    if eval_grid:
        *_, density_times_grid = measure_timing(compute_density, n_samples, measure_cuda, warm_start,
                                                particles, domain, kernel, adjacency=None)
        results.append(get_default_dict(nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
        results[-1].update({
            'operation': 'density',
            'adjacency': False,
            'times': density_times_grid,
            'fieldType': 'scalar',
        })
    return results



def measure_operation(particles, domain, kernel, adjacency, 
    field,
    nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start,
    operation = WarpOperation.Interpolate, gradientMode = GradientScheme.Difference, divergenceDotMode = True, positiveDivergence = False, laplacianScheme = LaplacianScheme.Brookshaw,
    eval_grid = True):
    if operation == WarpOperation.Interpolate:
        compute_func = compute_interpolation
    elif operation == WarpOperation.Gradient:
        compute_func = compute_gradient
    elif operation == WarpOperation.Divergence:
        compute_func = compute_divergence
    elif operation == WarpOperation.Laplacian:
        compute_func = compute_laplacian
    elif operation == WarpOperation.Curl:
        compute_func = compute_curl
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    results = []

    *_, times_adjacency = measure_timing(compute_func, n_samples, measure_cuda, warm_start,
        particles, domain, kernel, field=field, gradientMode=gradientMode, divergenceDotMode=divergenceDotMode, positiveDivergence=positiveDivergence, laplacianScheme=laplacianScheme, adjacency=adjacency)
    results.append(get_default_dict(nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
    results[-1].update({
        'operation': operation.name.lower(),
        'adjacency': True,
        'fieldType': 'scalar' if field.ndim == 1 else 'vector' if field.ndim == 2 else 'matrix',
        'times': times_adjacency,
    })

    if eval_grid:
        *_, times_grid = measure_timing(compute_func, n_samples, measure_cuda, warm_start,
            particles, domain, kernel, field=field, gradientMode=gradientMode, divergenceDotMode=divergenceDotMode, positiveDivergence=positiveDivergence, laplacianScheme=laplacianScheme, adjacency=None)
        results.append(get_default_dict(nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
        results[-1].update({
            'operation': operation.name.lower(),
            'adjacency': False,
            'fieldType': 'scalar' if field.ndim == 1 else 'vector' if field.ndim == 2 else 'matrix',
            'times': times_grid,
        })
    return results


def measure_timings(
    particles, adjacency, domain, kernel, scalar_field, vector_field, matrix_field,
    nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start,
    eval_grid = True,
    eval_adjacency = True,
    eval_crk = True,
    eval_renorm = True,
    eval_density = True,
    eval_interpolate = True,
    eval_gradient = True,
    eval_laplacian = True,
    eval_curl = True,
    eval_divergence = True,
    eval_vector_field = True,
    eval_matrix_field = True
):
    results = []
    if eval_adjacency:
        results.append(measure_adjacency(particles, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
    if eval_crk:
        results.append(measure_crk(particles, adjacency, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
    if eval_renorm:
        results.append(measure_renorm(particles, adjacency, domain, kernel, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start))
    if eval_density:
        results.extend(measure_density(particles, domain, kernel, adjacency, nx, dim, targetNumNeighbors, periodic, jitter, n_samples, measure_cuda, warm_start, eval_grid=eval_grid))
    if eval_interpolate:
        results.extend(measure_operation(particles, domain, kernel, adjacency, field=scalar_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Interpolate, eval_grid=eval_grid))
        if eval_vector_field:
            results.extend(measure_operation(particles, domain, kernel, adjacency, field=vector_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Interpolate, eval_grid=eval_grid))
        if eval_matrix_field:
            results.extend(measure_operation(particles, domain, kernel, adjacency, field=matrix_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Interpolate, eval_grid=eval_grid))
    if eval_gradient:
        results.extend(measure_operation(particles, domain, kernel, adjacency, field=scalar_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Gradient, eval_grid=eval_grid))
        if eval_vector_field:
            results.extend(measure_operation(particles, domain, kernel, adjacency, field=vector_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Gradient, eval_grid=eval_grid))

    if eval_laplacian:
        results.extend(measure_operation(particles, domain, kernel, adjacency, field=scalar_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Laplacian, eval_grid=eval_grid))
        if eval_vector_field:
            results.extend(measure_operation(particles, domain, kernel, adjacency, field=vector_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Laplacian, eval_grid=eval_grid))

    if eval_curl:
        results.extend(measure_operation(particles, domain, kernel, adjacency, field=vector_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Curl, eval_grid=eval_grid))

    if eval_divergence:
        results.extend(measure_operation(particles, domain, kernel, adjacency, field=vector_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Divergence, eval_grid=eval_grid))
        if eval_matrix_field:
            results.extend(measure_operation(particles, domain, kernel, adjacency, field=matrix_field, nx = nx, dim = dim, targetNumNeighbors = targetNumNeighbors, periodic = periodic, jitter = jitter, n_samples = n_samples, measure_cuda = measure_cuda, warm_start = warm_start, operation=WarpOperation.Divergence, eval_grid=eval_grid))

    return results
