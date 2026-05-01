import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import matplotlib.pyplot as plt
import os
import cProfile
import pstats
import io
import gc
import time
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'
import warp as wp; wp.init()

import torch
# from compressibleSPH.gencase import *
from compressibleSPH.sample import generateInitialVariables, SamplingScheme
from compressibleSPH.sampling import  sampleParticles
# from waves.util import plotState, plotInitialState
# from simulation import runSimulation
from compressibleSPH.utils import getCurrentTimestamp
from argparse import ArgumentParser
from compressibleSPH.casefile import argparse_defaults_from_casefile, load_casefile

# from compressibleSPH.systemv3 import WaveSystemStatev3
from compressibleSPH.sample import smoothState
# from compressibleSPH.systemv3 import WaveSystemv3, f_wave_equation
from integrators.integration import *
from compressibleSPH.utils import *
from sphWarpCore import *
# from compressibleSPH.systemv3 import WaveSystemStatev3

from sphWarpCore.radiusSearch.verlet import *
from sphWarpCore.radius import AdjacencyList
from sphWarpCore.operations import *
from sphWarpCore.enumTypes import *

from sphWarpCore import *
# from waves.sampling import sampleParticles
# from compressibleSPH.systemv3 import sampleInitialWaveState
# from compressibleSPH.systemv3 import computeDt
from compressibleSPH.sampling import finalizeWaveSystemSetup
from compressibleSPH.shape_generation import populateSourceObstacleGridsStructured

from compressibleSPH.config import SimulationConfig, CompressibleSPHConfig
from compressibleSPH.caseUtils.sod import *
from compressibleSPH.util import *
from compressibleSPH.schemes import *
from sphWarpCore.diffusion.viscosity import DiffusionParameters

from diffSPH.enums import ViscositySwitch, KernelType
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from tqdm.autonotebook import tqdm

import argparse

parser = ArgumentParser(description="Run Sod shock tube simulation with warpSPH core.")
parser.add_argument('--nx', type=int, default=128, help='Number of particles along x-axis')

args = parser.parse_args()

# ENABLE_STACK_PROFILE = os.getenv('SPH_STACK_PROFILE', '0') == '1'
# STACK_GROUP_N = int(os.getenv('SPH_STACK_GROUP_N', '8'))
# STACK_ROW_LIMIT = int(os.getenv('SPH_STACK_ROW_LIMIT', '60'))
# STACK_EXPORT_DIR = os.getenv('SPH_STACK_EXPORT_DIR', '/tmp/sod_stacks')
# ENABLE_CPROFILE = os.getenv('SPH_CPROFILE', '0') == '1'
# CPROFILE_SORT = os.getenv('SPH_CPROFILE_SORT', 'cumulative')
# CPROFILE_LIMIT = int(os.getenv('SPH_CPROFILE_LIMIT', '80'))
# PROFILE_WAIT_STEPS = int(os.getenv('SPH_PROFILE_WAIT_STEPS', '4'))
# PROFILE_WARMUP_STEPS = int(os.getenv('SPH_PROFILE_WARMUP_STEPS', '4'))
# PROFILE_ACTIVE_STEPS = int(os.getenv('SPH_PROFILE_ACTIVE_STEPS', '16'))
# PROFILE_REPEAT = int(os.getenv('SPH_PROFILE_REPEAT', '1'))
# DISABLE_GC_DURING_MEASURE = os.getenv('SPH_DISABLE_GC', '1') == '1'
# ENABLE_TIMING_SUMMARY = os.getenv('SPH_TIMING_SUMMARY', '1') == '1'
# TIMING_REPEATS = int(os.getenv('SPH_TIMING_REPEATS', '5'))
# TIMING_WARMUP_STEPS = int(os.getenv('SPH_TIMING_WARMUP_STEPS', '4'))


# def synchronize_device() -> None:
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()


# class measured_region:
#     def __enter__(self):
#         self.gc_was_enabled = gc.isenabled()
#         if DISABLE_GC_DURING_MEASURE and self.gc_was_enabled:
#             gc.disable()
#         synchronize_device()
#         self.start = time.perf_counter()
#         return self

#     def __exit__(self, exc_type, exc, tb):
#         synchronize_device()
#         self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
#         if DISABLE_GC_DURING_MEASURE and self.gc_was_enabled:
#             gc.enable()
#         return False


# def print_timing_summary(label: str, timings_ms: list[float]) -> None:
#     if not timings_ms:
#         return

#     timings = torch.tensor(timings_ms, dtype=torch.float64)
#     mean_ms = timings.mean().item()
#     median_ms = timings.median().item()
#     min_ms = timings.min().item()
#     max_ms = timings.max().item()
#     std_ms = timings.std(unbiased=False).item() if timings.numel() > 1 else 0.0
#     print(
#         f"\n=== {label} ===\n"
#         f"repeats={len(timings_ms)} mean_ms={mean_ms:.3f} median_ms={median_ms:.3f} "
#         f"std_ms={std_ms:.3f} min_ms={min_ms:.3f} max_ms={max_ms:.3f}"
#     )


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


nx = args.nx
gamma = 5/3
leftState = sodInitialState(1, 1, 0)
rightState = sodInitialState(0.1795, 0.25, 0)
samplingRatio = 1
smoothIC = True
timeLimit = 0.15

L = 2
dim = 1
n_h = 4
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

config = SimulationConfig(
    domain = buildDomainDescription(L, dim, True, device, dtype),
    dim = dim,
    kernel = KernelFunctions.Wendland2,
    targetNeighbors = n_h_to_nH(n_h, dim),
    supportMode = SupportScheme.Gather,
    gradientMode = GradientScheme.Difference,
    laplacianMode = LaplacianScheme.Brookshaw,
    integrationScheme = IntegrationSchemeType.rungeKutta4,
    samplingScheme = SamplingScheme.regular,
    device = device,
    dtype = dtype,
    dt = 1e-3,
    adaptiveDt = False,
    cflFactor=0.3,
)
# print('Hello world')

diffusionParams = DiffusionParameters()
diffusionParams.c_s = 1
diffusionParams.C_l = 1
diffusionParams.C_q = 0
diffusionParams.Cu_l = 1
diffusionParams.Cu_q = 0
diffusionParams.K = 1.0
diffusionParams.thermalConductivity = 0.5
diffusionParams.viscosityTerm = 7
diffusionParams.thermalConducitiyTerm = 9
diffusionParams.scaleBeta = False
diffusionParams.monaghanSwitch = True
diffusionParams.correctXi = True

compressibleSPHConfig = CompressibleSPHConfig(
    gamma = gamma,
    rho0 = leftState.rho,
    diffusionParams = diffusionParams,
)
integrator = getIntegrator(config.integrationScheme)

compSystem = buildSod1D(
    nx,
    samplingRatio,
    leftState,
    rightState,
    gamma, config,
    smoothIC
)

if torch.any(torch.isnan(compSystem.state.masses)):
    print("NaNs in initial mass")
if torch.any(torch.isnan(compSystem.state.densities)):
    print("NaNs in initial density")
if torch.any(torch.isnan(compSystem.state.supports)):
    print("NaNs in initial supports")
if torch.any(torch.isnan(compSystem.state.pressures)):
    print("NaNs in initial pressures")


# # Warm start
# for i in range(16):
#     update, adjacency, state = compressibleSPH_Monaghan(compSystem, config.dt, config, compressibleSPHConfig, verbose = True)


# def run_steps(num_steps: int, profiler=None):
#     for i in range(num_steps):
#         integrator.function(
#             state = compSystem,
#             f = compressibleSPH_Monaghan,
#             dt = config.dt,
#             config = config,
#             compParams = compressibleSPHConfig,
#             verbose = False,
#         )
#         if profiler is not None:
#             profiler.step()


# if ENABLE_TIMING_SUMMARY:
#     timing_runs = []
#     for _ in range(TIMING_REPEATS):
#         run_steps(TIMING_WARMUP_STEPS)
#         with measured_region() as timing_region:
#             run_steps(PROFILE_ACTIVE_STEPS)
#         timing_runs.append(timing_region.elapsed_ms)
#     print_timing_summary("TIMING SUMMARY", timing_runs)


# if ENABLE_CPROFILE:
#     profiler = cProfile.Profile()
#     with measured_region():
#         profiler.enable()
#         run_steps(PROFILE_ACTIVE_STEPS)
#         profiler.disable()

#     stats_stream = io.StringIO()
#     stats = pstats.Stats(profiler, stream=stats_stream).sort_stats(CPROFILE_SORT)
#     stats.print_stats(CPROFILE_LIMIT)
#     stats.print_callers(CPROFILE_LIMIT)
#     print('\n=== CPROFILE (Python call paths) ===')
#     print(stats_stream.getvalue())


# profile_schedule = schedule(
#     wait=PROFILE_WAIT_STEPS,
#     warmup=PROFILE_WARMUP_STEPS,
#     active=PROFILE_ACTIVE_STEPS,
#     repeat=PROFILE_REPEAT,
# )
# profile_total_steps = PROFILE_REPEAT * (PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS)


# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     with_stack=ENABLE_STACK_PROFILE,
#     schedule=profile_schedule,
# ) as prof:
#     with measured_region() as profiled_region:
#         run_steps(profile_total_steps, profiler=prof)

# print(
#     f"\n=== PROFILE WINDOW ===\n"
#     f"wait_steps={PROFILE_WAIT_STEPS} warmup_steps={PROFILE_WARMUP_STEPS} "
#     f"active_steps={PROFILE_ACTIVE_STEPS} repeat={PROFILE_REPEAT} "
#     f"wall_ms={profiled_region.elapsed_ms:.3f}"
# )

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

# if ENABLE_STACK_PROFILE:
#     print('\n=== STACK GROUPED (self_cpu_time_total) ===')
#     print(
#         prof.key_averages(group_by_stack_n=STACK_GROUP_N).table(
#             sort_by='self_cpu_time_total',
#             row_limit=STACK_ROW_LIMIT,
#         )
#     )

#     print('\n=== STACK GROUPED (cpu_time_total) ===')
#     print(
#         prof.key_averages(group_by_stack_n=STACK_GROUP_N).table(
#             sort_by='cpu_time_total',
#             row_limit=STACK_ROW_LIMIT,
#         )
#     )

#     os.makedirs(STACK_EXPORT_DIR, exist_ok=True)
#     cpu_stack_file = os.path.join(STACK_EXPORT_DIR, 'cpu_stacks.txt')
#     cuda_stack_file = os.path.join(STACK_EXPORT_DIR, 'cuda_stacks.txt')
#     prof.export_stacks(cpu_stack_file, 'self_cpu_time_total')
#     prof.export_stacks(cuda_stack_file, 'self_cuda_time_total')
#     print(f'Exported stack summaries to: {cpu_stack_file} and {cuda_stack_file}')