import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import os
import torch
if torch.cuda.is_available():
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'
import warp as wp; wp.init()

import torch
from waves.gencase import *
from waves.sample import generateInitialVariables, SamplingScheme
from waves.sampling import sampleParticles
# from waves.util import plotState, plotInitialState
# from simulation import runSimulation
from waves.utils import getCurrentTimestamp
from argparse import ArgumentParser
from waves.casefile import argparse_defaults_from_casefile, load_casefile

from waves.systemv3 import WaveSystemStatev3
from waves.sample import smoothState
from waves.systemv3 import WaveSystemv3, f_wave_equation
from integrators.integration import *
from waves.utils import *
from sphWarpCore import *
from waves.systemv3 import WaveSystemStatev3

from sphWarpCore.radiusSearch.verlet import *
from sphWarpCore.radius import AdjacencyList
from sphWarpCore.operations import *
from sphWarpCore.enumTypes import *

from sphWarpCore import *
from waves.systemv3 import sampleInitialWaveState
from waves.systemv3 import computeDt
from waves.sampling import finalizeWaveSystemSetup
from waves.shape_generation import populateSourceObstacleGridsStructured

from waves.systemv3 import SimulationConfig

from waves.config import *
from waves.casefile import build_configs_from_casefile



pre_parser = ArgumentParser(add_help=False)
pre_parser.add_argument('--casefile', type=str, default=None, help='Path to a TOML case file')
pre_args, remaining_args = pre_parser.parse_known_args()

casefile_defaults = {}
if pre_args.casefile:
    casefile_defaults = argparse_defaults_from_casefile(load_casefile(pre_args.casefile))

parser = ArgumentParser(parents=[pre_parser])

parser.add_argument('--nx', type=int, default=128, help='Number of grid points in one dimension')
parser.add_argument('--sampling', type=str, default='regular', help='Particle sampling scheme: regular, regular_jittered, glass, optimal, random')

parser.add_argument('--dt', type=float, default=0.0025, help='Time step size')
parser.add_argument('--timeLimit', type=float, default=4.0, help='Total simulation time')
parser.add_argument('--adaptiveDt', action='store_true', help='Enable adaptive time stepping')
parser.add_argument('--cflFactor', type=float, default=0.3, help='CFL factor for adaptive time stepping')
parser.add_argument('--nIter', type=int, default=1024, help='Number of iterations')

parser.add_argument('--uMagnitudes', type=float, nargs='*', default=[10], help='List of possible wave speeds for obstacles')
parser.add_argument('--uRandomMagnitude', action='store_true', help='Enable random magnitudes for sources')
parser.add_argument('--uRandomMin', type=float, default=-10.0, help='Minimum magnitude for sources when uRandomMagnitude is enabled')
parser.add_argument('--uRandomMax', type=float, default=10.0, help='Maximum magnitude for sources when uRandomMagnitude is enabled')

parser.add_argument('--smoothICs', action='store_true', help='Enable smoothing of initial conditions')
parser.add_argument('--smoothIters', type=int, default=4, help='Number of smoothing iterations for initial conditions')

parser.add_argument('--plotInterval', type=int, default=10, help='Plotting interval')
parser.add_argument('--export', action='store_true', help='Whether to export simulation data')
parser.add_argument('--exportImages', action='store_true', help='Whether to export simulation images')
parser.add_argument('--exportInitial', action='store_true', help='Whether to export initial conditions')

parser.add_argument('--filePrefix', type=str, default='waveEqn', help='Prefix for output files')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

parser.add_argument('--domainBox', action='store_true', help='Enable domain boundary box')
parser.add_argument('--domainDamping', action='store_true', help='Enable domain boundary damping')

parser.add_argument('--enableNoise', action='store_true', help='Enable noise addition to initial conditions')
parser.add_argument('--noiseType', type=str, default='perlin', help='Type of noise to add: perlin, uniform, normal')
parser.add_argument('--noiseAmplitude', type=float, default=0.02, help='Amplitude of noise to add to initial conditions')
parser.add_argument('--noiseSmoothIter', type=int, default=4, help='Number of smoothing iterations for noise')
parser.add_argument('--noiseSeed', type=int, default=42, help='Random seed for noise generation')

parser.add_argument('--boundarySpeed', type=float, default=0.01, help='Wave speed at boundaries')
parser.add_argument('--obstacleSpeeds', type=float, nargs='*', default=[0.5], help='List of possible wave speeds for obstacles')


parser.add_argument('--defaultSpeed', type=float, default=1.0, help='Default wave speed in the medium')
parser.add_argument('--randomObstacleSpeed', action='store_true', help='Enable random wave speeds for obstacles')
parser.add_argument('--obstacleSpeedMin', type=float, default=0.3, help='Minimum wave speed for obstacles when randomObstacleSpeed is enabled')
parser.add_argument('--obstacleSpeedMax', type=float, default=0.7, help='Maximum wave speed for obstacles when randomObstacleSpeed is enabled')
parser.add_argument('--figureDpi', type=int, default=200, help='DPI for saved figures')
parser.add_argument('--caseIndex', type=int, default=1, help='Index for the simulation case (used in file naming)')
parser.add_argument('--boundaryCaseIndex', type=int, default=1, help='Index for the simulation case (used in file naming)')


parser.add_argument('--exportFps', type=int, default=100, help='Frames per second for exported videos')

# parser.add_argument('--figureDpi --dt=0.01 --exportImages --exportInitial', type=int, default=200, help='DPI for saved figures')

import shlex
# splitArgs = shlex.split()

if casefile_defaults:
    parser.set_defaults(**casefile_defaults)

args = parser.parse_args(remaining_args)
args.casefile = pre_args.casefile


verbose = args.verbose
if verbose:
    print("Simulation Configuration:")
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

# exit()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if verbose:
    print(f'Using device: {device}')

plotInterval = args.plotInterval
nIter = args.nIter

dt = args.dt
nx = args.nx
# uMagnitude = args.uMagnitude
sampling = args.sampling
samplingScheme = None
export = args.export
timestamp = getCurrentTimestamp()


for scheme in SamplingScheme:
    if scheme.name == sampling:
        samplingScheme = scheme
        if verbose:
            print(f'Using sampling scheme: {samplingScheme.name}')
        break
if samplingScheme is None:
    raise ValueError(f'Unknown sampling scheme: {sampling}')

prefix = f'{args.filePrefix}_{args.caseIndex}_{args.boundaryCaseIndex}_{nx}_{sampling}'
os.makedirs('output', exist_ok=True)
folderName = f'{prefix}_{timestamp}'
os.makedirs(f'output/{folderName}', exist_ok=True)
if verbose:
    print(f'Output folder: output/{folderName}')

L = 2
dim = 2
n_h = 4
dtype = torch.float32

if args.casefile:
    config, caseConfig = build_configs_from_casefile(args.casefile, device=device, dtype=dtype)
    config.samplingScheme = samplingScheme
    if verbose:
        print(f"Loaded case file: {args.casefile}")
else:
    config = SimulationConfig(
        domain = buildDomainDescription(L, dim, True, device, dtype),
        dim = dim,
        kernel = KernelFunctions.Wendland4,
        targetNeighbors = n_h_to_nH(n_h, dim),
        supportMode = SupportScheme.SuperSymmetric,
        gradientMode = GradientScheme.Difference,
        laplacianMode = LaplacianScheme.Brookshaw,
        integrationScheme = IntegrationSchemeType.rungeKutta4,
        samplingScheme = samplingScheme,
        device = device,
        dtype = dtype,
        dt = args.dt,
        adaptiveDt = args.adaptiveDt,
        cflFactor=args.cflFactor,
    )

    caseConfig = CaseConfig(
        name = f'{prefix}_{timestamp}',
        description = 'Wave equation simulation with SPH method',
        domainBox = args.domainBox,
        domainDamping = args.domainDamping,
        smoothICs=args.smoothICs,
        smoothIterations=args.smoothIters,
        defaultSpeed = args.defaultSpeed,
        defaultBoundarySpeed=args.boundarySpeed,
        defaultObstacleSpeed=args.obstacleSpeeds,
        noisyICs = args.enableNoise,
        noiseType = args.noiseType,
        noiseAmplitude = args.noiseAmplitude,
        noiseSmoothIters = args.noiseSmoothIter,
        noiseSeed = args.noiseSeed,
    )

integrator = getIntegrator(config.integrationScheme)


if not args.casefile:
    if verbose:
        print("No case file provided, using default sources and obstacles")
    caseConfig.sources = [
        WaveSource(
            shapeSpec=ShapeSpec(
                kind='sphere',
                position=(-L / 4, 0.0),
                params={'radius': 0.1},
            ),
            magnitude=10.0,
        ),
    ]

    caseConfig.obstacles = [
        WaveBoundary(
            shapeSpec=ShapeSpec(
                kind='prism',
                position=(0.15, 0.0),
                params={
                    'side_length': 0.3,
                    'add_wall': True,
                    'wall_thickness': 0.04,
                },
            ),
            speed=0.5,
        ),
    ]
else:
    print(f"Using {len(caseConfig.sources)} source(s) and {len(caseConfig.obstacles)} obstacle(s) from case file")

particleState = sampleInitialWaveState(args.nx, config, caseConfig)

uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid = genInitial(
    particleState, config,
    args.nx,
    domainBox = args.domainBox,
    domainDamping = args.domainDamping,
)

uSourceGrid, cSourceGrid, sourceMagnitudes, obstacleSpeeds = populateSourceObstacleGridsStructured(
    particleState,
    config,
    caseConfig,
    uSourceGrid,
    cSourceGrid,
 )

if verbose:
    print(f"Generated {len(sourceMagnitudes)} source(s) and {len(obstacleSpeeds)} obstacle speed entry(ies).")


waveSystem, dt = finalizeWaveSystemSetup(
    particleState,
    uGrid, vGrid, cGrid, dampGrid,
    uSourceGrid, cSourceGrid,
    sourceMagnitudes, obstacleSpeeds,
    config, caseConfig,
)
runningState = waveSystem.initializeNewState(verbose=verbose)
# runningState.state.u[runningState.state.positions[:,0] > 0] = 0

# We need to ensure that the timestep is compatible with the export fps.
# The goal is to set the dt such that an integer multiple (the plot interval) of dt corresponds to the desired export frame rate.
if args.exportImages:
    desired_dt = 1.0 / args.exportFps
    current_ratio = desired_dt / dt
    if dt >= desired_dt:
        dt = desired_dt
        plotInterval = 1
    else:
        ratioUp = int(np.ceil(current_ratio))
        dt = desired_dt / ratioUp
        plotInterval = ratioUp
    if verbose:
        print(f"Adjusted time step to {dt:.6f} to match export frame rate of {args.exportFps} FPS with plot interval of {plotInterval}")


# ###############################################################################
#                              Visualization Setup                             #
# ###############################################################################
from warpPlot import *

gridVis = True
gridOptions = GridVisualization(
    resolution = args.nx,
)
markerSize = 16

plotter = visualize(
    particleState = runningState.state,
    domain = config.domain,
    quantities = {
        "A": uSourceGrid.to(torch.float32),
        "B": runningState.state.u,
        "C": runningState.state.v,
        "D": cSourceGrid.to(torch.float32),
        "E": runningState.state.c,
        "F": runningState.state.damping,
    },
    plotOptions = {
        "A": PlottingOptions( colorMap = QualitativeColorMap.tab10, markerSize = markerSize, plotTitle = "u Source Grid", gridVisualization = gridOptions if gridVis else None),
        "B": PlottingOptions( colorMap = DivergingColorMap.managua, markerSize = markerSize, plotTitle = "u", gridVisualization = gridOptions if gridVis else None),
        "C": PlottingOptions( colorMap = DivergingColorMap.vanimo,  markerSize = markerSize, plotTitle = "v", gridVisualization = gridOptions if gridVis else None),
        "D": PlottingOptions( colorMap = QualitativeColorMap.tab10, markerSize = markerSize, plotTitle = "c Source Grid", gridVisualization = gridOptions if gridVis else None),
        "E": PlottingOptions( colorMap = UniformColorMap.viridis, markerSize = markerSize, plotTitle = "c", gridVisualization = gridOptions if gridVis else None),
        "F": PlottingOptions( colorMap = UniformColorMap.cividis, markerSize = markerSize, plotTitle = "damping", gridVisualization = gridOptions if gridVis else None),
    },
    figTitle = "Wave Equation Example",
    mosaic = """ABC
    DEF""",
    figsize= (11,5),
    backend='matplotlib',
    # backend='pyVista',
    # backendOptions = {
    #     # In notebooks, use trame for reliable live updates.
    #     'jupyter_backend': 'trame',
    # }
)

if args.exportInitial:
    plotter.export(f'output/{folderName}/initial_visualization.png')

# if args.exportInitial:
#     plotter.fig.(f'output/{folderName}/initial_visualization.png', dpi = args.figureDpi)

markerSize = 0.5
plotter = visualize(
    particleState = runningState.state,
    domain = config.domain,
    quantities = {
        "A": runningState.state.u,
        "B": runningState.state.v,
    },
    plotOptions = {
        "A": PlottingOptions(
            colorMap = DivergingColorMap.managua,
            markerSize = markerSize,
            midPoint = 0.0,
            quantityScaling = PlotScaling.Symmetric,
            plotTitle = "u",
            gridVisualization = GridVisualization(
                resolution = args.nx,
            ),
        ),
        "B": PlottingOptions(
            colorMap = DivergingColorMap.vlag,
            markerSize = markerSize,
            midPoint = 0.0,
            quantityScaling = PlotScaling.Symmetric,
            plotTitle = "v",
            gridVisualization = GridVisualization(
                resolution = args.nx,
            ),
        ),
    },
    figTitle = "Wave Equation Example",
    mosaic = 'AB',
    figsize= (11,5),
    backend='vispy',
    # backend='pyVista',
    # backendOptions = {
    #     # In notebooks, use trame for reliable live updates.
    #     'jupyter_backend': 'trame',
    # }
)

plotter.show()


if args.exportImages:
    plotter.export(f'output/{folderName}/frame_00000.png', dpi = args.figureDpi)


    
import numpy as np
# Optional: Apply spectral filtering instead of (or in addition to) global damping
# Uncomment these lines in the integration loop below to use spectral filtering
use_spectral_filter = False  # Set to True to enable
k_cutoff_fraction = 0.7      # Start damping at 70% of max wavenumber
spectral_power = 4           # Sharpness of spectral cutoff

t = 0.0
# plotInterval = 10

initialUMagnitude = torch.sum(runningState.state.u.abs()).cpu().item()
initialVMagnitude = torch.sum(torch.abs(runningState.state.v)).cpu().item()
nSteps = int(args.timeLimit / dt)

for i in (tq := tqdm(range(nSteps), leave = True)):
    result = integrator.function(runningState,
                        dt = dt,
                        f = f_wave_equation,
                        verbose = False)

    runningState = result.state

    # Optional: Apply spectral filtering for periodic domains (alternative to global damping)
    if use_spectral_filter:
        runningState.state.u = apply_spectral_filter(
            runningState.state.u, nx, 2, k_cutoff_fraction, spectral_power)
        runningState.state.v = apply_spectral_filter(
            runningState.state.v, nx, 2, k_cutoff_fraction, spectral_power)
    
    t += dt
    tq.set_description(f"Simulating: t = {t:.4f}s, |u| = {torch.sum(runningState.state.u).cpu().item()/initialUMagnitude:.4f}(initial: {initialUMagnitude:.4f}), |v| = {torch.sum(torch.abs(runningState.state.v)).cpu().item():.4f}")


    if (i % plotInterval == 0 or i == nIter - 1) and i > 0:
        plotter.updateQuantities(
            {
                "A": runningState.state.u,
                "B": runningState.state.v,
            },
        )
        if args.exportImages:
            plotter.export(f'output/{folderName}/frame_{i:05d}.png', dpi = args.figureDpi)


imagePrefix = f'output/{folderName}'
import subprocess
import shlex
output = 'timestamp'
scale = 1280

if args.exportImages:

    command = '/usr/bin/ffmpeg -loglevel warning -hide_banner -y -framerate 50 -f image2 -pattern_type glob -i '+ imagePrefix + '/frame_*.png -c:v libx264 -pix_fmt yuv420p -b:v 20M -r 50 ' + imagePrefix + '/output.mp4'
    commandB = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -vf "fps=50,scale={scale}:-1:flags=lanczos,palettegen" {imagePrefix}/palette.png'
    commandC = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -i {imagePrefix}/palette.png -filter_complex "fps=50,scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse" {imagePrefix}/output.gif'

    print('Creating video from  frames (frame count: {})'.format(len(os.listdir(imagePrefix))))
    subprocess.run(shlex.split(command))
    print('Creating gif palette')
    subprocess.run(shlex.split(commandB))
    print('Creating gif')
    subprocess.run(shlex.split(commandC))
    print('Done')
