"""Build the opt-in prototype; CPU uses Torch's stable parallel_for API."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from torch.utils.cpp_extension import include_paths,library_paths

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda','hip'),default='cuda')
parser.add_argument('--arch',default='sm_120')
args=parser.parse_args()
source=Path(__file__).resolve().parent
prototype=source.parents[1]
includes=[prototype/'src/warpSPHCore/native/csrc',prototype/'src/warpSPHCore',
          args.core/'src/warpSPHCore/native/csrc',args.core/'src/warpSPHCore']
args.output.parent.mkdir(parents=True,exist_ok=True)
if args.backend=='cpu':
    command=['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',
             '-DTORCH_TARGET_VERSION=0x020a000000000000',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
             *[f'-I{p}' for p in includes+list(map(Path,include_paths()))],
             str(source/'cpu.cpp'),'-o',str(args.output),
             *[f'-L{p}' for p in library_paths()],'-ltorch_cpu','-lc10','-latomic',
             *[f'-Wl,-rpath,{p}' for p in library_paths()]]
elif args.backend=='cuda':
    compiler=shutil.which('nvcc') or str(Path(sys.prefix)/'bin/nvcc')
    command=[compiler,'-O3','-std=c++17','-shared','-lineinfo','--threads','4',
             f'-arch={args.arch}','-Xcompiler=-fPIC','--extended-lambda',
             *[f'-I{p}' for p in includes],str(source/'gpu.cu'),'-o',str(args.output)]
else:
    compiler=os.environ.get('HIPCC','/home/lu26029/miniconda3/envs/warpsph-native-hip/bin/hipcc')
    command=[compiler,'-O3','-std=c++17','--offload-arch='+args.arch,'-fPIC','-shared',
             '-x','hip',*[f'-I{p}' for p in includes],str(source/'gpu.cu'),'-o',str(args.output)]
print(json.dumps(command),flush=True)
subprocess.run(command,check=True)
