#!/usr/bin/env python3
"""Run the ordinary public benchmark against one explicitly saved native artifact.

The artifact must come from this experiment's compatible Torch/operator schema.
A fresh process loads exactly one library. The ordinary source hashes describe
this checkout; artifact SHA-256 and its supplied source revision identify the
executed native code. This bypass is confined to the experiment harness.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',type=Path,required=True)
parser.add_argument('--revision',required=True)
parser.add_argument('benchmark_args',nargs=argparse.REMAINDER)
args=parser.parse_args()
forward=args.benchmark_args
if forward and forward[0]=='--':forward=forward[1:]
os.environ.update(WARPSPHCORE_BACKEND='warp',warpSPHCore_INDEX_DTYPE='int64',
                  warpSPHCore_OFFSET_DTYPE='int64')
if '--precision' in forward:
    os.environ['warpSPHCore_PRECISION']=forward[forward.index('--precision')+1]
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'src'),str(root/'scripts/benchmarks')]
import torch
import warpSPHCore.native._build as build
torch.ops.load_library(str(args.artifact.resolve()))
build._extension=torch.ops.warpsph_native
import bench_backends as bench
import warpSPHCore as sph
assert sph.get_torch_index_dtype()==torch.int64
assert sph.get_torch_offset_dtype()==torch.int64
sys.argv=[str(root/'scripts/benchmarks/bench_backends.py'),*forward]
bench.main()
output=Path(forward[forward.index('--output')+1])
report=json.loads(output.read_text())
report['execution_artifact']=dict(path=str(args.artifact.resolve()),source_revision=args.revision,
    sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
    caveat='Executed native source revision above; source_sha256 describes current checkout and harness')
report['pid']=os.getpid()
bench.save(output,report)
