#!/usr/bin/env python3
"""Serial GPU study: small independent screens, then fresh-process fixtures."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--precisions',nargs='+',default=['float32','float64'])
parser.add_argument('--dims',type=int,nargs='+',default=[1,2,3])
parser.add_argument('--ns',type=int,nargs='+',default=[1000000,4000000])
parser.add_argument('--large-n',type=int,default=16000000)
args=parser.parse_args();args.archive.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH='/home/lu26029/miniconda3/envs/warp/lib')


def run(output,precision,dimension,nh,n,extra):
    if output.exists() and json.loads(output.read_text()).get('complete'):return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU has another compute process')
    command=[sys.executable,str(ROOT/'experiments/gradient_codecs/benchmark.py'),
        '--library',str(args.archive/'artifacts/gradient-codecs-v1.so'),
        '--native-artifact',str(args.baseline/'artifacts/native-torch-final.so'),
        '--planner-library',str(args.baseline/'artifacts/exploration-v2.so'),
        '--precision',precision,'--dimension',str(dimension),'--nh',str(nh),'--n',str(n),'--output',str(output),*extra]
    print('Running',output.name,flush=True)
    with output.with_suffix('.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert json.loads(output.read_text())['uncontended_process_snapshots']


for precision,dimension,nh in itertools.product(args.precisions,args.dims,(4.,2.57)):
    run(args.archive/f'screen-{precision}-d{dimension}-h{nh:g}.json',precision,dimension,nh,262144,
        ['--screen','--repeat','5','--target-ms','5'])
selection={}
for path in args.archive.glob('screen-*.json'):
    selection.update(json.loads(path.read_text())['selection'])
selection_path=args.archive/'selection.json'
selection_path.write_text(json.dumps(dict(selection=selection),indent=2)+'\n')
for precision,dimension,nh in itertools.product(args.precisions,args.dims,(4.,2.57)):
    sizes=args.ns+([args.large_n] if dimension==3 and args.large_n else [])
    for n in sizes:
        run(args.archive/f'{precision}-d{dimension}-h{nh:g}-n{n}.json',precision,dimension,nh,n,
            ['--selection',str(selection_path),'--repeat','7','--target-ms','10'])
print('Study complete',flush=True)
