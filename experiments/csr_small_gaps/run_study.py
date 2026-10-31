#!/usr/bin/env python3
"""Run one GPU task at a time; each held-out fixture gets a fresh process."""
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
parser.add_argument('--ns',type=int,nargs='+',default=[1000000,4000000,16000000])
parser.add_argument('--dims',type=int,nargs='+',default=[1,2,3])
parser.add_argument('--precisions',nargs='+',default=['float32','float64'])
args=parser.parse_args()
args.archive.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH='/home/lu26029/miniconda3/envs/warp/lib')


def run(output,extra,precision):
    if output.exists() and json.loads(output.read_text()).get('complete'):
        return
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if processes:
        raise RuntimeError(f'GPU has active compute processes: {processes}')
    command=[sys.executable,str(ROOT/'experiments/csr_small_gaps/benchmark.py'),
             '--library',str(args.archive/'artifacts/small-gaps-v1.so'),
             '--artifacts',str(args.baseline/'artifacts'),'--byte-selection',str(args.baseline/f'screen-{precision}.json'),
             '--precision',precision,'--output',str(output),*extra]
    print('Running',output.name,flush=True)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads(output.read_text())
    if not report['uncontended_process_snapshots']:
        raise RuntimeError(f'Other compute work contaminated {output}')


for precision in args.precisions:
    run(args.archive/f'screen-{precision}.json',
        ['--screen','--ns','262144','--dims',*[str(d) for d in args.dims],
         '--repeat','5','--iterations','3','--plan-repeat','1'],precision)
for precision,d,nh,n in itertools.product(args.precisions,args.dims,(4.,2.57),args.ns):
    run(args.archive/f'{precision}-d{d}-h{nh:g}-n{n}.json',
        ['--selection',str(args.archive/f'screen-{precision}.json'),'--ns',str(n),'--dims',str(d),
         '--n-h',str(nh),'--repeat','9','--iterations','5','--target-ms','30','--plan-repeat','1'],precision)
print('Study complete',flush=True)
