"""Serialize paired measurements; resume only reports marked complete."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core',type=Path,required=True)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=5)
parser.add_argument('--cpu',type=Path,required=True)
parser.add_argument('--gpu',type=Path,required=True)
args=parser.parse_args()
args.archive.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parent
configurations=[('float32',3,32768,4.)]
configurations+=list(itertools.product(('float32','float64'),(1,2,3),(32768,262144,1048576),(4.,)))
configurations+=list(itertools.product(('float32','float64'),(1,2,3),(262144,),(2.57,)))
configurations=list(dict.fromkeys(configurations))
environment=dict(os.environ,LD_LIBRARY_PATH=str(Path(sys.prefix)/'lib'))
for precision,dimension,n,nh in configurations:
    name=f'{precision}-d{dimension}-h{nh:g}-n{n}'
    output=args.archive/(name+'.json')
    if output.exists() and json.loads(output.read_text()).get('complete'):continue
    command=[sys.executable,str(source/'benchmark.py'),'--core',str(args.core),
        '--native',str(args.baseline/'native-torch-final.so'),'--planner',str(args.baseline/'exploration-v2.so'),
        '--cpu',str(args.cpu),'--gpu',str(args.gpu),
        '--precision',precision,'--n',str(n),'--dimension',str(dimension),'--nh',str(nh),
        '--output',str(output),'--repeats',str(args.repeats)]
    print('Starting '+name,flush=True)
    with (args.archive/(name+'.log')).open('w') as stream:
        subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Completed '+name,flush=True)
print('Study complete',flush=True)
