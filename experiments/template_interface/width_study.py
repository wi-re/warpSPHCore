"""Qualify and compare GPU group widths without changing operation policies."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import torch
from fixtures import load_core,fixture
from packing_study import paired
from template_runtime import Library

parser=argparse.ArgumentParser(description=__doc__)
for name in ('core','baseline','cpu','gpu','output'):parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--precision',required=True);args=parser.parse_args()
torch.set_num_threads(24)
sph=load_core(args.core,args.baseline/'native-torch-final.so',args.precision)
state,scalar,vector,domain,adjacency,csr,tiles,data,planning=fixture(sph,args.core,
    args.baseline/'exploration-v2.so',1048576,3,getattr(torch,args.precision),4.)
library=Library(args.cpu,args.gpu)
report=dict(precision=args.precision,planning=planning,records=[],activity=[],complete=False)
def activity():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip().splitlines()
    others=[line for line in processes if line.split(',')[0].strip()!=str(os.getpid())]
    report['activity'].append(dict(processes=processes,other_processes=others))
    if others:raise RuntimeError(f'Concurrent GPU workload: {others}')
activity()
for work in range(7):
    fields=vector if work==3 else scalar
    current=dict(data,qvalue=fields if work in (2,3) else None,
        rvalue=fields if work in (1,2,3,5) else None)
    expected=library.forward(current,csr,work,width=16).values
    calls={}
    for width in (8,16,32):
        actual=library.forward(current,csr,work,width=width).values
        error=float((actual-expected).double().norm()/expected.double().norm().clamp_min(1e-30))
        assert error<(4e-6 if args.precision=='float32' else 2e-12),error
        calls[str(width)]=lambda count,width=width:library.forward(current,csr,work,width=width)
    report['records'].append(dict(work=work,results=paired(calls,'cuda',False)))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
activity();report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
