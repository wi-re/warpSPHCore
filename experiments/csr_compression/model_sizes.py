#!/usr/bin/env python3
"""Sample exact neighbor rows and compare codec storage models outside timing."""
import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--ns',type=int,nargs='+',default=[1000000,16000000])
parser.add_argument('--sampled-rows',type=int,default=32768)
args=parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp',warpSPHCore_PRECISION='float32',
                  warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64')
sys.path.insert(0,str(ROOT/'src'))
import torch
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch
from codec_models import storage_models,verify_models
spec=importlib.util.spec_from_file_location('model_planner',ROOT/'experiments/cluster_pairs/runtime.py')
planner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=planner
spec.loader.exec_module(planner)
torch.set_num_threads(16)
torch.ops.load_library(str(args.artifacts/'native-torch-final.so'))
build._extension=torch.ops.warpsph_native
library=planner.Library(args.artifacts/'exploration-v2.so')
verify_models()
results=[]
for n in args.ns:
    for nh in (4.,2.57):
        dimension=3
        generator=torch.Generator().manual_seed(9067+n*3+dimension)
        x=torch.rand((n,dimension),generator=generator,dtype=torch.float64).to('cuda',torch.float32)
        side=round(n**(1/dimension))
        spacing=1/side if side**dimension==n else n**(-1/dimension)
        h=x.new_full((n,),nh*spacing)
        state=sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
        cells=planner.build_cells(library,state,x,float(h[0]),8)
        domain=sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
        rows=np.sort(np.random.default_rng(51031).choice(n,min(args.sampled_rows,n),replace=False))
        selected=torch.as_tensor(rows,device='cuda')
        query=replace(cells.state,**{name:getattr(cells.state,name)[selected] for name in ('positions','supports','masses','densities','kinds')})
        # Search only the selected queries, keeping the full sorted reference
        # state. This avoids creating billions of unused edges for a size model.
        adjacency=radiusSearch(query,domain,referenceParticles=cells.state)
        counts=adjacency.numNeighbors.cpu().numpy().astype(np.int64)
        width=int((counts.max()+31)//32*32)
        offsets=adjacency.edgeOffsets
        column=torch.arange(width,device='cuda')
        slot=torch.minimum(column[None,:],torch.as_tensor(counts-1,device='cuda')[:,None])
        ids=adjacency.paddedJ[offsets[:,None]+slot].cpu().numpy()
        result=dict(particles=n,dimension=dimension,nh=nh,precision='float32',seed=51031,**storage_models(ids,counts))
        results.append(result)
        args.output.write_text(json.dumps(dict(complete=False,cases=results),indent=2)+'\n')
        print(json.dumps(result),flush=True)
        del ids,slot,offsets,query,adjacency,selected,cells,state,x,h,domain
        torch.cuda.empty_cache()
args.output.write_text(json.dumps(dict(complete=True,cases=results),indent=2)+'\n')
