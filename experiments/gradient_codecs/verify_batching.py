#!/usr/bin/env python3
"""Check query-batched Warp against full-array calls on a manageable graph."""
import argparse
import json
import os
from pathlib import Path
import sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-artifact',type=Path,required=True)
parser.add_argument('--precision',choices=('float32','float64'),required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp',warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64',WARPSPHCORE_NEIGHBOR_LANES='32')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
import warp as wp
import warpSPHCore as sph
import warpSPHCore.native._build as build
from common import fixture,properties,crk_factors,WarpBatches
from warpSPHCore.native.api import radiusSearch
wp.init();torch.ops.load_library(str(args.native_artifact));build._extension=torch.ops.warpsph_native
for module in ('wp_gradient','wp_divergence','wp_covariance'):
    wp.set_module_options({'enable_backward':False},module=f'warpSPHCore.coreOperations.native.{module}')
for module in ('crk_volume','crk_moments'):
    wp.set_module_options({'enable_backward':False},module=f'warpSPHCore.crk.{module}')
state,scalar,vector,domain=fixture(1024,3,getattr(torch,args.precision),2.57)
adjacency=radiusSearch(state,domain)
plan=WarpBatches(state,adjacency,domain,max_rows=257)
volumes,correction,statistics=crk_factors(state,domain,adjacency)
chunk_volumes,chunk_correction,chunk_statistics=crk_factors(state,domain,adjacency,batch_plan=plan)
torch.testing.assert_close(chunk_volumes,volumes,rtol=0,atol=0)
for name in ('A','B','gradA','gradB'):
    torch.testing.assert_close(getattr(chunk_correction,name),getattr(correction,name),rtol=0,atol=0)
for work in range(4):
    for corrected in (False,True):
        values=scalar if work==0 else vector
        kwargs=dict(queryVolumes=volumes,referenceVolumes=volumes,crkState=correction) if corrected else {}
        expected=sph.warpOperation(state,properties(work),domain,values,adjacency=adjacency,**kwargs)
        actual=plan.operation(properties(work),values,volumes if corrected else None,correction if corrected else None)()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
args.output.write_text(json.dumps(dict(complete=True,batches=len(plan.batches),bit_exact=True,
    operators=8,precision=args.precision,full_factor_statistics=statistics,batched_factor_statistics=chunk_statistics),indent=2)+'\n')
print('Batched factors and all eight operator cases are bit exact',flush=True)
