"""Sorted simulation fixtures and cell tiles; all planning is outside timings."""
import importlib.util
import math
from pathlib import Path
import sys
import time
import torch
from template_runtime import Plan


def load_core(root,native_artifact,precision):
    import os
    os.environ.update(warpSPHCore_PRECISION=precision,WARPSPHCORE_BACKEND='warp',
        warpSPHCore_INDEX_DTYPE='int64',warpSPHCore_OFFSET_DTYPE='int64',WARPSPHCORE_NEIGHBOR_LANES='32')
    sys.path.insert(0,str(root/'src'))
    import warpSPHCore as sph
    import warpSPHCore.native._build as build
    torch.ops.load_library(str(native_artifact))
    build._extension=torch.ops.warpsph_native
    return sph


def fixture(sph,root,planner_library,n,dimension,dtype,nh):
    from warpSPHCore.native.api import radiusSearch
    start=time.perf_counter()
    generator=torch.Generator().manual_seed(271828+n+dimension)
    x=torch.rand((n,dimension),generator=generator,dtype=torch.float64).to('cuda',dtype=dtype)
    h=x.new_full((n,),nh*n**(-1/dimension))
    state=sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device=x.device,dtype=torch.int32),h.new_ones(n))
    spec=importlib.util.spec_from_file_location('template_cell_planner',root/'experiments/cluster_pairs/runtime.py')
    planner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=planner;spec.loader.exec_module(planner)
    library=planner.Library(planner_library)
    cells=planner.build_cells(library,state,x.new_ones((n,1)),float(h[0]))
    state=cells.state
    vector=torch.sin(state.positions*(2*math.pi))+.2*torch.cos(state.positions*(4*math.pi))
    scalar=(vector*torch.arange(1,dimension+1,device=x.device,dtype=x.dtype)).sum(-1,keepdim=True)
    domain=sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),torch.ones(dimension,device=x.device,dtype=torch.bool),dimension)
    adjacency=radiusSearch(state,domain)
    csr=Plan(dict(indices=adjacency.paddedJ.to(torch.int32),counts=adjacency.numNeighbors,offsets=adjacency.edgeOffsets))
    tasks=cells.tasks(8);neighbors=cells.neighbors[tasks.cells.long()].long()
    valid=neighbors>=0;safe=neighbors.clamp_min(0)
    begins=torch.where(valid,cells.offsets[safe],0).int().flatten()
    ends=torch.where(valid,cells.offsets[safe+1],0).int().flatten()
    ntask,stencil=neighbors.shape
    tiles=Plan(dict(offsets=torch.arange(ntask,device=x.device,dtype=torch.int64)*stencil,
        counts=torch.full((ntask,),stencil,device=x.device,dtype=torch.int32),begins=begins,ends=ends,
        query_begin=tasks.begin,query_end=tasks.end),2,ntask)
    torch.cuda.synchronize()
    planning=dict(wall_ms=(time.perf_counter()-start)*1000,edges=adjacency.numEdges,
                  mean_neighbors=adjacency.numEdges/n,csr32_bytes=csr.bytes(),cell_tile_bytes=tiles.bytes())
    data=dict(qx=state.positions,rx=state.positions,h=state.supports,mass=state.masses,density=state.densities,
              lower=domain.min,upper=domain.max,periodic=domain.periodic)
    return state,scalar,vector,domain,adjacency,csr,tiles,data,planning


def to_device(data,device):
    return {name:value.to(device) if isinstance(value,torch.Tensor) else value for name,value in data.items()}
