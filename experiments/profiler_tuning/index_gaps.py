"""Offline CSR encoding analysis. No code runs inside an SPH operation."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', type=Path, required=True,
                    help='Directory containing native-torch-final.so and exploration-v2.so')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION='float32',
                  warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments/cluster_pairs')]
import torch
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch
from runtime import Library, build_cells
ART = args.artifacts
torch.set_num_threads(16)
torch.ops.load_library(str(ART/'native-torch-final.so'))
build._extension = torch.ops.warpsph_native
library = Library(ART/'exploration-v2.so')
report = dict(particles=1000000, sampled_rows=32768, seed=51031, precision='float32',
              distribution='uniform random; periodic unit box; refined Morton sorting',
              artifacts=str(ART.resolve()), cases=[])
for dim in (1, 2, 3):
    for nh in (4., 2.57):
        n = report['particles']
        generator = torch.Generator().manual_seed(9067+n*3+dim)
        x = torch.rand((n,dim), generator=generator, dtype=torch.float64).to('cuda',torch.float32)
        side = round(n**(1/dim))
        spacing = 1/side if side**dim == n else n**(-1/dim)
        h = x.new_full((n,),nh*spacing)
        state = sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
        domain = sph.DomainDescription(x.new_zeros(dim),x.new_ones(dim),torch.ones(dim,device='cuda',dtype=torch.bool),dim)
        cells = build_cells(library,state,x,float(h[0]),8)
        adj = radiusSearch(cells.state,domain)
        # Sample rows uniformly without replacement. All measurements below are
        # offline distribution scans, explicitly excluded from runtime kernels.
        rng = np.random.default_rng(report['seed'])
        rows = np.sort(rng.choice(n,report['sampled_rows'],replace=False))
        rows_gpu = torch.as_tensor(rows,device='cuda')
        counts = adj.numNeighbors[rows_gpu].cpu().numpy().astype(np.int64)
        offsets = adj.edgeOffsets[rows_gpu]
        width = int((counts.max()+31)//32*32)
        column = torch.arange(width,device='cuda')
        # Repeat the last valid ID in padded lanes; masks exclude these entries.
        slot = torch.minimum(column[None,:],torch.as_tensor(counts-1,device='cuda')[:,None])
        ids = adj.paddedJ[offsets[:,None]+slot].cpu().numpy()
        valid = np.arange(width)[None,:] < counts[:,None]
        gaps = np.diff(ids,axis=1)
        valid_gap = np.arange(width-1)[None,:] < (counts-1)[:,None]
        deltas = gaps[valid_gap]
        assert np.all(deltas > 0), 'CSR must have distinct sorted neighbors'
        relative = np.abs(ids-rows[:,None])[valid]
        occupancy = cells.counts.cpu().numpy()
        blocks = ids.reshape(len(rows),-1,32)
        block_count = np.clip(counts[:,None]-np.arange(width//32)[None,:]*32,0,32)
        occupied = block_count > 0
        spans = (blocks[:,:,-1]-blocks[:,:,0])[occupied]
        used = block_count[occupied]
        selected_edges = int(counts.sum())
        # Payload estimates: row base uint32 + byte deltas; 255 is an escape
        # followed by a full uint32 delta. Row byte-offset metadata is excluded.
        escaped = int(np.count_nonzero(deltas >= 255))
        row_delta_bytes = 4*len(rows)+len(deltas)+4*escaped
        # Independent 32-entry blocks: uint32 base, uint8/16/32 offsets,
        # one-byte width tag, uint32 block byte offset. Ignore allocator padding.
        widths = np.where(spans<=255,1,np.where(spans<=65535,2,4))
        block_bytes = int(np.sum(8+1+(used-1)*widths))
        # Independent block-delta encoding: a uint32 base and byte offset per
        # block, then byte deltas with uint32 escape payloads after the codes.
        block_deltas = np.diff(blocks,axis=2)
        block_valid = np.arange(31)[None,None,:] < (block_count-1)[:,:,None]
        block_escapes = int(np.count_nonzero((block_deltas>=255) & block_valid))
        block_delta_bytes = selected_edges + 7*len(used) + 4*block_escapes
        record = dict(dim=dim,nh=nh,edges=adj.numEdges,mean_neighbors=adj.numEdges/n,
            sampled_edges=selected_edges,max_cell_occupancy=int(occupancy.max()),
            cells_exceeding_256_particles=int(np.count_nonzero(occupancy>256)),
            query_distance_fits={str(limit):float(np.mean(relative<=limit)) for limit in (127,255,65535)},
            successive_gap_fits={str(limit):float(np.mean(deltas<=limit)) for limit in (1,15,254,255,65535)},
            maximum_sampled_gap=int(deltas.max()),
            independent_32_block_span_fits={str(limit):float(np.mean(spans<=limit)) for limit in (255,65535)},
            estimated_row_delta_payload_bytes_per_edge=row_delta_bytes/selected_edges,
            estimated_block_offsets_bytes_per_edge=block_bytes/selected_edges,
            estimated_block_delta_bytes_per_edge=block_delta_bytes/selected_edges,
            estimated_block_offsets_metadata='uint32 base + uint32 byte offset + uint8 width per block',
            estimated_row_delta_metadata='uint32 base per row; excludes row byte offsets/counts',
        )
        report['cases'].append(record)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        del ids, valid, gaps, valid_gap, deltas, relative, blocks, block_deltas, block_valid, x,h,state,domain,cells,adj,slot,offsets
        torch.cuda.empty_cache()
