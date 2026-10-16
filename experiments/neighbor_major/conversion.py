"""Alternative CSR-J -> neighbor-major builders, isolated from direct search."""

import torch
import warp as wp

import layout as experiment
from warpSPHCore.autograd.stream import currentTorchWarpStream


@wp.kernel(module='unique',enable_backward=False)
def rowsToColumns(counts:wp.array(dtype=wp.int32),offsets:wp.array(dtype=wp.int32),
                  source:wp.array(dtype=wp.int64),target:wp.array(dtype=wp.int64)):
    row=wp.tid()
    n=wp.int64(counts.shape[0])
    begin=wp.int64(offsets[row])
    for k in range(counts[row]):
        target[wp.int64(k)*n+wp.int64(row)]=source[begin+wp.int64(k)]


@wp.kernel(module='unique',enable_backward=False)
def tileRowsToColumns(counts:wp.array(dtype=wp.int32),offsets:wp.array(dtype=wp.int32),
                      source:wp.array(dtype=wp.int64),target:wp.array(dtype=wp.int64),maximum:int):
    rowTile,slotTile,lane=wp.tid()
    r=lane//32
    k=lane%32
    row=rowTile*32+r
    slot=slotTile*32+k
    n=counts.shape[0]
    value=wp.int64(-1)
    if row<n:
        if slot<counts[row]:
            value=source[wp.int64(offsets[row])+wp.int64(slot)]
    tile=wp.tile_reshape(wp.tile(value,preserve_type=True),shape=(32,32))
    transposed=wp.tile_transpose(tile)
    result=transposed[r,k]  # all lanes read before the output bounds branch
    destinationRow=rowTile*32+k
    destinationSlot=slotTile*32+r
    if destinationRow<n and destinationSlot<maximum:
        target[wp.int64(destinationSlot)*wp.int64(n)+wp.int64(destinationRow)]=result


def allocate(csr,method):
    if csr.layout!='csr':raise TypeError('conversion requires experimental CSR-J')
    n,k=csr.numRows,csr.maximum
    target=torch.empty((n*k,),dtype=torch.int64,device=csr.paddedJ.device)
    return experiment.Neighbors(target,csr.numNeighbors,torch.empty(0,dtype=torch.int32,device=target.device),
                                 n,csr.numCols,csr.numEdges,'neighbor-major',k,torch.zeros_like(csr.overflow))


def fill(csr,target,method):
    if not target.paddedJ.numel():return
    stream=currentTorchWarpStream(csr.paddedJ.device)
    arrays=[wp.from_torch(t) for t in (csr.numNeighbors,csr.edgeOffsets,csr.paddedJ,target.paddedJ)]
    if method=='rows':
        target.paddedJ.fill_(-1)
        wp.launch(rowsToColumns,csr.numRows,inputs=arrays,device=arrays[0].device,stream=stream)
    elif method=='tile':
        if target.paddedJ.device.type!='cuda':raise NotImplementedError('block-cooperative transpose requires CUDA')
        wp.launch(tileRowsToColumns,dim=((csr.numRows+31)//32,(csr.maximum+31)//32,1024),
                  inputs=arrays+[csr.maximum],block_dim=1024,device=arrays[0].device,stream=stream)
    else:raise ValueError(method)


def convert(csr,method='rows'):
    target=allocate(csr,method)
    fill(csr,target,method)
    return target
