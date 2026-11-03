"""Failure-only memory receipts for bulk row-chunk allocations."""
import json
import torch


def noteAllocationFailure(error, device, *, kind, requestedBytes, **details):
    # No CUDA queries on the successful path and no speculative empty_cache().
    receipt = dict(kind=kind, requestedBytes=requestedBytes, device=str(device), **details)
    try:
        if torch.device(device).type == 'cuda':
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            free, total = torch.cuda.mem_get_info(device)
            receipt.update(allocatedBytes=allocated, reservedBytes=reserved,
                           cachedBytes=reserved-allocated, deviceFreeBytes=free, deviceTotalBytes=total)
    except Exception as secondary:
        receipt['accountingError'] = f'{type(secondary).__name__}: {secondary}'
    error.add_note('Row-chunk allocation: ' + json.dumps(receipt, sort_keys=True))
