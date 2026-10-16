"""Per-search execution choices; no mutable process-wide policy."""
import torch

NEIGHBOR_COLLECTION_MODES = ('auto', 'fused', 'split')
NEIGHBOR_QUERY_ORDERS = ('auto', 'input', 'spatial')


def resolveCollection(collection: str, device) -> str:
    if collection not in NEIGHBOR_COLLECTION_MODES:
        raise ValueError(f'collection must be one of {NEIGHBOR_COLLECTION_MODES}')
    if collection == 'auto':
        return 'split' if torch.device(device).type == 'cuda' else 'fused'
    return collection


def resolveQueryOrder(queryOrder: str, queryPositions, referencePositions) -> str:
    if queryOrder not in NEIGHBOR_QUERY_ORDERS:
        raise ValueError(f'queryOrder must be one of {NEIGHBOR_QUERY_ORDERS}')
    same = queryPositions is referencePositions or queryPositions.is_set_to(referencePositions)
    if queryOrder == 'spatial' and not same:
        raise ValueError("queryOrder='spatial' requires query/reference positions to share the same layout; use 'auto' for mixed searches")
    return 'spatial' if same and queryOrder != 'input' else 'input'
