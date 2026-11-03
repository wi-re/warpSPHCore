"""Typed resident pair fields: one allocation per owned adjacency chunk.

There is deliberately no flattened tensor view. The adjacency identity and its
storage lease pin pair ordering for consumers; a new plan requires replay or
an explicit migration. Policy bytes account for J/I plus this field.
"""
from dataclasses import dataclass
from math import prod

import torch

from .chunked import ChunkedAdjacency
from .allocation import noteAllocationFailure
from ..type_config import get_torch_index_dtype


@dataclass(frozen=True, slots=True)
class ChunkedPairField:
    adjacency: ChunkedAdjacency
    buffers: tuple[torch.Tensor, ...]
    trailingShape: tuple[int, ...]
    dtype: torch.dtype

    @staticmethod
    def _checkLayout(adjacency, trailingShape, dtype):
        adjacency.assertStorageValid()
        if not isinstance(trailingShape, tuple) or any(type(n) is not int or n < 1 for n in trailingShape):
            raise ValueError('Pair-field trailingShape must contain positive integer extents.')
        width = prod(trailingShape)
        policy = adjacency.plan.policy
        indexBytes = 4 if get_torch_index_dtype()==torch.int32 else 8
        pairBytes = indexBytes * (1 + adjacency.storeI) + width * torch.empty(0, dtype=dtype).element_size()
        if policy.maxElementsPerEdge < width or policy.bytesPerEdge < pairBytes:
            raise ValueError('RowChunkPolicy must account for pair-field scalar extents and J/I plus field bytes.')

    @classmethod
    def empty(cls, adjacency, *, trailingShape, dtype):
        cls._checkLayout(adjacency, trailingShape, dtype)
        buffers = []
        try:
            for chunk in adjacency:
                buffers.append(torch.empty((chunk.metadata.capacity, *trailingShape),
                    dtype=dtype, device=adjacency.queryPositions.device))
        except (MemoryError, torch.OutOfMemoryError) as error:
            requested = sum(c.metadata.capacity for c in adjacency) * prod(trailingShape) * torch.empty(0, dtype=dtype).element_size()
            noteAllocationFailure(error, adjacency.queryPositions.device, kind='pair-field',
                requestedBytes=requested, completedChunks=len(buffers), generation=adjacency.plan.generation.number)
            buffers.clear()
            raise
        return cls(adjacency, tuple(buffers), trailingShape, dtype)

    def validate(self, adjacency):
        if self.adjacency is not adjacency:
            raise ValueError('Pair field belongs to another chunked adjacency generation/layout; replay or migrate explicitly.')
        self._checkLayout(adjacency, self.trailingShape, self.dtype)
        if not isinstance(self.buffers, tuple) or len(self.buffers) != len(adjacency.chunks):
            raise ValueError('Pair field must supply exactly one buffer per owned chunk.')
        storages = set()
        for chunk, tensor in zip(adjacency, self.buffers):
            if (not isinstance(tensor, torch.Tensor)
                or tensor.shape != (chunk.metadata.capacity, *self.trailingShape)
                or tensor.dtype != self.dtype or tensor.device != adjacency.queryPositions.device
                or not tensor.is_contiguous()):
                raise ValueError('Pair-field buffer shape, dtype, device or stride does not match its owned chunk.')
            if tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.numel()*tensor.element_size():
                raise ValueError('Pair-field buffers require dedicated chunk storage, without a larger hidden pair allocation.')
            if tensor.numel():
                key = (tensor.device, tensor.untyped_storage().data_ptr())
                if key in storages:
                    raise ValueError('Pair-field buffers cannot share storage across owned chunks.')
                storages.add(key)
        return self

    def memoryUsage(self):
        self.validate(self.adjacency)
        stride = prod(self.trailingShape) * torch.empty(0, dtype=self.dtype).element_size()
        return dict(liveBytes=sum(c.metadata.liveEdges for c in self.adjacency) * stride,
                    capacityBytes=sum(t.numel() * t.element_size() for t in self.buffers))
