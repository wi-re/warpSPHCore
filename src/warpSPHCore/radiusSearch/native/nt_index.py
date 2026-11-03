"""Retained compact index for bounded whole-row count/collect passes."""
import torch


class SearchIndex:
    def __init__(self,q,r,qh,rh,domain,mode):
        from ...native._build import build
        self.reference,self.referenceSupports,self.domain,self.mode = r,rh,domain,mode
        self.metadata = torch.empty(21,dtype=torch.float64,device='cpu')
        self.arrays = build().search_index([q,r,qh,rh,domain.min,domain.max,domain.periodic],self.metadata,mode.value)
        self._ready = None
        self._sortIndex = None
        if q.is_cuda:
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(q.device))

    @property
    def sortIndex(self):
        self._wait()
        if self._sortIndex is None:self._sortIndex = self.arrays[0][:,2].contiguous()
        return self._sortIndex

    @property
    def nbytes(self):
        return self.metadata.numel()*self.metadata.element_size()+sum(t.numel()*t.element_size() for t in self.arrays)

    def _wait(self):
        if self._ready is not None:
            stream = torch.cuda.current_stream(self.reference.device)
            stream.wait_event(self._ready)
            for tensor in self.arrays:tensor.record_stream(stream)

    def apply(self,query,supports,counts,offsets=None,neighbors=None):
        from ...native._build import build
        self._wait()
        geometry = [query,self.reference,supports,self.referenceSupports,
                    self.domain.min,self.domain.max,self.domain.periodic]
        build().search_index_apply(geometry,self.metadata,self.arrays,counts,offsets,neighbors,self.mode.value)
