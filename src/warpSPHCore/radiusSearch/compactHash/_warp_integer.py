"""Retained integer scratch for the typed compact-hash builder.

The block scan follows the FluxBridge foundation layout. Sorting is stable for
repeated cell keys: GPU merge ranking uses strict/weak opposite-half bounds;
CPU uses eight finite byte-radix passes in a Warp CPU kernel. All scratch is
owned by the workspace; no CUB/global allocator cache or host library is used.
"""
import warp as wp

BLOCK=256


@wp.kernel(enable_backward=False)
def _scan_blocks(source: wp.array(dtype=wp.int64),out: wp.array(dtype=wp.int64),sums: wp.array(dtype=wp.int64)):
    block=wp.tid();total=wp.int64(0)
    for i in range(block*BLOCK,wp.min((block+1)*BLOCK,source.shape[0])):
        out[i]=total;total=total+source[i]
    sums[block]=total


@wp.kernel(enable_backward=False)
def _prefix(out: wp.array(dtype=wp.int64),prefix: wp.array(dtype=wp.int64)):
    i=wp.tid();out[i]=out[i]+prefix[i//BLOCK]


class Scan64:
    def __init__(self,size,device):
        self.device=device;self.levels=[]
        while True:
            blocks=(size+BLOCK-1)//BLOCK
            self.levels.append((wp.empty(size,dtype=wp.int64,device=device),wp.empty(blocks,dtype=wp.int64,device=device)))
            if blocks<=1:break
            size=blocks
        self.output=self.levels[0][0]
        self.array_bytes=sum(a.capacity for level in self.levels for a in level)

    def run(self,source):
        for out,sums in self.levels:
            wp.launch(_scan_blocks,sums.shape,inputs=[source,out,sums],device=self.device);source=sums
        for i in range(len(self.levels)-2,-1,-1):
            wp.launch(_prefix,self.levels[i][0].shape,inputs=[self.levels[i][0],self.levels[i+1][0]],device=self.device)


@wp.kernel(enable_backward=False)
def _merge(width: int,keys: wp.array(dtype=wp.int64),values: wp.array(dtype=int),out_keys: wp.array(dtype=wp.int64),out_values: wp.array(dtype=int)):
    i=wp.tid();start=(i//(2*width))*(2*width)
    middle=wp.min(start+width,keys.shape[0]);end=wp.min(start+2*width,keys.shape[0])
    lo=middle;hi=end;own=i-start;right=i>=middle
    if right:lo=start;hi=middle;own=i-middle
    first=lo;key=keys[i]
    while lo<hi:
        mid=lo+(hi-lo)//2
        before=keys[mid]<key
        if right:before=keys[mid]<=key
        if before:lo=mid+1
        else:hi=mid
    position=start+own+lo-first
    out_keys[position]=key;out_values[position]=values[i]


@wp.kernel(enable_backward=False)
def _radix_cpu(keys0: wp.array(dtype=wp.int64),values0: wp.array(dtype=int),keys1: wp.array(dtype=wp.int64),values1: wp.array(dtype=int),histogram: wp.array(dtype=int)):
    # Nonnegative integer keys only. Two byte passes return to the first buffer.
    for pair in range(4):
        for digit in range(2):
            shift=8*(2*pair+digit)
            for bucket in range(256):histogram[bucket]=0
            for i in range(keys0.shape[0]):
                key=keys0[i]
                if digit==1:key=keys1[i]
                bucket=int((key>>wp.int64(shift))&wp.int64(255));histogram[bucket]=histogram[bucket]+1
            total=int(0)
            for bucket in range(256):
                count=histogram[bucket];histogram[bucket]=total;total=total+count
            for i in range(keys0.shape[0]):
                key=keys0[i];value=values0[i]
                if digit==1:key=keys1[i];value=values1[i]
                bucket=int((key>>wp.int64(shift))&wp.int64(255));slot=histogram[bucket];histogram[bucket]=slot+1
                if digit==0:keys1[slot]=key;values1[slot]=value
                else:keys0[slot]=key;values0[slot]=value


class StablePairs:
    def __init__(self,size,device):
        self.device=device;self.size=size
        self.keys=[wp.empty(size,dtype=wp.int64,device=device) for _ in range(2)]
        self.values=[wp.empty(size,dtype=int,device=device) for _ in range(2)]
        self.steps=(size-1).bit_length() if size and device.is_cuda else 0
        self.histogram=wp.empty(0 if device.is_cuda else 256,dtype=int,device=device)
        self.sorted_keys=self.keys[self.steps%2];self.sorted_values=self.values[self.steps%2]
        self.array_bytes=sum(a.capacity for a in self.keys+self.values)+self.histogram.capacity

    def run(self):
        if not self.size:return
        if not self.device.is_cuda:
            wp.launch(_radix_cpu,1,inputs=[self.keys[0],self.values[0],self.keys[1],self.values[1],self.histogram],device=self.device)
        else:
            for step in range(self.steps):
                src=step%2;dst=1-src
                wp.launch(_merge,self.size,inputs=[1<<step,self.keys[src],self.values[src],self.keys[dst],self.values[dst]],device=self.device)
