"""Typed Warp specialization of the existing fixed-slot compact hash.

No host numerical processing or global scalar configuration. The public entry
points remain makeHashMapSlots/buildFixed/GraphedBuild. Currently qualified only
for nonperiodic 2D producer/FluxBridge consumption; old particle operators that
expect a host hCell are not consumers of this device-parameter contract.
"""
from functools import lru_cache
from itertools import product
from types import SimpleNamespace
import operator
import warp as wp
from ._warp_integer import Scan64,StablePairs
from .slots import HashMapSlots

wp.set_module_options({'enable_backward':False,'fuse_fp':False})


def _view(value):
    if isinstance(value,wp.array):return value
    if type(value).__module__.startswith('torch'):return wp.from_torch(value)
    raise ValueError('typed grid data must be Warp arrays or exact-type zero-copy Torch views')


def _require(array,shape,dtype,device,name):
    if (not isinstance(array,wp.array) or array.shape!=shape or not wp.types.types_equal(array.dtype,dtype)
            or array.device!=device or not array.is_contiguous):
        raise ValueError(f'{name} must be contiguous {dtype} {shape} on {device}')


@wp.func
def _hash(linear: wp.int64,nx: int,length: int):
    x=wp.uint32(linear%wp.int64(nx));y=wp.uint32(linear//wp.int64(nx))
    return int((x*wp.uint32(73856093)+y*wp.uint32(19349663))%wp.uint32(length))


@lru_cache(None)
def kernels(precision):
    real=wp.float32 if precision=='float32' else wp.float64
    vec2=wp.types.vector(length=2,dtype=real)
    max_cells=2**24 if precision=='float32' else 2**31-1

    @wp.kernel(module='unique',enable_backward=False)
    def parameters(qmin: wp.array(dtype=real),qmax: wp.array(dtype=real),periodic: wp.array(dtype=bool),
                   radius: wp.array(dtype=real),width: wp.array(dtype=real),cells: wp.array(dtype=int),status: wp.array(dtype=int)):
        valid=wp.isfinite(radius[0]) and radius[0]>real(0.0)
        extent=vec2(real(0.0))
        count=extent;h=extent
        for j in range(2):
            valid=valid and wp.isfinite(qmin[j]) and wp.isfinite(qmax[j])
            extent[j]=qmax[j]-qmin[j]
            valid=valid and wp.isfinite(extent[j]) and extent[j]>real(0.0)
        status[0]=0;width[0]=real(1.0)
        for j in range(2):cells[j]=1
        if periodic[0] or periodic[1]:status[0]=6
        elif not valid:status[0]=4
        else:
            # Same finite cell-width construction as Core compute_h, in the
            # input precision throughout; no extent.double()/host scalar.
            shrink=bool(False)
            for j in range(2):
                ratio=extent[j]/radius[0]
                valid=valid and wp.isfinite(ratio)
                count[j]=wp.max(real(1.0),wp.floor(ratio))
                h[j]=extent[j]/count[j]
                shrink=shrink or extent[j]/h[j]-count[j]>real(0.0)
            if shrink:
                for j in range(2):count[j]=wp.max(real(1.0),count[j]-real(1.0));h[j]=extent[j]/count[j]
            expand=bool(False)
            for j in range(2):expand=expand or wp.ceil(extent[j]/h[j])>extent[j]/h[j]
            cell_width=wp.max(h[0],h[1])
            if expand:cell_width=cell_width*real(1.0001)
            valid=valid and wp.isfinite(cell_width) and cell_width>real(0.0)
            for j in range(2):
                raw=wp.max(real(1.0),wp.ceil(extent[j]/cell_width-real(1e-6)))
                valid=valid and wp.isfinite(raw) and raw<=real(max_cells)
                if valid:cells[j]=int(raw)
            if valid:width[0]=cell_width
            else:status[0]=4

    @wp.kernel(module='unique',enable_backward=False)
    def particle_keys(q: wp.array2d(dtype=real),supports: wp.array(dtype=real),qmin: wp.array(dtype=real),
                      cells: wp.array(dtype=int),width: wp.array(dtype=real),bound: wp.array(dtype=real),
                      keys: wp.array(dtype=wp.int64),order: wp.array(dtype=int),status: wp.array(dtype=int)):
        i=wp.tid();valid=wp.isfinite(supports[i]) and supports[i]>real(0.0) and supports[i]<=bound[0]
        cell=wp.vec2i(0)
        for j in range(2):
            t=(q[i,j]-qmin[j])/width[0]
            valid=valid and wp.isfinite(q[i,j]) and wp.isfinite(t)
            if wp.isfinite(t):cell[j]=int(wp.clamp(wp.floor(t),real(0.0),real(cells[j]-1)))
        key=wp.int64(cell[0])+wp.int64(cells[0])*wp.int64(cell[1])
        keys[i]=key;order[i]=i
        if not valid:wp.atomic_max(status,0,4)

    @wp.kernel(module='unique',enable_backward=False)
    def reorder(q: wp.array2d(dtype=real),h: wp.array(dtype=real),keys: wp.array(dtype=wp.int64),order: wp.array(dtype=int),
                sorted_q: wp.array2d(dtype=real),sorted_h: wp.array(dtype=real),sort_index: wp.array(dtype=int),flags: wp.array(dtype=wp.int64)):
        i=wp.tid();source=order[i]
        for j in range(2):sorted_q[i,j]=q[source,j]
        sorted_h[i]=h[source];sort_index[i]=source
        flag=wp.int64(1)
        if i>0 and keys[i]==keys[i-1]:flag=wp.int64(0)
        flags[i]=flag
    return SimpleNamespace(**locals())


@wp.kernel(enable_backward=False)
def _cell_starts(keys: wp.array(dtype=wp.int64),flags: wp.array(dtype=wp.int64),prefix: wp.array(dtype=wp.int64),
                 linear: wp.array(dtype=wp.int64),starts: wp.array(dtype=int)):
    i=wp.tid()
    if i<keys.shape[0] and flags[i]!=wp.int64(0):
        c=int(prefix[i]);linear[c]=keys[i];starts[c]=i
    if i==keys.shape[0]:starts[int(prefix[i])]=i


@wp.kernel(enable_backward=False)
def _cell_keys(prefix: wp.array(dtype=wp.int64),linear: wp.array(dtype=wp.int64),cells: wp.array(dtype=int),
               length: int,keys: wp.array(dtype=wp.int64),order: wp.array(dtype=int)):
    i=wp.tid();bucket=length
    if wp.int64(i)<prefix[prefix.shape[0]-1]:bucket=_hash(linear[i],cells[0],length)
    keys[i]=wp.int64(bucket);order[i]=i


@wp.kernel(enable_backward=False)
def _clear_hash(table: wp.array2d(dtype=int)):
    i=wp.tid();table[i,0]=-1;table[i,1]=0


@wp.kernel(enable_backward=False)
def _publish_cells(keys: wp.array(dtype=wp.int64),order: wp.array(dtype=int),prefix: wp.array(dtype=wp.int64),
                   linear: wp.array(dtype=wp.int64),starts: wp.array(dtype=int),table: wp.array2d(dtype=wp.int64),hash_table: wp.array2d(dtype=int)):
    i=wp.tid();count=int(prefix[prefix.shape[0]-1]);table[i,0]=wp.int64(-1);table[i,1]=wp.int64(keys.shape[0]);table[i,2]=wp.int64(0)
    if i<count:
        cell=order[i];bucket=int(keys[i]);table[i,0]=linear[cell];table[i,1]=wp.int64(starts[cell]);table[i,2]=wp.int64(starts[cell+1]-starts[cell])
        if i==0 or keys[i-1]!=keys[i]:hash_table[bucket,0]=i
        if i==count-1 or keys[i+1]!=keys[i]:hash_table[bucket,1]=i+1


@wp.kernel(enable_backward=False)
def _finish(table: wp.array2d(dtype=int),status: wp.array(dtype=int),fits: wp.array(dtype=bool)):
    i=wp.tid();begin=table[i,0]
    if begin>=0:table[i,1]=table[i,1]-begin
    if i==0:fits[0]=status[0]==0


class WarpHashMapSlots(HashMapSlots):
    """Private selected-type storage behind the existing fixed-slot factory."""
    resident_warp=True

    def __init__(self,n,domain,radius,*,device,dtype,mode,hashMapLength):
        self.device=wp.get_device(device);self.n=operator.index(n);self.dim=2;self.mode=mode
        if domain.dim!=2:raise NotImplementedError('typed fixed-slot producer currently supports nonperiodic 2D only')
        if not 0<=self.n<=2**31-2:raise ValueError('particle count must fit int32 with one sentinel')
        radius=_view(radius)
        if radius.dtype not in (wp.float32,wp.float64):raise ValueError('radius must be float32/float64')
        self.real=radius.dtype;self.precision='float32' if self.real==wp.float32 else 'float64'
        if dtype not in (None,self.real,self.precision):raise ValueError('dtype must match the typed radius')
        low=_view(domain.min);high=_view(domain.max);periodic=_view(domain.periodic)
        for name,a,shape,t in (('radius',radius,(1,),self.real),('domain.min',low,(2,),self.real),('domain.max',high,(2,),self.real),('periodic',periodic,(2,),wp.bool)):
            _require(a,shape,t,self.device,name)
        self.hashMapLength=self.n+1 if hashMapLength is None else operator.index(hashMapLength)
        if not 1<=self.hashMapLength<=2**31-1:raise ValueError('hashMapLength must be positive int32')
        if self.n*(8*wp.types.type_size_in_bytes(self.real)+128)+self.hashMapLength*8>2**30:raise ValueError('typed hash workspace exceeds 1 GiB budget')
        kw=dict(device=self.device)
        self.qMin=wp.clone(low);self.qMax=wp.clone(high);self.periodicT=wp.clone(periodic);self.periodic=self.periodicT;self.radius=wp.clone(radius)
        self.hCell=wp.empty(1,dtype=self.real,**kw);self.numCellsT=wp.empty(2,dtype=int,**kw);self.numCells=self.numCellsT
        self.parameter_status=wp.empty(1,dtype=int,**kw);self.status=wp.empty_like(self.parameter_status);self.fits=wp.zeros(1,dtype=bool,**kw)
        self.kernels=kernels(self.precision)
        wp.launch(self.kernels.parameters,1,inputs=[self.qMin,self.qMax,self.periodicT,self.radius,self.hCell,self.numCellsT,self.parameter_status],**kw)
        self.positions=wp.empty((self.n,2),dtype=self.real,**kw);self.supports=wp.empty(self.n,dtype=self.real,**kw)
        self.sortedPositions=wp.empty_like(self.positions);self.sortedSupports=wp.empty_like(self.supports)
        self.sortIndex=wp.empty(self.n,dtype=int,**kw)
        self.hashTableBuffer=wp.empty((self.hashMapLength,2),dtype=int,**kw);self.sortedCellTable=wp.empty((self.n,3),dtype=wp.int64,**kw)
        self.particle_sort=StablePairs(self.n,self.device);self.cell_sort=StablePairs(self.n,self.device)
        self.flags=wp.zeros(self.n+1,dtype=wp.int64,**kw);self.scan=Scan64(self.n+1,self.device)
        self.cell_linear=wp.empty(self.n,dtype=wp.int64,**kw);self.cell_starts=wp.empty(self.n+1,dtype=int,**kw)
        self.cellOffsets=wp.array([[*c,0] for c in product(range(-1,2),repeat=2)],dtype=int,**kw)
        from ...dataTypes import CompactHashMap
        self.hashMap=CompactHashMap(sortedPositions=self.sortedPositions,sortedSupports=self.sortedSupports,sortIndex=self.sortIndex,
                                   hashTable=self.hashTableBuffer,sortedCellTable=self.sortedCellTable,qMin=self.qMin,qMax=self.qMax,
                                   hCell=self.hCell,numCells=self.numCellsT,mode_uint=mode.value,D=2,searchRadius=1,numOffsets=9,cellOffsets=self.cellOffsets)
        self.hashMap.status=self.status;self.hashMap.resident_warp=True
        self.array_bytes=sum(getattr(self,k).capacity for k in ('qMin','qMax','periodicT','radius','hCell','numCellsT','parameter_status','status','fits',
                         'positions','supports','sortedPositions','sortedSupports','sortIndex','hashTableBuffer','sortedCellTable','flags','cell_linear','cell_starts','cellOffsets'))+self.scan.array_bytes+self.particle_sort.array_bytes+self.cell_sort.array_bytes
        wp.copy(self.status,self.parameter_status)
        self._built=False

    def build(self):
        kw=dict(device=self.device);n=self.n;p=self.particle_sort;c=self.cell_sort
        wp.copy(self.status,self.parameter_status)
        if n:wp.launch(self.kernels.particle_keys,n,inputs=[self.positions,self.supports,self.qMin,self.numCellsT,self.hCell,self.radius,p.keys[0],p.values[0],self.status],**kw)
        p.run()
        if n:wp.launch(self.kernels.reorder,n,inputs=[self.positions,self.supports,p.sorted_keys,p.sorted_values,self.sortedPositions,self.sortedSupports,self.sortIndex,self.flags],**kw)
        self.scan.run(self.flags)
        wp.launch(_cell_starts,n+1,inputs=[p.sorted_keys,self.flags,self.scan.output,self.cell_linear,self.cell_starts],**kw)
        if n:wp.launch(_cell_keys,n,inputs=[self.scan.output,self.cell_linear,self.numCellsT,self.hashMapLength,c.keys[0],c.values[0]],**kw)
        c.run()
        wp.launch(_clear_hash,self.hashMapLength,inputs=[self.hashTableBuffer],**kw)
        if n:wp.launch(_publish_cells,n,inputs=[c.sorted_keys,c.sorted_values,self.scan.output,self.cell_linear,self.cell_starts,self.sortedCellTable,self.hashTableBuffer],**kw)
        wp.launch(_finish,self.hashMapLength,inputs=[self.hashTableBuffer,self.status,self.fits],**kw)
        self._built=True
        return self.hashMap

    def update_device(self,positions,supports):
        positions=_view(positions);supports=_view(supports)
        _require(positions,(self.n,2),self.real,self.device,'positions');_require(supports,(self.n,),self.real,self.device,'supports')
        wp.copy(self.positions,positions);wp.copy(self.supports,supports)
        return self.build()

    def check(self):
        if not self._built:raise RuntimeError('build the particle grid before checking it')
        code=int(self.status.numpy()[0])
        if code==6:raise NotImplementedError('typed fixed-slot periodic producer is not implemented')
        if code:raise ValueError('invalid grid domain, precision/index range, particle data or support bound')
        return int(self.scan.output.numpy()[-1])
