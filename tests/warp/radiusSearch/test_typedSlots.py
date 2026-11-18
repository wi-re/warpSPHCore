"""Typed fixed-slot producer: independent layout oracle and resident lifetime."""
from importlib import import_module
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from warpSPHCore.radiusSearch.compactHash.slots import makeHashMapSlots,buildFixed,GraphedBuild


def domain(real,device):
    return SimpleNamespace(dim=2,min=wp.array([-2.,-2.],dtype=real,device=device),
                           max=wp.array([6.,6.],dtype=real,device=device),periodic=wp.zeros(2,dtype=bool,device=device))


def oracle(q,h,*,low=None,width=None,cells=None):
    low=np.asarray([-2.,-2.] if low is None else low,dtype=q.dtype)
    width=np.asarray(1. if width is None else width,dtype=q.dtype)
    cells=np.asarray([8,8] if cells is None else cells,dtype=np.int32);nx=int(cells[0])
    n=len(q);cell=np.minimum(np.maximum(np.floor((q-low)/width).astype(np.int64),0),cells-1);linear=cell[:,0]+np.int64(nx)*cell[:,1]
    order=np.argsort(linear,kind='stable');keys=linear[order]
    first=np.flatnonzero(np.r_[True,keys[1:]!=keys[:-1]]) if n else np.empty(0,dtype=int)
    count=np.diff(np.r_[first,n]);ids=keys[first]
    bucket=np.asarray([(((int(c)%nx)*73856093+(int(c)//nx)*19349663)&0xffffffff)%(n+1) for c in ids],dtype=np.int64)
    permutation=np.argsort(bucket,kind='stable');cells=np.zeros((n,3),dtype=np.int64);cells[:,0]=-1;cells[:,1]=n
    cells[:len(first)]=np.column_stack((ids[permutation],first[permutation],count[permutation]))
    table=np.zeros((n+1,2),dtype=np.int32);table[:,0]=-1
    for i,p in enumerate(permutation):
        b=bucket[p]
        if table[b,0]<0:table[b,0]=i
        table[b,1]+=1
    return dict(sortedPositions=q[order],sortedSupports=h[order],sortIndex=order,sortedCellTable=cells,hashTable=table)


def reject(*args,**kwargs):raise AssertionError('numerical host work/readback/synchronization/allocation')


class NoNumerics:
    def __getattr__(self,name):reject()


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('case',['random','one-cell','empty'])
def test_typed_slot_layout_and_changed_inputs(device,precision,case):
    device=wp.get_device(str(device));real=wp.float32 if precision=='float32' else wp.float64
    rng=np.random.default_rng(992);n=0 if case=='empty' else 257
    q=rng.uniform(-4,8,(n,2)).astype(precision)
    if case=='one-cell':q[:]=.2
    h=np.full(n,.7,dtype=precision);x=wp.array(q,dtype=real,device=device);r=wp.array(h,dtype=real,device=device)
    s=makeHashMapSlots(n,domain(real,device),wp.ones(1,dtype=real,device=device),device=device,dtype=real)
    grid=s.update_device(x,r);s.check()
    addresses=[getattr(grid,k).ptr for k in oracle(q,h)]
    for k,v in oracle(q,h).items():np.testing.assert_array_equal(getattr(grid,k).numpy(),v)
    q=q[::-1].copy()+np.asarray([.17,-.13],dtype=precision);h[:]=.6
    wp.copy(x,wp.array(q,dtype=real,device=device));wp.copy(r,wp.array(h,dtype=real,device=device));s.update_device(x,r);s.check()
    for k,v in oracle(q,h).items():np.testing.assert_array_equal(getattr(grid,k).numpy(),v)
    assert addresses==[getattr(grid,k).ptr for k in oracle(q,h)]


@pytest.mark.parametrize('precision',['float32','float64'])
def test_typed_slot_no_host_work_and_reusable_scratch(device,precision,monkeypatch):
    device=wp.get_device(str(device));real=wp.float32 if precision=='float32' else wp.float64
    d=domain(real,device);radius=wp.ones(1,dtype=real,device=device)
    x=wp.full((17,2),.2,dtype=real,device=device);h=wp.full(17,.7,dtype=real,device=device)
    warm=makeHashMapSlots(17,d,radius,device=device);warm.update_device(x,h)
    with monkeypatch.context() as patch:
        patch.setattr(import_module('warpSPHCore.radiusSearch.compactHash.slots'),'np',NoNumerics())
        patch.setattr(wp.array,'numpy',reject);patch.setattr(wp,'synchronize',reject);patch.setattr(wp,'synchronize_device',reject)
        import torch
        for name in ('cpu','numpy','item','double','float','to','__array__'):patch.setattr(torch.Tensor,name,reject)
        for name in ('argsort','cumsum','floor','matmul','einsum'):patch.setattr(torch,name,reject)
        s=makeHashMapSlots(17,d,radius,device=device);s.update_device(x,h)
        d.min.fill_(0.);radius.fill_(0.)  # owned immutable parameter snapshots
    s.check();np.testing.assert_array_equal(s.qMin.numpy(),[-2.,-2.]);np.testing.assert_array_equal(s.radius.numpy(),[1.])
    with monkeypatch.context() as patch:
        patch.setattr(device.get_allocator(),'allocate',reject);patch.setattr(wp.utils,'array_scan',reject);patch.setattr(wp.utils,'radix_sort_pairs',reject)
        patch.setattr(wp.array,'numpy',reject);patch.setattr(wp,'synchronize',reject);patch.setattr(wp,'synchronize_device',reject)
        s.update_device(x,h);buildFixed(s,backend='warp')
    s.check()
    # The existing graph wrapper specializes too, without Torch sync/graphs.
    if device.is_cuda:
        graph=GraphedBuild(s,backend='warp');graph.replay();graph.replay()
        x.fill_(real(3.2));s.update_device(x,h);graph.replay();s.check()
        np.testing.assert_array_equal(s.sortedPositions.numpy(),x.numpy())
        with wp.ScopedCapture(device=device) as capture:s.update_device(x,h)
        x.fill_(real(-.5));wp.capture_launch(capture.graph);s.check()
        np.testing.assert_array_equal(s.sortedPositions.numpy(),x.numpy())


@pytest.mark.parametrize('precision',['float32','float64'])
def test_typed_slot_guards_recover_without_partial_cast(device,precision):
    device=wp.get_device(str(device));real=wp.float32 if precision=='float32' else wp.float64
    d=domain(real,device);s=makeHashMapSlots(3,d,wp.ones(1,dtype=real,device=device),device=device)
    x=wp.full((3,2),.2,dtype=real,device=device);h=wp.full(3,.7,dtype=real,device=device)
    for bad in (-1.,float('nan'),float('inf'),1.1):
        h.fill_(real(bad));s.update_device(x,h)
        with pytest.raises(ValueError):s.check()
    h.fill_(real(.7));s.update_device(x,h);s.check()
    pointer=s.positions.ptr
    with pytest.raises(ValueError):s.update_device(wp.zeros((3,2),dtype=wp.float64 if real==wp.float32 else wp.float32,device=device),h)
    assert s.positions.ptr==pointer
    d.periodic.fill_(True);periodic=makeHashMapSlots(3,d,wp.ones(1,dtype=real,device=device),device=device);periodic.update_device(x,h);periodic.check()
    x.fill_(real(8.2));periodic.update_device(x,h);periodic.check()
    np.testing.assert_allclose(periodic.sortedPositions.numpy(),.2,atol=1e-6 if precision=='float32' else 1e-14)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_grid_parameters_keep_the_input_precision(device,precision):
    import torch
    from warpSPHCore.radiusSearch.compactHash.grid import compute_h
    device=wp.get_device(str(device));real=wp.float32 if precision=='float32' else wp.float64
    torch_real=torch.float32 if precision=='float32' else torch.float64
    low=np.asarray([-.123,.217],dtype=precision);high=np.asarray([7.137,5.913],dtype=precision);radius=np.asarray([.317],dtype=precision)
    d=SimpleNamespace(dim=2,min=wp.array(low,dtype=real,device=device),max=wp.array(high,dtype=real,device=device),periodic=wp.zeros(2,dtype=bool,device=device))
    rng=np.random.default_rng(867);q=rng.uniform(low,high,(139,2)).astype(precision);r=np.full(139,.211,dtype=precision)
    s=makeHashMapSlots(139,d,wp.array(radius,dtype=real,device=device),device=device)
    grid=s.update_device(wp.array(q,dtype=real,device=device),wp.array(r,dtype=real,device=device));s.check()
    # Independent execution of the retained Core cell-width formula; validation
    # only. No Torch arithmetic is reachable from the typed producer.
    h=compute_h(torch.tensor(low,dtype=torch_real),torch.tensor(high,dtype=torch_real),torch.tensor(radius[0],dtype=torch_real)).numpy()
    expected=np.maximum(1,np.ceil((high-low)/h-np.asarray(1e-6,dtype=precision))).astype(np.int32)
    np.testing.assert_array_equal(s.hCell.numpy(),[h]);np.testing.assert_array_equal(s.numCellsT.numpy(),expected)
    for k,v in oracle(q,r,low=low,width=h,cells=expected).items():np.testing.assert_array_equal(getattr(grid,k).numpy(),v)


@pytest.mark.parametrize('real', [wp.float32, wp.float64])
def test_periodic_slots_keep_local_coordinates_and_wrap_both_edges(device, real):
    device = wp.get_device(str(device))
    bounds = SimpleNamespace(
        dim=2,
        min=wp.full(2, -0.5, dtype=real, device=device),
        max=wp.full(2, 0.5, dtype=real, device=device),
        periodic=wp.array([True, False], dtype=bool, device=device),
    )
    points = [[0.01, 0.01], [-0.625, 0.0], [0.625, 0.0], [0.5, 0.0], [-0.5, 0.0]]
    expected = [[0.01, 0.01], [0.375, 0.0], [-0.375, 0.0], [-0.5, 0.0], [-0.5, 0.0]]
    positions = wp.array(points, dtype=real, device=device)
    supports = wp.full(len(points), 0.125, dtype=real, device=device)
    slots = makeHashMapSlots(
        len(points), bounds, wp.full(1, 0.25, dtype=real, device=device), device=device,
    )
    grid = slots.update_device(positions, supports)
    slots.check()
    sorted_positions = grid.sortedPositions.numpy()
    np.testing.assert_array_equal(
        sorted_positions, np.asarray(expected, dtype=sorted_positions.dtype)[grid.sortIndex.numpy()],
    )
