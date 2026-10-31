"""Large-dispatch interpolation and VJP against independent FP64 equations."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph

if sph.backend != 'native':
    pytest.skip('native process only', allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('offset_dtype', [torch.int32, torch.int64])
def test_packed_interpolation_dense_dispatch_and_vjp(device, dtype, dimension, rank, index_dtype, offset_dtype):
    if device == 'cpu':
        pytest.skip('packed GPU execution')
    from _support.operators import dense
    generator = torch.Generator().manual_seed(3907)
    nq, nr, active = 32769, 193, 1057

    def random(*sizes):
        return torch.rand(sizes, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

    qx, rx = random(nq, dimension)*.2-.4, random(nr, dimension)*.2+.2
    qx[:, 0] += 6
    rx[:, 0] -= 4
    inputs = tuple(t.detach().requires_grad_() for t in
                   (qx, rx, random(nq)*.4+1.4, random(nr)*.4+.7,
                    random(nr)+.3, random(nr)+.8, random(nr, *([dimension]*rank))-.5))
    qx, rx, qh, rh, mass, density, field = inputs
    query = sph.ParticleState(qx, qh, torch.ones_like(qh),
        torch.zeros(nq, device=device, dtype=torch.int32), torch.ones_like(qh))
    reference = sph.ParticleState(rx, rh, mass,
        torch.zeros(nr, device=device, dtype=torch.int32), density)
    domain = sph.DomainDescription(qx.new_full((dimension,), -1), qx.new_full((dimension,), 1),
        torch.tensor([True]+[False]*(dimension-1), device=device), dimension)
    counts = torch.zeros(nq, device=device, dtype=torch.int32)
    counts[:active] = nr
    offsets = (counts.long().cumsum(0)-counts).to(offset_dtype)
    # A valid overallocated CSR buffer exercises the dense scheduling estimate
    # while keeping the independent oracle and backward calculation small.
    ids = torch.zeros(nq*128, device=device, dtype=index_dtype)
    ids[:active*nr] = torch.arange(nr, device=device, dtype=index_dtype).repeat(active)
    adjacency = sph.AdjacencyList(None, ids, counts, offsets, nq, nr, numEdges=active*nr)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)
    actual = sph.warpOperation(query, properties, domain, None, field,
                               referenceParticles=reference, adjacency=adjacency)
    q64 = replace(query, positions=qx[:active].double(), supports=qh[:active].double(),
                  masses=query.masses[:active].double(), densities=query.densities[:active].double())
    r64 = replace(reference, positions=rx.double(), supports=rh.double(),
                  masses=mass.double(), densities=density.double())
    d64 = replace(domain, min=domain.min.double(), max=domain.max.double())
    query_shape = field.new_zeros((active, *field.shape[1:]), dtype=torch.float64)
    small = dense(q64, r64, d64, properties, query_shape, field.double(), differentiable=True)
    expected = torch.cat([small, small.new_zeros((nq-active, *small.shape[1:]))])
    tolerance = 3e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    cotangent = random(*actual.shape)-.5
    got = torch.autograd.grad(actual, inputs, cotangent, allow_unused=True)
    want = torch.autograd.grad(expected, inputs, cotangent.double(), allow_unused=True)
    for source, left, right in zip(inputs, got, want):
        left = torch.zeros_like(source) if left is None else left
        right = torch.zeros_like(source) if right is None else right
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_fixed_adjacency_graph_replays_changed_fields(device, dtype, dimension):
    if device == 'cpu':
        pytest.skip('CUDA graph execution')
    nq, nr = 32769, 193
    qx = torch.linspace(.1, .7, nq, device=device, dtype=dtype)[:, None].repeat(1, dimension)
    rx = torch.linspace(-.3, .3, nr, device=device, dtype=dtype)[:, None].repeat(1, dimension)
    query = sph.ParticleState(qx, qx.new_full((nq,), .8), qx.new_ones(nq),
        torch.zeros(nq, device=device, dtype=torch.int32), qx.new_ones(nq))
    reference = sph.ParticleState(rx, rx.new_full((nr,), .8), rx.new_full((nr,), 1/nr),
        torch.zeros(nr, device=device, dtype=torch.int32), rx.new_ones(nr))
    domain = sph.DomainDescription(qx.new_full((dimension,), -1), qx.new_full((dimension,), 1),
                                  torch.ones(dimension, device=device, dtype=torch.bool), dimension)
    adjacency = sph.AdjacencyList(None, torch.arange(nr, device=device).repeat(nq),
        torch.full((nq,), nr, device=device, dtype=torch.int32),
        torch.arange(nq, device=device)*nr, nq, nr, numEdges=nq*nr)
    field = torch.cos(rx)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)

    def operation():
        return sph.warpOperation(query, properties, domain, None, field,
                                 referenceParticles=reference, adjacency=adjacency)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = operation()
    graph.replay()
    torch.testing.assert_close(captured, operation())
    field.add_(.125)
    reference.masses.mul_(.8)
    graph.replay()
    torch.testing.assert_close(captured, operation())
