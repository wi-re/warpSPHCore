"""Cooperative GPU rows against an independent differentiable edge-sum oracle."""
import pytest
import torch
import warpSPHCore as sph

if sph.backend != 'native':
    pytest.skip('native process only', allow_module_level=True)

from _support.kernels import shape
from _support.operators import normalization, OPS

@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('policy', ['gather', 'symmetric', 'filtered'])
@pytest.mark.parametrize('nq', [67,32769])
def test_ragged_interpolation_values_and_adjoints(device, dtype, dimension, rank, index_dtype, policy, nq):
    if device == 'cpu':
        pytest.skip('cooperative device execution')
    generator = torch.Generator().manual_seed(9021)
    nr = 97  # Partial final thread block and multiple neighbor tiles.

    def random(*sizes):
        return torch.rand(sizes, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

    qx, rx = random(nq, dimension) - .5, random(nr, dimension) - .5
    qx[:, 0] += 8
    rx[:, 0] -= 6  # Multiple periodic images must survive query-side caching.
    qh, rh = random(nq) * .5 + .55, random(nr) * .5 + .6
    mass, density = random(nr) + .2, random(nr) + .7
    field = random(nr, *([dimension] * rank)) - .5
    inputs = tuple(t.detach().requires_grad_() for t in (qx, rx, qh, rh, mass, density, field))
    qx, rx, qh, rh, mass, density, field = inputs
    qkind = torch.arange(nq, device=device, dtype=torch.int32) % 3
    rkind = torch.arange(nr, device=device, dtype=torch.int32) % 3
    query = sph.ParticleState(qx, qh, torch.ones_like(qh), qkind, torch.ones_like(qh))
    reference = sph.ParticleState(rx, rh, mass, rkind, density)
    domain = sph.DomainDescription(qx.new_full((dimension,), -1), qx.new_full((dimension,), 1),
                                  torch.tensor([True] + [False] * (dimension-1), device=device), dimension)
    pattern = torch.tensor([0, 1, 7, 8, 9, 31, 32, 33, 65, 97], device=device, dtype=torch.int32)
    counts = pattern[torch.arange(nq, device=device) % len(pattern)]
    # A partial set of populated rows exercises the large dispatch regime
    # without growing the independent edge oracle's work with every padded row.
    counts[1057:] = 0
    offsets = counts.long().cumsum(0) - counts
    edges = int(counts.sum())
    rows = torch.repeat_interleave(torch.arange(nq, device=device), counts.long(), output_size=edges)
    neighbors = (torch.arange(edges, device=device) + rows * 17) % nr
    adjacency = sph.AdjacencyList(None, neighbors.to(index_dtype), counts, offsets.to(index_dtype),
                                 nq, nr, numEdges=edges)
    kernel, support, direction = {
        'gather': (sph.KernelFunctions.Wendland2, sph.SupportScheme.Gather,
                   sph.OperationDirection.TrueAllToToAll),
        'symmetric': (sph.KernelFunctions.Wendland4, sph.SupportScheme.KernelMeanSymmetric,
                      sph.OperationDirection.AllToAll),
        'filtered': (sph.KernelFunctions.CubicSpline, sph.SupportScheme.SuperSymmetric,
                     sph.OperationDirection.FluidToBoundary),
    }[policy]
    properties = sph.OperationProperties(kernel, operation=sph.WarpOperation.Interpolate,
                                         supportMode=support, operationMode=direction)
    actual = sph.warpOperation(query, properties, domain, None, field,
                               referenceParticles=reference, adjacency=adjacency)

    # This oracle uses float64 arithmetic even for rounded FP32 inputs and is
    # differentiated by Torch, independently of the manual native adjoints.
    displacement = qx.double()[rows] - rx.double()[neighbors]
    length = (domain.max - domain.min).double()
    displacement = torch.where(domain.periodic,
        displacement - length * torch.floor((displacement + length / 2) / length), displacement)
    radius = displacement.norm(dim=-1)

    def weights(h):
        z = radius / h
        return torch.where(z <= 1, normalization(kernel, dimension) * shape(z, kernel, dimension)
                           / h.pow(dimension), 0)

    if policy == 'gather':
        w = weights(qh.double()[rows])
    else:
        w = (weights(qh.double()[rows]) + weights(rh.double()[neighbors])) / 2
    if policy == 'symmetric':
        w = w * (rkind[neighbors] != 2)
    elif policy == 'filtered':
        w = w * ((qkind[rows] == 1) & (rkind[neighbors] == 0))
    columns = dimension ** rank
    contribution = (w * mass.double()[neighbors] / density.double()[neighbors])[:, None]
    contribution = contribution * field.double().reshape(nr, columns)[neighbors]
    expected = torch.zeros((nq, columns), device=device, dtype=torch.float64).index_add(0, rows, contribution)
    expected = expected.reshape(actual.shape)
    tolerance = 3e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    cotangent = random(*actual.shape)
    actual_vjp = torch.autograd.grad(actual, inputs, cotangent, allow_unused=True)
    expected_vjp = torch.autograd.grad(expected, inputs, cotangent.double(), allow_unused=True)
    for source, got, want in zip(inputs, actual_vjp, expected_vjp):
        got = torch.zeros_like(source) if got is None else got
        want = torch.zeros_like(source) if want is None else want
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('n', [133,32769])
def test_interpolation_uses_nondefault_stream(device, dtype, n):
    if device == 'cpu':
        pytest.skip('CUDA/HIP stream contract')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.linspace(-.4, .4, n, device=device, dtype=dtype)[:, None]
        state = sph.ParticleState(x, x.new_full((len(x),), .2), x.new_ones(len(x)),
                                  torch.zeros(len(x), device=device, dtype=torch.int32), x.new_ones(len(x)))
        domain = sph.DomainDescription(x.new_tensor([-1]), x.new_tensor([1]),
                                       torch.tensor([True], device=device), 1)
        prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,
                                       operation=sph.WarpOperation.Interpolate)
        adjacency = None
        if n >= 32768:
            # Bounded degree keeps this stream-lifetime test linear in n.
            adjacency = sph.AdjacencyList(None,torch.arange(8,device=device).repeat(n),
                torch.full((n,),8,device=device,dtype=torch.int32),
                torch.arange(n,device=device)*8,n,n,numEdges=n*8)
        actual = sph.warpOperation(state, prop, domain, torch.ones_like(x),adjacency=adjacency)
        assert torch.isfinite(actual).all()
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_periodic_halfbox_ties_and_images(device, dtype, operation):
    length = torch.tensor(1.25, device=device, dtype=dtype)
    half = length / 2
    below = torch.nextafter(half, half.new_tensor(0))
    above = torch.nextafter(half, half.new_tensor(float('inf')))
    positions = torch.stack([half * 0, half, below, above, 3 * length + half,
                             -4 * length + half, length])[:, None]
    nq = len(positions)
    qh = torch.full((nq,), 2.5, device=device, dtype=dtype)
    kinds = torch.zeros(nq, device=device, dtype=torch.int32)
    query = sph.ParticleState(positions, qh, torch.ones_like(qh), kinds, torch.ones_like(qh))
    reference = sph.ParticleState(positions.new_zeros((1, 1)), qh[:1], qh.new_ones(1),
                                  kinds[:1], qh.new_ones(1))
    domain = sph.DomainDescription(length.new_tensor([0]), length[None],
                                   torch.tensor([True], device=device), 1)
    adjacency = sph.AdjacencyList(None, torch.zeros(nq, device=device, dtype=torch.int64),
                                  torch.ones(nq, device=device, dtype=torch.int32),
                                  torch.arange(nq, device=device), nq, 1, numEdges=nq)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Naive, operationMode=sph.OperationDirection.TrueAllToToAll)
    actual = sph.warpOperation(query, properties, domain, positions.new_zeros(nq),
                               positions.new_ones(1), referenceParticles=reference, adjacency=adjacency)
    # Keep the shifted scalar when selecting the integer wrap. It captures the
    # existing rounding convention immediately adjacent to the half-box tie.
    projected = positions[:, 0] - length * torch.floor(positions[:, 0] / length)
    dx = projected - length * torch.floor((projected + half) / length)
    t = 1 - dx.abs() / qh
    if operation == sph.WarpOperation.Interpolate:
        expected = 1.25 * t.pow(3) * (1 + 3 * dx.abs() / qh) / qh
    else:
        expected = (-15 * t.square() / qh.pow(3) * dx)[:, None]
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_search_preserves_sorted_rows_across_sort_thresholds(device, dtype):
    # Odd/even original IDs lie on opposite sides of a cell boundary. Cell
    # traversal therefore collects an unsorted interleaving for every row.
    targets = torch.tensor([0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 97, 97], device=device)
    ids = torch.arange(97, device=device)
    radius = (ids.to(dtype) + .5) * .005
    positions = (10 + torch.where(ids % 2 == 0, radius, -radius))[:, None]
    supports = targets.to(dtype) * .005
    supports[-1] = 1  # Fix cell width to one, placing the queries at a cell boundary.
    query = sph.ParticleState(positions.new_full((len(targets), 1), 10), supports,
                              supports.new_ones(len(targets)), targets.new_zeros(len(targets), dtype=torch.int32),
                              supports.new_ones(len(targets)))
    reference = sph.ParticleState(positions, positions.new_full((97,), .01),
                                  positions.new_ones(97), ids.new_zeros(97, dtype=torch.int32),
                                  positions.new_ones(97))
    domain = sph.DomainDescription(positions.new_tensor([0]), positions.new_tensor([20]),
                                   torch.tensor([False], device=device), 1)
    adjacency = sph.radiusSearch(query, domain, referenceParticles=reference)
    torch.testing.assert_close(adjacency.numNeighbors.long(), targets, rtol=0, atol=0)
    for row, count in enumerate(targets.tolist()):
        start = int(adjacency.edgeOffsets[row])
        torch.testing.assert_close(adjacency.paddedJ[start:start+count].long(),
                                   torch.arange(count, device=device), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('operation', OPS)
@pytest.mark.parametrize('index_dtype', [torch.int32,torch.int64])
@pytest.mark.parametrize('offset_dtype', [torch.int32,torch.int64])
def test_packed_primal_and_vjp_against_dense_oracle(device, dimension, rank, operation, index_dtype, offset_dtype):
    """Exercise the packed-size regime against independent FP64 Torch equations."""
    if device == 'cpu':
        pytest.skip('packed device execution')
    if rank == 0 and (operation == sph.WarpOperation.Divergence or
                      (dimension > 1 and operation == sph.WarpOperation.Curl)):
        pytest.skip('operator requires vector or matrix field')
    from _support.operators import dense
    generator = torch.Generator().manual_seed(7039)
    nq, nr, active = 32769, 17, 1057

    def random(*sizes):
        return torch.rand(sizes, generator=generator, dtype=torch.float32).to(device)

    # Separate the clouds from coincident-pair/near-zero-distance conditioning.
    # This gate checks packed execution; the original stress geometry and its
    # identical checkpoint discrepancies are retained in the experiment archive.
    qx, rx = random(nq, dimension)*.2-.4, random(nr, dimension)*.2+.2
    qx[:, 0] += 6
    rx[:, 0] -= 4
    tensors = [qx, rx, random(nq)*.4+1.4, random(nr)*.4+.7, random(nr)+.3,
               random(nr)+.8, random(nq, *([dimension]*rank))-.5,
               random(nr, *([dimension]*rank))-.5]
    inputs = [value.detach().requires_grad_() for value in tensors]
    qx, rx, qh, rh, mass, density, qf, rf = inputs
    query = sph.ParticleState(qx, qh, torch.ones_like(qh),
        torch.arange(nq, device=device, dtype=torch.int32)%4, torch.ones_like(qh))
    reference = sph.ParticleState(rx, rh, mass,
        torch.arange(nr, device=device, dtype=torch.int32)%4, density)
    domain = sph.DomainDescription(qx.new_full((dimension,),-1),qx.new_full((dimension,),1),
        torch.tensor([True]+[False]*(dimension-1),device=device),dimension)
    counts=torch.zeros(nq,device=device,dtype=torch.int32)
    counts[:active]=nr
    offsets=(counts.long().cumsum(0)-counts).to(offset_dtype)
    adjacency = sph.AdjacencyList(None,torch.arange(nr,device=device,dtype=index_dtype).repeat(active),
        counts,offsets,nq,nr,numEdges=active*nr)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=operation,
        gradientMode=sph.GradientScheme.Difference,laplacianMode=sph.LaplacianScheme.Brookshaw,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    actual = sph.warpOperation(query,prop,domain,qf,rf,referenceParticles=reference,adjacency=adjacency)
    from dataclasses import replace
    q64 = replace(query,positions=qx[:active].double(),supports=qh[:active].double(),masses=query.masses[:active].double(),
                  densities=query.densities[:active].double())
    r64 = replace(reference,positions=rx.double(),supports=rh.double(),masses=mass.double(),densities=density.double())
    d64 = replace(domain,min=domain.min.double(),max=domain.max.double())
    small = dense(q64,r64,d64,prop,qf[:active].double(),rf.double(),differentiable=True)
    expected = torch.cat([small,small.new_zeros((nq-active,*small.shape[1:]))])
    torch.testing.assert_close(actual.double(),expected,rtol=3e-4,atol=3e-4)
    cotangent = random(*actual.shape)-.5
    got = torch.autograd.grad(actual,inputs,cotangent,allow_unused=True)
    want = torch.autograd.grad(expected,inputs,cotangent.double(),allow_unused=True) if expected.requires_grad else [None]*len(inputs)
    for source, left, right in zip(inputs,got,want):
        left = torch.zeros_like(source) if left is None else left
        right = torch.zeros_like(source) if right is None else right
        torch.testing.assert_close(left,right,rtol=5e-4,atol=5e-4)
