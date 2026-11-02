"""Owned field derivatives: independent equations and fixed-topology lifetimes."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from warpSPHCore.native import CSRTopology, linearizeFields
from _support.operators import fixture
from _support.corrected import remaining_oracle
from test_corrected_primal import correction_fixture, oracle
from test_crk_preparation import full_csr

OPS = [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient,
       sph.WarpOperation.Divergence, sph.WarpOperation.Curl, sph.WarpOperation.Laplacian]
WIDTHS = [(torch.int32, torch.int32), (torch.int32, torch.int64),
          (torch.int64, torch.int32), (torch.int64, torch.int64)]


def problem(device, dimension, dtype, rank):
    q, r, domain, fi, fj = fixture(device, dimension, rank)
    def cast(state):
        return replace(state, **{name: getattr(state, name).to(dtype) for name in
            ('positions', 'supports', 'masses', 'densities')})
    return cast(q), cast(r), replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype)), fi.to(dtype), fj.to(dtype)


def check_product(device, dimension, dtype, rank, operation, case, kernel):
    q, r, domain, fi, fj = problem(device, dimension, dtype, rank)
    state, (_, volumes), omega, R = correction_fixture(q, r)
    prop = sph.OperationProperties(kernel, operation=operation,
        gradientMode=list(sph.GradientScheme)[case % 4],
        supportMode=list(sph.SupportScheme)[:5][case % 5],
        laplacianMode=list(sph.LaplacianScheme)[case % 4],
        divergenceDotMode=bool(case % 2),
        operationMode=sph.OperationDirection.FluidToBoundary if case % 2 else sph.OperationDirection.TrueAllToToAll)
    if rank == 0 and dimension > 1 and prop.laplacianMode == sph.LaplacianScheme.Dot:
        prop = replace(prop, laplacianMode=sph.LaplacianScheme.Brookshaw)
    options = dict(referenceVolumes=volumes)
    if case % 3 != 0: options['crkState'] = state
    if operation != sph.WarpOperation.Interpolate:
        if case % 3 != 1: options['gradHState'] = omega
        if case % 3 != 2: options['renormalizationState'] = R
    if operation == sph.WarpOperation.Divergence:
        options['consistentDivergence'] = bool((case // 2) % 2)
    topology = CSRTopology(full_csr(q, r, *WIDTHS[case % 4]))
    linear = linearizeFields(q, prop, domain, fi, fj, referenceParticles=r, adjacency=topology, **options)
    equation = oracle if operation in OPS[:2] else remaining_oracle
    aq, ar = fi.detach().double().requires_grad_(), fj.detach().double().requires_grad_()
    expected = equation(q, r, domain, prop, aq, ar, **options)
    tolerance = 1e-3 if dtype == torch.float32 else 5e-9
    torch.testing.assert_close(linear.value.double(), expected.detach(), rtol=tolerance, atol=tolerance)
    tq, tr = fi.sin().contiguous(), fj.cos().contiguous()
    expected_jvp = equation(q, r, domain, prop, tq, tr, **options)
    actual_jvp = linear.jvp(query=tq, reference=tr)
    torch.testing.assert_close(actual_jvp.double(), expected_jvp, rtol=tolerance, atol=tolerance)
    bar = (linear.value + .31).cos().contiguous()
    target = (expected * bar.double()).sum()
    if target.requires_grad:
        eq, er = torch.autograd.grad(target, (aq, ar), allow_unused=True)
    else: eq = er = None
    bq, br = linear.vjp(bar)
    for actual, wanted, prototype in ((bq, eq, fi), (br, er, fj)):
        if actual is None:
            assert wanted is None or torch.count_nonzero(wanted) == 0
        else:
            wanted = torch.zeros_like(prototype).double() if wanted is None else wanted
            torch.testing.assert_close(actual.double(), wanted, rtol=tolerance, atol=tolerance)
    right = (br * tr).sum() + (0 if bq is None else (bq * tq).sum())
    torch.testing.assert_close((actual_jvp * bar).sum(), right, rtol=tolerance, atol=tolerance)
    assert not actual_jvp.requires_grad and not br.requires_grad
    assert topology.transpose() is topology.transpose()
    # Explicit nullable tangents mean exactly zero, with no dummy field arrays.
    torch.testing.assert_close(linear.jvp(), torch.zeros_like(linear.value), rtol=0, atol=0)
    torch.testing.assert_close(linear.jvp(query=tq) + linear.jvp(reference=tr), actual_jvp,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('operation', OPS)
@pytest.mark.parametrize('case', range(8))
def test_field_products(device, dimension, dtype, rank, operation, case):
    if rank == 0 and dimension > 1 and operation in OPS[2:4]: pytest.skip('vector field required')
    check_product(device, dimension, dtype, rank, operation, case, sph.KernelFunctions.Wendland2)


@pytest.mark.extended
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('operation', OPS)
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_field_kernel_products(device, dimension, dtype, operation, kernel):
    check_product(device, dimension, dtype, 1, operation, kernel.value, kernel)


@pytest.mark.parametrize('width', WIDTHS)
def test_transpose_gaps_duplicates_empty_and_cache(device, width):
    ids = torch.tensor([-1, 2, 0, 2, -1, 1, -1, -1], device=device, dtype=width[0])
    counts = torch.tensor([3, 0, 1], device=device, dtype=torch.int32)
    offsets = torch.tensor([1, 4, 5], device=device, dtype=width[1])
    source = sph.AdjacencyList(None, ids, counts, offsets, 3, 4)
    topology = CSRTopology(source)
    csc = topology.transpose()
    assert csc.numRows == 4 and csc.numCols == 3
    assert csc.numNeighbors.tolist() == [1, 1, 2, 0]
    assert csc.edgeOffsets.tolist() == [0, 1, 2, 4]
    assert csc.paddedJ.tolist() == [0, 2, 0, 0, -1, -1, -1, -1]
    assert csc is topology.transpose()
    # Changes to exposed cached storage trigger reconstruction, not silent reuse.
    csc.paddedJ[0] = 99
    assert topology.transpose() is not csc
    prior = topology.transpose()
    source.paddedJ[1] = 1
    assert topology.transpose() is not prior
    for rows, cols in ((0, 4), (3, 0), (0, 0)):
        a = sph.AdjacencyList(None, ids[:0], counts.new_zeros(rows), offsets.new_zeros(rows), rows, cols)
        t = CSRTopology(a).transpose()
        assert t.numRows == cols and t.numCols == rows
        assert torch.count_nonzero(t.numNeighbors) == 0


@pytest.mark.parametrize('operation', OPS)
def test_autograd_aliasing_and_cache_lifetime(device, operation):
    q, _, domain, fi, _ = problem(device, 3, torch.float64, 1)
    field = fi.requires_grad_()
    graph = full_csr(q, q)
    topology = CSRTopology(graph)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Difference, operationMode=sph.OperationDirection.TrueAllToToAll)
    linear = linearizeFields(q, prop, domain, field, adjacency=topology)
    bar = linear.value.detach().sin().contiguous()
    bq, br = linear.vjp(bar)
    expected = br if bq is None else bq + br
    actual, = torch.autograd.grad(linear.value, field, bar)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda x: linear.apply(x), (field,), fast_mode=True)
    other = linearizeFields(q, replace(prop, operation=sph.WarpOperation.Gradient), domain,
                            field, adjacency=topology)
    other.vjp(torch.ones_like(other.value))
    assert other.topology.transpose() is topology.transpose()
    graph.paddedJ.add_(0)
    with pytest.raises(RuntimeError, match='topology changed'): linear.jvp(reference=field.detach())


def test_fixed_state_guards_and_optional_pullbacks(device):
    q, r, domain, fi, fj = problem(device, 2, torch.float64, 1)
    state, (_, volumes), _, _ = correction_fixture(q, r)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient)
    topo = CSRTopology(full_csr(q, r))
    linear = linearizeFields(q, prop, domain, fi, fj, referenceParticles=r,
                            adjacency=topo, crkState=state, referenceVolumes=volumes)
    assert topo._transpose is None
    qbar, rbar = linear.vjp(torch.ones_like(linear.value), reference=False)
    assert qbar is not None and rbar is None and topo._transpose is None
    assert linear.vjp(torch.ones_like(linear.value), query=False, reference=False) == (None, None)
    qbar, rbar = linear.vjp(torch.ones_like(linear.value), query=False)
    assert qbar is None and rbar is not None
    state.A.add_(0)
    with pytest.raises(RuntimeError, match='Fixed native'): linear.jvp(reference=fj)
    q = replace(q, positions=q.positions.detach().requires_grad_())
    with pytest.raises(NotImplementedError, match='autograd'):
        linearizeFields(q, prop, domain, fi, fj, referenceParticles=r, adjacency=topo)
    with torch.no_grad():
        frozen = linearizeFields(q, prop, domain, fi, fj, referenceParticles=r, adjacency=topo)
    assert frozen.jvp(reference=fj).shape == frozen.value.shape
    with pytest.raises(NotImplementedError): frozen.jvp(reference=fj.requires_grad_())


def test_transpose_stream_reuse(device):
    if device == 'cpu': pytest.skip('CUDA streams')
    q, r, domain, fi, fj = problem(device, 3, torch.float64, 1)
    topology = CSRTopology(full_csr(q, r))
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient)
    linear = linearizeFields(q, prop, domain, fi, fj, referenceParticles=r, adjacency=topology)
    bar = torch.ones_like(linear.value)
    expected = linear.vjp(bar)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): actual = linear.vjp(bar)
    torch.cuda.current_stream().wait_stream(stream)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('operation', OPS)
@pytest.mark.parametrize('population', ['query', 'reference'])
def test_field_empty_populations(device, operation, population):
    q, r, domain, fi, fj = problem(device, 3, torch.float64, 1)
    if population == 'query':
        q = replace(q, **{name: getattr(q, name)[:0] for name in
            ('positions', 'supports', 'masses', 'densities', 'kinds')})
        fi = fi[:0]
    else:
        r = replace(r, **{name: getattr(r, name)[:0] for name in
            ('positions', 'supports', 'masses', 'densities', 'kinds')})
        fj = fj[:0]
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation)
    linear = linearizeFields(q, prop, domain, fi, fj, adjacency=full_csr(q, r), referenceParticles=r)
    assert torch.count_nonzero(linear.value) == 0
    for gradient in linear.vjp(torch.ones_like(linear.value)):
        if gradient is not None: assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(linear.jvp(query=fi, reference=fj)) == 0


@pytest.mark.parametrize('width', WIDTHS)
def test_parallel_transpose_long_rows(device, width):
    # More than one CPU scheduling grain and more than 64 entries per CSC
    # row exercise parallel fill and the large-row GPU sorting fallback.
    n, m = 257, 4
    counts = torch.full((n,), m + 1, device=device, dtype=torch.int32)
    offsets = (torch.arange(n, device=device) * (m + 2) + 1).to(width[1])
    ids = torch.full((n * (m + 2) + 3,), -1, device=device, dtype=width[0])
    for row in range(n):
        at = 1 + row * (m + 2)
        ids[at:at + m + 1] = ids.new_tensor([3, 1, 2, 1, 0])
    graph = sph.AdjacencyList(None, ids, counts, offsets, n, m)
    csc = CSRTopology(graph).transpose()
    assert csc.numNeighbors.tolist() == [n, 2 * n, n, n]
    expected = torch.arange(n, device=device)
    for row in range(m):
        length = n * (2 if row == 1 else 1)
        start = int(csc.edgeOffsets[row])
        torch.testing.assert_close(csc.paddedJ[start:start + length],
            expected.repeat_interleave(2) if row == 1 else expected)


@pytest.mark.parametrize('operation', OPS)
def test_irregular_graph_pullback(device, operation):
    q, r, domain, fi, fj = problem(device, 3, torch.float64, 1)
    # Isolated rows, repeated edges, inactive pairs and unused storage. The
    # dense field Jacobian here is assembled from primal basis evaluations.
    rows = [[2, 0, 2], [], [1, 4], [3], [0, 2], [], [4, 1, 1]]
    ids, offsets = [-1], []
    for row in rows:
        offsets.append(len(ids))
        ids.extend(row + [-1])
    graph = sph.AdjacencyList(None, torch.tensor(ids, device=device, dtype=torch.int32),
        torch.tensor([len(row) for row in rows], device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int64), len(rows), len(fj))
    state, (_, volumes), omega, R = correction_fixture(q, r)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Symmetric, operationMode=sph.OperationDirection.TrueAllToToAll)
    options = dict(referenceParticles=r, adjacency=graph, referenceVolumes=volumes, crkState=state)
    if operation != sph.WarpOperation.Interpolate:
        options.update(gradHState=omega, renormalizationState=R)
    linear = linearizeFields(q, prop, domain, fi, fj, **options)
    bar = torch.cos(linear.value).contiguous()
    bq, br = linear.vjp(bar)
    zeros_q, zeros_r = torch.zeros_like(fi), torch.zeros_like(fj)
    for side, prototype, wanted in (('query', fi, bq), ('reference', fj, br)):
        if wanted is None: continue
        expected = torch.empty_like(prototype)
        for k in range(prototype.numel()):
            basis = torch.zeros_like(prototype)
            basis.view(-1)[k] = 1
            output = sph.warpOperation(q, prop, domain,
                basis if side == 'query' else zeros_q,
                basis if side == 'reference' else zeros_r, **options)
            expected.view(-1)[k] = (output * bar).sum()
        torch.testing.assert_close(wanted, expected, rtol=3e-12, atol=3e-12)


def test_field_metadata_and_inference_contract(device):
    q, r, domain, fi, fj = problem(device, 2, torch.float64, 1)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient)
    graph = full_csr(q, r)
    linear = linearizeFields(q, prop, domain, fi, fj, adjacency=graph, referenceParticles=r)
    with pytest.raises(RuntimeError, match='shape|rank'): linear.jvp(reference=fj[:1])
    with pytest.raises(RuntimeError, match='shape|rank'): linear.vjp(linear.value[:1])
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(fj, torch.ones_like(fj))
        with pytest.raises(NotImplementedError): linear.jvp(reference=dual)
    with torch.inference_mode():
        inference_graph = replace(graph, paddedJ=graph.paddedJ.clone())
    with pytest.raises(ValueError, match='versioned'): CSRTopology(inference_graph).transpose()
    # Replacing a state attribute leaves the map bound to its original storage.
    expected = linear.jvp(reference=fj)
    q.positions = q.positions + .1
    torch.testing.assert_close(linear.jvp(reference=fj), expected)
