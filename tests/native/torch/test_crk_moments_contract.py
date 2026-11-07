"""Complete raw moment outputs and the first reverse of their explicit JVP."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.crk_moments_contract import case, equation, product_equation, flat


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('explicit', [False, True])
def test_all_kernel_actual_gradcheck(device, kernel, explicit):
    i = list(sph.KernelFunctions).index(kernel)
    *_, values, states, functions = case(device, 1 + i % 3, kernel=kernel, explicit=explicit)
    def run(*x):
        outputs = functions('native')(*x)
        return outputs if explicit else outputs[:6]
    assert torch.autograd.gradcheck(run, values, fast_mode=True, check_forward_ad=not explicit, atol=8e-5, rtol=2e-3)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('explicit,support', [(False, support) for support in sph.SupportScheme] + [(True, sph.SupportScheme.Scatter)])
def test_scalar_and_weighted_products_independent_displacement_equation(device, dimension, support, explicit):
    q, r, domain, props, graph, values, states, functions = case(device, dimension, support=support, explicit=explicit)
    inputs = [x for x in values if x.requires_grad]
    for weighted in (False, True):
        actual = functions('native')(*values)
        expected = product_equation(values, domain, props, graph) if explicit else equation(values, domain, props, graph)
        torch.testing.assert_close(flat(actual[:6]), flat(expected), rtol=2e-7, atol=2e-7)
        seeds = tuple(torch.linspace(-.3, .7, x.numel(), device=device, dtype=x.dtype).reshape_as(x)
                      if weighted else torch.ones_like(x) for x in actual[:6])
        bars = torch.autograd.grad(actual[:6], inputs, seeds, allow_unused=True)
        wanted = torch.autograd.grad(expected, inputs, seeds, allow_unused=True)
        for a, b, x in zip(bars, wanted, inputs):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,
                                       torch.zeros_like(x) if b is None else b, rtol=2e-7, atol=2e-7)
        if not explicit:
            assert not actual[-1].requires_grad
            torch.testing.assert_close(actual[-1], graph.numNeighbors)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_volume_only_product_actual_reverse_gradcheck(device, dimension, kernel):
    *_, values, states, functions = case(device, dimension, kernel=kernel, explicit=True, geometry=False)
    assert torch.autograd.gradcheck(functions('native'), values, fast_mode=True, atol=8e-5, rtol=2e-3)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('role', [4, 7, 8, 9, 11])
def test_sole_unread_input_retains_zero_reverse_dependency(device, explicit, role):
    if explicit and role == 11:
        pytest.skip('query volumes are not an explicit moment API argument')
    *_, values, states, functions = case(device, explicit=explicit)
    values = tuple(x.detach().requires_grad_(k == role) for k, x in enumerate(values))
    outputs = functions('native')(*values)[:6]
    bar = torch.autograd.grad(sum(x.sum() for x in outputs), values[role])[0]
    torch.testing.assert_close(bar, torch.zeros_like(bar), rtol=0, atol=0)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('index,offset', [(torch.int32, torch.int32), (torch.int32, torch.int64),
                                        (torch.int64, torch.int32), (torch.int64, torch.int64)])
def test_sparse_strided_deterministic_reverse_and_topology_guard(device, explicit, index, offset):
    q, r, domain, props, graph, values, states, functions = case(device, explicit=explicit, graph_types=(index, offset))
    graph.paddedJ = torch.tensor([-1, -1, 0, 2, 2, -1], device=device, dtype=index)
    graph.edgeOffsets = torch.tensor([2, 5], device=device, dtype=offset)
    graph.numNeighbors = torch.tensor([3, 0], device=device, dtype=torch.int32)
    def strided(x):
        storage = torch.empty((*x.shape, 2), device=x.device, dtype=x.dtype); storage[..., 0] = x.detach()
        return storage[..., 0].requires_grad_(x.requires_grad)
    values = tuple(strided(x) for x in values); inputs = [x for x in values if x.requires_grad]
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for attempt in range(3):
            outputs = functions('native')(*values)[:6]
            bars = torch.autograd.grad(outputs, inputs, tuple(torch.ones_like(x) for x in outputs), allow_unused=True)
            assert all(x is None or x.isfinite().all() for x in bars)
            if attempt == 0:
                first = bars
            else:
                for a, b in zip(first, bars):
                    if a is not None: torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)
    outputs = functions('native')(*values)[:6]; graph.paddedJ[2] = 1
    with pytest.raises(RuntimeError, match='topology changed|modified by an inplace'):
        torch.autograd.grad(outputs, inputs, tuple(torch.ones_like(x) for x in outputs), allow_unused=True)


@pytest.mark.parametrize('mode', ['query', 'none'])
def test_volume_fallbacks_keep_counts_and_actual_forward_reverse_gradcheck(device, mode):
    *_, values, states, functions = case(device, volume_mode=mode)
    assert torch.autograd.gradcheck(lambda *x: functions('native')(*x)[:6], values, fast_mode=True,
                                   check_forward_ad=True, atol=8e-5, rtol=2e-3)


@pytest.mark.parametrize('explicit', [False, True])
def test_efficient_zero_cotangents_are_materialized(device, explicit):
    *_, values, states, functions = case(device, explicit=explicit)
    outputs = functions('native')(*values)[:6]
    bars = torch.autograd.grad(outputs, [x for x in values if x.requires_grad],
        tuple(torch._efficientzerotensor(x.shape, device=device, dtype=x.dtype) for x in outputs), allow_unused=True)
    for bar in bars:
        if bar is not None: torch.testing.assert_close(bar, torch.zeros_like(bar), rtol=0, atol=0)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_full_rank_factor_pipeline_actual_reverse_gradcheck(device, kernel, dimension):
    from _support.crk_factors_contract import case as factor_case
    values, functions = factor_case(device, dimension, kernel=kernel)
    outputs = functions('native')(*values)
    assert not torch.allclose(outputs[2], torch.ones_like(outputs[2]))
    assert outputs[7].abs().max() > 0
    assert torch.autograd.gradcheck(functions('native'), values, fast_mode=True, atol=2e-4, rtol=3e-3)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('sizes', [(0, 3), (2, 0), (0, 0)])
def test_empty_populations_and_reverse(device, explicit, sizes):
    from warpSPHCore.crk.native.nt_moments import computeCRKMoments
    from warpSPHCore.crk.native.nt_jvp import computeCRKMomentsGeometryJVP
    q, r, d, p, g, v, states, _ = case(device, explicit=explicit)
    q, r, d = states(v[:10]); n, m = sizes
    q = replace(q, **{a: getattr(q, a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r = replace(r, **{a: getattr(r, a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g = sph.AdjacencyList(None, torch.empty(0, device=device, dtype=torch.int64),
        torch.zeros(n, device=device, dtype=torch.int32), torch.zeros(n, device=device, dtype=torch.int64), n, m)
    if explicit:
        outputs = computeCRKMomentsGeometryJVP(q, d, p.kernel, g, v[10][:m], v[16][:m],
            sph.ParticleTangentState(v[12][:n], v[14][:n], None), r,
            sph.ParticleTangentState(v[13][:m], v[15][:m], None), p.operationMode)
    else:
        outputs = computeCRKMoments(q, p, d, None, v[10][:m], g, r)[:6]
    assert all(len(x) == n and not x.count_nonzero() for x in outputs)
    bars = torch.autograd.grad(outputs, [x for x in v if x.requires_grad], tuple(torch.ones_like(x) for x in outputs), allow_unused=True)
    assert all(x is None or not x.count_nonzero() for x in bars)


@pytest.mark.parametrize('explicit', [False, True])
def test_three_step_feedback_reverse(device, explicit):
    q, r, d, p, g, v, states, functions = case(device, explicit=explicit)
    def rollout(evaluate):
        position = v[0]
        for _ in range(3):
            fields = evaluate(position, *v[1:])[:6]
            position = position + .002*flat(fields).sum(-1)[:, None].tanh()*position.cos()
        return torch.cat((position.flatten(), flat(fields).flatten()))
    actual = rollout(functions('native'))
    expected = rollout(lambda *x: product_equation(x, d, p, g) if explicit else equation(x, d, p, g))
    seed = torch.linspace(-.3, .7, actual.numel(), device=device, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-7)
    inputs = [x for x in v if x.requires_grad]
    bars = torch.autograd.grad(actual, inputs, seed, allow_unused=True)
    wanted = torch.autograd.grad(expected, inputs, seed, allow_unused=True)
    for a, b, x in zip(bars, wanted, inputs):
        torch.testing.assert_close(torch.zeros_like(x) if a is None else a,
                                   torch.zeros_like(x) if b is None else b, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize('explicit', [False, True])
def test_native_search_without_warp_and_shared_role_gradcheck(device, explicit, monkeypatch):
    import builtins
    from warpSPHCore.crk.native.nt_moments import computeCRKMoments
    from warpSPHCore.crk.native.nt_jvp import computeCRKMomentsGeometryJVP
    q, r, d, p, g, v, states, functions = case(device, explicit=explicit)
    original = builtins.__import__
    def guarded(name, *a, **k):
        if name == 'warp' or name.startswith('warp.'): raise AssertionError('Native moments imported Warp')
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guarded)
    for a, b in zip(functions('native', adjacency=None)(*v), functions('native')(*v)):
        torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-7)
    q, _, d = states(v[:10])
    # Cross edges keep finite differences off the self-pair custom-Hessian
    # boundary while both geometric roles still alias the same leaf tensors.
    g = sph.AdjacencyList(None, torch.tensor([1,0], device=device),
        torch.ones(2, device=device, dtype=torch.int32), torch.tensor([0,1], device=device), 2, 2)
    shared = (v[0], v[2], v[10][:2], *(v[k] for k in (12,14,16))) if explicit else (v[0], v[2], v[10][:2])
    def run(x, h, volume, *t):
        state = replace(q, positions=x, supports=h)
        if not explicit: return computeCRKMoments(state, p, d, None, volume, g)[:6]
        return computeCRKMomentsGeometryJVP(state, d, p.kernel, g, volume, t[2][:2],
            sph.ParticleTangentState(t[0], t[1], None), referenceTangentState=sph.ParticleTangentState(t[0]*.3, t[1]*-.2, None),
            operationMode=p.operationMode)
    assert torch.autograd.gradcheck(run, shared, fast_mode=True, atol=8e-5, rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('mode', ['ordinary', 'explicit'])
def test_nondefault_stream_capture_replay_with_reverse(device, dimension, mode):
    if device != 'cuda': pytest.skip('CUDA execution contract')
    import gc
    gc.collect(); enabled = gc.isenabled(); gc.disable()
    try:
        *_, v, states, functions = case(device, dimension, torch.float32, explicit=mode == 'explicit')
        inputs = [x for x in v if x.requires_grad]; run = functions('native')
        def products():
            outputs = run(*v)[:6]
            seeds = tuple(torch.linspace(-.3, .7, x.numel(), device=device, dtype=x.dtype).reshape_as(x) for x in outputs)
            bars = torch.autograd.grad(outputs, inputs, seeds, allow_unused=True)
            return (*outputs, *(torch.zeros_like(x) if bar is None else bar for x, bar in zip(inputs, bars)))
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): products()
        torch.cuda.current_stream().wait_stream(stream); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): captured = products()
        with torch.no_grad(): v[0][:, 0].add_(.001); v[2].mul_(1.01)
        graph.replay(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): wanted = products()
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(captured, wanted): torch.testing.assert_close(a, b, rtol=7e-4, atol=7e-4)
    finally:
        if enabled: gc.enable()


@pytest.mark.gpu
@pytest.mark.parametrize('dimension', [1,2,3])
def test_factor_capture_matches_shared_torch_pseudoinverse_limit(device, dimension):
    if device != 'cuda': pytest.skip('CUDA execution contract')
    import gc
    from _support.crk_factors_contract import case as factor_case
    values, functions = factor_case(device, dimension, torch.float32)
    matrix = (torch.eye(dimension, device=device)[None]*2).requires_grad_()
    # The shared coefficient solve calls this same Torch pseudoinverse in
    # both backends. Diagnose that primitive directly as well as the complete
    # native fixed-CSR pipeline; do not add a native first-order/capture guard.
    for run in (lambda: torch.linalg.pinv(matrix), lambda: functions('native')(*values)):
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): run()
        torch.cuda.current_stream().wait_stream(stream)
        gc.collect(); enabled = gc.isenabled(); gc.disable()
        try:
            with pytest.raises(RuntimeError, match='Cannot copy between CPU and CUDA tensors during CUDA graph capture'):
                with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream): run()
        finally:
            if enabled: gc.enable()
