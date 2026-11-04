"""One focused check of exclusive attribution in the benchmark instrumenter."""
import importlib.util
from pathlib import Path
from collections import Counter, defaultdict


def test_nested_stage_and_dispatch_intervals_do_not_double_count(tmp_path):
    source = Path(__file__).resolve().parents[2] / 'scripts/benchmarks/stage_timing.py'
    spec = importlib.util.spec_from_file_location('stage_timing', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    timing = module.StageTiming('cpu')
    clock = iter(range(8))
    timing.stamp = lambda: next(clock)
    timing.frame = dict(boundaries=[], calls=Counter())
    timing.boundary()  # t=0, integration begins
    with timing.region(stage='CRK factors'):  # t=1
        with timing.region(engine='native', operation='moments'):  # t=2..3
            pass
        with timing.region(engine='Warp', operation='coefficient solve'):  # t=4..5
            pass
    timing.boundary()  # t=7
    costs = defaultdict(float)
    boundaries = timing.frame['boundaries']
    for (start, _, _), (end, stage, engine) in zip(boundaries, boundaries[1:]):
        costs[(stage, engine)] += end - start
    assert costs == {('layout / integration', 'Torch/Python'): 2,
                     ('CRK factors', 'Torch/Python'): 3,
                     ('CRK factors', 'native'): 1,
                     ('CRK factors', 'Warp'): 1}
    assert sum(costs.values()) == 7
    assert timing.frame['calls'] == {'native:moments': 1, 'Warp:coefficient solve': 1}
    assert (timing.stage, timing.engine) == ('layout / integration', 'Torch/Python')


def test_tiled_launch_attribution_counts_nested_dispatch_once():
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[2] / 'scripts/benchmarks/stage_timing.py'
    spec = importlib.util.spec_from_file_location('stage_timing', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    timing = module.StageTiming('cpu')
    clock = [0.]
    timing.stamp = lambda: clock[0]
    timing.frame = dict(boundaries=[], calls=Counter())
    physical = []
    def launch(kernel):
        physical.append(kernel.key)
        clock[0] += 2.
        return 17
    wp = SimpleNamespace(launch=launch)
    def launch_tiled(kernel):
        clock[0] += 1.
        result = wp.launch(kernel)
        clock[0] += 1.
        return result
    wp.launch_tiled = launch_tiled
    timing.install_warp_launches(wp)
    timing.boundary()
    assert wp.launch_tiled(SimpleNamespace(key='tile')) == 17
    timing.boundary()
    assert timing.frame['calls'] == {'Warp:tile': 1}
    assert physical == ['tile']
    assert timing.warp_launch_depth == 0
    costs = defaultdict(float)
    b = timing.frame['boundaries']
    for (start, _, _), (end, stage, engine) in zip(b, b[1:]):
        costs[engine] += end-start
    assert costs == {'Torch/Python': 0., 'Warp': 4.}
    for obj, name, value in reversed(timing.restores):
        setattr(obj, name, value)
    assert wp.launch is launch and wp.launch_tiled is launch_tiled
