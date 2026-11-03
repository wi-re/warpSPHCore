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
