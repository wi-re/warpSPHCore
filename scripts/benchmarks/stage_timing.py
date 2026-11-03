"""Benchmark-only partition of runner timers using existing frontend stage gates.

Each boundary closes the previous interval; nested regions never double count.
CUDA intervals are elapsed stream time (including host submission gaps), not
kernel-active time. Events are read only after the runner's existing sync.
No production profiling/validation switch or per-stage synchronization is added.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
import importlib
import json
import re
import sys
import time


class StageTiming:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.cuda = device == 'cuda'
        self.frames = []
        self.frame = None
        self.stage = 'layout / integration'
        self.engine = 'Torch/Python'
        self.step = -1
        self.restores = []
        self.event_pool = []
        self.event_index = 0

    def stamp(self):
        if self.cuda:
            if self.event_index == len(self.event_pool):
                self.event_pool.append(self.torch.cuda.Event(enable_timing=True))
            event = self.event_pool[self.event_index]
            self.event_index += 1
            event.record()
            return event
        return time.perf_counter()

    def boundary(self):
        if self.frame is not None:
            self.frame['boundaries'].append((self.stamp(), self.stage, self.engine))

    @contextmanager
    def region(self, stage=None, engine=None, operation=None):
        if self.frame is None:
            yield
            return
        self.boundary()
        previous = self.stage, self.engine
        self.stage = stage or self.stage
        self.engine = engine or self.engine
        if operation:
            self.frame['calls'][f'{self.engine}:{operation}'] += 1
        try:
            yield
        finally:
            self.boundary()
            self.stage, self.engine = previous

    def gate(self, name):
        if name == '[runner] integrator step':
            if self.frame is not None:
                self.frame['integrator'] = True
            return nullcontext()
        if re.match(r'\[warpSPH\] - \[[^]]+ - \d+[a-z]?\]', name):
            return self.region(stage=name)
        return nullcontext()

    def patch(self, obj, name, value):
        self.restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def install(self):
        import warp as wp
        import warpSPHCore.profiling as profiling
        from warpSPHCore.native import _build
        runner = importlib.import_module('warpSPH.runner.runner')
        original_gate = self.original_gate = profiling.record_function
        # Already-imported modules hold aliases; later imports use the replaced
        # core gate. Ignore Torch profiler's separate integrator bookkeeping.
        for name, module in list(sys.modules.items()):
            if name.startswith(('warpSPH', 'warpSPHCore')) and getattr(module, 'record_function', None) is original_gate:
                self.patch(module, 'record_function', self.gate)
        original_launch = wp.launch

        @wraps(original_launch)
        def launch(*args, **kwargs):
            kernel = kwargs.get('kernel', args[0] if args else None)
            if self.frame is not None and self.cuda:
                stream = kwargs.get('stream')
                if stream is None:
                    stream = wp.get_stream('cuda')
                expected = self.torch.cuda.current_stream().cuda_stream
                if stream.cuda_stream != expected:
                    raise RuntimeError('Stage timing requires Warp and Torch on the same stream')
            with self.region(engine='Warp', operation=getattr(kernel, 'key', str(kernel))):
                return original_launch(*args, **kwargs)
        self.patch(wp, 'launch', launch)
        original_build = _build.build
        timing = self

        class NativeNamespace:
            def __init__(self, namespace):
                self.namespace = namespace
                self.cache = {}

            def __getattr__(self, name):
                if name not in self.cache:
                    operator = getattr(self.namespace, name)

                    @wraps(operator)
                    def call(*args, **kwargs):
                        with timing.region(engine='native', operation=name):
                            return operator(*args, **kwargs)
                    self.cache[name] = call
                return self.cache[name]

        proxies = {}

        def build(*args, **kwargs):
            namespace = original_build(*args, **kwargs)
            if id(namespace) not in proxies:
                proxies[id(namespace)] = NativeNamespace(namespace)
            return proxies[id(namespace)]
        self.patch(_build, 'build', build)

        # The wave RHS has no numbered frontend gate. Add the corresponding
        # benchmark-only scope around its sole spatial operator, for both paths.
        wave = importlib.import_module('warpSPH.schemes.pde.waveEquation')
        original_operation = wave.warpOperation

        @wraps(original_operation)
        def wave_operation(*args, **kwargs):
            with self.region(stage='wave Laplacian'):
                return original_operation(*args, **kwargs)
        self.patch(wave, 'warpOperation', wave_operation)
        original_timer = runner._Timer

        class Timer(original_timer):
            def __enter__(timer):
                result = super().__enter__()
                if timing.frame is not None:
                    raise RuntimeError('Nested runner timers are not supported')
                timing.event_index = 0
                timing.stage, timing.engine = 'layout / integration', 'Torch/Python'
                timing.frame = dict(boundaries=[], calls=Counter(), integrator=False)
                timing.boundary()
                return result

            def __exit__(timer, *exc):
                timing.boundary()
                frame, timing.frame = timing.frame, None
                result = super().__exit__(*exc)
                if exc[0] is not None:
                    return result
                if frame['integrator']:
                    timing.step += 1
                frame['step'] = timing.step
                frame['timer_ms'] = timer.elapsed_ms
                costs = defaultdict(float)
                boundaries = frame.pop('boundaries')
                for (start, _, _), (end, stage, engine) in zip(boundaries, boundaries[1:]):
                    elapsed = start.elapsed_time(end) if timing.cuda else (end - start) * 1000
                    costs[(stage, engine)] += elapsed
                frame['costs'] = [dict(stage=s, engine=e, ms=ms) for (s, e), ms in costs.items()]
                frame['partition_ms'] = sum(costs.values())
                timing.frames.append(frame)
                return result
        self.patch(runner, '_Timer', Timer)
        return self

    def save(self, path, warmup, steps):
        frames = [f for f in self.frames if warmup <= f['step'] < warmup + steps]
        accepted = {f['step'] for f in frames if f['integrator']}
        if len(accepted) != steps:
            raise AssertionError(f'Expected {steps} profiled steps, got {len(accepted)}')
        totals = defaultdict(float)
        for frame in frames:
            for item in frame['costs']:
                totals[(item['stage'], item['engine'])] += item['ms'] / steps
        data = dict(method='Exclusive boundary intervals; CUDA elapsed current-stream time, CPU wall time',
                    mean_partition_ms=sum(f['partition_ms'] for f in frames) / steps,
                    mean_timer_ms=sum(f['timer_ms'] for f in frames) / steps,
                    mean_costs=[dict(stage=s, engine=e, ms=ms) for (s, e), ms in totals.items()], frames=frames)
        path.write_text(json.dumps(data, indent=2))
        return data

    def close(self):
        # Modules imported during the run may have bound the temporary gate.
        for name, module in list(sys.modules.items()):
            if name.startswith(('warpSPH', 'warpSPHCore')) and getattr(module, 'record_function', None) == self.gate:
                setattr(module, 'record_function', self.original_gate)
        for obj, name, value in reversed(self.restores):
            setattr(obj, name, value)
