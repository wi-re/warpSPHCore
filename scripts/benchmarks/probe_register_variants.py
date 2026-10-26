#!/usr/bin/env python3
"""Experimental forward-only register probes; never alters production dispatch.

Clone a captured production kernel using an AST-transformed source. Each
variant changes one compiler input while retaining the original argument ABI,
pair functions, fields and iteration order. Resource/timing evidence is local
to the selected fixture and installed Warp/CUDA versions, not a new default.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _operator_cases import Case, Workload, check_output, tensors
from bench_core_operators import save

def source_hashes(root):
    paths = sorted((root / "src/warpSPHCore").rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Constants(ast.NodeTransformer):
    def __init__(self, traversal, states):
        self.traversal, self.states = traversal, states

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id == 'useAdjacency' and self.traversal is not None:
                return ast.copy_location(ast.Constant(self.traversal), node)
            if node.id in self.states:
                return ast.copy_location(ast.Name('_probe_' + node.id, ast.Load()), node)
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        prefix = []
        for name, values in self.states.items():
            text = f'_probe_{name} = {name}\n'
            text += ''.join(f'_probe_{name}.{key} = {value!r}\n' for key, value in values.items())
            prefix.extend(ast.parse(text).body)
        node.body = prefix + node.body
        return ast.fix_missing_locations(node)


@contextmanager
def launches():
    import warp as wp
    original, seen = wp.launch, []
    def capture(kernel, dim, inputs=None, outputs=None, **kwargs):
        seen.append((kernel, dim, list(inputs or []) + list(outputs or []), kwargs))
        return original(kernel, dim, inputs=inputs, outputs=outputs or [], **kwargs)
    wp.launch = capture
    try:
        yield seen
    finally:
        wp.launch = original


def clone(kernel, args, variant):
    import warp as wp
    indices = kernel.arg_indices
    traversal = bool(args[indices['useAdjacency']]) if variant in ('traversal', 'policy') else None
    states = {}
    if variant == 'policy':
        # Scalar policy, not particle data: no tensor readback or state caching.
        for name, fields in {
            'kernelProperties': ('kernelFunction', 'supportMode', 'gradientMode', 'laplacianMode',
                                 'positiveDivergenceMode', 'divergenceMode', 'operationMode', 'calibrateNormalization'),
            'correctionData': ('useVolume', 'useCRK', 'useGradHTerms', 'useGradientRenormalization'),
        }.items():
            state = args[indices[name]]
            states[name] = {key: getattr(state, key) for key in fields}
    signature = repr((kernel.key, traversal, states, variant))
    suffix = hashlib.sha256(signature.encode()).hexdigest()[:12]
    module = wp.get_module('register_probe_' + suffix)
    module.options.update(kernel.module.options)
    module.options['enable_backward'] = False
    options = dict(kernel.options)
    if variant == 'lean':
        options['grid_stride'] = False
    elif variant.startswith('cap'):
        options['cuda_max_registers'] = int(variant[3:])
    # Passing transformers alone is insufficient: this Warp version's generic
    # add_overload reconstructs from adj.source without its transformers.
    # Materialize the transformed source so concrete overloads retain it.
    tree = Constants(traversal, states).visit(ast.parse(textwrap.dedent(inspect.getsource(kernel.func))))
    return wp.Kernel(kernel.func, key=kernel.key + '_probe_' + suffix, module=module,
                     options=options, source=ast.unparse(tree))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT, help='isolated checkout containing src/warpSPHCore')
    p.add_argument('--dim', type=int, choices=[2, 3], default=3)
    p.add_argument('--n', type=int, default=262144)
    p.add_argument('--neighbors', type=int, default=55)
    p.add_argument('--operations', nargs='+', default=['density', 'gradient', 'covariance', 'crk_moments'])
    p.add_argument('--traversals', nargs='+', choices=['adjacency', 'grid'], default=['adjacency', 'grid'])
    p.add_argument('--lanes', nargs='+', type=int, default=[1, 32])
    p.add_argument('--variants', nargs='+', choices=['runtime', 'traversal', 'policy', 'lean', 'cap64', 'cap96', 'cap128'], default=['runtime', 'traversal', 'policy', 'lean'])
    p.add_argument('--repeat', type=int, default=9)
    p.add_argument('--profile-range', action='store_true')
    p.add_argument('--save-outputs', action='store_true', help='save untimed original outputs for cross-source comparisons')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.source_root = a.source_root.resolve()
    sys.path.insert(0, str(a.source_root / 'src'))
    import torch
    import warp as wp
    from warpSPHCore import setNeighborLanes
    from warpSPHCore.autograd.stream import torchWarpStream
    from warpSPHCore.type_config import get_torch_precision
    wp.init()
    torch.set_num_threads(2)
    wp.config.kernel_cache_dir = str(a.output.parent / 'probe-cache')
    saved_outputs = {}
    report = dict(complete=False, source_root=str(a.source_root), command=sys.argv, precision=str(get_torch_precision()),
                  warp=wp.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  source_sha256=source_hashes(a.source_root), probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='experimental forward-only cloned kernels; public output qualification against original; CUDA event intervals around one direct launch exclude public wrapper/allocation but include launch gaps', records=[])
    save(a.output, report)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w = Workload(a.n, a.dim, a.neighbors, 'cuda:0')
        for op in a.operations:
            for traversal in a.traversals:
                for lanes in a.lanes:
                    case = Case(a.n, a.dim, a.neighbors, op, traversal=traversal, lanes=lanes)
                    call = w.call(case)  # correction preparation may reset lane policy
                    setNeighborLanes(lanes)
                    with launches() as seen:
                        expected = call()
                    if len(seen) != 1:
                        raise RuntimeError(f'expected one operator kernel, got {len(seen)}')
                    original, dim, args, kw = seen[0]
                    # Keep the tensors whose arrays we captured alive throughout timing.
                    outputs = expected
                    if a.save_outputs:
                        saved_outputs[json.dumps(case.dictionary(), sort_keys=True)] = [t.detach().cpu().clone() for t in tensors(expected)]
                    for variant in a.variants:
                        kernel = original if variant == 'runtime' else clone(original, args, variant)
                        def direct():
                            with torchWarpStream('cuda:0'):
                                wp.launch(kernel, dim, inputs=args, **kw)
                        for _ in range(4):
                            direct()
                        stream.synchronize()
                        # Compare cloned outputs with a fresh original output.
                        check_output(outputs, call())
                        stream.synchronize()
                        concrete = kernel.add_overload(kernel.infer_argument_types(args)) if kernel.is_generic else kernel
                        props = wp.get_cuda_kernel_properties(concrete, device='cuda:0', block_dim=kw.get('block_dim', 256))
                        row = case.dictionary() | dict(variant=variant, kernel=concrete.key, resources=props, block_dim=kw.get('block_dim', 256), options=dict(kernel.options), concrete_source_sha256=hashlib.sha256(concrete.adj.source.encode()).hexdigest())
                        if a.profile_range:
                            wp.cuda_profiler_start()
                            direct()
                            stream.synchronize()
                            wp.cuda_profiler_stop()
                            row['status'] = 'profiled'
                        else:
                            samples = []
                            for _ in range(a.repeat):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                direct()
                                end.record()
                                end.synchronize()
                                samples.append(start.elapsed_time(end))
                            row.update(status='qualified', kernel_interval_ms=samples, median_kernel_interval_ms=statistics.median(samples))
                        report['records'].append(row)
                        save(a.output, report)
                        print(op, traversal, lanes, variant, row.get('median_kernel_interval_ms'), props, flush=True)
    if a.save_outputs:
        torch.save(saved_outputs, a.output.with_suffix('.pt'))
    report['complete'] = True
    report['source_hashes_match_at_end'] = report['source_sha256'] == source_hashes(a.source_root)
    save(a.output, report)


if __name__ == '__main__':
    main()
