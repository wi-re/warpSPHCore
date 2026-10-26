"""Benchmark contracts before any GPU sweeps; never imports CUDA at collection."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/benchmarks'))
from _operator_cases import Case, cases, work_counts


def test_local_work_has_no_fabricated_interactions():
    c = Case(1024, operation='pinv_symmetric')
    assert work_counts(c, 1024 * 57) == dict(pair_passes=0, evaluated_nonself_pairs=None,
                                            candidate_entries=None, matrices=1024)


def test_fused_moments_and_full_crk_counts():
    assert work_counts(Case(1024, operation='crk_moments'), 58368)['evaluated_nonself_pairs'] == 57344
    assert work_counts(Case(1024, operation='crk_factors'), 58368)['evaluated_nonself_pairs'] == 3 * 57344
    assert work_counts(Case(1024, traversal='grid'), 58368)['candidate_entries'] is None
    assert work_counts(Case(1024, traversal='rebuild'), 58368)['candidate_entries'] is None


def test_irrelevant_matrix_options_are_deduplicated():
    result = list(cases([1024], operations=['pinv_symmetric'], forms=['plain', 'renorm', 'crk']))
    assert len(result) == 1
    result = list(cases([1024], operations=['gradient'], neighbors=[55], forms=['plain', 'renorm', 'crk']))
    assert len(result) == 12


@pytest.mark.parametrize('kwargs', [dict(n=10), dict(n=0), dict(n=1024, lanes=3), dict(n=1024, neighbors=0)])
def test_bad_case_rejected(kwargs):
    with pytest.raises(ValueError):
        Case(**kwargs)


def test_output_checker_checks_dataclasses_and_arity():
    import torch
    from dataclasses import dataclass
    from _operator_cases import check_output
    @dataclass
    class Output:
        value: object
    check_output(Output(torch.ones(2)), Output(torch.ones(2)))
    with pytest.raises(AssertionError):
        check_output(Output(torch.tensor([float('nan')])))
    with pytest.raises(AssertionError):
        check_output((torch.ones(1),), (torch.ones(1), torch.ones(1)))


def test_physical_and_useful_occupancy_are_distinct():
    from survey_operator_resources import sm120_upper_bound
    result = sm120_upper_bound(40, 8)
    assert result['physical_warp_occupancy_upper_bound'] == pytest.approx(32 / 48)
    assert result['useful_thread_occupancy_upper_bound'] == pytest.approx(8 / 32 * 32 / 48)
    assert sm120_upper_bound(128, 256)['resident_blocks_upper_bound'] == 2


def test_parse_compiler_resources_and_spills():
    from survey_operator_resources import parse_ptxas
    log = """ptxas info : Compiling entry function 'forward' for 'sm_120'
ptxas info : Function properties for forward
    64 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
ptxas info : Used 96 registers, 256 bytes smem
ptxas info : Compiling entry function 'backward' for 'sm_120'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 32 registers
"""
    result = parse_ptxas(log)
    assert result['forward'] == dict(registers_per_thread=96, stack_frame_bytes=64, spill_store_bytes=16,
                                   spill_load_bytes=8, static_shared_bytes=256)
    assert result['backward']['spill_load_bytes'] == 0
    assert result['backward']['static_shared_bytes'] == 0


def test_ptxas_helper_properties_do_not_overwrite_entry_spills():
    from survey_operator_resources import parse_ptxas
    log = """ptxas info : Compiling entry function 'forward' for 'sm_120'
ptxas info : Function properties for forward
    216 bytes stack frame, 280 bytes spill stores, 192 bytes spill loads
ptxas info : Used 255 registers, 216 bytes cumulative stack size
ptxas info : Function properties for __internal_accurate_pow
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Compiling entry function 'backward' for 'sm_120'
ptxas info : Function properties for backward
    288 bytes stack frame, 432 bytes spill stores, 120 bytes spill loads
ptxas info : Used 255 registers, 288 bytes cumulative stack size
ptxas info : Function properties for another_helper
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""
    result = parse_ptxas(log)
    assert set(result) == {'forward', 'backward'}
    assert result['forward']['stack_frame_bytes'] == 216
    assert result['forward']['cumulative_stack_bytes'] == 216
    assert result['forward']['spill_store_bytes'] == 280
    assert result['forward']['spill_load_bytes'] == 192
    assert result['backward']['spill_store_bytes'] == 432
    assert result['backward']['spill_load_bytes'] == 120


def test_plot_normalization_uses_matching_denominator():
    from plot_core_operators import normalize
    row = dict(n=1024, operation='renorm', wall_ms=[1., 3., 2.], median_wall_ms=2.,
               counts=dict(evaluated_nonself_pairs=2048, matrices=1024))
    result = list(normalize(dict(records=[row]), 'fixture'))[0]
    assert result['rate'] == 1024000
    assert result['unit'] == 'interactions/s'
    row['counts']['evaluated_nonself_pairs'] = None
    result = list(normalize(dict(records=[row]), 'fixture'))[0]
    assert result['rate'] == 512000
    assert result['unit'] == 'matrices/s'
    row['median_wall_ms'] = 1.
    with pytest.raises(ValueError, match='median'):
        list(normalize(dict(records=[row]), 'fixture'))


def test_profile_counts_only_raw_kernel_events():
    from profile_sph_stages import summarize
    result = summarize([
        dict(cat='kernel', name='warp_kernel', dur=12., args={'registers per thread': 64}),
        dict(cat='kernel', name='warp_kernel', dur=8., args={}),
        dict(cat='cpu_op', name='wrapper', dur=1000.),
        dict(cat='cuda_runtime', name='cudaStreamSynchronize', dur=100.),
        dict(cat='user_annotation', name='aggregate_with_device_time', dur=5000.),
    ], repeat=2)
    assert result['kernel_launches_per_call'] == 1
    assert result['summed_kernel_ms_per_call'] == .01
    assert result['kernels'][0]['count'] == 2
    assert result['kernels'][0]['launch_attributes']['registers per thread'] == 64
    assert result['cuda_runtime_counts']['cudaStreamSynchronize'] == 1


def test_keep_going_preserves_numerical_failures(monkeypatch, tmp_path):
    import json
    import warp as wp
    import bench_core_operators as benchmark
    class Fixture:
        def __init__(self, n, dim, neighbors, device):
            self.edges = 2 * n
    def qualify(fixture, case):
        if case.operation == 'density':
            raise AssertionError('numerical mismatch fixture')
        return lambda: None, {}
    output = tmp_path / 'qualification.json'
    monkeypatch.setattr(benchmark, 'Workload', Fixture)
    monkeypatch.setattr(benchmark, 'qualify', qualify)
    monkeypatch.setattr(wp, 'init', lambda: None)
    monkeypatch.setattr(wp.config, 'kernel_cache_dir', str(tmp_path / 'cache'))
    monkeypatch.setattr(sys, 'argv', ['benchmark', 'validate', '--ns', '1024', '--neighbors', '55',
                                    '--operations', 'density', 'gradient', '--traversals', 'adjacency',
                                    '--lanes', '1', '--keep-going', '--output', str(output)])
    benchmark.main()
    result = json.loads(output.read_text())
    assert result['complete']
    assert result['numerical_check_failures'] == 1
    assert [r['status'] for r in result['records']] == ['failed_numerical_check', 'qualified']
    assert 'numerical mismatch fixture' in result['records'][0]['error']


def test_counter_attribution_checks_case_order_and_ignores_auxiliary_kernels():
    from analyze_sph_counters import associate
    document = dict(records=[dict(status='profiled_without_timing', operation='density', dim=2)])
    auxiliary = {'Kernel Name': 'torch_fill', 'ID': '0', 'Block Size': '(128,1,1)', 'Grid Size': '(8,1,1)'}
    kernel = {'Kernel Name': 'computeSPHDensity_Kernel_hash_cuda_kernel_forward', 'ID': '1',
              'Block Size': '(256,1,1)', 'Grid Size': '(4,1,1)', 'registers': '64'}
    rows = associate(document, [auxiliary, kernel], {'registers': 'register/thread'}, 'fixture')
    assert len(rows) == 1
    assert rows[0]['case']['operation'] == 'density'
    assert rows[0]['metrics']['registers'] == dict(value=64., unit='register/thread')
    with pytest.raises(ValueError, match='order mismatch'):
        associate(document, [kernel | {'Kernel Name': 'computeCovariance_Kernel_hash_cuda_kernel_forward'}], {}, 'fixture')
    with pytest.raises(ValueError, match='unassigned'):
        associate(document, [kernel, kernel], {}, 'fixture')
