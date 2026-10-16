"""Independent work counts protect comparable interaction-rate denominators."""

from pathlib import Path

import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts/benchmarks'))
    import _work_metrics
    return _work_metrics


def test_directed_pairs_verification_and_fused_outputs(metrics):
    # Three particles, each with self + two directed neighbors. Two fused
    # multi-output update passes and one verification visit 27 CSR entries,
    # evaluate 18 non-self pairs, and take 2ms for the entire window.
    ledger = metrics.WorkLedger().pair('fused moments', evaluated_nonself_pairs=6, candidate_entries=9, passes=2)
    ledger.pair('residual', evaluated_nonself_pairs=6, candidate_entries=9, role='verification')
    report = ledger.report(wall_ms=2, timing_scope='entire test step')
    assert report['evaluated_nonself_pairs'] == 18
    assert report['candidate_entries'] == 27
    assert report['directed_interactions_per_second'] == 9000
    assert report['candidate_entries_per_second'] == 13500
    assert report['entries'][0]['items_per_second'] == 6000


def test_unknown_masks_do_not_become_evaluated_pairs(metrics):
    ledger = metrics.WorkLedger().pair('Verlet superset', evaluated_nonself_pairs=None, candidate_entries=100)
    report = ledger.report(wall_ms=1, timing_scope='masked action')
    assert not report['pair_count_complete']
    assert report['evaluated_nonself_pairs'] is None
    assert report['directed_interactions_per_second'] is None
    assert report['candidate_entries_per_second'] == 100000


def test_particle_and_local_matrix_work_stays_separate(metrics):
    ledger = metrics.WorkLedger().particle('EOS', particles=10, passes=2)
    ledger.local_matrix('CRK pinv', matrices=10, rows=3, cols=3, method='SVD')
    report = ledger.report(wall_ms=4, timing_scope='step')
    assert report['particle_pass_items'] == 20
    assert report['local_matrix_items'] == 10
    assert report['entries'][0]['items_per_second'] == 5000
    assert report['entries'][1]['items_per_second'] == 2500
    assert report['entries'][1]['method'] == 'SVD'
    assert report['directed_interactions_per_second'] is None


@pytest.mark.parametrize('bad', [-1, 1.5, True])
def test_invalid_counts_are_rejected(metrics, bad):
    with pytest.raises(ValueError):
        metrics.WorkLedger().pair('bad', evaluated_nonself_pairs=bad, candidate_entries=10)


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan')])
def test_invalid_timing_is_rejected(metrics, bad):
    with pytest.raises(ValueError):
        metrics.WorkLedger().report(wall_ms=bad, timing_scope='test')


def test_cg_counts_include_initial_and_final_verification(metrics):
    row = dict(n=3, neighborhood=dict(edges=9, nonself_edges=6), matvecs=[10, 10],
               verifications=2, median_wall_ms=5, isolated_matvec=None)
    report = metrics.cgWork(row, engine='native-cached')
    assert report['evaluated_nonself_pairs'] == 60
    assert report['entries'][0]['evaluated_nonself_pairs'] == 48
    assert report['entries'][1]['evaluated_nonself_pairs'] == 12
    assert report['directed_interactions_per_second'] == 12000
    assert report['particle_pass_items'] == 30
    row['matvecs'] = [10, 11]
    with pytest.raises(ValueError, match='varies between samples'):
        metrics.cgWork(row, engine='native-cached')
