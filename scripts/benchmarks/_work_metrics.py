"""Benchmark work ledgers: pair passes, particle passes and local matrices.

Counts describe algorithmic work, not FLOPs or instruction counts. Collect them
from actual solver diagnostics or a separate accounting pass, outside timing.
Never infer evaluated pairs from CSR capacity, a Verlet superset, or N*k when
direction/support masks are unknown. Use None for an unavailable count.
"""

from dataclasses import dataclass, field
import math
import operator


def _count(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a nonnegative integer')
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a nonnegative integer') from error
    if result < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return result


@dataclass
class WorkLedger:
    entries: list = field(default_factory=list)

    def pair(self, operation, *, evaluated_nonself_pairs, candidate_entries, passes=1,
             role='numerical', basis='observed'):
        """One directed (i,j) per pass; (i,j) and (j,i) count separately.

        Candidate entries can include self and rejected/skin pairs. Evaluated
        pairs must describe the actual operator's eligibility, excluding self.
        A fused multi-output traversal counts once, not once per output.
        """
        passes = _count(passes, 'passes')
        pairs = None if evaluated_nonself_pairs is None else _count(evaluated_nonself_pairs, 'pairs') * passes
        candidates = None if candidate_entries is None else _count(candidate_entries, 'candidate entries') * passes
        if pairs is not None and candidates is not None and pairs > candidates:
            raise ValueError('evaluated pairs cannot exceed candidate entries')
        self.entries.append(dict(kind='pair', operation=operation, role=role, passes=passes,
                                 evaluated_nonself_pairs=pairs, candidate_entries=candidates, basis=basis))
        return self

    def particle(self, operation, *, particles, passes=1, role='numerical', basis='observed'):
        """A named particle pass; one pass may contain many scalar operations."""
        passes = _count(passes, 'passes')
        self.entries.append(dict(kind='particle', operation=operation, role=role, passes=passes,
                                 particle_items=_count(particles, 'particles') * passes, basis=basis))
        return self

    def local_matrix(self, operation, *, matrices, rows, cols, method, passes=1,
                     role='numerical', basis='observed'):
        passes = _count(passes, 'passes')
        rows, cols = _count(rows, 'rows'), _count(cols, 'cols')
        if not rows or not cols:
            raise ValueError('local matrix dimensions must be positive')
        self.entries.append(dict(kind='local_matrix', operation=operation, role=role, passes=passes,
                                 matrices=_count(matrices, 'matrices') * passes, rows=rows, cols=cols,
                                 method=method, basis=basis))
        return self

    def report(self, *, wall_ms, timing_scope):
        if not math.isfinite(wall_ms) or wall_ms <= 0:
            raise ValueError('wall_ms must be finite and positive')
        seconds = wall_ms * 1e-3
        pairs = [entry for entry in self.entries if entry['kind'] == 'pair']
        complete = bool(pairs) and all(entry['evaluated_nonself_pairs'] is not None for entry in pairs)
        candidatesComplete = bool(pairs) and all(entry['candidate_entries'] is not None for entry in pairs)
        evaluations = sum(entry['evaluated_nonself_pairs'] for entry in pairs) if complete else None
        candidates = sum(entry['candidate_entries'] for entry in pairs) if candidatesComplete else None
        entries = []
        for entry in self.entries:
            row = dict(entry)
            countKey = {'pair': 'evaluated_nonself_pairs', 'particle': 'particle_items', 'local_matrix': 'matrices'}[entry['kind']]
            row['items_per_second'] = None if entry[countKey] is None else entry[countKey] / seconds
            entries.append(row)
        return dict(schema_version=1, wall_ms=wall_ms, timing_scope=timing_scope, entries=entries,
                    pair_count_complete=complete, evaluated_nonself_pairs=evaluations,
                    directed_interactions_per_second=None if evaluations is None else evaluations / seconds,
                    candidate_entries=candidates,
                    candidate_entries_per_second=None if candidates is None else candidates / seconds,
                    particle_pass_items=sum(entry['particle_items'] for entry in self.entries if entry['kind'] == 'particle'),
                    local_matrix_items=sum(entry['matrices'] for entry in self.entries if entry['kind'] == 'local_matrix'),
                    rate_scope='all entry rates use the same measured denominator; entries are work categories, not time shares',
                    normalization='algorithmic counts, unweighted; compare matching operator, dtype, dimension and traversal')


def cgWork(row, *, engine):
    """Account an existing SPH CG row; no estimates of scalar/vector FLOPs."""
    neighborhood = row['neighborhood']
    if neighborhood is None:
        return None
    counts = row['matvecs']
    if len(set(counts)) != 1:
        raise ValueError('work varies between samples; pair counts must be matched to each sample')
    # The private Warp adapter always evaluates A*x0 and then the final A*x.
    # Core diagnostics count both as original-system verifications.
    verifications = 2 if engine.startswith('warp-') else row['verifications']
    ledger = WorkLedger()
    pairKw = dict(evaluated_nonself_pairs=neighborhood['nonself_edges'], candidate_entries=neighborhood['edges'],
                  basis='actual fixed all-to-all CSR, equal supports, no skin; solver matvec count')
    ledger.pair('Brookshaw Laplacian', passes=counts[0] - verifications, role='scheduled_updates', **pairKw)
    ledger.pair('Brookshaw Laplacian', passes=verifications, role='initial_and_final_verification', **pairKw)
    # I - dx^2*L has one named particle scale/add pass per matvec. This is
    # algorithmic work, not a statement that Torch emits exactly one kernel.
    ledger.particle('identity_minus_scaled_laplacian', particles=row['n'], passes=counts[0],
                    basis='defined SPH action, one scale/add pass per matvec')
    report = ledger.report(wall_ms=row['median_wall_ms'], timing_scope='whole warmed CG call including certification and observation')
    isolated = row.get('isolated_matvec')
    if isolated:
        report['isolated_matvec'] = WorkLedger().pair('Brookshaw Laplacian', **pairKw).particle(
            'identity_minus_scaled_laplacian', particles=row['n']).report(
                wall_ms=isolated['median_graph_device_ms_per_call'], timing_scope='device interval per matvec in a warmed batch graph')
    return report
