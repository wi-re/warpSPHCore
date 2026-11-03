"""Optional, synchronizing input verification for native-operation debugging.

These functions deliberately scan contents. Call them explicitly from a test or
an interactive debugging session, outside the simulation or measured interval.
"""
import torch


def verify_finite(values, name='values'):
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f'{name} must be finite')


def verify_geometry(query, reference, domain):
    for name, particles in [('query', query), ('reference', reference)]:
        verify_finite(particles.positions, f'{name} positions')
        verify_finite(particles.supports, f'{name} supports')
        if not bool((particles.supports >= 0).all()):
            raise ValueError(f'{name} supports must be nonnegative')
    verify_finite(domain.min, 'domain minimum')
    verify_finite(domain.max, 'domain maximum')
    if not bool((domain.max > domain.min).all()):
        raise ValueError('domain maximum must exceed its minimum')


def verify_csr(offsets, counts, indices, references):
    if not bool((counts >= 0).all()):
        raise ValueError('CSR counts must be nonnegative')
    wide_counts = counts.to(torch.int64)
    expected = torch.cumsum(wide_counts, 0) - wide_counts
    if not torch.equal(offsets.to(torch.int64), expected):
        raise ValueError('CSR offsets must be the exclusive scan of counts')
    total = int(wide_counts.sum())
    if total > indices.numel():
        raise ValueError('CSR capacity does not cover live entries')
    live = indices[:total]
    if not bool(((live >= 0) & (live < references)).all()):
        raise ValueError('CSR contains an invalid live reference index')
