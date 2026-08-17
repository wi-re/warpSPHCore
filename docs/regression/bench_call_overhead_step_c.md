# bench_call_overhead: after Step C (null fields replace dummy tensors)

precision=float32 device=cuda

Comparison point: `bench_call_overhead_baseline.md`, recorded before warpier_fields.md's Steps
A-C landed. Same script (`scripts/bench_call_overhead.py`), same device (RTX PRO 6000
Blackwell), same N/dim grid, `--warmup 20 --iters 200`.

## Headline: no-grad path, dim=2 (Section 1's reference configuration)

| N | convert: baseline | convert: after Step C | total: baseline | total: after Step C |
|---|---|---|---|---|
| 2025 | 298.7 us | 62.9 us (4.7x) | 1197.2 us | 380.5 us (3.1x) |
| 19881 | 291.8 us | 62.3 us (4.7x) | 1212.1 us | 362.8 us (3.3x) |
| 199809 | 319.8 us | 64.0 us (5.0x) | 1412.8 us | 453.9 us (3.1x) |

`convert` is the flat-tensor -> wp.array conversion loop Step C targets directly: the disabled
correction slots (no CRK/gradH/volumes/renormalization in this configuration) no longer pay a
`from_torch` each call, only a Field-registry dict lookup. Real tensors (positions, supports,
masses, densities, kinds) still convert fresh every call -- that's Step D's job, not this one's.
`total` drops by more than `convert` alone accounts for; some of that is measurement variance
across the two runs (different process, same GPU), not attributed further here.

## Full sweep

| dim | N | traversal | grad | extract | convert | build_fn | allocate | other | total |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | grid | False | 11.3 | 62.7 | 52.7 | 14.4 | 304.7 | 445.8 |
| 1 | 2000 | grid | True | 11.7 | 131.0 | 68.5 | 15.0 | 333.0 | 559.2 |
| 1 | 20000 | grid | False | 11.3 | 63.5 | 52.0 | 13.9 | 234.1 | 374.8 |
| 1 | 20000 | grid | True | 11.6 | 134.4 | 67.5 | 15.1 | 300.2 | 528.7 |
| 1 | 200000 | grid | False | 11.5 | 63.8 | 56.4 | 14.6 | 305.6 | 451.9 |
| 1 | 200000 | grid | True | 11.5 | 117.1 | 71.8 | 15.6 | 376.6 | 592.7 |
| 2 | 2025 | grid | False | 11.3 | 62.9 | 52.7 | 13.6 | 239.9 | 380.5 |
| 2 | 2025 | grid | True | 12.4 | 137.2 | 65.7 | 15.0 | 304.9 | 535.1 |
| 2 | 19881 | grid | False | 11.3 | 62.3 | 53.3 | 13.3 | 222.7 | 362.8 |
| 2 | 19881 | grid | True | 12.5 | 1518.9\* | 74.2 | 15.6 | 334.4 | 1955.6\* |
| 2 | 199809 | grid | False | 11.6 | 64.0 | 58.0 | 14.6 | 305.7 | 453.9 |
| 2 | 199809 | grid | True | 11.7 | 116.7 | 72.4 | 15.1 | 377.9 | 593.6 |
| 3 | 2197 | grid | False | 11.2 | 63.8 | 53.2 | 13.6 | 215.4 | 357.2 |
| 3 | 2197 | grid | True | 11.5 | 141.8 | 58.3 | 14.8 | 323.6 | 549.9 |
| 3 | 19683 | grid | False | 11.6 | 61.9 | 53.4 | 13.3 | 229.5 | 369.8 |
| 3 | 19683 | grid | True | 12.4 | 138.4 | 66.0 | 15.2 | 311.3 | 543.3 |
| 3 | 195112 | grid | False | 11.7 | 63.9 | 58.3 | 14.5 | 337.1 | 485.5 |
| 3 | 195112 | grid | True | 13.1 | 117.3 | 72.6 | 15.5 | 390.4 | 608.9 |

\* Reproduced across two full-sweep runs at exactly this (dim=2, N=19881, grad=True) position in
the sweep, but **not** reproduced when that single configuration is benchmarked in isolation
(`--dims 2 --ns 20000 --grad`, same warmup/iters), where convert measures 130.9 us -- in line with
every other grad=True row. Looks like a one-time Warp kernel-cache/autograd-graph compile
triggered by that specific dim/grad transition partway through the sweep, not a Step C
regression. Worth a look if it starts showing up elsewhere, but not chased further here.

## Grad path

Grad-path convert also drops sharply (e.g. dim=2 N=19881: 546.9 us baseline -> 137.2 us at N=2025,
116.7-141.8 us across the sweep) even though Step C's null-field change applies equally on both
grad and no-grad paths today -- Step D/E's *view-reuse* gating (no-grad-only, then grad after the
twice-in-process gate) hasn't landed yet, so this number will move again once it does.
