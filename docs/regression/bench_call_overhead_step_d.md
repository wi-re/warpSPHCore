# bench_call_overhead: after Step D (view reuse, no-grad path only)

precision=float32 device=cuda

Comparison points: `bench_call_overhead_baseline.md` (pre-Steps A-C) and
`bench_call_overhead_step_c.md` (after Step C, before Step D). Same script, same device (RTX PRO
6000 Blackwell), same N/dim grid, `--warmup 20 --iters 200`.

## Headline: no-grad path, dim=2

| N | convert: baseline | convert: Step C | convert: Step D | total: baseline | total: Step D |
|---|---|---|---|---|---|
| 2025 | 298.7 us | 62.9 us | 24.3 us (12.3x) | 1197.2 us | 275.5 us (4.3x) |
| 19881 | 291.8 us | 62.3 us | 24.4 us (12.0x) | 1212.1 us | 274.0 us (4.4x) |
| 199809 | 319.8 us | 64.0 us | 24.0 us (13.3x) | 1412.8 us | 374.2 us (3.8x) |

Step C removed the disabled-correction slots' reconversion cost; Step D adds view reuse for the
*real* tensors (positions, supports, masses, densities, kinds, adjacency, domain) on calls where
nothing requires grad -- which is every call in this configuration, so `convert` now reflects an
almost-all-cache-hit steady state. No outliers this run (Step C's recorded run had one -- see that
doc's footnote; not reproduced here).

## Grad path: unchanged by design

| N (dim=2) | convert: Step C (grad) | convert: Step D (grad) |
|---|---|---|
| 2025 | 137.2 us | 133.8 us |
| 19881 | 116.7-141.8 us | 141.6 us |
| 199809 | 116.7 us | 145.8 us |

Step D gates caching to `not ctx.any_requires_grad` -- the grad path still builds a fresh wrapper
every call, exactly as before Step D, because reusing a wrapper's `.grad` buffer across calls
without zeroing it is the specific bug that got the original cache deleted (Section 3.3). Step E
brings the grad path in, behind the twice-in-process gradcheck gate and a zero-on-acquire
contract, only after the full gradcheck suite passes it twice in the same process.

## Full sweep

| dim | N | traversal | grad | extract | convert | build_fn | allocate | other | total |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | grid | False | 11.3 | 24.2 | 48.9 | 14.4 | 236.5 | 335.2 |
| 1 | 2000 | grid | True | 11.8 | 127.0 | 70.7 | 15.6 | 339.2 | 564.3 |
| 1 | 20000 | grid | False | 11.1 | 23.8 | 48.4 | 14.0 | 186.4 | 283.8 |
| 1 | 20000 | grid | True | 11.8 | 143.1 | 58.1 | 15.1 | 327.2 | 555.4 |
| 1 | 200000 | grid | False | 11.4 | 24.4 | 52.7 | 15.0 | 253.3 | 356.9 |
| 1 | 200000 | grid | True | 11.6 | 141.9 | 58.0 | 15.0 | 361.3 | 587.9 |
| 2 | 2025 | grid | False | 11.2 | 24.3 | 50.3 | 14.1 | 175.6 | 275.5 |
| 2 | 2025 | grid | True | 12.7 | 133.8 | 75.0 | 15.3 | 310.3 | 547.1 |
| 2 | 19881 | grid | False | 11.3 | 24.4 | 49.7 | 13.9 | 174.7 | 274.0 |
| 2 | 19881 | grid | True | 11.7 | 141.6 | 57.9 | 14.9 | 319.2 | 545.3 |
| 2 | 199809 | grid | False | 11.8 | 24.0 | 52.1 | 15.1 | 271.2 | 374.2 |
| 2 | 199809 | grid | True | 12.2 | 145.8 | 61.3 | 15.5 | 377.8 | 612.5 |
| 3 | 2197 | grid | False | 11.6 | 23.9 | 49.5 | 14.3 | 177.7 | 277.0 |
| 3 | 2197 | grid | True | 12.2 | 123.8 | 73.3 | 15.1 | 320.4 | 544.8 |
| 3 | 19683 | grid | False | 12.1 | 24.7 | 51.7 | 14.4 | 186.7 | 289.6 |
| 3 | 19683 | grid | True | 11.9 | 143.3 | 58.9 | 15.4 | 300.8 | 530.3 |
| 3 | 195112 | grid | False | 12.4 | 25.7 | 54.3 | 16.2 | 288.0 | 396.7 |
| 3 | 195112 | grid | True | 12.7 | 148.2 | 63.0 | 16.0 | 417.2 | 657.2 |

## Note: `build_fn` has not moved

`build_fn` still costs ~48-75 us -- the per-call struct-write closure Step F replaces with
`StateBundle.refresh`. Steps C/D only touch the conversion loop, not struct assembly, so this
number is expected to hold until Step F lands.
