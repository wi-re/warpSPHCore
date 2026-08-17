# bench_call_overhead: after Step E (view reuse, grad path)

precision=float32 device=cuda

Comparison points: `bench_call_overhead_baseline.md` (pre-Steps A-C), `bench_call_overhead_step_c.md`,
`bench_call_overhead_step_d.md`. Same script, same device (RTX PRO 6000 Blackwell), same N/dim
grid, `--warmup 20 --iters 200`.

## Headline: grad path, dim=2

| N | convert: baseline | convert: Step D | convert: Step E | total: baseline | total: Step E |
|---|---|---|---|---|---|
| 2025 | 542.7 us | 133.8 us | 38.9 us (13.9x) | 1681.9 us | 366.2 us (4.6x) |
| 19881 | 546.9 us | 141.6 us | 38.6 us (14.2x) | 1753.7 us | 346.4 us (5.1x) |
| 199809 | 596.5 us | 145.8 us | 38.8 us (15.4x) | 1995.4 us | 414.5 us (4.8x) |

Step D deliberately left the grad path uncached (Section 3.3's gating). Step E brings the same
view-reuse cache to the grad path, protected by zero-on-acquire; `convert` on the grad path is now
within ~1.6x of the no-grad path's 24 us (Step D doc), rather than ~6x higher.

## No-grad path: unchanged from Step D, as expected

| N (dim=2) | convert: Step D (no-grad) | convert: Step E (no-grad) |
|---|---|---|
| 2025 | 24.3 us | 23.7 us |
| 19881 | 24.4 us | 24.0 us |
| 199809 | 24.0 us | 24.8 us |

Step E only changes `use_cached_views` for the grad path (`(not any_requires_grad) or
_field_cache_grad_enabled()`); the no-grad path was already unconditionally `True` before this
step and stays so. Numbers match within noise, as expected for an unmodified code path.

## Full sweep

| dim | N | traversal | grad | extract | convert | build_fn | allocate | other | total |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | grid | False | 11.0 | 23.5 | 49.3 | 14.3 | 236.6 | 334.7 |
| 1 | 2000 | grid | True | 11.5 | 39.0 | 62.8 | 15.1 | 238.1 | 366.4 |
| 1 | 20000 | grid | False | 11.4 | 24.0 | 48.6 | 14.1 | 189.7 | 287.9 |
| 1 | 20000 | grid | True | 11.7 | 39.4 | 59.9 | 15.2 | 225.1 | 351.3 |
| 1 | 200000 | grid | False | 11.0 | 24.1 | 50.9 | 15.0 | 245.7 | 346.7 |
| 1 | 200000 | grid | True | 11.1 | 38.6 | 61.4 | 16.1 | 284.6 | 411.8 |
| 2 | 2025 | grid | False | 10.9 | 23.7 | 48.6 | 14.0 | 171.4 | 268.7 |
| 2 | 2025 | grid | True | 11.7 | 38.9 | 58.0 | 15.3 | 242.2 | 366.2 |
| 2 | 19881 | grid | False | 11.2 | 24.0 | 48.6 | 13.7 | 171.7 | 269.3 |
| 2 | 19881 | grid | True | 11.3 | 38.6 | 58.4 | 14.9 | 223.1 | 346.4 |
| 2 | 199809 | grid | False | 11.6 | 24.8 | 52.7 | 15.9 | 270.0 | 375.0 |
| 2 | 199809 | grid | True | 11.5 | 38.8 | 62.1 | 15.1 | 287.0 | 414.5 |
| 3 | 2197 | grid | False | 11.3 | 23.9 | 49.0 | 14.2 | 174.0 | 272.3 |
| 3 | 2197 | grid | True | 11.2 | 37.5 | 57.9 | 15.4 | 221.1 | 343.1 |
| 3 | 19683 | grid | False | 10.7 | 23.7 | 48.8 | 14.2 | 179.4 | 276.6 |
| 3 | 19683 | grid | True | 11.5 | 37.7 | 59.6 | 14.8 | 218.9 | 342.5 |
| 3 | 195112 | grid | False | 11.2 | 24.0 | 50.8 | 14.9 | 240.4 | 341.2 |
| 3 | 195112 | grid | True | 11.4 | 39.9 | 54.5 | 24.1 | 288.2 | 418.2 |

## Correctness cost of getting here

This step's implementation surfaced two real bugs before it shipped as default-on, both caught by
the twice-in-process gradcheck gate this step's own design called for (not by the bench numbers,
which look identical whether the underlying value is right or silently 2x wrong):

1. **Grad-buffer double-counting when one source tensor fills two kernel roles.** The common case
   `referenceParticles=None` makes `qPos`/`rPos` (and `qSup`/`rSup`, `qMas`/`rMas`) literally the
   same tensor object. Under caching they now map to the *same* wp.array, so warp's adjoint
   kernels correctly sum both roles' contributions into that one shared `.grad` buffer -- but
   `backward()` was reading that buffer once per flat-tensor *position*, reporting the same total
   twice, which PyTorch then summed again at the leaf. Fixed by deduplicating on `id(wa)`: only the
   first flat-tensor position referencing a given cached wp.array reports its gradient; later
   positions sharing it report `None`. Caught by `gradcheck_density_native.py` failing with
   analytical exactly 2x numerical.
2. **A latent detach() bug in `fieldRegistry.acquireView`'s two fallback branches** (cache-disabled,
   non-contiguous) -- both built from the bare tensor `t` instead of `t.detach()`, unlike the main
   cached-view path. Harmless before this step (those branches only ever saw non-grad tensors:
   null fields, and Step D's no-grad-only gate); Step E is the first caller that can hand
   `acquireView` a `requires_grad=True` tensor, and building off an undetached tensor let that
   conversion sit in torch's own autograd graph in addition to warp's tape, again doubling the
   reported gradient. Caught by cross-checking a cached run against a
   `WARPSPHCORE_DISABLE_FIELD_CACHE=1` run that disagreed by 2x.

Both are fixed in the landed code; see `stateAwareWarpFunction.py` and `fieldRegistry.py` for the
comments at each fix site.
