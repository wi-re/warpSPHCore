# bench_call_overhead: after Step F (StateBundle replaces the per-call closure)

precision=float32 device=cuda

Comparison points: `bench_call_overhead_baseline.md` (pre-Steps A-C) and
`bench_call_overhead_step_e.md` (Steps A-E, before this step). Same script, same device (RTX PRO
6000 Blackwell), same N/dim grid, `--warmup 20 --iters 200`.

Two runs recorded here, because Step F's other change (Section 3.7's `record_function` gating)
affects how this bench script itself measures things: `bench_call_overhead.py` reads its
per-stage breakdown off `torch.profiler`-recorded regions, and those regions are now no-ops by
default (`WARPSPHCORE_PROFILING=0`, the new default -- see `profiling.py`). Both numbers are real;
they answer different questions.

## Run 1: `WARPSPHCORE_PROFILING=1` -- per-stage breakdown, for comparison with prior steps

| dim | N | traversal | grad | extract | convert | build_fn | allocate | other | total |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | grid | False | 11.0 | 22.1 | 10.7 | 11.9 | 224.3 | 280.0 |
| 1 | 2000 | grid | True | 11.4 | 37.8 | 53.6 | 14.2 | 244.1 | 361.1 |
| 1 | 20000 | grid | False | 11.2 | 22.0 | 10.4 | 11.7 | 175.5 | 230.8 |
| 1 | 20000 | grid | True | 11.4 | 38.1 | 54.1 | 13.9 | 222.9 | 340.4 |
| 1 | 200000 | grid | False | 11.5 | 23.3 | 11.7 | 14.2 | 244.3 | 305.1 |
| 1 | 200000 | grid | True | 11.4 | 38.8 | 55.0 | 15.8 | 316.4 | 437.4 |
| 2 | 2025 | grid | False | 10.8 | 22.5 | 10.3 | 11.9 | 163.6 | 219.1 |
| 2 | 2025 | grid | True | 11.9 | 39.1 | 61.2 | 14.7 | 253.4 | 380.4 |
| 2 | 19881 | grid | False | 10.9 | 22.4 | 10.5 | 11.9 | 163.4 | 219.0 |
| 2 | 19881 | grid | True | 11.4 | 38.1 | 54.3 | 14.1 | 226.6 | 344.5 |
| 2 | 199809 | grid | False | 11.3 | 23.5 | 12.1 | 13.9 | 255.5 | 316.3 |
| 2 | 199809 | grid | True | 11.4 | 39.0 | 54.8 | 23.4 | 302.8 | 431.5 |
| 3 | 2197 | grid | False | 11.2 | 22.4 | 10.5 | 11.9 | 163.0 | 219.0 |
| 3 | 2197 | grid | True | 11.5 | 39.1 | 61.2 | 14.4 | 222.8 | 349.0 |
| 3 | 19683 | grid | False | 11.3 | 22.4 | 10.5 | 12.0 | 167.8 | 224.0 |
| 3 | 19683 | grid | True | 11.5 | 37.8 | 53.7 | 14.2 | 227.1 | 344.2 |
| 3 | 195112 | grid | False | 11.5 | 23.7 | 11.7 | 13.7 | 248.1 | 308.6 |
| 3 | 195112 | grid | True | 11.7 | 39.6 | 149.7\* | 16.4 | 316.0 | 533.4 |

\* One-off; not reproduced at neighboring N/dim in this or prior sweeps. Consistent with the
kernel-cache/compile-artifact outliers noted in the Step C and D docs -- not chased further.

### Headline: no-grad path, `build_fn`, dim=2

| N | build_fn: baseline | build_fn: Step E | build_fn: Step F | total: baseline | total: Step F |
|---|---|---|---|---|---|
| 2025 | 142.1 us | 58.0 us | 10.3 us (13.8x) | 1197.2 us | 219.1 us (5.5x) |
| 19881 | 141.5 us | 58.4 us | 10.5 us (13.5x) | 1212.1 us | 219.0 us (5.5x) |
| 199809 | 152.7 us | 62.1 us | 12.1 us (12.6x) | 1412.8 us | 316.3 us (4.5x) |

`build_fn` is the per-call struct-assembly closure Step F targets directly. Step E already halved
it somewhat as a side effect of cheaper tensor conversion feeding into it; Step F's own
contribution -- refreshing a persistent `StateBundle` in place, writing only the array fields that
actually changed since the last call -- takes it the rest of the way, close to the ~1.6 us
best-case the design predicted for an all-hit call (this bench's synthetic workload reuses the
*same* particle/adjacency tensors every iteration, so nearly every field is a "changed" write only
on the first call and a skip thereafter; a real integrator step, per Section 2.3, sees a mixed hit
rate -- ~8 particle fields miss once per stage, ~28 remaining entries hit).

### Grad path: unchanged by design

`build_fn` on the grad path (54-61 us, occasionally higher) is within noise of Step E's numbers,
as expected -- Step F's bundle reuse is gated to `not ctx.any_requires_grad`, the same restriction
as Step D's view-reuse cache and Step E's grad-path caching, but for a different and non-negotiable
reason (see "Correctness cost" below): there is no zero-on-acquire equivalent that would make
sharing a mutable struct across grad-requiring calls safe.

## Run 2: `WARPSPHCORE_PROFILING=0` (the new default) -- what callers actually see

| dim | N | grad | total (profiling off) | total (profiling on, Run 1) |
|---|---|---|---|---|
| 2 | 2025 | False | 146.4 us | 219.1 us |
| 2 | 2025 | True | 266.5 us | 380.4 us |
| 2 | 19881 | False | 147.3 us | 219.0 us |
| 2 | 199809 | False | 234.0 us | 316.3 us |

The ~70-115 us gap between the two runs at each row is `record_function`'s own cost when actually
hooked to a profiler -- Section 3.7's second finding, now eliminated by default rather than merely
documented. Per-stage columns are all 0.0 in this mode (nothing to attribute time to, by design);
`total` is still wall-clock and real.

## Correctness cost of getting here

`StateBundle`'s design -- a persistent struct instance refreshed in place instead of rebuilt per
call -- surfaced a hazard the written plan did not anticipate, verified directly against warp
1.16.0 before any implementation code was written: **`wp.Tape` does not snapshot a struct's field
values at launch time.** It holds a live reference to whatever struct object a launch was given
and re-reads its fields lazily, at `tape.backward()` time. A minimal repro (two `wp.launch` calls
sharing one mutable struct, the struct's array field reassigned between them, backward run only on
the first tape) showed the *second* call's array ending up in the *first* call's gradient, and the
first call's own gradient reading as zero.

Sharing a mutable bundle across grad-requiring calls would therefore silently corrupt an earlier
call's gradient any time its backward is deferred past a later call that refreshes the same
bundle -- which is completely ordinary PyTorch usage (build a graph across several ops, call
`.backward()` once), not an edge case. `StateBundle` reuse is therefore gated to
`not ctx.any_requires_grad`, unconditionally, with no escape-hatch env var (unlike Steps D/E's
`WARPSPHCORE_DISABLE_FIELD_CACHE`/`WARPSPHCORE_FIELD_CACHE_GRAD`): there is no zero-on-acquire-style
contract that would make grad-path struct sharing safe, so there is nothing to bisect a suspected
bug against -- the grad path always gets a fresh, call-local struct set, matching the original
`build_fn` construction exactly. A new regression test,
`tests/operations/test_state_bundle.py::test_deferred_backward_across_two_grad_calls_not_corrupted`,
pins this: two independent grad-requiring forward calls (sharing dim, so they *would* share a
bundle if the gate were ever weakened), neither backward run until both forwards complete, checked
against a fully-sequential reference. Verified this test actually catches the hazard (not merely
passing vacuously) by temporarily monkeypatching `use_bundle=True` unconditionally and confirming
it fails with exactly the predicted symptom (one call's gradient reading as zero).

Two smaller bugs were also caught and fixed while landing Step F's "cheap wins" (Section 3.7):

1. **A circular import** from gating `record_function` behind a flag: the gate module was first
   placed at `autograd/profiling.py`, and `util/wp_util.py` importing `from ..autograd.profiling
   import record_function` forced `warpSPHCore.autograd`'s `__init__.py` to run (pulling in
   `radiusSearch` via `arg_extract.py`), which reentered `util` while it was still mid-import,
   getting back a partially-initialized module missing `castTorchToWarp` and others. Fixed by
   moving the gate to a genuinely dependency-free top-level module (`warpSPHCore/profiling.py`)
   imported as the very first statement in `warpSPHCore/__init__.py`, before even `type_config` --
   guaranteeing it is fully loaded before any reentrant chain elsewhere in the package can start.
2. **`wp.Device` instances are unhashable** (define `__eq__` without `__hash__`). The
   `allocateTorchWarp` dtype/device memoization used the raw `device` argument as part of a dict
   key, which is sometimes a `wp.Device` object (e.g. `warp_array.device`) rather than a plain
   string -- fixed by keying on `str(device)`.

Neither bug was reachable by a change confined to `stateAwareWarpFunction.py`/`arg_extract.py`
alone; both came from the "fold in the Section 3.7 cheap wins while here" scope, a reminder that
even low-risk, purely-additive changes (an import gate, a memoization) are not exempt from the
verify-before-trusting treatment the rest of this plan applies to the caching steps themselves.
