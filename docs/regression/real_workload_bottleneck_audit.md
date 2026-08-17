# Step H: real-workload bottleneck audit

**Date:** 2026-08-17. **Deliverable for** `warpier_fields.md` Step H.
**Harness:** `scripts/bench_real_workload.py` (added by this step -- see
[Re-running with other scenarios](#re-running-with-other-scenarios), and note that the
bottleneck is strongly scenario-dependent).
**Machine:** RTX PRO 6000 Blackwell (96 GB), warp 1.16.0, torch 2.13.0+cu130, float32.

## The question

`bench_call_overhead.py` showed Steps A-F cutting fixed per-operator dispatch cost
~4-6x. A real ~70k-particle run then improved ~7% end to end (45 -> 42 min). Both
numbers are correct and they answer different questions. This audit answers the second:
**of a real solver step's wall clock, what fraction is the dispatch overhead Steps A-F
targeted, and what dominates instead?**

## Workload and method

The frontend's `dambreak` case configured as fully-periodic **Kolmogorov flow** -- the
invocation actually used for data generation -- driven through `warpSPH.runner.run`
(deltaSPH, RK2, adaptive dt, 16 SPH operator calls per step). Plot/store/video off.

Two measurements per scale:

1. **Ground truth:** per-step wall clock from the runner's own CUDA-event timing, *no
   profiler attached*. This is what a real run pays.
2. **Attribution:** `torch.profiler` (CPU+CUDA) over 15 steady-state steps after a
   12-step warmup, stepped once per simulation step through the case's `postStep` hook,
   with `WARPSPHCORE_PROFILING=1` so the `record_function` regions Step F gated off are
   real. Profiling inflates host time, so shares are the meaningful figures there.

Host time is attributed to the **innermost enclosing `record_function` region**, not to
the event's own name. This is not a detail -- see [Two attribution
bugs](#two-attribution-bugs-found-while-building-this-both-of-which-reversed-a-conclusion);
naming alone gave the opposite conclusion.

## Result 1 -- the audit's headline finding, now fixed

The audit found the **Kolmogorov forcing dominating from ~57k particles up**, at 66-68% of
host self-time. `caseUtils/weaklyCompressible.py`'s `forcing()` evaluated a **scipy
`RegularGridInterpolator` on the host** every stage:

```python
u_y = noiseGen(pos.detach().cpu()).to(dtype=x.dtype, device=x.device) * noiseLevel
```

so every stage shipped the whole position array device -> host, interpolated on one CPU
core, and shipped the result back. At 155k particles that was 16.4 ms/step inside the
`computeForcing` region (the scipy call), 9.1 ms/step in `aten::copy_`, and 3.7 ms/step in
`cudaStreamSynchronize` underneath it.

**This has since been fixed in the frontend** (`d9ad712`, "add torch interpolator"):
`warpSPH.math.interpolation.RegularGridInterpolator` is a torch-native, device-resident
port, so the sampling stays on the GPU. Measured before and after, same harness, same
scales, forcing on in both:

| particles | forcing, scipy (ms/step) | forcing, torch (ms/step) | step, scipy (ms) | step, torch (ms) | step speedup |
|----------:|-------------------------:|-------------------------:|-----------------:|-----------------:|-------------:|
| 2,401     | 2.10                     | 2.06                     | 9.69             | 9.50             | 1.02x        |
| 5,476     | 2.38                     | 2.23                     | 10.92            | 10.66            | 1.02x        |
| 19,044    | 3.47                     | 2.10                     | 12.35            | 10.75            | 1.15x        |
| 57,121    | **26.36**                | 2.27                     | 20.56            | 11.67            | **1.76x**    |
| 155,236   | **35.13**                | 5.62                     | 48.03            | 17.11            | **2.81x**    |

**The forcing has left the scaling cost.** It is now flat at ~2.1-2.3 ms/step through 57k
and rises to 5.6 ms at 155k -- i.e. it scales like the device work it now is, rather than
like a per-stage host round-trip. At 155k the whole step is now within 5% of the same case
with the forcing removed entirely (17.11 vs 16.35 ms).

**Confirmed end-to-end on the real run, not just in this harness:** the full ~70k-particle
production run went from **42 to 16 minutes** (user-reported, same case, immediately after
the interpolator change) -- **2.6x**, sitting inside the 1.76x (57k) / 2.81x (155k) bracket
measured above and close to what interpolating between them predicts. Worth stating plainly
because it is the second time this audit's two measurement scales have agreed with a full
run: the same workload is what earlier pinned Steps A-F's own contribution at ~7%. A
profiled 15-step window on this case predicts full-run wall clock well.

## Result 2 -- the current picture: dispatch is the largest host bucket again

Kolmogorov forcing **ON**, torch interpolator (i.e. the frontend as it now stands):

| particles | step (ms) | GPU busy (ms) | GPU share | dispatch | neighbour | forcing | integrator | solver |
|----------:|----------:|--------------:|----------:|---------:|----------:|--------:|-----------:|-------:|
| 2,401     | 9.50      | 3.68          | 38.7%     | **31.5%**| 12.9%     | 16.5%   | 9.6%       | 25.4%  |
| 5,476     | 10.66     | 3.83          | 35.9%     | **32.3%**| 13.1%     | 16.5%   | 8.6%       | 25.4%  |
| 19,044    | 10.75     | 4.00          | 37.3%     | **31.8%**| 12.1%     | 16.5%   | 8.3%       | 27.1%  |
| 57,121    | 11.67     | 4.75          | 40.7%     | **30.8%**| 12.2%     | 17.0%   | 8.0%       | 27.7%  |
| 155,236   | 17.11     | 10.41         | 60.8%     | 21.6%    | 9.0%      | 29.2%   | 6.4%       | 30.6%  |

Same case with the forcing removed entirely, as a reference for what the solver alone
costs:

| particles | step (ms) | GPU busy (ms) | GPU share | dispatch | neighbour | integrator | solver |
|----------:|----------:|--------------:|----------:|---------:|----------:|-----------:|-------:|
| 2,401     | 8.71      | 3.85          | 44.2%     | 36.5%    | 14.9%     | 9.9%       | 34.0%  |
| 5,476     | 9.38      | 4.03          | 43.0%     | 35.7%    | 15.2%     | 10.2%      | 33.9%  |
| 19,044    | 9.61      | 4.12          | 42.9%     | 35.6%    | 15.1%     | 9.7%       | 34.9%  |
| 57,121    | 10.66     | 4.97          | 46.6%     | 34.0%    | 14.3%     | 9.7%       | 36.9%  |
| 155,236   | 16.35     | 10.98         | 67.2%     | 22.3%    | 10.6%     | 7.4%       | 56.5%  |

Both tables show the shape Section 1.1's baseline predicted: **step time is nearly flat
from 2.4k to 57k particles** (8.71 -> 10.66 ms without forcing, a 1.2x rise for 24x the
particles; per-particle cost falls 19x) because fixed per-step cost dominates, and the run
only becomes genuinely GPU-bound at ~150k (61-67% device-busy). **Dispatch is the largest
single host bucket at every scale up to 57k, at a flat ~4.0-4.4 ms/step** -- N-independent,
as designed.

`solver` at 155k is not solver Python: `blocking_ms` there is 8.3-8.8 of 18-19 ms host
(43-48%), i.e. the host correctly waiting on ~11 ms of real GPU work. Below 57k blocking is
19-24% of host time across ~41 `cudaStreamSynchronize` calls per step.

## Result 3 -- in-situ before/after: what Steps A-F actually bought here

Per-step wall clock as landed vs. with every Steps A-F caching layer disabled
(`WARPSPHCORE_DISABLE_FIELD_CACHE=1`, `WARPSPHCORE_FIELD_CACHE_GRAD=0`,
`WARPSPHCORE_DISABLE_BUNDLE=1`), 40 measured steps:

| particles | landed (ms) | caching off (ms) | delta | speedup | forcing off: speedup |
|----------:|------------:|-----------------:|------:|--------:|---------------------:|
| 2,401     | 8.93        | 10.08            | +1.15 | 1.13x   | 1.12x                |
| 19,044    | 9.52        | 10.70            | +1.17 | 1.12x   | 1.16x                |
| 57,121    | 9.88        | 10.66            | +0.78 | 1.08x   | 1.11x                |
| 155,236   | 15.72       | 16.32            | +0.59 | 1.04x   | 1.06x                |

**1.04-1.16x on the full solver loop, against 4-6x on the isolated dispatch
microbenchmark.** Both are real. The microbenchmark measures the marshalling Steps A-F
removed; the loop also pays the parts they did not touch -- ~350 `cudaLaunchKernel`s per
step, warp's own launch machinery, the Python call chain through 16 operator wrappers, ~41
host stalls, and the neighbour/integrator/solver work around it.

What the saving *is*, in absolute terms, is a near-constant **~0.6-1.2 ms/step** at every
scale -- the number to reason about, since the percentage depends entirely on what else the
step is paying for. Against the *scipy-era* step of 20.6 ms at 57k, ~1 ms is ~5%; the
reported real run improved ~7% at ~70k. **The two agree to within the precision either
measurement supports, so the 45 -> 42 min observation is fully accounted for -- no missing
regression, and no shortfall against Steps A-F's own gates.** With the forcing fixed, the
same ~1 ms is now ~8-12% of a step instead of ~5%, purely because the denominator shrank.

`WARPSPHCORE_DISABLE_BUNDLE` did not exist -- Step F ships no hatch by design, since the
grad path is gated on `requires_grad` rather than a variable and there is nothing to
bisect. Step H's methodology anticipated this ("note as a finding if it's missing"); it
was added as a **measurement-only** hatch (disabling it only falls back to the original
per-call struct construction, so it can never be the unsafe direction), and verified
non-inert by checking that `_BUNDLE_CACHE` stays empty with it set.

## Re-running with other scenarios

**The bottleneck is scenario-dependent, so do not extrapolate from the table above --
re-run.** Same core, same harness, five scenarios, and the dominant host bucket differs in
every one:

| scenario | particles | step (ms) | op calls | launches | syncs | allocs | dominant host buckets |
|----------|----------:|----------:|---------:|---------:|------:|-------:|-----------------------|
| dambreak + Kolmogorov, RK2, 2D (default) | 5,476 | 10.66 | 16 | 348 | 41 | 409 | dispatch 32%, solver 25% |
| ... with `integrationScheme=rungeKutta4` | 5,476 | 17.61 | 30 | 908 | 86 | 1060 | dispatch 33%, solver 26% |
| `--case tgv` | 4,096 | 9.24 | 23 | 308 | 42 | 715 | **dispatch 47%**, integrator 28% |
| `--case sod2d --spec scheme=CRKSPH` | 1,200 | 15.71 | 18 | 695 | 50 | 758 | **integrator 43%**, dispatch 27%, neighbour 22% |
| `--case sod3d` | 16,011 | 75.69 | 14 | 1024 | 82 | 825 | **integrator 45%**, neighbour 29%, sync 15%, dispatch 5% |

(Indicative single-scale probes at 6 warmup + 6 measured steps, not gate-quality sweeps.
Their purpose is to show how far the profile moves, not to characterise those cases.)

Two things worth reading off that table. RK4 doubles the operator calls (16 -> 30) and
2.6x's the allocations (409 -> 1060) at identical particle count, so it is the scenario for
anything about Section 7's integrator churn. And **3D at only 16k particles is
integrator- and neighbour-bound with dispatch at 5%** -- the opposite of the 2D conclusion,
which is exactly why this is a harness rather than a one-off script.

### Invocations

```bash
# the audit's own sweep, and the two comparisons that interpret it
python scripts/bench_real_workload.py --nx 39 64 128 229 384
python scripts/bench_real_workload.py --nx 39 64 128 229 384 --no-forcing
python scripts/bench_real_workload.py --nx 39 128 229 384 --hatches

# what cases exist; and the raw profiler event names behind the buckets
python scripts/bench_real_workload.py --listCases
python scripts/bench_real_workload.py --dump 128 --top 40
```

| flag | effect |
|------|--------|
| `--nx N [N ...]` | lattice resolutions to sweep. Particles ~ `1.34 * nx^2` in 2D for this case; the table prints the actual count. |
| `--warmup` / `--active` | steps discarded / measured. Warmup must cover warp's kernel compilation and allocator growth; 12 is enough here, 15+ for the `--hatches` medians. |
| `--case NAME` | frontend case by registered name (default `dambreak`). |
| `--preset kolmogorov\|none` | case-parameter preset. **Use `none` for any case other than `dambreak`** -- the preset sets dambreak-specific knobs. |
| `--param NAME=VALUE` | case knob, repeatable (lands in `CaseSpec.params`). |
| `--spec NAME=VALUE` | `CaseSpec` field, repeatable: `scheme`, `integrationScheme`, `kernel`, `supportMode`, `n_h`, `adaptiveDt`, `dt`, ... Enum fields take their names as strings, as on the frontend's own CLI. |
| `--no-forcing` | drop the Kolmogorov forcing and noise IC (dambreak only). |
| `--hatches` | in-situ before/after with every Steps A-F caching layer off. Spawns two subprocesses per scale, since the hatches are read at import time. |
| `--dump NX` | print the profiler's own top events at one scale. This is how the bucket classifier was built; use it before trusting a new bucket. |

Recipes for particular paths:

```bash
... --spec integrationScheme=rungeKutta4          # 4 stages: integrator churn
... --case sod2d --preset none --spec scheme=CRKSPH   # the CRK correction path
... --case sod3d --preset none                    # 3D: neighbour/compute balance
... --spec n_h=6.0                                # bigger neighbourhoods, same N
... --spec kernel=Wendland6 --spec supportMode=SuperSymmetric
```

### Reading the output

* **`step_ms` (unprofiled) is the only absolute number to trust.** Everything in the HOST
  table is profiler-inflated; compare shares, not milliseconds. Host totals run ~12-19 ms
  against a real 9-17 ms step for that reason.
* **`gpu_share` = device-busy / `step_ms`.** Low means host-bound (the GPU idles waiting);
  as it approaches 100% the run is genuinely GPU-limited and no dispatch-level work can
  help. This is the single number that says whether further core-side effort is worth
  anything at a given scale.
* **`blocking_ms` is cross-cutting, not a bucket.** It counts host stalls wherever they
  occur, so it deliberately overlaps the buckets instead of partitioning with them. When
  the run is GPU-bound a large `blocking_ms` is correct behaviour, not a pathology.
* **`op_calls` / `launches` / `syncs` / `allocs` per step** are often more diagnostic than
  the timings when comparing two scenarios -- they are exact counts, not inflated.
* Buckets: `dispatch` is what Steps A-F targeted (`extractStateInfo`, `SAWF.forward`,
  `warpWrapper2`, `launch_kernel`, `cudaLaunchKernel`); `neighbour` includes the Verlet
  validity check; `integrator` is the `[Integration]` regions; `solver` is frontend module
  work outside the core; `forcing` is broken out because it used to dominate. `other`
  should stay ~1%; if it grows, run `--dump` and add a rule.

### Adding a bucket

`REGION_RULES` maps `record_function` region names to buckets and is consulted for a
region's own self-time *and* for everything nested inside it; `HOST_RULES` is the fallback
for events with no enclosing region; `DEVICE_RULES` classifies GPU kernels by name. Add to
`REGION_RULES` first -- attributing by region is what makes the numbers mean anything (see
below). Verify with `--dump` that the names you matched are the names the profiler actually
emits.

## Two attribution bugs found while building this, both of which reversed a conclusion

Recorded because either would have produced a confident, wrong recommendation.

1. **Device time triple-counted.** `key_averages()` merges the CPU and CUDA aspects of a
   key into one entry labelled `DeviceType.CUDA`, so a `record_function` region arrives
   carrying the device time of every kernel launched inside it. Summing device time over
   it gave 18.05 ms/step against a real 3.75, i.e. a **GPU share of 175%** -- impossible,
   which is the only reason it got caught. Fixed by aggregating raw events and filtering
   on `is_user_annotation`; the totals now match the profiler's own "Self CUDA time
   total" exactly.
2. **`aten::copy_` charged to the integrator.** Bucketing host events by their own name
   put 18 of 20 ms/step at 57k into `integrator` and made the integrator look like the
   dominant cost -- which would have promoted Section 7's buffer-pool work on false
   evidence. Walking the parent chain showed 9 of those ms inside
   `[warpSPH] - computeForcing` and only ~0.05 ms/step actually inside `[Integration]`
   regions. A follow-on bug in the same fix charged a region's *own* self-time to its
   parent, hiding half the forcing cost (16.4 ms/step) in the generic `solver` bucket.
3. **Region attribution misread the sync census too** -- the same failure mode a third time,
   and worth stating as a general caution about this harness. Region-based attribution said
   15 readbacks/step were in "the Verlet validity check", which reads naturally as *the
   rebuild decision*; the decision is 2.9/step and the other 11.5 were
   `_minimum_image_delta` nested inside the same region. Acting on the region number would
   have meant restructuring the rebuild logic (hard, and it is the one genuinely
   irreducible sync) instead of deleting a per-axis `.item()` (eleven lines, 1.2x).
   **When a bucket is the target of actual work, re-derive it by counting in Python** --
   wrap `Tensor.item`/`__bool__` and record call sites -- rather than trusting the region
   totals, which attribute to the nearest enclosing region and cannot distinguish a
   region's own cost from its callees'.

## Recommendation

**"Nothing further needed before Steps I/J", with two high-value fixes already landed (the
forcing, and the core-side host stalls) and two of Step H's candidate follow-ons explicitly
ruled out.**

* **Host-stall removal: done, both repos.** 39.1 -> 8.6 readbacks/step, worth 1.27-1.40x of
  per-step wall clock across the two passes. What remains is load-bearing (the rebuild
  decision, the NaN check) or setup-only. This turned out to be the best value-per-line in
  the whole of Step H after the forcing fix -- eleven lines of core code alone beat every
  caching layer in Steps A-F -- and it is now finished rather than outstanding.

* **Steps I/J are not blocked, and dispatch is now the clear target below ~57k.** No
  bottleneck found needs an architectural change to the core before the interface moves.
  After the forcing and sync fixes, dispatch is a flat ~4.1 ms/step and **35-37% of host
  time -- the largest single bucket** -- so the interface work's own goals (retiring the 27
  hot-path dtype probes, the per-call `additionalArguments` re-analysis, somewhere to put
  `ExecutionMode`) sit directly on top of what now dominates. Note what that does *not*
  mean: the caching layers already took the marshalling out of dispatch, and the measured
  value of turning them off is ~0.6-1.2 ms/step, so the remaining ~4 ms is warp's launch
  machinery and the Python call chain, not argument conversion. Steps I/J should be scoped
  on their ergonomic and forward-mode merits, with any further dispatch win measured rather
  than assumed.
* **The forcing fix is done** (`d9ad712`). It was worth 1.76x at 57k and 2.81x at 155k in
  this harness and **2.6x on the real ~70k production run (42 -> 16 min)** -- an order of
  magnitude more than any remaining core-side dispatch work. It also *gains*
  differentiability through the forcing, which the scipy version could not offer.
* **Section 7's integrator buffer pool is NOT the bottleneck in 2D -- do not promote it on
  this evidence.** `[Integration]` regions are 6-10% of host time and 0.4-0.9 ms/step of
  GPU time, and the integrator's own `aten::copy_` traffic is ~0.05 ms/step. Section 7.2's
  "~3% drag" estimate holds up. **But note the scenario table:** under CRKSPH and in 3D the
  integrator is the *largest* bucket (43-45%), so Section 7 should be re-profiled against
  those before it is scoped, not dismissed outright. The 2D-only conclusion is what this
  audit supports.
* **Neighbour-list rebuild is not the bottleneck in 2D either, and the Verlet scaffolding
  is live, not dead.** 1.6-1.9 ms/step host, 10-15% of host time and *falling* as N grows;
  `[Verlet] Checking validity of prior neighborhood` runs ~3x per step and is doing its
  job -- answering Step H's open question about `adjacency_t.py`'s rebuild-threshold
  fields. In 3D it is 29% of host time, so the same caveat applies.
* **GPU kernel compute dominates above ~150k** (61-67% device-busy), so headroom from
  dispatch-level work shrinks past that scale, as expected. CUDA-graph capture (Section
  7.3) is the remaining dispatch-level lever; at ~350 launches and ~41 host stalls per step
  it looks worthwhile in principle. What blocks it is **not** the integrator -- see below.

### What actually blocks CUDA-graph capture

Section 7.3 says capture is gated on a "fixed step shape", and earlier drafts of this
report repeated that as "gated on the integrator". Measured at 19,044 particles over 20
steady-state steps, that is wrong on both counts:

| blocker | measured | in whose code |
|---|---|---|
| **Host readbacks (`.item()`)** -- illegal inside a captured region, since capture forbids synchronization | **49 per step** | solver modules, the Verlet check, the forcing |
| **`aten::nonzero`** -- output shape depends on data, so launch and allocation sizes are not known at capture time | **12 per step** | `computeForcing` (6), `[Integration] Finalize` (1), `computeDeltaShift` (1), unattributed (4) |
| Data-dependent neighbour rebuild branch (`shouldRebuild = bool(shouldRebuild_t.item())`, `radiusSearch/verlet/build.py`) | validity check runs 6x/step; **rebuild fired 0 times in 20 steps** | core |
| Integrator's out-of-place allocation churn | 3 fresh tensors per component per stage | integrator |

### The sync census, and the two core-side sites now fixed

Profiler region attribution turned out to be misleading here too: it put 15
readbacks/step in "the Verlet validity check", which read as *the rebuild decision*. Counting
in Python instead -- `scripts/count_host_syncs.py`, which wraps `Tensor.item` / `__bool__` /
`tolist` / `numpy` and records the calling site -- showed the decision is only 2.9/step and the
real cost was `_minimum_image_delta`, nested inside the same region, at 11.5/step. **Census
total: 39.1 host readbacks per step** at 19,044 particles (the earlier profiler figure of 49
double-counted `aten::item` against its inner `aten::_local_scalar_dense`).

```bash
python scripts/count_host_syncs.py --nx 128 --steps 5          # every site, per step
python scripts/count_host_syncs.py --filter warpSPHCore        # only this repo's
```

Two of them were core, and both are now fixed:

| per step | site | what it was | fix |
|---------:|------|-------------|-----|
| **11.5** | `radiusSearch/verlet/util.py` `_minimum_image_delta` | `if bool(periodicity[d].item())` once per axis per call, reading a **domain property fixed for the whole run** | branch-free: wrap every axis and select with `torch.where`, keeping `periodicity` on device |
| **2.0** | `renorm.py` low-neighbour fallback | `if torch.any(lowNbrMask):` guarding a masked assignment -- and a boolean-mask `index_put_` **is itself a synchronizing op**, so the guard paid a stall to *maybe* avoid a stall | `torch.where(mask, identity, C)`; no guard, no clone, no scatter |

Both verified bit-identical to what they replaced before being trusted: `_minimum_image_delta`
across dims 1-3, every periodicity pattern, and motion from 0.01x to 3x the box length (max
difference exactly 0.0, including the seam-crossing case); the renorm fallback on values *and*
gradients, with the masked path fully exercised. Pinned by
`tests/operations/test_no_host_sync.py`, which counts readbacks rather than timing and was
confirmed non-vacuous by restoring both old implementations (2 of 6 tests fail).

**Measured effect -- 39.1 -> 25.5 readbacks/step (-35%), and per-step wall clock:**

| particles | before | after | speedup |
|----------:|-------:|------:|--------:|
| 19,044    | 10.64 ms | 8.89 ms | **1.20x** |
| 57,121    | 11.07 ms | 9.78 ms | **1.13x** |
| 155,236   | 16.38 ms | 15.19 ms | **1.08x** |

(60 measured steps per point, medians, forcing on.) **Two changes touching eleven lines of core
code are worth more than every caching layer in Steps A-F combined** (1.19-1.75 ms/step against
~0.6-1.2 ms/step), which is worth sitting with: the win came from deleting stalls, not from
making anything faster. Core suite 112 passed, 2D+3D operation matrix 258 OK / 0 HIGH each, all
15 gradcheck scripts green, frontend suite 104 passed / 1 skipped / 0 failed.

### Frontend follow-up: the rest of the removable syncs are gone too

The frontend's straightforward, non-load-bearing readbacks were removed after this audit
(`3d4646a`..`c3238c3`, touching `math/__init__.py`, the three mDBC modules,
`shifting/delta.py`, `schemes/deltaSPH.py`, `schemes/dfsph.py`,
`systems/weaklyCompressible.py`, plus a shared `mdbc/_util.py` helper). Re-measured with the
same census and harness:

**39.1 -> 25.5 (core fixes) -> 8.6 readbacks per step: a 78% reduction.** Per-step wall clock
across the whole of Step H:

| particles | scipy era | + forcing fix | + core sync fixes | + frontend sync fixes | total |
|----------:|----------:|--------------:|------------------:|----------------------:|------:|
| 19,044    | 12.35 ms  | 10.75 ms      | 8.89 ms           | **8.25 ms**           | **1.50x** |
| 57,121    | 20.56 ms  | 11.67 ms      | 9.78 ms           | **8.35 ms**           | **2.46x** |
| 155,236   | 48.03 ms  | 17.11 ms      | 15.19 ms          | **13.49 ms**          | **3.56x** |

The two sync passes together are 1.27-1.40x; the frontend half alone is 1.08-1.17x (not
purely attributable -- two unrelated frontend commits landed in the same range). Frontend
suite green throughout: 104 passed, 1 skipped, 0 failed.

**Step time is now essentially flat from 2.4k to 57k particles** (7.91 -> 9.37 ms for 24x the
particles; 19k and 57k are within 2% of each other), so below ~57k the run is now almost
entirely fixed-overhead-bound. And **dispatch is now unambiguously the largest host bucket**
at 35-37% of host time, a flat ~4.1 ms/step, against a blocking cost that has fallen from
19-24% of host time to 11-15%:

| particles | step (ms) | GPU share | dispatch | neighbour | forcing | integrator | solver | blocking |
|----------:|----------:|----------:|---------:|----------:|--------:|-----------:|-------:|---------:|
| 2,401     | 7.91      | 45.5%     | **37.3%**| 11.1%     | 14.1%   | 8.3%       | 23.9%  | 10.8%    |
| 5,476     | 9.15      | 41.1%     | **35.9%**| 10.6%     | 14.7%   | 7.9%       | 24.6%  | 12.8%    |
| 19,044    | 9.19      | 42.4%     | **37.0%**| 11.0%     | 13.8%   | 8.2%       | 24.2%  | 10.8%    |
| 57,121    | 9.37      | 49.6%     | **34.9%**| 10.6%     | 15.3%   | 8.0%       | 24.3%  | 15.2%    |
| 155,236   | 14.18     | 72.5%     | 26.1%    | 8.3%      | 32.3%   | 6.1%       | 22.1%  | 36.0%    |

At 155k the run is now **72.5% device-busy** -- genuinely GPU-bound, where no host-side work
can help much. Below that, dispatch is the target, which is what Steps I/J address.

### Where the remaining readbacks are

8.6/step remain and are close to irreducible. Traced to the issuing op (19,044 particles):

| per step | site | repo | status |
|---------:|------|------|--------|
| 2.88 | `radiusSearch/verlet/build.py:74` `bool(shouldRebuild_t.item())` | core | **load-bearing** -- selects a Python code path (rebuild or reuse). Only removable by restructuring; at 2.9/step, leave it. |
| 1.00 | `runner/runner.py:306` NaN divergence check | frontend | load-bearing (stops a diverged run), but amortisable to every k steps if it ever matters |
| ~0.35 each | `compactHash/{grid,buildHashmap,search,indexing}.py` | core | only fire on an actual hashmap rebuild, so this is a fraction of a step, not a per-step cost |
| ~0.5 each | `sample/regular.py`, `regions/contour.py` | frontend | **setup only** -- these are one-time costs divided over the measured steps, not per-step work |

For historical reference, the pre-fix census that this list replaced:

| per step | issuing op | site | repo | why it syncs |
|---------:|------------|------|------|--------------|
| **15** | `.item()` | `[Verlet] Checking validity of prior neighborhood` | core | `shouldRebuild = bool(shouldRebuild_t.item())` -- one per check, and the check runs 15x/step across adjacency pairs and support schemes. Inherent to deciding *in Python* whether to rebuild. |
| **8** | `aten::is_nonzero` | `[warpSPH] - computeForcing` | frontend | a Python truth test on a tensor (`if t:`), which forces a host read |
| **6** | `aten::_index_put_impl_` | `[deltaSPH - 17] - enforce updates` | frontend | boolean-mask `index_put_`, which reads the mask count on the host |
| 2 | `aten::is_nonzero` | `[warpSPH] - Renorm - Covariance Postprocess` | **core** | `if torch.any(lowNbrMask):` in `renorm.py`'s low-neighbour-count fallback |
| 2 | `.item()` | `[warpSPH] - (shift) - computeDeltaShift` | frontend | |
| 1 each | `aten::is_nonzero` | mDBC density, boundary velocities | frontend | same `if t:` pattern |
| ~5 | mixed | unattributed / runner | frontend | the `t` readback, NaN check, diagnostics |

Two things follow. The `is_nonzero` group (~12/step) is the cheapest to remove -- an `if
tensor:` where a `.numel()` test or a Python flag would do is pure waste, and removing it
costs nothing in behaviour. The Verlet group (15/step) is the largest single source and is
*core* code; the check is one host read per call with no redundancy to squeeze, so
shortening it means either amortising the decision (check every k steps), or sharing one
decision across the adjacency pairs that currently each make their own, rather than
micro-optimising the check itself.

Three consequences:

1. **The step shape is already stable in steady state.** The Verlet rebuild -- the one
   genuinely data-dependent branch in the shape -- did not fire once in 20 measured steps;
   the validity check is doing its job and reusing the list. So "fixed step shape" is not
   what is missing. A capture scheme could capture the no-rebuild fast path and fall back
   to eager execution on the rare rebuild step.
2. **The blocker is the ~49 host readbacks and 12 data-dependent-shape ops, and they are
   almost all in frontend solver modules**, not in the integrator and not in this repo. Each
   would have to move on-device (a device-side flag rather than a Python `if`) or out of the
   captured region before any of the step is capturable.
3. **The integrator's allocation churn does not forbid capture -- it makes capture
   worthless without a buffer pool.** Repeated identical allocation sequences are fine
   (that is what a torch graph memory pool is for), but a captured graph bakes in device
   addresses, so the buffers it writes through must stay valid across replays. That is
   exactly what Section 7.3's `PooledState` provides. So the integrator work is a
   *prerequisite for capture to be worth anything*, not the thing standing in its way --
   the opposite of the dependency direction Section 7.3's wording implies.

## Notes

**Noise-resolution fix.** Arbitrary `--nx` works only because this step also fixed the
frontend's noise generator: `perlinNoise*D` requires the grid resolution to be an exact
multiple of every octave's lattice frequency (16 for the default 4 octaves at base
frequency 2), so `nx=39` crashed with a shape mismatch several frames deep and the sweep
was initially limited to multiples of 16. `warpSPH/math/noiseFunctions/generator.py` now
generates at `paddedNoiseResolution(n, ...)` and resamples down to `n` with a wrap-aware
separable linear resample (`resampleNoise`), leaving already-valid resolutions
bit-identical.

**Historical caveat, now resolved.** In the scipy era the `--hatches` arm with forcing on
reported the *hatched* run as faster at 57k and 155k (0.91x, 0.82x). That was noise, not a
result: the forcing's own cost varied 24-91 ms across consecutive steps, swamping a ~1 ms
effect. With the torch interpolator the forcing-on and forcing-off arms now agree to within
0.04x at every scale, so the comparison is stable either way.
