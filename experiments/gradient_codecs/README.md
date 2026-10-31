# Heavier operators on exact compressed neighbor lists

GPU primal comparison of scalar gradients, vector Jacobians, vector divergence,
and covariance matrices, using Wendland2, Gather support, Difference fields,
uniform support/mass/density, and smooth periodic fields. Every operator is measured with and without the CRK
kernel-gradient correction. Production dispatch and AD are unchanged.

The comparison includes original-order Warp, refined Morton-order Warp,
production native/Torch for uncorrected operators, and experimental native
CSR64, CSR32, byte-delta, two-bit-gap and unity/nibble consumers. Warp uses its
existing 32-lane launcher and CSR64 IDs/offsets. Matching native CSR64 controls
separate backend/scheduling effects from the additional saving of CSR32 IDs.
The production native wrapper rejects CRK corrections; corrected native curves
are explicitly the new experimental primal, never a fallback to Warp.

Each CRK consumer evaluates the same four terms as `crk/kernel.py`, retaining
the derivative-first `gradB[g,c]` convention. Plain operators use mass/density;
corrected operators use the apparent volumes used to construct their moments.
All source records are packed once per call and reused across incident edges.
The compressed readers are the exact formats from `csr_small_gaps`; the byte
reader extends the existing independent-chunk format to two adjacent codes per
lane. Encoders and interpolation prototypes remain shared source dependencies.

## CRK factor construction

Factors are actual apparent-volume factors computed from Warp volume/moment
kernels. The coefficient solve is performed in dimensionless coordinates, in
batches, to avoid the existing absolute `det(m2)<1e-14` threshold silently
limiting most large 3D fixtures to identity corrections merely because their
physical supports became small. Holding each query support fixed:

* Normalize moments as `(m0, m1/h, m2/h²)`.
* Normalize their coordinate derivatives as `(h*dm0, dm1, dm2/h)`.
* Call the unchanged CRK term solver.
* Transform the results back as `(A, B/h, gradA/h, gradB/h²)`.

Small fixtures check equivalence to the ordinary physical solve when its
limiter is inactive. Reports retain the physical limiter fraction, actual
identity-correction fraction, coefficient finiteness and normalized constant/
linear reproduction residuals. Factor construction is separate from operator
timing; its wall cost excludes the unrelated diagnostic-density pass of the
higher-level `computeCRKFactors` convenience function.

Exactly the same factors and volumes are permuted with the particles. They are
not recomputed after sorting, which isolates the change in traversal order.
The shared factor construction runs on the original-order graph; its reported
setup cost is not a measurement of a sorted CRK pipeline. Sorting membership is checked by full per-particle counts; permutation parity
is checked on a deterministic particle sample before timings. The standalone
numerical suite additionally uses independently constructed dense CSR graphs.

## Measurement boundaries

All curves use allocating calls, including field packing and Python/ctypes or
public-wrapper dispatch. Search, sorting, conversion, compression, factor
construction, qualification, and host readbacks occur outside those timings.
The screen uses resident buffers to select subgroup schedules at 262,144
particles; selections are frozen for the held-out 1M/4M fixtures in 1D–3D and
16M 3D fixtures. Both precisions and `n_h=4/2.57` are included.

Original and sorted states run in separate phases so two large CSR64 graphs do
not have to coexist. Comparisons among sorted backends are randomized paired
batches; original-versus-sorted timings are phase-separated. Every fixture has
a fresh process and activity snapshots; the runner refuses other GPU jobs.

At 16M particles with `n_h=4` in 3D, the CSR has about 4.3 billion
entries. Warp's array wrapper rejects a dimension above 2^31-1, even though
neighbor offsets are int64. Only these oversized graphs use an explicit Warp
query-batching adapter: four approximately 4M-row queries share full reference
arrays, each CSR view has rebased offsets, and concatenating outputs is included
in operator timing. Batch planning and its scalar sizing reads are outside it.
`verify_batching.py` establishes bit-exact factors and all eight operator results
against ordinary full-array calls in both precisions. Smaller graphs retain the
ordinary single-call Warp path; all native formats use one full CSR call.

The inherited FP32 periodic 1D Warp geometry computes `(dx+L/2)-L/2`, which can
lose a material fraction of a very small separation. Native minimum-image
geometry retains an already in-range delta. Disagreements in those fixtures
are recorded as unqualified comparisons rather than passing a relaxed global
tolerance. Independent dense native qualification still covers all dimensions,
precisions, mixed periodicity, variable/zero supports and nonsymmetric CRK data.

## Reproduce

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -lineinfo -Xcompiler=-fPIC -shared --threads 4 \
  -Isrc/warpSPHCore/native/csrc experiments/gradient_codecs/kernels.cu \
  -o /tmp/gradient-codecs.so
python experiments/gradient_codecs/verify.py \
  --library /tmp/gradient-codecs.so --output /tmp/gradient-verify.json
```

`run_study.py --archive <output> --baseline <previous-scale-archive>` expects
`artifacts/gradient-codecs-v1.so` in the output and the earlier immutable native
Torch/planner binaries in the baseline archive. `analyze.py <archive>` produces
zero-based throughput and relative-throughput plots plus JSON/CSV summaries.
Sanitizer runs can use `verify.py --quick --sanitizer`, with filters for
`gradient_codecs.*consume.*Li16ELi2ELi1E` and `gradient_codecs7prepare`, covering
every workload, correction mode and codec at a representative schedule.
