# Repository survey and cleanup plan

Surveyed 23 October 2026 at `8f5dd44` on `main`, version 0.6.2, including
the existing working tree. The core capability pass is substantially complete.
The immediate priorities are a newly reproduced support-conversion defect,
coherent delivery of the existing changes, and reconciliation of the planning
documents. Most historical implementation plans can then leave the checkout.

This is an execution proposal. The survey adds this file; retirement, numerical
fixes, commits, publication and changes to sibling repositories are separate work.

## Repository baseline

The starting tree contains 570 tracked files, including 181 Python files under
`src/`, 83 under `tests/`, and 81 Markdown files. Twenty planning documents
contain 10,940 lines, approximately 912 KB. These counts exclude untracked files.
The working tree starts with nine modified tracked files and sixteen untracked
files. `main` and the locally recorded `origin/main` point to the same commit;
this survey does not establish remote freshness or package publication status.

| Area | Current capability and evidence | Remaining boundary |
| --- | --- | --- |
| Neighbor search | Compact hash and CSR traversal, independent particle/offset widths, padded capacity, in-place reuse, fixed hash slots, Verlet validity and empty populations; `tests/radiusSearch/` | Graph lifetime and invalidation must be qualified in each consumer. Flat-array backend limits still apply. |
| SPH operators | Seven public operations, CSR/grid paths, kinds/directions, corrections, reverse AD and scoped value/geometry JVPs; `tests/operations/` | Grad-h geometry tangents and generic higher derivatives are not established. Density has an explicit position HVP. |
| CRK and renormalization | Correction construction, selected-rank behavior and derivative contracts; public CRK/renorm/autograd documentation | Branch changes and degenerate stencils retain explicit limitations. |
| Polynomial reconstruction | Production `PolyFit` and `Reconstructor`, MLS/LABFM, TENO/WENO, field and scoped geometry derivatives, fallback contracts | Nonlinear sector reconstruction is 1D/2D. 3D sectors and derivatives at stencil/rank transitions remain separate work. |
| Linear algebra | Matrix-free Krylov/stationary solvers, constraints, nullspaces, preconditioners and optional native workspaces | Actual pressure and implicit/vector applications need their own residual, boundary and adjoint qualification. |
| MFM and MFV | Torch reference, forward-only Warp backend, guarded geometry, projected/unprojected faces, closure solvers and rate/update examples | First-moment faces and complete Warp flux/closure AD remain open. Full moving schemes, boundaries and timestep policy belong to consumers. |
| Delivery | CPU CI, operation matrices, isolated wheel/examples and Docusaurus build are configured | No dedicated hosted GPU job; metadata ranges do not certify every dependency combination. No lint configuration or lint job. |
| Research | Reusable static/frozen/PDE harness and D&A replication scripts | Unfinished scientific comparisons must remain visible after the old plans are removed. |

The [recorded October 16 qualification](docs/development-artifacts.md) reports
2,568 CPU passes and 2,429 skips, plus full CUDA gates in both precisions.
Those numbers describe source commit `2034a4e`, not the later working tree.
The [public examples](docs/examples.md) and public capability pages are the
maintained contracts.

### Existing changes awaiting coherent delivery

The current edits form two related but separable groups:

- MFM architecture and correctness: move `PreparedLauncher` into
  `autograd.prepared`, retain the `mfm.fastLaunch` compatibility import, expose
  traversal metadata independently of AD eligibility, suppress unused backward
  generation for forward-only entry kernels, and carry momentum between grid
  cells in `mfmFlux_Func_i`.
- Shared operator evaluation: ten new benchmark/profiling tools, three reports,
  benchmark contracts, a raw CRK-moment derivative canary, and static tensor
  extents in outer products, divergence and CRK moments.

The new `autograd.prepared` module is untracked but imported by modified tracked
code. A commit containing only tracked modifications would be incomplete.
The [register report](regression/core_register_pass_2026_10_21.md) records focused
qualification and measured improvements; it does not claim a full suite rerun.
The sibling MFM performance report records completed staged qualification and
explicitly defers further focused tuning.

## Fresh survey checks

The usable environment is `/home/lu26029/miniconda3/envs/warp/bin/python`, with
Torch 2.13.0+cu130, Warp 1.17.0 and NumPy 2.5.2. The default shell Python lacks
Torch. CUDA is unavailable to this session.

| Check | Result |
| --- | --- |
| Full current-tree CPU pytest suite | 2,584 passed, 2,429 skipped, 327 warnings in 559.01 seconds; includes isolated float64 derivative scripts. |
| Public documentation production build | Passed with `npm run build` in `docs/`. |
| Existing diff whitespace check | `git diff --check` passed. |
| October 16 archive integrity | All 544 manifest entries exist and match their SHA-256 values. |
| Public support-conversion helper | Fresh float32 and float64 CPU probes reproduce the 3D mismatch below. |

Survey logs are `/tmp/warpsphcore-survey-pytest-2026-10-23.log` and
`/tmp/warpsphcore-survey-docs-2026-10-23.log`. Temporary logs are not durable
qualification records; retain required evidence with provenance during delivery.

The passing suite does not catch the support-conversion defect below. GPU
qualification was not rerun. The test invocation used
`python -m pytest tests/ -q` in the environment above, with BLAS/OpenMP thread counts set to two and
compilation caches directed to `/tmp/warpsphcore-survey-warp-cache` and
`/tmp/warpsphcore-survey-cache`.

### Correctness issue to address first

`src/warpSPHCore/util/support.py` defines `_VOL_3D_FACTOR = 3*pi/4` and divides
by it in `volumeToSupport_warp`. The sphere relation requires division by
`4*pi/3`. The Python and tensor helpers already use the latter relation.

For volume `0.125` and target neighbor count `55`, a fresh float64 CPU kernel
returns `1.42896148102002` in 3D; the Python helper and independent sphere
formula give `1.17958171295021`. The relative error is `0.211413729`, or
21.14%. The same probe matches in 1D and 2D. The helper is publicly exported;
the source search finds no in-repository callers beyond its export.

Correct the factor and comment, and add a numerical regression against the
independent dimensional volume formula for 1D/2D/3D in isolated float32 and
float64 processes. Exercise CPU and available CUDA. This issue is distinct
from the traced float32-constant precision problem fixed in September.
The original constant-arithmetic note still says the support precision fix
is pending, so it needs reconciliation as well.

## Planning discrepancies to resolve

- `REPOSITORY_PLAN.md` says performance is deferred, then reopens a register
  pass in its October 21 additions. It also mixes current priorities with
  completion narrative. Fold the later decisions into one dated current state.
- `higher_order.md` opens with “Phases 4–7 have not started”, despite later
  sections recording production MLS/LABFM and TENO/WENO implementations.
  Its Phase 4 summary says forward mode is open while its core-port section
  describes the implemented geometry/field JVPs.
- The same roadmap says MFM frontend registration is outstanding. The sibling
  `warpSPH/src/warpSPH/schemes/__init__.py` exports `meshlessStep`, and the
  October 21 performance report records MFM/MFV consumer runs. Registration
  is distinct from completing the original matched scientific comparison.
- The October 16 delivery records also qualify 24 production frozen-advection
  cases for cubic MLS/LABFM. Separate that completed evidence from any still
  missing frozen-diffusion or matched-scheme study before carrying forward the
  older roadmap's generic frozen-leg task.
- `ADJACENCY_CAPACITY_PLAN.md` has an updated implemented-contract banner but
  retains an older numbered design with different defaults and speculative
  rebuild behavior. Its remaining frontend tasks must not become core tasks
  merely because the document lives here.
- `warpier_residual_open_problems_plan.md` mixes closed August investigations,
  downstream work, optional capabilities and CI hygiene. It still describes
  a six-operator JVP scope, although public dispatch now includes covariance.
  It calls JFNK unstarted, while the sibling integrator source and current
  roadmap establish that it exists.
- The D&A `PLAN.md` retains an open cubic long-wave discrepancy that its
  completed `closeout_plan.md` resolves as a phase-reference bug. Closeout
  items 1–7 are complete; Phase 7 dynamic paper tests remain unfinished.
- The D&A README still says no scripts exist. The harness plan still speaks
  about components to build although production code and tests exist.
- The register report ends with MFM architecture/qualification “active”,
  whereas the later root-plan addition and sibling report record completion.
  Keep the chronology in the archive and make the maintained status consistent.

The existing external archive is intact, but none of the twelve historical
plans has a same-path copy there. Do not infer their preservation from the
October 16 migration. Historical-plan names occur in 48 source files, 20 test
files, 61 script files, public docs and a CI comment outside those plans.
Many are bare filenames, so a successful Docusaurus build cannot catch them.

## File dispositions

“Remove” means remove from the current checkout after extracting useful
contracts, reconciling outstanding items and preserving a recoverable snapshot.
It does not mean rewrite Git history. This table covers all twenty original
planning documents.

| Existing file | Disposition and destination |
| --- | --- |
| `REPOSITORY_PLAN.md` | Keep as the sole core execution queue. Rewrite its status, current priority, bounded open capabilities and downstream dependencies; link evidence instead of appending session logs. |
| `ADJACENCY_CAPACITY_PLAN.md` | Retire and remove. Put live capacity, aliasing, eager rebuild and recapture contracts in `docs/docs/neighbor-search.md`; carry only verified remaining core requirements into the queue. |
| `warpier_residual_open_problems_plan.md` | Reconcile each item into the queue, downstream pointers or explicit deferrals, then remove. Preserve issue drafts with their repros or archived records. |
| `higher_order.md` | Extract a concise completed-capability map and unresolved scientific questions into `higherOrderSPH/README.md`, then remove the root narrative. |
| `higherOrderSPH/harness/PLAN.md` | Retire and remove. Keep usage, frozen-baseline/versioning rules and test commands in the existing harness README. |
| `higherOrderSPH/dehnen2012_convergence-without-pairing-instability/PLAN.md` | Replace with a compact Phase 7 research backlog in that folder's README, then remove. Keep paper notation, findings and reproduction instructions as reference material. |
| `higherOrderSPH/dehnen2012_convergence-without-pairing-instability/closeout_plan.md` | All seven closeout items are complete. Preserve findings and the final closeout verdict, then remove. |
| `higherOrderSPH/dehnen2012_convergence-without-pairing-instability/phase6_shock_capturing_prompt.md` | Retire and remove the completed frontend handoff. Keep an owning-repository evidence pointer. |
| `docs/historic_plans/warpier_core.md` | Remove after extracting any still-needed ABI/call-graph explanation into maintained architecture docs. |
| `docs/historic_plans/warpier_fields.md` | Remove after preserving current state/cache/ownership contracts in maintained architecture docs. |
| `docs/historic_plans/warpier_adjoint.md` | Remove the plan after preserving the useful JVP derivations as a clearly scoped technical reference. |
| `docs/historic_plans/warpier_forward_mode_plan.md` | Closed implementation/comparison narrative; remove and retain downstream result pointers. |
| `docs/historic_plans/warpier_tier2_operators_plan.md` | Closed operator rollout; remove. Current operator documentation and tests own support claims. |
| `docs/historic_plans/warpier_tier2_combined_jvp_plan.md` | Closed combined-tangent rollout; remove after checking current combined-tangent documentation. |
| `docs/historic_plans/warpier_tier2_jvp_csr_backend_plan.md` | Remove. The old unstarted benchmark was subsequently waived; record that disposition rather than reviving it. |
| `docs/historic_plans/warpier_tier2_jvp_reverse_mode_plan.md` | Remove. Preserve the reverse-over-JVP contract and reconcile its historical bug against later correction work. |
| `docs/historic_plans/warpier_tier2_jvp_remaining_work_plan.md` | Superseded consolidation; remove. |
| `docs/historic_plans/warpier_tier2_correction_jvp_plan.md` | Closed correction rollout and follow-ups; remove after preserving relevant derivation and branch limitations. |
| `docs/historic_plans/warpier_unified_operator_wrapper_plan.md` | Remove. Preserve current dual dispatch and the demonstrated higher-derivative composition limits. |
| `docs/historic_plans/warpier_jvp_dual_argument_pruning_plan.md` | Remove. Preserve the corrected benchmark conclusion and inert-argument ownership rationale; do not revive abandoned phases. |

Technical explanations should live in the existing public autograd/operator
pages where concise. Longer enduring derivations can use `docs/design/` with
explicit links from those pages. Do not copy entire execution journals into
a newly named plan.

## Remaining work and ownership

| Work | Treatment after reconciliation | Completion or reopening condition |
| --- | --- | --- |
| 3D Warp support conversion | Immediate core correctness task | Independent volume-formula regression passes in both precisions; corrected docs agree. |
| Existing MFM/register/evaluation changes | Immediate delivery task | All new imports/scripts/tests are included coherently; numerical changes have focused regression coverage and provenance. |
| Capacity and fixed-grid graph integration | Core contracts plus owning frontend work | Verify current consumer state before carrying forward old tasks. Require a concrete failing/missing core contract for a new core implementation task. |
| Pressure, compressible boundary/support updates, implicit/vector composition | Downstream consumer qualification with core support as needed | Record exact operator, boundary, topology lifetime, precision/device and residual/adjoint acceptance. Existing kernel tests alone do not close a consumer. |
| First-moment MFM faces and Warp flux/closure AD | Deferred core capability boundaries | A named consumer and independent consistency or derivative acceptance test. Preserve current guards. |
| Grad-h tangents, non-density HVPs and AD outside current operators | Deferred capabilities | A concrete consumer and explicit mathematical scope. Existing negative composition results stay documented. |
| 3D nonlinear reconstruction and transition derivatives | Deferred reconstruction capability | Defined sectors/coverage, rank/fallback behavior and independent numerical/AD evidence. |
| Higher-order scientific comparisons | Research backlog in `higherOrderSPH/README.md` | Matched schemes/configurations and frozen baselines; retain genuine LABFM boundary/hyperviscosity/disorder, reconstruction coupling/comparison and MFM face/disorder questions. |
| D&A dynamic figures 7–13 | Deferred paper-replication backlog in its README | Individually specified pairing/Gresho/Sod/cost reproductions. Do not mark the whole paper complete because static closeout is done. |
| Upstream Warp issue drafts | Revalidate, then decide disposition | Reproduce on the supported runtime and check the upstream tracker before any submission. Retain repros and workarounds regardless of filing status. |
| CI and static hygiene | Small separate maintenance queue | Keep existing CPU/wheel/matrix gates; add static checks incrementally, and scope GPU coverage to available infrastructure. |
| Further layout/register/traversal tuning | Deferred measurement-driven work | A consumer-relevant bottleneck, preserved periodic/stencil semantics, numerical parity and repeated timing evidence. No blanket register caps or changed defaults from the old experiments. |

“Deferred” is a valid disposition. Cleanup does not require implementing every
idea ever mentioned. Treat cross-repository status as a dated dependency;
the current integrator roadmap has already retired much of its own old plan
structure. Audit sibling sources read-only and update core pointers without
editing sibling queues as a side effect.

## Execution sequence

### Batch 1 Establish the numerical and source baseline

1. Record HEAD, initial status, source hashes, environment and fresh test results.
2. Correct the 3D support factor with the independent regression described above.
3. Review the existing MFM and register work in coherent groups. Include the
   untracked prepared module, benchmark imports, tests and reports together with
   their callers. Preserve the compatibility import.
4. Check or add focused coverage for the multi-cell MFM momentum accumulation
   fix and prepared traversal metadata/fallback behavior. Existing MFM tests
   largely exercise CSR; the sibling qualification does not replace a durable
   regression for a public core path.

Exit: a reviewable source baseline, distinct recorded and fresh qualifications,
and no falsely closed numerical defect. A documentation cleanup can proceed
independently of unavailable GPU timing; release qualification must state it.

### Batch 2 Reconcile the current queue and research backlogs

1. Rewrite `REPOSITORY_PLAN.md` around the final baseline, preserving the existing
   October 21 work and reconciling its performance priority statements.
2. Resolve each old residual item as complete, superseded, deferred, downstream
   or actionable core work, with an evidence path and next acceptance condition.
3. Update the higher-order and D&A READMEs from the later implementation and
   qualification records. Keep missing research comparisons explicit.
4. Replace stale “not started”, “open” and “pending” summaries; distinguish
   registration, implementation, qualification and paper-comparison completion.

Exit: every old unchecked task has a disposition; one core queue owns executable
work. No task depends only on an old narrative to explain its current status.

### Batch 3 Preserve reference material and remove retired plans

1. Snapshot every retiring document, including relevant uncommitted reports,
   with original path, source revision/dirty state, SHA-256 and destination.
   Use a new dated retirement manifest in the development archive; leave the
   verified October 16 manifest and its files unchanged.
2. Extract the enduring ABI, cache/aliasing, differentiation and numerical
   explanations. Preserve the useful equations and negative findings.
3. Update references in public docs, source/test/script docstrings, CI comments,
   READMEs and the lessons file in the same batch as deletion. Replace an old
   plan citation with its durable contract or derivation, not a missing basename.
4. Remove the nineteen superseded plan files identified above. Remove
   `docs/historic_plans/` once its surviving reference material has a destination.
   Record the exact removal commit and path mapping for historical retrieval.

Exit: no live local/GitHub-main link or unexplained bare citation points at a
deleted plan; all needed historical bytes have a verified recovery location.

### Batch 4 Align reports and reproduction tools

1. Reconcile the three October 21 reports into a concise checked-in results/index
   summary plus detailed external evidence. Preserve preparation versus fresh
   scaling versus retained optimization chronology, source hashes, excluded
   cases and parser corrections.
2. Replace workstation-specific absolute artifact links with portable archive
   locations/regeneration instructions. Preserve original historical commands
   in immutable archived copies.
3. Update `warp_float32_binop_constants.md`: remove the stale pending-support-fix
   statement, preserve the traced-constant lesson and independently track the
   newly found volume-factor defect. Relocate it beside the reproduction tools
   or maintained authoring notes if that avoids another root document.
4. Inventory script entry points before pruning. Six `scripts/spikes/` files
   are standing subprocess gates in `test_gradcheck_scripts.py`; retain them or
   move them with their caller. Keep the legacy Krylov test oracle, small
   distribution fixtures, kernel specs and source-hash-guarded experiments.
5. Keep generated runs external under the existing artifact policy. Review
   ignored build/cache/scratch files separately; do not treat local data as
   disposable merely because Git ignores it.

Exit: a new clone can read supported contracts and run normal tests without the
archive, while optional historical reproduction has clear prerequisites.

### Batch 5 Validate the cleanup and retire this temporary plan

1. Run a reference/path audit across tracked and newly added files, including
   bare plan names and GitHub `blob/main` links. Check retained derivation anchors.
2. Run the documentation production build, diff checks and relevant package,
   benchmark, search, operator and derivative gates. For source changes, retain
   the full CPU result and focused float64 checks; use available real CUDA for
   changed kernels and record any missing qualification explicitly.
3. Run isolated wheel/examples checks before delivery of the combined source
   changes. Confirm that untracked imports and auxiliary scripts are packaged
   or intentionally repository-only.
4. Add a small static CI gate as separate maintenance work after triaging its
   findings. Do not mix a repository-wide formatting rewrite or new speculative
   GPU/nightly infrastructure into document retirement.
5. Leave a short completion/evidence pointer in `docs/development/README.md` and
   the current queue. Archive and remove this temporary cleanup plan once its
   batches are complete, so it does not become the next stale plan.

Completion means one current core roadmap, short folder-owned research backlogs,
nineteen original plans retired, preserved scientific/architecture explanations,
working references and reproducible validation. Performance experiments and
deferred capabilities remain explicitly scoped rather than silently closed.
