# Repository working rules

## Keep research and run artifacts outside this checkout

- Start disposable probes and one-off troubleshooting scripts under `/tmp`.
  Do not create them in this checkout and plan to clean them up later.
- Keep exploratory kernels, parameter sweeps, research notebooks and temporary
  analysis tools in a separate research Git repository. The local native study
  repository is `$WARPSPH_DEVELOPMENT_ROOT/core/research/native-backend`.
- Save retained simulation states, profiler captures, raw timings, logs and
  generated research plots under
  `$WARPSPH_DEVELOPMENT_ROOT/core/experiments/<study>/<run>/`. The shared root
  defaults to the sibling `warpSPHDevelopment` workspace; set it explicitly
  for worktrees. That evidence workspace is not itself a Git repository.
- Record the core/research commits, arguments, runtime and relevant source
  hashes with retained evidence. Check copies before removing original files.
  Never overwrite an existing run to replace a failed or superseded result.
- Promote useful work into core as maintained code, a minimal regression test,
  a reusable documented benchmark/diagnostic, or concise public documentation.
  Do not promote the surrounding campaign scripts and output collections.
- Keep ordinary tests, builds, imports and documentation independent of the
  external research/evidence workspace. Test a focused owner/device selection;
  avoid multiplying equivalent cases across suites.
- Preserve curated examples, documentation figures and small regression fixtures
  that have a maintained consumer. They are distinct from raw experiment output.

Read `docs/docs/development-artifacts.md` for locations and provenance. Run
`python scripts/check_repository_layout.py` before committing. CI checks tracked
paths; `.gitignore` alone cannot prevent accidentally force-added artifacts.
