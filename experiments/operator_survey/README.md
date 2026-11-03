# Warp operator source survey

Static inventory and manual accumulator assessments for the native traversal design. No kernels are launched and neither package is imported.

The snapshot covers all `wp_*.py` files under the core and frontend source packages, plus all other source files containing `wp.func`, `wp.kernel`, `wp.func_grad` or `wp.func_replay` definitions. The current core checkout is compared separately with the native worktree, accounting for moved primal implementations.

The report is in `docs/regression/native_operator_accumulators_2026_10_31.md`; the adjacent catalog includes every operation record, file assessment and decorated definition.

Run from the native worktree:

```bash
python experiments/operator_survey/survey_wp.py \
    --core "$PWD" \
    --frontend /home/lu26029/dev/warpSPH \
    --core-reference /home/lu26029/dev/warpSPHCore \
    --output /tmp/native-operator-inventory.json

python experiments/operator_survey/render_report.py \
    --inventory /tmp/native-operator-inventory.json \
    --report-directory /tmp/native-operator-report
```

`assessments.json` contains the reviewed semantics. `inventory.json` contains the source snapshot, definition-level AST evidence and the read-only current-core comparison. Source hashes in the assessments prevent silently applying an old semantic assessment to changed code. Update assessments after reviewing changed source; adding files or renaming operation anchors also requires updating coverage.

The renderer targets this reviewed snapshot. New semantic records or source changes can require updating its narrative and aggregate tables. The ratings describe source-level integration complexity, not measured development time. Inventory assertions check coverage and anchors; they are not numerical correctness tests.

Indexed writes include local by-value vector/matrix assignments and local adjoints. Inspect the operation store contract before treating them as global output arrays. Automatically linked passes follow calls within a file; helper functions are described by their file assessment. Runtime overload/factory instantiations and generated Warp backward kernels are not counted as separate source definitions.
