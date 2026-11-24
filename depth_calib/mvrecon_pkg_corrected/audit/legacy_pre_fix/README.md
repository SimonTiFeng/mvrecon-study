# Legacy v1 artefacts (invalid scores, kept for audit)

These are the exact files behind the first handoff. Their rotation/translation numbers are
NOT valid: the scorer compared the cumulative transform M to the incremental A, and the
COLMAP relative-pose formula was inverted. See `../COORDINATE_AUDIT.md`.

`task.json` and `gt_transforms.json` here are the v1 task (post-D-injection, pre-pose-fix).
`baseline_table.json`, `baseline_A.json`, `agent_runs.json`, `agent_scores.json`,
`run_c_electro.json` are the v1 result records.

The v1 tarball itself (`mvrecon_handoff.tar.gz`) is unchanged and is the same file that was
delivered earlier; it is not duplicated here to keep this package shippable.
