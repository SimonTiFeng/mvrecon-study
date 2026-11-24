# MV-Recon interactive registration study — run manifest

## Protocol

- **Question.** Given two independently mono-depth-predicted views placed at a deliberate
  injected displacement D, can a language model operating an interactive workbench choose
  observations and registration actions that beat strong traditional pipelines?
- **Task set.** 40 target instances over 7 ETH3D scenes, nested subsets v2/v3/v4.
  These instances repeat **19 unique scene/reference/target view pairs**; D differs between
  nested instances. Both counts are reported.
- **Initial state.** The target cloud's cumulative transform is D (25–66° rotation,
  metre-scale translation). The reference cloud is fixed at identity. D comes from
  `eval_data/task.json`; it is agent-visible as the current state, and contains no
  information about T_true beyond the task construction.
- **Scoring.** After the agent finishes, `eval/protocol.py` compares the final cumulative
  matrix to T_true (camera-target → camera-reference), stripping uniform scale.
  Metrics: rotation error (deg), lateral translation error (m, x-y of the reference frame),
  full translation error (m). **Success** = rotation < 2° and lateral < 0.5 m.
- **Full set, no GT filtering.** All methods run on all 40 instances. The historical
  “coarse rotation < 20°” view is produced only as a *reporting* subset, with its rule
  stated.
- **Baselines.** identity; FPFH+RANSAC+ICP; SIFT rigid RANSAC (with/without sky mask);
  FPFH rotation + SIFT translation; + iterated small-rotation correction; + reprojection
  refinement; SIFT rigid + reprojection refinement. All start from the same D, see the same
  RGB/depth/K, and are scored by the same code. Baseline code: `study/baselines_full.py`.
- **Agent.** Claude Sonnet 5, one fresh Claude Code subagent per case
  (`Agent` tool, `subagent_type=Explore`). The Explore type was chosen because it omits
  CLAUDE.md/memory injection; the plain Agent tool does not relay the parent user prompt.
  Each subagent receives only its own `sessions/<case>/TASK.md` (task + workbench manual).
  Budget: 30 charged operations, with per-tool caps; observations (stats/state), undo and
  finish are free.
- **Workbench.** `raw/workbench_cli_frozen.py`, SHA-256
  `2c67867a32e0de73b16813053d3b06cac91e371bf4d3e56c129a18b6718ccd1d`.
  Tools: photos/photo/render/closeup; SIFT/ORB/AKAZE matching with optional reference and
  moving pixel boxes; translation/rigid-3D/small-rotation/reprojection solvers; point-to-plane
  ICP; FPFH+RANSAC+ICP; manual rotate/translate; point masks; undo; finish/abstain.
  Every state change is logged with its exact increment matrix. Parallel CLI calls are
  serialised by a per-session lock.

## Evidence layout

- `raw/transcripts/<case>.jsonl` — full subagent JSONL (system prompt, model output, tool calls).
- `raw/final_reports/<case>.txt` — the agent's final message.
- `raw/launches/batch*.json` — case → subagent-id mapping and launch metadata.
- `raw/workbench_cli_frozen.py`, `raw/sandbox_manifest.json` — frozen tool provenance.
- `eval/baselines.json` — 360 baseline rows (9 methods × 40 instances).
- `eval/agent_results.json` — 40 agent rows with final matrices and metrics.
- `eval/reconstruction_check.json` — replay of each trace against the final state.
- `eval/leakage_audit.json` — transcript audit (memory/parent-prompt/eval_data/paths/tools).
- `eval/per_sample.csv`, `eval/summary.json`, `eval/stats_tests.json`,
  `eval/attribution*.json` — analysis.
- `report/SUMMARY.md` — the report.
- Sandbox sessions (trace + state + rendered images): `%TEMP%\reg_wb_study_20261123\sessions`.

## Excised pilots (not in the reported results)

1. `study/pilot_contaminated/` — first workflow pilot. The workflow harness relayed the
   parent user prompt into the subagent and injected the project MEMORY.md index, which names
   the rotation finding and SIFT. Excluded as contaminated.
2. `study/pilot_explore_race/` — Explore pilot that exposed a lost-update race between
   parallel CLI processes. Excluded; the lock was added and the sandbox reset.
3. `study/pilot_batch1_oldcli/` — a completed case from the batch started before the
   `closeup` serialisation fix. Excluded so every reported session uses one frozen CLI.

## Interruptions

- The upstream gateway returned HTTP 429 at two points during the run. Affected agents were
  resumed from their existing workbench state (same subagent session) or, if no state
  existed, restarted; every case reports its final status and number of attempts.
- No agent received T_true, T_error, oracle output, or a GT-derived recommendation.
  The audit checks every tool input, every Read path, every Bash command and every assistant
  text block for forbidden tokens and out-of-sandbox access.
