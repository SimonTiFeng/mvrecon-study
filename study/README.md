# Interactive registration study — reproducibility

## 1. Build the agent sandbox

```powershell
cd C:\Users\Simon\Downloads\Git\research\music\depth_calib\mvrecon
python study/prepare_study.py
```

This creates `%TEMP%\reg_wb_study_20261123` with `workbench_cli.py`, hard-linked
`data/<scene>/`, and one `sessions/<case>/` per target containing `TASK.md`, a pristine
`session.json` (initial D) and an empty trace. The hash of the frozen CLI is recorded in
`study/sandbox_manifest.json`.

## 2. Baselines (all 40 instances)

```powershell
python study/baselines_full.py
```

Writes `out/agent_study_20261123/eval/baselines.json` (9 methods × 40 instances, same D,
same RGB/depth/K, same scorer).

## 3. Agent runs

Each case is a separate Claude Code subagent (`Agent` tool, `subagent_type=Explore`,
model Claude Sonnet 5) whose prompt is only:

> 读取并严格执行 `<sandbox>/sessions/<case>/TASK.md` 中的任务说明……

The case → agent-id mapping and launch batches are in
`out/agent_study_20261123/raw/launches/batch*.json`. The subagent's JSONL records live at
`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`; the Agent tool's
`.output` path is a zero-byte placeholder.

## 4. Collect and score

```powershell
python study/collect_agents.py
python study/audit_leakage.py out/agent_study_20261123/raw/transcripts out/agent_study_20261123/eval/leakage_audit.json
python study/score_study.py
python study/stats_tests.py
python study/attribution.py
python study/make_report.py
```

- `score_study.py` replays every trace (initial D + logged increments/undos) and refuses to
  call a case valid unless the replay equals the saved final matrix to 1e-8.
- `audit_leakage.py` checks every tool input, Read path and Bash command for forbidden
  tokens (`eval_data`, `T_true`, `task.json`, memory index, parent prompt) and out-of-sandbox
  access.
- `make_report.py` writes `out/agent_study_20261123/report/SUMMARY.md`.

## 5. Raw evidence locations

- Workbench traces/state/renders: `%TEMP%\reg_wb_study_20261123\sessions\<case>\`
- Agent transcripts: `out/agent_study_20261123/raw/transcripts/<case>.jsonl`
- Agent final messages: `out/agent_study_20261123/raw/final_reports/<case>.txt`
- Frozen workbench: `out/agent_study_20261123/raw/workbench_cli_frozen.py`
- Protocol and exclusions: `out/agent_study_20261123/raw/RUN_MANIFEST.md`
