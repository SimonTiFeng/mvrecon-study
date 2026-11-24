# World-knowledge metric depth calibration (Codex-free execution)

Implements the protocol in `../Codex_Metric_Depth_Agent_Handoff.md` (v0.2) with a Claude vision
frontend instead of Codex, and runs it on NYUv2 + ETH3D with frozen DA3 Mono/Metric backends.

## Layout

```
configs/        pilot.yaml (frozen parameters), checkpoint_manifest.json
prompts/        claude_harness.txt / claude_direct.txt (frozen agent instructions)
schemas/        tools.json (frozen four-tool submission contract)
src/mcp/        session/store/handlers/server (server-side validation, budgets, atomic evidence)
src/frontends/  agent_client.py (vision agent's only interface to the contract)
src/geometry/   camera (incl. pycolmap fisheye), metric_scale, depth_affine,
                inverse_depth_affine, quality, pipeline
src/backends/   da3.py (vendored official code at pinned commit)
src/data/       NYUv2 / ETH3D ingest, manifest builder (scene-disjoint splits)
src/evaluation/ metrics, oracle diagnostics
scripts/        inference, evaluation, reports, HPC job scripts, measurement workflow
manifests/      pilot/dev/smoke JSONL + hashes (scene-disjoint, seed 42)
results/        runs, evaluations, REPORT.md, RUNBOOK_LEDGER.md, CONFIG_FREEZE.json
cache/          depth caches (hpc = formal, local = cross-platform check)
vendor/         Depth-Anything-3 @ 3d835ec1…
```

## Reproduce

```bash
# 1. environment (isolated venv reusing anaconda site-packages)
python -m venv --system-site-packages .venv
.venv/Scripts/pip install addict omegaconf evo pycolmap opencv-python-headless safetensors einops

# 2. geometry + protocol tests (no data needed)
.venv/Scripts/python -m pytest tests/ -q

# 3. data (one-time)
.venv/Scripts/python src/data/ingest_nyuv2.py --mat <nyu_depth_v2_labeled.mat> --out <processed/nyuv2>
.venv/Scripts/python src/data/ingest_eth3d.py --extracted <eth3d_extracted> --scenes ... --out <processed/eth3d>
.venv/Scripts/python scripts/finalize_eth_meta.py --inputs ... --out <processed/eth3d_final>
.venv/Scripts/python scripts/build_manifests.py --datasets nyuv2=... eth3d=...

# 4. frozen depth inference
.venv/Scripts/python scripts/infer_depths.py --manifest manifests/pilot.jsonl \
  manifests/dev.jsonl manifests/smoke.jsonl --backends A B --out cache/depths

# 5. measurement (vision agent; run per split/mode)
.venv/Scripts/python scripts/prepare_run.py --manifest manifests/pilot.jsonl \
  --run-id pilot_f1_001 --mode harness
#    then invoke scripts/measure_workflow.js with {run_dir, mode, samples:[{sample_id,w,h}]}

# 6. evaluation + report
.venv/Scripts/python scripts/run_eval.py --manifest manifests/pilot.jsonl \
  --run results/pilot_f1_001 --depth-cache cache/hpc/depths --out results/pilot_f1_001/eval
.venv/Scripts/python scripts/make_report.py --eval-dir results/pilot_f1_001/eval
```

## Frozen artifacts

- Config hash recorded in `results/CONFIG_FREEZE.json` (configs, schemas, prompts, protocol docs).
- Backend provenance: `configs/checkpoint_manifest.json` (repo revisions + weight SHA-256).
- HPC jobs and costs: `results/RUNBOOK_LEDGER.md`.
- Deviations, limitations and results: `results/REPORT.md`.
