# `out/` — analysis arms outside the 40-instance study

`agent_study_20261123/` holds the published interactive-registration study (see the root
README). Everything else here is the research log that led to it: per-arm result JSONs, logs
and small session archives, all from 2026-11-22/23. They are diagnostics; the producer of
each is listed below. `FINDING_*.md` in the repo root interprets them.

| group | files | produced by | finding it feeds |
|---|---|---|---|
| rotation is the bottleneck | `rotation_basin.json`, `rotation_profile.json`, `rotation_leak.json`, `masked_basin.json`, `masked_rotation.json`, `plane_rotation.json`, `plane_match_rotation.json`, `plane_refine.json`, `multistart_rotation.json` (+ logs) | `eval/rotation_basin.py`, `eval/rotation_profile.py`, `eval/rotation_leak.py`, `eval/masked_basin.py`, `eval/masked_rotation.py`, `eval/plane_rotation.py`, `eval/plane_match_rotation.py`, `eval/plane_refine.py`, `eval/multistart_rotation.py` | `FINDING_rotation.md` — residual error is rotation |
| translation given rotation | `lateral_search.json`, `coverage_refine.json`, `scale_coupling.json`, `anchor_*.json`, `selfconsistency_width.json`, `degeneracy.json` | `eval/lateral_search.py`, `eval/coverage_refine.py`, `eval/scale_coupling.py`, `eval/anchor_*.py`, `eval/selfconsistency_width.py`, `eval/degeneracy.py` | `FINDING_translation.md` |
| SIFT as the identity source | `sift_identity.json`, `sift_log.txt`, `identity_corrected.json`, `baseline_corrected.json`, `baseline_rescored.json`, `baseline_table.json`, `baseline_A.json`, `agent_*` (pre-study agent arms) | `eval/sift_identity.py`, `eval/run_baseline_corrected.py`, `eval/run_identity_corrected.py`, `eval/rescore_saved.py`, `out/make_table.py` | `FINDING_identity.md` |
| correspondence sweep / HPC job 14069627 | `results_14069627.jsonl` (raw), `corr_sweep.md`, `reproj_refine*.json`, `reproj_orb.log` | `hpc/exp_corr.py`, `hpc/analyse_corr.py` on the CPU partition; `eval/reproj_refine.py` | `FINDING_sweep_hpc2.md` — ORB matches SIFT at 1/5 cost; reprojection beats 3D-3D |
| candidate selection | `select_by_reproj.json`, `select_log.txt`, `stage_chain.json`, `chain_log.txt`, `frame_candidates.json`, `frame_plus_identity.json`, `replay_recipe.json` (+ logs) | `eval/select_by_reproj.py`, `eval/stage_chain.py`, `eval/frame_candidates.py`, `eval/frame_plus_identity.py`, `eval/replay_recipe.py` | `FINDING_sweep_hpc2.md` — GT-free selection still unsolved |
| multi-view fusion (draft) | `multiview_fuse.json`, `multiview_log.txt` | `eval/multiview_fuse.py` | not written up |
| VLM tooling | `vlm_anchors.json`, `vlm_direct/`, `vlm_sessions/`, `tools_smoke.json` | `eval/vlm_anchor_tool.py`, `eval/vlm_direct.py`, `workbench/vlm_loop.py` | `FINDING_vlm_direct.md` — model-direct registration fails (rank-1 solve) |
| misc | `correction_audit.json/.py`, `real_convention_check.json`, `run_c_electro.json`, `summarize_*.py`, `sheets/` (contact sheets used to check the data by eye) | — | — |

Notes: these files were produced against the fuller working tree (`workbench/`, `eval/`,
`hpc/` are included); paths inside them point at the local sandbox and at `E:\` data roots and
are not needed to read the numbers. `sheets/` and `vlm_direct/*.jpg` are inspection crops,
not results.
