# HPC2 job ledger (depth calibration)

Protocol: HPC2_AGENT_RUNBOOK.md (sha256 360e52a4…). Cost accounting by partition.
debug: free per runbook. Paid A40: single card, low tier; actual billing via SSO portal
(not resolvable from SSH — no estimates presented as fact).

| Job | Partition | Purpose | State | Elapsed | Cost | Notes |
|---|---|---|---|---|---|---|
| 13973167 | debug (free) | qualification gate: torch/CUDA, model load, 2 real inference steps, cache write/read | FAILED exit 43 | 00:00:02 | ¥0 | CAUGHT BLOCKER: shared anaconda torch 2.3.0 is CPU-only (`torch.version.cuda=None`). Gate failed exactly as designed; no paid job submitted. |

## Environment fixes after gate failure
- Installed `torch==2.3.0+cu121`, `torchvision==0.18.0+cu121` into
  `/hpc2hdd/home/hourujiang/jhspoolers/depthcalib/venv` (isolated venv; shared env untouched).
- Gate will be re-run before any paid submission.
| 13973212 | debug (free) | gate retry after torch cu121 install | FAILED exit 44 | 00:00:05 | ¥0 | cv2 missing in remote venv |
| 13973232 | debug (free) | gate retry after cv2/safetensors install | FAILED exit 44 | ~00:00:10 | ¥0 | einops missing in remote venv |
| 13973265 | debug (free) | gate retry after einops install | FAILED exit 44 | ~00:00:10 | ¥0 | local model dir missing config.json (from_pretrained looked for pytorch_model.bin) |
| 13973271 | debug (free) | gate retry after config.json upload | FAILED exit 44 | ~00:00:15 | ¥0 | remote huggingface_hub 0.15.1 too old: local-dir load looked for pytorch_model.bin |
| 13973283 | debug (free) | gate retry after pinning huggingface_hub==0.36.0 + infer_depths key fix | FAILED exit 45 | ~00:00:20 | ¥0 | inference OK (A 0.5s, B 0.24s); cache re-read lacked exported GATE_OUT |
| 13973288 | debug (free) | final gate | **GATE_OK** | ~00:00:25 | ¥0 | CUDA A40 OK, A/B load+infer, cache write/read verified, GPU released to 11 MiB |
| (13973295) | i64m1tga40u | formal inference (all 105 images × A,B) | CANCELLED before start | 0 | ¥0 | --test-only start estimate 2026-12-15; A40 backlog 46 jobs → switched |
| 13973314 | i64m1tga800u | formal inference (all 105 images × A,B) | **COMPLETED exit 0** | 00:01:11 | 1×A800 + 8 CPU-min ≈ 1.2 min; exact ¥ per portal (SSO) | started 10:14:09, ended 10:15:20; 210 cache entries, INFER_OK |

## Cost summary (as of 2026-11-17)

- Free debug partition: 7 gate attempts, total wall < 2 min; ¥0 by runbook.
- Paid: one A800 card for 00:01:11 (job 13973314) plus 8 CPU-min; exact ¥ billed via SSO portal
  (not resolvable from SSH) — no estimate is presented as a bill.
- No other paid jobs; the A40 job 13973295 was cancelled before start (0 elapsed, no allocation).
- Local workstation GPU was used only for engineering smoke/cross-platform check (not a billed resource).
- VLM token usage (recorded from completed workflows): dev NYU 1.74M tokens / 12 min; dev ETH 1.06M / 12.5 min.
  Pilot F1/F0 batches were stopped/resumed; per-agent token totals for stopped segments were not reported,
  so total VLM cost is only partially recorded (declared as such, not estimated).

## Measurement batch status (VLM frontend)

- Dev (NYU 20 + ETH 10): complete; 1 ETH image submission-missing after contract rejections.
- Pilot F1 (60): 55 committed, 5 submission-missing (2 exhausted contract attempts, 3 ended without submit).
- Pilot F0 (60): **partial 25 committed**, batch stopped when model-service throughput stalled;
  remaining 35 images recorded as not run. Comparisons use the 23 images common to F1 and F0.
| 13974196 | i64m1tga800u | exploratory 20-image inference (A,B) | **COMPLETED exit 0** | 00:00:32 | 1×A800 + 8 CPU-min ≈ 0.5 min | INFER_OK; 40 cache entries |

## Exploratory round — model-service accounting

- Both paired conditions consumed the Claude vision frontend, not HPC GPU time.
- Direct condition: 9/20 agents failed on the model-service 5-hour quota (HTTP 429) and never ran;
  these are recorded as **not run**, not as condition failures. Usable denominator = 11.
- Points condition: 20/20 completed (2.18M tokens, 1206 tool calls, ~2.6 h wall).
- Direct condition (completed subset): 1.41M tokens, 1892 tool calls, ~2.6 h wall.
