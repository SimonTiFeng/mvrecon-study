# Codex-free execution report — world-knowledge metric depth calibration

Protocol source: `Codex_Metric_Depth_Agent_Handoff.md` v0.2 (sha256 a0a79c86…).
Resource protocol: `HPC2_AGENT_RUNBOOK.md` (sha256 360e52a4…).

## 1. Deviations from the handoff (all recorded, none silent)

| Item | Handoff plan | Executed | Reason |
|---|---|---|---|
| VLM frontend | Codex (`codex exec`, MCP tools) | **Claude Code vision subagents** (`claude-sonnet-5`), same four-tool submission contract via `agent_client.py` | Codex CLI on this machine routes through a local proxy; both `gpt-5.5` and `gpt-5-codex` are rejected by the account ("not supported when using Codex with a ChatGPT account", HTTP 400). User directed: complete directly without Codex. Handoff §3.2 allows using a vision model actually available to the user. |
| MCP transport | stdio MCP server | Same server/handlers, invoked per call by the vision agent through a thin CLI client | Preserves server-side budget, validation, atomic evidence storage and idempotency exactly as specified. |
| Daily checkpoint D/E | DA2 metric indoor/outdoor | Not used in round 1 | `active_backends: [A, B]` per handoff §7. |
| ARKitScenes | 20 pilot images | **Not run** | Deferred explicitly (see §6); pilot = NYUv2 40 + ETH3D 20 as permitted by handoff §8.3. |

## 2. Data

- NYUv2 labeled (official `nyu_depth_v2_labeled.mat`, v7.3): 1449 images, 464 true scenes.
  K: official labeled-image intrinsics constants (documented in `meta.json`); GT = `depths` (metres, z-depth);
  mask = finite and >0. Scene-disjoint splits by `scenes` identifiers.
- ETH3D DSLR (7 scenes: courtyard, delivery_area, electro, office, facade, terrace, playground):
  GT depth = headerless float32 metres, invalid = +inf; depth convention verified in the official
  `ground_truth_creator.cc` (`gt_depth_map = image_point.z()` → z-depth). Cameras are
  `THIN_PRISM_FISHEYE`; pixel→ray unprojection uses pycolmap's exact camera model (no pinhole
  approximation). Images resized to long side ≤1280 with K/params scaled; depth/mask nearest.
- Splits (scene-disjoint, seed 42, sha256 ordering): pilot 60 (NYU 40 + ETH 20), dev 30 (NYU 20 + ETH 10),
  smoke 15 (NYU 10 + ETH 5). Manifests and hashes in `manifests/`.
- GT mask coverage: NYUv2 100% of pixels; ETH3D 6.9–14.5% (laser-scanned sparse GT). All metrics are
  computed on the valid mask only.

## 3. Depth backends (frozen)

- A = `depth-anything/DA3MONO-LARGE` @ f465978e…
- B = `depth-anything/DA3METRIC-LARGE` @ 4010e39f…
- Vendored official code at commit 3d835ec1… (`vendor/Depth-Anything-3/PINNED_COMMIT.txt`).
- DA3 Metric focal/300 conversion verified empirically on 5 NYUv2 images
  (`results/metric_factor_check.json`): raw prediction ≈ GT / 1.36 (median ratio 1.387); applying
  `fx_processed/300` once gives median ratio 1.018 → decision `apply_focal_over_300_once`, matching the
  config field (no double conversion).

## 4. HPC2 execution

See `results/RUNBOOK_LEDGER.md`. Summary: free debug gate iterated 7 times (each failure fixed a real
blocker: CPU-only shared torch, missing cv2/einops/safetensors, stale huggingface_hub that ignored
`safetensors`, missing local `config.json`, script defects). Final gate `13973288`: GATE_OK.
Formal inference ran on one A800 card (`i64m1tga800u`, authorized low tier); A40 was cancelled
before start because its queue estimate was a month out.

## 5. Measurement protocol

- F1 (harness): up to 4 observation calls (≤2 crops), ≤6 anchors, 2 submission attempts — enforced
  server-side by the tool service; evidence stored only by the handler.
- F0 (direct): same submission contract, observation tools not exposed.
- Note: the configured 180 s per-sample wall-clock timeout was implemented for the Codex runner;
  with subagent measurement it was not enforced (no equivalent process-level timeout), which is recorded
  as a protocol deviation rather than claimed as done.
- Endpoint quality gate (depth-based sensitivity, 2 px, >1% warn/reject) and solver identifiability
  checks are applied evaluator-side; applied thresholds are frozen in `configs/pilot.yaml`.

## 6. Known limitations / unfinished

- No filesystem-level isolation between measurement agents and evaluator data on this workstation;
  GT lives outside the project tree and was never referenced in agent prompts, but this is weaker than
  the mount/permission isolation the handoff requires. Recorded as a validity caveat.
- ARKitScenes pilot subset not executed (access/scale); explicit block, not silent omission.
- Human-verified oracle subsets (correct-length/correct-endpoint) not produced this round.

## 7. Results

### 7.1 Dev set (29 measured: NYUv2 20 + ETH3D 9; 1 ETH submission-missing) — acceptance-rule decision

AbsRel on the valid GT mask, no per-image scaling (bootstrap 2000, seed 42):

| System | AbsRel (all) | NYUv2 | ETH3D | 95% CI |
|---|---|---|---|---|
| B metric raw (adapter-native) | 0.0701 | 0.077 | 0.054 | [0.060, 0.081] |
| B + anchors, plain median | 0.2353 | 0.228 | 0.251 | [0.171, 0.302] |
| B + anchors, consistency (applies single-anchor scales) | 0.1548 | 0.174 | 0.112 | [0.103, 0.216] |
| B + anchors, conservative (requires ≥2 agreeing anchors) | 0.0697 | 0.077 | 0.053 | [0.060, 0.081] |
| B oracle global scale (AbsRel-optimal) | 0.0502 | 0.062 | 0.024 | [0.040, 0.062] |

Solver outcomes on dev: 15 single-anchor, 9 conflict, 3 insufficient, **2 ok**. The conservative system
is statistically indistinguishable from the metric baseline (0.0697 vs 0.0701); applying sparse anchors
without the agreement rule is strongly harmful (0.235). The oracle ceiling (0.050) shows limited headroom
on these sets.

Protocol-integrity note: during pilot inspection a stricter acceptance rule (requiring ≥2 distinct
knowledge groups, to prevent agreement among correlated anchors on the same object) appeared useful.
Because it was motivated by pilot observations, it is **not** applied to the pilot main analysis; the
pre-registered v0.2.1 rule (≥2 agreeing anchors) is kept, and the group rule is reported only as a
post-hoc exploratory comparison for the next round.

Two protocol-relevant corrections made during dev (config v0.2.1-devfix, hash recorded):
1. The DA3-Metric focal/300 conversion is part of the adapter; the earlier baseline that omitted it
   inflated apparent calibration gains. The baseline is now the adapter-native metric depth.
2. The conservative acceptance variant (handoff §9.1) was implemented and promoted as the main
   calibrated row; consistency/plain rows are reported alongside for transparency.

### 7.2 Pilot (scene-disjoint, 60 images: NYUv2 40 + ETH3D 20)

Coverage: 55/60 committed (5 submission-missing: 2 exhausted both contract attempts, 3 ended without
submitting). Metrics on the valid GT mask, no per-image scaling; paired bootstrap over images.

| System | n | AbsRel | 95% CI | paired Δ vs B metric raw |
|---|---|---|---|---|
| B metric raw | 55 | 0.1040 | [0.094, 0.114] | — |
| B + anchors, conservative (pre-registered) | 55 | 0.1103 | [0.098, 0.123] | −0.0063 [−0.0151, −0.0002] |
| B + anchors, consistent | 55 | 0.1916 | [0.149, 0.240] | −0.0876 [−0.136, −0.050] |
| B + anchors, plain | 55 | 0.2399 | [0.189, 0.294] | −0.1359 [−0.189, −0.089] |
| A + affine (coverage-limited) | 55 | 0.6245 | [0.578, 0.668] | −0.5205 |
| A + affine, B fallback | 55 | 0.1889 | [0.147, 0.237] | −0.0849 |
| B oracle global scale (AbsRel-optimal) | 55 | 0.0519 | [0.044, 0.062] | +0.0520 [+0.041, +0.064] |

**Outcome:** with this vision model, world-knowledge anchors do not improve the frozen DA3-Metric
baseline. The pre-registered conservative system is slightly worse (−0.006 AbsRel, CI excludes 0), and
aggressive application is much worse. The oracle gap (+0.052) shows the global-scale parameterization
itself has headroom, so the loss is in anchor quality, not in the solver.

Anchor-scale diagnostics (95 anchors, 55 images; evaluator-side):
- Median knowledge scale `s_knowledge` = **0.927** (p10–p90 0.51–1.26): claimed lengths are ~7% small,
  with large spread; `identified_standard` (0.948) only slightly better than `category_prior` (0.919).
- Median endpoint/backend ratio = **1.088**: the model's depth between anchor endpoints is ~9% shorter
  than GT, so knowledge bias (low) and backend bias (low predictions) partially cancel.
- Solver outcomes: dominated by single-anchor/conflict; few samples satisfy the 2-anchor agreement rule.

Illustrative cases:
- **Degradation**: `eth3d_0085` (electro) — three anchors on glass double doors (2.0 m width, 2.1 m
  heights) agree with each other (s≈0.80) but the GT-optimal scale is 1.16; AbsRel 0.13→0.30. All three
  anchors describe the same door family, so agreement was not independent evidence.
- **Mild degradation despite agreement**: `nyuv2_0844` — microwave/countertop/cooktop anchors agree
  (s≈0.95) but GT-optimal is 1.05; AbsRel 0.074→0.117.
- **Insufficient evidence**: the majority of images yield 1 anchor or mutually inconsistent anchors →
  conservative system falls back to identity (coverage of applied corrections ≈ 3–5% of images).
- **Submission contract**: 2 images exhausted the two-attempt budget on payload-format errors
  (point_id/xy misuse), showing the server-side validation works but the agent sometimes ignores it.

F0 (direct submission) arm status: **partial** — 25/60 images measured before the batch stalled on
model-service throughput; the arm is recorded as not fully run (remaining 35 images pending), and all
F0-vs-F1 comparisons below use only the 23 images measured in both arms. F0 partial metrics (25 images):
B metric raw 0.0922, conservative 0.0970, oracle 0.0608 — the same pattern as F1.

F1 (harness) vs F0 (direct submission), on the 23 pilot images measured in both arms:
- Anchor quality: F1 median knowledge scale 0.943, log-IQR 0.336; F0 0.887, log-IQR 0.474
  (F1 anchors are more consistent and slightly less biased).
- Final metric: mean AbsRel difference (F1 − F0) = +0.0022 (median 0.0) — no practical difference.
  Tools improve anchor consistency, not the calibrated metric.


## 8. Interpretation against the handoff decision framework (§10.5)

- "B oracle has almost no headroom" — **not the case**: oracle gains +0.052 AbsRel; the global-scale
  parameterization has real headroom, but the anchors do not capture it.
- "B improves, A affine often fails" — B does not improve; A affine fails and is harmful when applied.
- "Correct sizes still do not improve" — closest description: anchors are systematically ~7% small and
  heterogeneous, while the backend is ~9% short between anchor points; corrections cancel on average.
- "A few large gains but many mild mis-edits" — no; edits are uniformly neutral-to-negative.

## 9. What to try next (one hypothesis, highest value)

Anchor **precision**, not more anchors: enforce internal consistency across *independent* objects and
reject single-object agreement. The two clean degradations both came from correlated anchors on one
object family (glass double doors; one kitchen appliance run). A next round should (a) require ≥2
distinct object groups, (b) weight by the anchor's declared interval, and (c) add an evaluator-side
oracle check on a small human-verified subset to separate knowledge error from endpoint error.
Category priors should be demoted unless their interval is narrow and the object identity is verified
by a legible marking (e.g., a model label), as in the handoff's "identified_standard" definition.

## 10. Unresolved gaps / blocks

- ARKitScenes pilot subset: **not executed** (access/scale), explicit block.
- Filesystem isolation between measurement agents and GT: prompt/path isolation only (see §6).
- Smoke split (15 images) was not run through the VLM; engineering smoke was covered by synthetic
  tests, dev runs and the HPC gate. Recorded rather than silently omitted.
- Human-verified correct-length / correct-endpoint oracle subsets: not produced.
- VLM token accounting for stopped/resumed pilot batches is partial (see ledger).
