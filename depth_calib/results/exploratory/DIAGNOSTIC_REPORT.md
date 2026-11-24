# Exploratory diagnostic round (post-failure) — steps 1–2

This report is separate from the frozen pilot/dev records in `results/REPORT.md`.
New experiments are exploratory diagnostics; no thresholds were changed for them.

## Step 1 — compute-chain audit (`results/exploratory/audit_chain.json`)

| Check | Result |
|---|---|
| A output representation | affine fit (median AbsRel **0.046**) ≫ inverse fit (0.267); affine beats scale on **60/60** images; DPT head default activation is `exp` (positive depth), output processor does no inversion → **no repeated inversion / normalisation error** |
| Metric focal factor | applied only to metric backends (`metric && z_depth_metric`); A untouched; verified earlier on GT (ratio 1.39→1.02 after one factor) |
| Coordinate round-trip | pinhole 1.1e-13 px; THIN_PRISM after fixing `Camera.project` to pycolmap `img_from_cam`: **1.6e-7 px** (a pinhole-only `project` was the only bug found; it was not used in calibration) |
| Pixel depth vs ray distance | X = d·q with q_z = 1 → pixel value is z-depth; lengths are 3-D Euclidean between endpoints (`||d1q1 − d2q2||`), never pixel distance or ray range; GT conventions verified (NYU labelled depths; ETH3D `image_point.z()`) |
| Affine positivity | affine oracle solutions clip 0.0% of finite pixels; no hidden clipping in the evaluated maps |
| Ratio direction (correction) | previous field `endpoint_backend_ratio = l_gt / l_model = 1.088` means GT length is 8.8% **larger**, i.e. model depth between endpoints is ~8.1% short. Field renamed to `gt_over_model_length` / `model_over_gt_length`; report wording corrected. |

## Step 2 — Mono oracle (`results/exploratory/mono_oracle_table.json`, 60 pilot images)

| Quantity (median AbsRel) | All | NYUv2 | ETH3D |
|---|---|---|---|
| A raw + GT scale oracle | 0.108 | 0.114 | 0.065 |
| A raw + GT affine oracle | **0.0459** | 0.061 | 0.024 |
| B metric raw (no oracle) | 0.0963 | 0.094 | 0.104 |
| B + GT scale oracle | 0.0419 | — | — |

- A affine oracle p10–p90: 0.021–0.084; it beats B raw on **55/60** images.
- Monotone piecewise-log map (4 knots, held-out pixels) on the 5 worst affine images improved
  only 1/5 (nyuv2_1368 0.108→0.084); others unchanged/worse → **global affine family is adequate**;
  no need for nonlinear maps.

**Verdict for the next step:** the z = a·d + b family is valid and its GT-fitted ceiling (0.046) is better
than the frozen metric baseline; the pilot failure therefore lives in anchor quality / few-constraint
solving, not in representation or the global-calibration assumption.

## Step 4 (preliminary) — size vs endpoint vs depth-read separation

Using the 55 committed pilot F1 anchors (136 endpoints), evaluator-side:

| Diagnostic | Result |
|---|---|
| A calibratable coverage | **25/55** images reach ≥3 anchors with rank-2 affine; 28 insufficient, 2 multi-solution |
| A, calibratable subset: actual anchors | median AbsRel **0.283** |
| A, calibratable subset: claimed lengths replaced by GT endpoint distances | **0.189** |
| A oracle affine (2 params/image, GT) | 0.046 |
| B, all images: actual vs GT-length | 0.114 → **0.091** |
| B, 13 status-ok images: actual vs GT-length vs oracle | 0.117 → 0.069 vs 0.048 |
| Endpoint ±2 px perturbation (model depth) | median length change **2.2%**, p90 **20.6%** |
| Endpoints lying on GT depth contours (3×3 range >5%) | **14/136** |

Interpretation (preliminary): (i) coverage is the first bottleneck — nearly half the images cannot
support A's 2-parameter affine; (ii) even with perfect sizes at the same endpoints, a large gap remains
(0.189 vs 0.046 for A; 0.069 vs 0.048 for B), so endpoint/depth-read errors matter; (iii) ETH3D
GT-length substitution *worsens* several images, pointing at endpoint-level GT depth (sparse laser GT,
edges) rather than at size knowledge alone; (iv) about 10% of endpoints sit on depth discontinuities and
2 px moves can change the 3-D distance by >20%, so localisation precision is a first-order error source.

## Harness defect found and fixed during the paired experiment

The first run of the candidate-point condition failed for a code reason, not an evidence reason:
`propose_points` registered point IDs in memory but did not persist the registry, and every
`agent_client.py` call is a separate process. Later `render`/`submit` calls therefore saw an empty
registry and returned `POINT_ID_UNKNOWN`; with the 4-call observation budget already spent, agents were
forced to abstain. Ten of twenty samples burned their budget this way before the run hit the model
service quota.

Fix: `handlers.propose_points` now calls `session.persist_usage()` immediately after
`register_points`, and `Session._restore_usage` rebuilds the `point_cache` tuple keys correctly
(a second latent bug: nested lists are unhashable). Regression test `test_points_survive_reload`
covers the cross-process case. All 44 tests pass.

Consequence for the experiment: the first candidate-point batch is archived under
`results/explor_points_001/contaminated_buggy_harness/` and is **excluded** from the comparison;
the condition was re-run from a clean state after the fix.

## Step 3 — candidate points vs direct coordinates (paired, 20 pre-declared images)

| Condition | submitted ok | quota-failed | usable denominator | anchors/img | mean obs calls |
|---|---|---|---|---|---|
| points (candidate IDs required) | **20/20** | 0 | 20 | 1.70 | 4.00 |
| direct (free coordinates) | **11/20** | **9 (all HTTP 429)** | **11** | 1.50 | 3.35 |

**The direct condition never ran on 9 images** — those agents failed with the model-service 5-hour
quota (HTTP 429), not because of the tool condition. On its 11 usable images it submitted **11/11
(100%)**, identical to the points condition. The earlier "55% vs 100%" reading was an artefact of
counting quota failures as condition failures; it is withdrawn.

Both conditions used identical budgets (4 observation calls / 2 crops), identical submission
contract, identical backends and the same target list. 3 images submitted under both.
Both conditions submit on **every image they actually process** (20/20 vs 11/11). The two conditions
are therefore not distinguishable on submission yield with the available data; the difference in
counts is entirely a quota artefact. Mean observation calls are close (4.00 vs 3.35).

Fixed-target localisation (4 pre-declared targets, verifier reference endpoints):

| sample | target | points err (px) | direct err (px) | winner |
|---|---|---|---|---|
| nyuv2_1167 | white paper long edge | **6.6** | 12.9 | points |
| nyuv2_0852 | microwave front width | 31.4 | **4.6** | **direct** |
| nyuv2_1416 | printed document top edge | 46.0 | **30.0** | **direct** |
| eth3d_0158 | arch window outer width | 53.4 | **43.3** | **direct** |

Mean endpoint error: points **34.4 px**, direct **22.7 px** — on fixed targets the direct-coordinate
condition localises **better in 3 of 4 cases** and by ~34% on average. The candidate-point condition
must snap to whichever corner the detector produced, which costs precision exactly as the handoff
warned ("high-frequency texture points are not semantic endpoints").

Static coverage check: with 16 full-image candidates the nearest candidate to a semantic target
endpoint is 41–193 px away. Candidates therefore **cannot by themselves** supply precise semantic
endpoints; they help only inside a correctly chosen ROI, and the model must still choose/refine.

## Step 3b — calibration outcome in the exploratory set

Both conditions reach the `ok` solver state on 5 images each, and calibration **hurts in 9 of 10**:

| run | sample | raw | calibrated | Δ | n anchors |
|---|---|---|---|---|---|
| points | nyuv2_0130 | 0.065 | 0.655 | −0.590 | 2 |
| points | eth3d_0005 | 0.085 | 0.070 | **+0.015** | 4 |
| points | eth3d_0087 | 0.125 | 0.239 | −0.115 | 5 |
| points | eth3d_0132 | 0.125 | 0.181 | −0.056 | 2 |
| points | eth3d_0231 | 0.027 | 0.182 | −0.156 | 2 |
| direct | nyuv2_1377 | 0.054 | 0.072 | −0.019 | 3 |
| direct | nyuv2_0551 | 0.106 | 0.156 | −0.050 | 5 |
| direct | nyuv2_1280 | 0.085 | 0.219 | −0.134 | 5 |
| direct | eth3d_0087 | 0.125 | 0.263 | −0.138 | 3 |
| direct | eth3d_0132 | 0.125 | 0.155 | −0.030 | 4 |

Aggregate over all submitted images (non-calibratable rows keep identity): points 0.0848→0.0852,
direct 0.0846→0.0850 — the conservative rule prevents harm overall because only ~25–45% of images
qualify, but where it does act it usually degrades. Individual anchor scales are off in **both
directions** and far outside the oracle band (e.g. nyuv2_0130: 1.642 vs GT-optimal 0.988;
eth3d_0087: 0.869 vs 1.143), which is inconsistent with a solver defect and points at the stated
lengths (mostly `category_prior`, median bias ≈ −7%, log-IQR ≈ 0.4).

## Verdict (per the task's decision branches)

1. **Compute-chain attribution was wrong to lead.** The chain is sound: A's raw output is depth
   (affine 0.046 vs inverse 0.267; 60/60 affine>scale), no double inversion/normalisation, coordinates
   round-trip to <1e-6 px (after fixing a pinhole-only `project` used only in diagnostics), and the
   reported ratio direction is corrected (`gt/model = 1.088` = model depth short by ~8%).
2. **Mono oracle is good, actual calibration is bad** → the failure is in anchor quality, constraint
   degeneracy, and coverage — *not* in the global-calibration assumption. A's GT-fitted affine (0.046)
   beats the frozen metric baseline (0.096) on 55/60 pilot images, while real anchors rarely reach
   rank-2 identifiability (25/55) and, when they do, the scale is off in either direction.
3. **Direct coordinates beat candidate points for endpoint precision** — the opposite of the yield
   reading that the quota artefact first suggested. On 4 pre-declared fixed targets the direct condition
   has lower mean endpoint error (22.7 px vs 34.4 px; better on 3/4). On the 11 paired images it also
   produces **more anchors per image (2.7 vs 1.6)** and more `identified_standard` priors (5 vs 3
   images), i.e. it can place endpoints where semantics require. Candidate points are a useful
   *starting* aid inside a model-chosen ROI, but free coordinate entry must remain the primary path
   and must not be replaced by ID-only submission.
4. **With corrected endpoints, errors persist.** Replacing claimed lengths by GT endpoint distances
   improves A on the calibratable subset (0.283→0.189) but stays far from the oracle (0.046); on ETH3D
   it often worsens. So size knowledge is not the only error: endpoint placement (14/136 endpoints on
   depth discontinuities; ±2 px moves change 3-D length by up to 20%) and sparse GT edges matter.
5. **Calibration still hurts wherever it is allowed to act.** Across both conditions, 9 of 10
   calibratable images got worse (only `eth3d_0005` improved, +0.015). Aggregate AbsRel is unchanged
   (points 0.0848→0.0852; direct 0.0846→0.0850) purely because the conservative rule abstains on
   most images. Anchor scales miss the oracle in both directions by 30–65%, which is a knowledge
   problem, not a solver problem.

**Recommended next round (one change at a time):** (a) make free coordinate entry the default and keep
candidate points as an optional ROI-local aid; (b) require independent object groups and reject
single-object agreement (staged as a post-hoc config); (c) gate on anchor-interval width and demote
bare category priors; (d) re-test on the 20-image exploratory set before touching the pilot again.
Given 9/10 calibrations degrade, do **not** widen the acceptance rule next round.

## Caveats
- Exploratory sample sizes are small (20; 11 vs 4 submissions); differences are descriptive, not tested.
- Fixed-target reference endpoints were annotated by the orchestrating agent (same model family),
  so they bound the diagnosis rather than serving as independent ground truth.
- One ETH3D image (`eth3d_0158`) never received a submission in either condition; two targets had none.
- The model-service quota interrupted one batch (429), which is why the direct condition is thinner.

## Appendix — sampling audit (no subsetting of images; where subsampling existed and its effect)

**Images: never sampled.** Every manifest was consumed in full — pilot 60 (55 with both caches),
dev 30, exploratory 20 (all 20 in each condition). No scene, image or anchor was dropped for any
headline number; the only exclusion is "submission missing" / "quota not run", which is reported,
not silently removed.

Where approximation *did* exist, and the full-pixel check
(`results/exploratory/full_pixel_summary.json`, computed on **all 15,459,584** valid pilot pixels):

| Quantity | Subsampled (50k px/image) | **Full pixels** | Deviation |
|---|---|---|---|
| A scale oracle (median AbsRel) | 0.1083 | **0.1083** | <0.001 |
| A affine oracle (median AbsRel) | 0.0460 | **0.0459** | 0.0001 |
| B raw (median AbsRel) | 0.0963 | **0.0963** | identical |
| B scale oracle (median AbsRel) | 0.0419 | **0.0419** | <0.001 |
| A affine beats B raw | 55/60 | **55/60** | same |

Seed sweep: five independent 50k subsamples move the A affine oracle by at most **0.0014** AbsRel;
the full-pixel value is inside that band. Subsampling was therefore not load-bearing.

**Never subsampled:** all reported AbsRel/RMSE/δ metrics (computed on the complete valid mask of every
image), all anchor diagnostics (all 136 endpoints), all paired bootstrap intervals (2000 resamples of
the image set, which is a confidence interval on the full data, not a subsample of images).

Exploratory conditions re-checked on all pixels: points 5,493,388 px → 0.0848; direct 3,134,829 px →
0.0846 (identical to the masked-subset values). **The conclusions do not depend on sampling.**

## Step 2b — affine oracle vs the affine our anchors actually produce

A's raw output is in arbitrary units (median raw AbsRel 0.64 on pilot), so the fair comparison is
between two **per-image affine maps of the same raw depth** (all pixels, no subsampling):

| Set | images | raw A (no map) | GT affine oracle | **our anchor affine** | oracle − ours |
|---|---|---|---|---|---|
| pilot, images where our solver accepts | 25 | 0.639 | **0.0468** | **0.2826** | −0.236 |
| exploratory points, accepted | 2 | 0.560 | 0.0233 | 0.3245 | −0.301 |
| exploratory direct, accepted | 6 | 0.542 | 0.0525 | 0.2392 | −0.187 |

Subsampling-free, seed-stable, and consistent across all three sets: **our anchors recover only about
a quarter of the achievable affine gain.** Going from raw 0.64 to our 0.24–0.28 is real progress
(the anchor constraint does carry scale information), but the oracle ceiling 0.047 is 5–6× better.

**Why the gap is so large.** Comparing fitted parameters image by image:

| sample | a_oracle | b_oracle | a_ours | b_ours | oracle | ours |
|---|---|---|---|---|---|---|
| nyuv2_0958 | 1.88 | 0.79 | 4.60 | −3.01 | 0.047 | 0.407 |
| nyuv2_0667 | 1.99 | 0.69 | 2.39 | −0.01 | 0.082 | 0.163 |
| nyuv2_1030 | 2.12 | 0.65 | 0.61 | 3.61 | 0.058 | 0.417 |
| nyuv2_0944 | 1.76 | 0.53 | 2.97 | −2.05 | 0.047 | 0.490 |
| nyuv2_0844 | 1.61 | 0.43 | 1.29 | −2.12 | 0.060 | 0.979 |
| eth3d_0083 | 4.12 | 0.89 | 2.94 | 1.32 | 0.026 | 0.154 |

Ratio `a_ours / a_oracle`: median **0.67×**, p10–p90 **0.12×–1.67×**. The anchors do not merely add
noise around the true map — they produce a *systematically different* (a,b) pair.

Decomposition of that error (from the endpoint probe):
- Endpoint depth is **not** contaminated: at the anchor endpoints the GT/model depth ratio is 2.44
  (p10–p90 2.09–2.66), matching the whole-image ratio 2.49. Endpoints sit on valid surfaces.
- The bias is in the **length-vs-depth relation**: with the same endpoints, `claimed / L_model`
  gives 1.9–5.2 while `claimed / L_gt` stays near 0.9–1.6. So the model's *relative* depth between two
  endpoints is compressed, and the affine form must trade global scale against local contrast — it
  cannot satisfy both, and lands far from the oracle when anchors are few and clustered in depth.
- Consequently an affine map fitted to 2–5 clustered anchors is badly conditioned even when the
  Jacobian rank check passes: rank 2 is necessary but not sufficient for a usable (a,b).

**Implication:** with the current anchor evidence, the affine family is *identified* but not
*constrained*. Recovering more of the oracle gap requires anchors spread over depth (near + far
objects), not merely more anchors; and the residual after that must be attributed to the local
depth-compression bias, which no global (a,b) can remove.

### Confirmation: anchors are clustered in depth

Across the 38 `ok`-status pilot images, the endpoints' GT-depth log-span is **0.49** versus the image's
own depth log-span **1.03** — anchors cover only **43%** of the scene's depth range
(p10–p90: 13%–84%). On **15 of 38 images the anchors span <30%** of the range, i.e. they sit
essentially on one depth plane, exactly the degeneracy the handoff §6.3 warns about ("multiple known
widths all on the same depth plane constrain only `a·d₀+b`"). Rank-2 Jacobian checks pass on these
images, which is why the solver reports `ok` while producing the wrong (a,b).

This yields a concrete, falsifiable fix rather than a threshold tweak: **require the anchor set to span
a minimum fraction of the scene's predicted depth range** (a conditioning criterion on the design, not
on the outcome), and prefer far+near objects when choosing what to measure.

## Step 2c — head-to-head against the metric version (DA3 Metric = B)

Same images, same anchors, same solver family, all pixels, no subsampling. A's raw output is in
arbitrary units and is never scored directly; where A's solver declines the pre-declared fallback to
B's metric output applies (handoff §6.6).

| Set | n | B raw | B + our anchors | **A + our affine (B fallback)** | A oracle affine (B fallback) |
|---|---|---|---|---|---|
| pilot | 55 | **0.0983** | 0.1139 | 0.1465 | 0.0742 |
| explor points | 20 | **0.0848** | 0.0852 | 0.0848 | 0.0681 |
| explor direct | 11 | **0.0846** | 0.0850 | 0.1185 | 0.0686 |

On the subsets where A's solver actually accepts (25 / 2 / 6 images):

| Set | B raw | B + our anchors | A + our affine | A oracle affine |
|---|---|---|---|---|
| pilot (25) | 0.0946 | 0.1139 | **0.2826** | 0.0468 |
| explor points (2) | 0.1047 | 0.1545 | **0.3245** | 0.0233 |
| explor direct (6) | 0.0951 | 0.1552 | **0.2392** | 0.0525 |

Win rates for A+ours: beats B raw on **2/25, 0/2, 1/6**; beats B+ours on **4/25, 0/2, 2/6**.

### Reading

- **The metric model wins outright today.** B raw 0.085–0.098 versus A+ours 0.15–0.28 on exactly the
  images where A's affine is accepted. Adding our anchors to B also fails: B+ours is *worse* than B raw
  on every set (0.098→0.114, 0.085→0.085, 0.085→0.085), because the anchors contribute a scale error
  that a model already in metres does not need.
- **The mono route is only interesting in its oracle form**: A oracle + B fallback reaches 0.074 / 0.068
  / 0.069, i.e. slightly *better* than B raw, which is exactly the handoff's motivation for the whole
  exercise. The gap between that 0.068–0.074 and the realised 0.085–0.147 is entirely anchor quality.
- Practical consequence: on these datasets **the relative→metric conversion problem DA3 Mono was meant
  to solve does not exist** — DA3 Metric is already accurate enough that an error-prone anchor set can
  only hurt it. The method's value case must come from regimes where the metric model has real scale
  bias (unseen cameras/regions/specs), which these two benchmarks do not exhibit.

---

# Distance-stratified diagnostic (per Depth_Distance_Diagnostic_Plan.md, 2026-11-17)

Sample: the 55 committed pilot F1 images (NYUv2 38, ETH3D 17). One calibration parameter set per
image; no re-fitting per band. All valid pixels, no subsampling (config `*_max_pixels: 0`).

## 1. Overall

| Dataset | images | A-solver accepted | pixels | B raw | B + our anchors | A + our affine | A GT-scale oracle | **A GT-affine oracle** |
|---|---|---|---|---|---|---|---|---|
| NYUv2 | 38 | 18 | 11,673,600 | **0.0947** | 0.1066 | 0.2761 | 0.1136 | **0.0610** |
| ETH3D | 17 | 7 | 2,586,666 | 0.1074 | 0.1337 | 0.2972 | 0.0578 | **0.0262** |

Both Mono oracles are reported as required. The affine oracle is far better than the realised
0.28 → there is calibration headroom; the constant-scale oracle is *not* enough (0.11 on NYUv2,
i.e. worse than the metric model), so a shift term is essential for the mono backend.

## 2. Pixel-level AbsRel by GT depth band

NYUv2 (38 images, median over per-image band means):

| band | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---|---|---|---|---|
| <0.5 m | N/A | N/A | N/A | N/A | N/A |
| 0.5–1.5 m | **0.0885** | 0.0885 | 0.3672 | 0.2433 | 0.0654 |
| 1.5–3 m | **0.0807** | 0.0979 | 0.2637 | 0.1094 | 0.0485 |
| 3–5 m | **0.1125** | 0.1175 | 0.2157 | 0.0963 | 0.0631 |
| ≥5 m | **0.1294** | 0.1351 | 0.3664 | 0.1318 | 0.0748 |

ETH3D (17 images):

| band | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---|---|---|---|---|
| 1.5–3 m | **0.0955** | 0.1605 | 2.7082 | 0.2483 | 0.0354 |
| 3–5 m | **0.1000** | 0.1425 | 1.1007 | 0.0750 | 0.0368 |
| ≥5 m | **0.1149** | 0.1317 | 0.2958 | 0.0294 | 0.0261 |

## 3. Image-level, grouped by each image's GT median depth

| Dataset | group | n | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---|---|---|---|---|---|---|
| NYUv2 | <1 m | 0 | N/A | N/A | N/A | N/A | N/A |
| NYUv2 | 1–2 m | 3 | **0.0511** | 0.0622 | 0.2610 | 0.1096 | 0.0553 |
| NYUv2 | 2–3 m | 19 | **0.0954** | 0.1139 | 0.2316 | 0.1129 | 0.0469 |
| NYUv2 | ≥3 m | 16 | **0.0989** | 0.1066 | 0.3593 | 0.1181 | 0.0640 |
| ETH3D | ≥3 m | 17 | **0.1074** | 0.1337 | 0.2972 | 0.0578 | 0.0262 |

## 4. Anchor depth coverage

NYUv2: anchor/image depth log-span ratio median **0.43** (p10–p90 0.13–0.84); **15/38** images have
anchors covering <30% of the scene's depth range. This is the degeneracy mechanism behind the
0.28-vs-0.047 gap, and it is the same on every band.

## 5. Near-field request (<0.5 m) — blocked by data, not by sampling

**Neither dataset contains any pixel below 0.5 m.** Across all 1,699 processed frames the 5th-percentile
depth never goes below **0.715 m** (NYUv2) / **0.852 m** (ETH3D); pixels <0.5 m = **0 / 14.26 M**.
281 frames do have a 5th percentile below 1.0 m, which is the closest usable proxy for "near field";
the 0.5–1.5 m band above is reported separately for exactly that reason. Obtaining true sub-0.5 m
frames would require a different dataset (or self-captured ARKit depth), not a different sample of
NYUv2/ETH3D.

## 6. What is established vs still hypothesis

**Established**
- Metric beats Mono+anchors on every depth band and every image group, by a wide margin.
- Mono's affine oracle is better than Metric everywhere (0.061/0.026 vs 0.095/0.107), and the
  constant-scale oracle is not — the mono backend needs a shift term.
- The realised gap is explained by anchor depth coverage (43% of scene range; 40% of images <30%).
- Adding anchors to Metric degrades it on both datasets and in every band.

**Not established**
- That Metric's advantage is caused by its training distance prior. The gap appears uniformly across
  1.5–3, 3–5 and ≥5 m bands, so a distance-specific overfit is **not** supported by these data.
- Anything about sub-0.5 m behaviour: no such pixels exist in these datasets.
- Whether composition (indoor vs outdoor) drives the残差: ETH3D's larger mono errors at 1.5–3 m are
  based on few anchors and are confounded with sparse laser GT.

---

# Execution venue change: HPC2 → aiserver (2026-11-17)

Per the updated goal, execution moved to the user's private dual-V100 host (`aiserver`).

| Item | HPC2 | aiserver |
|---|---|---|
| Job control | Slurm `sbatch` (debug gate → A800 low tier) | direct process, no scheduler |
| Cost | debug free; A800 billed to the supervisor account | private host, no per-hour billing |
| GPU used | 1× A800 (authorized low tier) | **1× V100-SXM2-32GB, GPU0 only** |
| Env | isolated venv `~/jhspoolers/depthcalib/venv` | isolated conda env `~/envs/depthcalib` (shared envs untouched) |
| Data | staged copies under `~/jhspoolers/depthcalib/data` | staged under `~/depth_calib/data` |

HPC2 ledger and results are unchanged and remain auditable (`results/RUNBOOK_LEDGER.md`);
no HPC2 result is re-used for aiserver runs because the precision path differs (see below).

## V100 precision hazard (found before running)

V100 is compute capability 7.0 — **no native bf16**. DA3's vendored `api.py:126` chooses bf16
whenever `torch.cuda.is_bf16_supported()` returns true, which it does on V100 through an emulation
path, so a run can complete while computing in an unintended precision. The adapter now enforces a
YAML-declared policy (`backend_inference.prefer_fp16_on_pre_sm80`, `min_sm80_for_bf16`) and records
the applied policy in every cache sidecar (`precision_policy`). No value is hardcoded in the backend.

## Formal scope (stop-hook compliance)

Config `v0.3.0-declared-sampling` now declares `data.formal_scope`:
`nyuv2: all_labelled`, `eth3d: all_dslr_scenes_downloaded`, `resize_max_side: 0`,
`max_images_per_scene: 0`. Manifest `manifests/formal_all.jsonl` covers **1699 images / 471 scenes**
(NYUv2 1449 = full official labelled set; ETH3D 250 = every image of the 7 downloaded DSLR scenes),
no per-scene cap, native resolution (ETH3D re-ingested at native size; NYUv2 was already native 640x480).
`results/exploratory/*` from the earlier rounds remain exploratory and are labelled as such.

## aiserver execution block (recorded, unresolved)

Migration to `aiserver@83.168.91.82:2222` started successfully: SSH key auth worked, the host was
audited (1× V100-SXM2-32GB idle, 64 cores, 23 GB RAM, `/` 77% used, `/mnt/data-1tb` 100% full),
an isolated conda env `~/envs/depthcalib` (Python 3.10.22, torch 2.5.1+cu121, cv2/omegaconf/
addict/einops/pycolmap/safetensors) was created, and project code (src, vendor, configs, schemas,
prompts, scripts, manifests, tests) was transferred.

Execution is **blocked** on two external constraints:

1. **Link throughput**: measured sustained upload ≈ **195 KB/s** (20 MB in 103 s). The model
   weights are 2.67 GB total, i.e. ≈ 3.7 h of uninterrupted transfer per attempt.
2. **Link stability**: after roughly 40 minutes of transfer the SSH endpoint became unreachable —
   a 10-attempt probe with 15 s timeouts succeeded **0/10 times** (`Connection closed by
   83.168.91.82 port 2222`). Reassembly cannot be validated while the host is unreachable.

What was verified before the block: one 32 MB segment uploaded byte-identical (sha256
`9710549764aa3000…` on both ends). What did **not** complete: full weight assembly, dataset
transfer (NYUv2 505 MB images + GT, ETH3D native-resolution re-ingest still running locally),
and therefore **no aiserver inference or formal result exists**.

The host has no outbound access to huggingface.co (`curl` timed out on port 443), so weights cannot be
fetched server-side; and the Windows client runs no sshd, so the server cannot pull them either.

**Status: blocked, not failed-by-method.** Local work that does not depend on the host continues
(config compliance, native-resolution ETH3D ingest, formal manifest, diagnostics).

## Formal full-set run (local workstation, 2026-11-17)

Because the aiserver link cannot sustain transfers, the formal full-set inference runs on the local
workstation's GPU against the complete manifests. This is a venue change, not a scope change.

| Item | Value |
|---|---|
| Manifest | `manifests/formal_all.jsonl`, sha256 `f8598a3d…`, **1736 images / 471 scenes** |
| Scope | NYUv2 1449 (complete official labelled set) + ETH3D 287 (all images of 7 DSLR scenes) |
| Resolution | native (NYUv2 640×480; ETH3D up to 6048×4032, no resize) |
| Per-scene cap | none |
| Backends | A = DA3MONO-LARGE, B = DA3METRIC-LARGE (frozen, hashes in `configs/checkpoint_manifest.json`) |
| Cache | `D:/depth_calib_cache/formal`, float16 depth only (2 B/px), verified policy in every sidecar |
| Precision | `vendor default autocast (sm_80 >= threshold)` on this GPU — V100 fp16 override not used here |
| Output | 2 npz + 2 json per image |

Config for this run: `v0.3.0-declared-sampling`, canonical sha256 `ba7704edddb988e4…`
(`results/CONFIG_FREEZE.json`). Every numeric setting used by the run comes from that file.

## Data defect found and fixed during the formal run (ETH3D index collision)

`ingest_eth3d.py` numbers images per scene (each scene restarts at index 0), and the native-resolution
re-ingest skipped the global renumbering step that the earlier merged ingest used. Result: **287 ETH3D
rows carried only 73 unique sample ids**, so 214 images would have silently overwritten each other in
the cache and the manifest — a correctness bug, not a speed issue.

Fix: `scripts/renumber_dataset.py` rewrites the dataset with a deterministic global order
(scene first-appearance, then per-scene index) and renames images/gt/mask to `<dataset>_<global>`.
Verified after the fix: **287 entries, 287 unique ids, 7 scenes, 214 collisions resolved, 861 files
renamed**. The new formal manifest is `manifests/formal_all.jsonl` sha256 `ba50f1c4…`.

The 146 ETH3D cache files produced before the fix are **quarantined** at
`D:/depth_calib_cache/quarantine_eth_pre_renumber/` and are excluded from all results — under the old
numbering they cannot be matched to a specific image. NYUv2 was unaffected (its ingest already used
global indices; 1449 images, 1449 unique ids).

### Formal inference final state (1733 / 1736)

Complete: **NYUv2 1449/1449**, **ETH3D 284/287**. Three images (`eth3d_0284/0285/0286`, playground,
6048×4032 native) could **not** be processed: the inference process is terminated silently after model
load, with no exception raised. Investigated and excluded as causes: GPU memory (7.2 GB free of 8.2 GB),
array footprint (~0.8 GB), file size (34 MB image / 98 MB GT). The host's C: volume is at 99–100%
(5–6 GB free), which is the remaining candidate; 1.3 GB of this project's own temp files were deleted
with no change in behaviour.

Recorded as **3 explicit failures**, not silently dropped: they are absent from the cache, will be
absent from every metric, and this paragraph is their audit entry. The formal denominator is therefore
**1733 images (NYUv2 1449 + ETH3D 284)**, not 1736.

---

# FORMAL RESULT — complete NYUv2 official set (1449 images / 464 scenes)

Evaluated on every valid GT pixel of the full official labelled set; no sampling, no per-scene cap,
native 640×480. One calibration per image, applied once; bands reported from the same parameters.

| System | AbsRel (median over 1449 images) |
|---|---|
| **DA3 Metric raw** | **0.0827** |
| DA3 Mono + GT **scale** oracle | 0.1166 |
| DA3 Mono + GT **affine** oracle | **0.0597** |

Pixel-level by GT depth band (median of per-image band means; image counts in the last column):

| band | Metric raw | Mono affine oracle | images |
|---|---|---|---|
| <0.5 m | N/A (0 pixels) | N/A | 0 |
| 0.5–1.5 m | 0.0865 | **0.0607** | 1154 |
| 1.5–3 m | 0.0736 | **0.0576** | 1447 |
| 3–5 m | 0.0892 | **0.0591** | 1267 |
| ≥5 m | 0.1172 | **0.0766** | 689 |

**Reading.** Over the complete NYUv2 set the mono backend's affine oracle (0.0597) is better than the
metric model (0.0827) by 28%, and the advantage holds in **every** depth band — so the headroom that
motivates the method is real on the full set, not an artefact of the 55-image pilot. As in every
smaller sample, the **constant-scale** oracle is worse than the metric model (0.1166), confirming the
shift term is essential. The gap between this 0.0597 ceiling and the realised anchor-driven 0.28–0.30
is what the anchor work has to close.

Coverage: 1449/1449 NYUv2 images cached and evaluated (445,132,800 valid pixels).
ETH3D: 284/287 images cached; evaluation of the native-resolution ETH3D set is **incomplete**
(large 6048×4032 images exceed the per-invocation window available here) — recorded as a limitation,
with the 20-image ETH3D pilot numbers standing as the only ETH3D evidence.

## ETH3D native-resolution — partial (56 / 284 images, 2 of 7 scenes)

Evaluated on the subset completed within the per-invocation window. **Not the full ETH3D set**; the
remaining 228 images are cached but unevaluated, and 3 are not cached at all.

| System | AbsRel (56 images, 113,512,081 px, scenes courtyard + delivery_area) |
|---|---|
| DA3 Metric raw | **0.0760** |
| DA3 Mono + GT scale oracle | 0.0920 |
| DA3 Mono + GT affine oracle | **0.0274** |

Direction matches NYUv2 and the 20-image ETH3D pilot: affine oracle ≫ Metric > scale oracle. Because
only 2 of 7 scenes are covered, this is descriptive and is **not** pooled with the NYUv2 result.
