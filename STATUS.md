# MV-Recon Agent — status

## 2026-11-24 round: first cluster run, Pareto front, and a better estimator

Ran on HPC2 (job **14069627**, CPU partition `i64m512u`, ~7 min, 3752 records) — the project's
first cluster experiment. Swept 4 matchers × 3 masks × 2 solves × 4 injected-outlier rates over all
40 targets. Details and reproduction in `FINDING_sweep_hpc2.md`.

**Accuracy is saturated; the gain is cost.** On the 25 rotation-successful targets, rotation-fixed:

| matcher | rot med | lateral med | < 0.5 m | cost |
|---|---|---|---|---|
| ORB | 1.65° | 0.31 m | 20/25 | **165 ms** |
| AKAZE | 1.65° | 0.27 m | **21/25** | 475 ms |
| SIFT | 1.65° | **0.25 m** | **21/25** | 819 ms |

ORB reaches SIFT's accuracy band at **one fifth** the cost; SIFT is Pareto-dominated by AKAZE
(same 21/25 and 0.27 m, 42% cheaper); the two-matcher union is strictly worse. Nothing moves the
0.25 m floor.

**Three further measurements from the sweep:**
- *Masks barely matter at the correspondence level* (≤0.02 m, ≤1 target). That qualifies rather than
  contradicts the earlier cloud-level sky finding: fabricated far-field surfaces distorted the
  overlap objective, not the match set.
- *The rotation correction is conditional*: it helps a good start (SIFT 14→16 of 25 under 2°) and
  hurts a bad one — on the full 40-target set it is clearly worse than leaving rotation alone.
- *Essentially flat to 30% gross outlier correspondences* (lat 0.59 → 0.62 m). The IRLS weighting
  does its job.
- *Cross-scene spread is the headline*: courtyard 0.10 m on 5/5 targets, terrace 1.65 m, facade
  1.32 m, with identical parameters. Domain shift inside one dataset, measured.

**A better estimator found meanwhile:** minimising *reprojection* error (robust, 6-DOF) instead of
3D–3D distances — lateral 0.94 → **0.29 m, 10 → 21 of 25**, improving 22 of 25 targets. It depends
on correspondence quality: with AKAZE it is much worse (0.47 m / 15-of-25), so SIFT correspondences
plus reprojection refinement is the current best pairing.

**The current method, at full scale:** front-end initialisation → sky-masked SIFT → robust
reprojection refinement reaches **27 of 40 targets under 0.5 m, median 0.28 m**, against 6 of 40 for
the geometric front end alone.

**Two negatives from this round, both recorded rather than buried:**

- *A selector that looked like a win was not one.* Choosing between the two pose candidates by the
  spatial coherence of the reprojection residual appeared to improve rotation to 1.83° median
  (17/32 under 2°) from 2.82°/2.09° for the fixed choices. Checked directly, coherence agrees with
  the better candidate in only **17 of 32** cases and the two candidates' coherence differs by
  **0.012** at the median — it is a coin flip, and the aggregate gain was an artefact of which
  choices happened to fall. Five GT-free selectors have now failed (fitness, spread, depth
  agreement, footprint, coherence).
- *Multi-view fusion is much worse.* Chaining through an intermediate view and combining robustly
  gives rotation 2.43° → **18.93°** and under-0.5 m 27/40 → **13/40** (26 worsened, 7 improved),
  because a chain is only as good as its worst link and the non-reference pairs initialise badly.

Together these say the binding problem is **selection, not the number of constraints**: the project
can generate candidate poses but cannot tell a good one from a bad one without reference geometry.


## 2026-11-23, last: the model-direct route was attempted and fails at correspondence

"Can the model do the registration itself?" was tested properly (`eval/vlm_direct.py`): the model
commits its named correspondences to disk, and the scorer refuses to run until that file exists, so
the pose can only come from what it said.

Result on playground v2: the model localised **one** feature in both views (the thick post's cap).
It could not establish a second — the antenna mast, crisp in the reference, is not visible in the
moving view at all, and finding the post cap in the moving view took five attempts with 50–170 px
errors, each caught only by rendering a crosshair. A two-pair answer it did produce scored 174°/48.8 m,
but that is not an accuracy measurement: two pairs give a **rank-1** cross-covariance, so the
rotation about their joining axis is free. The scorer now refuses <3 pairs or collinear ones.

**So: a model that cannot establish correspondences cannot do registration directly.** SIFT gets a
median of 306 correspondences per pair. The earlier 0.27 m result used SIFT for correspondences and
the model only for deciding *what to do with them* — which is a real contribution (0.94 → 0.27 m)
but not what "LLM 配准" means. Details in `FINDING_vlm_direct.md`.


## 2026-11-23, later: a model in the loop made a measured difference — with SIFT doing the correspondences

**Read the caveat first.** In this section "VLM loop" means a model choosing actions; the
correspondences came from SIFT, and the choice of tool in the single session reported here was
informed by a ground-truth decomposition (disclosed in `FINDING_vlm_loop.md` §4b). Whether a model
can do the registration *itself* was tested separately and fails — see the section at the top of
this file and `FINDING_vlm_direct.md`.

With that stated: `workbench/vlm_loop.py` gives a model real control — it renders a view, looks at
the PNG, and issues one action per turn, with every reason, feedback and state logged. On
`playground_v2_eth3d_0286` a five-step session — exclude the ~35-40% sky points, restrict to the
near field, solve translation on masked correspondences, fit a small rotation correction instead of
searching, stop — reached **1.28° / 0.052 m** against the baseline's 5.69° / 3.78 m.

Replaying those decisions across all 25 rotation-successful targets (`eval/replay_recipe.py`, no
per-target tuning, no GT in the loop):

| | rotation median | < 2° | lateral median | < 0.5 m |
|---|---|---|---|---|
| baseline A | 1.65° | 14/25 | 0.94 m | 10/25 |
| + sky mask and iterative correction | 1.75° | **18/25** | **0.27 m** | **20/25** |

It **solves the narrow-basin targets** that defeated every geometric refinement: playground
5.69°/3.11 m → **0.90°/0.01 m** (and v3, v4 likewise). Regressions are reported: facade v2/v4
eth3d_0209 0.19 → 0.97 m, terrace v3 0.27 → 0.37 m, delivery_area 0081 0.21 → 0.36 m.

Credit, stated carefully: the mechanisms (sky exclusion, small-rotation fit) are numerical steps a
scripted agent could call. What the model supplied was the decision to do them. Two GT-free
selectors for *when* to apply the correction both fail (spread: 23/25 accepted, no change; sky
fraction: 13/25). Details in `FINDING_vlm_loop.md`.


## Round of 2026-11-23: the error was rotation all along

Restricted to the 25 targets whose coarse rotation is already inside 20° (the only ones where a
translation question is well posed). All numbers are lateral translation error against the
reference transform, median over those 25.

| configuration | rotation median | rotation < 2° | translation median | < 0.5 m |
|---|---|---|---|---|
| FPFH/RANSAC/ICP end to end | 1.65° | 14/25 | 0.94 m | 10/25 |
| **rotation from FPFH, translation from SIFT correspondences** | 1.65° | 14/25 | **0.23 m** | **21/25** |
| + dominant-frame rotation candidates, selected by translation agreement | 1.49° | 16/25 | 0.26 m | 21/25 |
| oracle selection over the same candidates | 1.49° | 17/25 | 0.20 m | 22/25 |

Three things are established by measurement, each with a script:

1. **Translation is not the hard part.** With a correct rotation, ONE correct correspondence
   determines translation exactly — residual 0.00 m on 25/25 (`eval/rotation_leak.py`). With the
   estimated rotation the same perfect correspondence leaves 0.17 m, and that residual correlates
   with rotation error at +0.78. So the "lateral translation failure" that earlier rounds tried to
   fix was rotation error in disguise: 9.2° at an 11 m lever arm is 1.76 m.
2. **All translation-specific remedies were therefore aimed at the wrong variable** and their
   negative results (similarity ICP, anchor-precision sweeps, anchor-placement rules, co-visibility
   fixed-point iteration, depth-agreement search, footprint coverage) say nothing about those
   methods' merits. See `FINDING_rotation.md`.
3. **Classical appearance matching already supplies identity.** SIFT + ratio test + mutual check,
   backprojected with each view's own depth, reaches 0.27 m / 21-of-25. The hypothesis that
   semantic understanding supplies correspondence information that geometry cannot is **not
   supported on this data** and has been dropped. See `FINDING_identity.md`.

The method that this round actually validates is unglamorous but well-founded: **take rotation from
the geometric front end, take translation from appearance correspondences.** It is the single
largest improvement measured (0.94 → 0.23 m, 10 → 21 of 25) and it follows directly from the
decomposition.

### Where it still fails, and why that is the interesting part

Four targets remain above 0.5 m: delivery_area v3 eth3d_0057 (0.88 m) and three facade pairs
(1.21, 1.42, 1.42 m). Their rotations are 4.47°, 9.25°, 1.65°, 1.65°.

- delivery_area 0057 is fixed by oracle frame selection (0.88 → 0.06 m), so its rotation *is* among
  the frame candidates; only the selection missed it.
- **facade is not fixed by any rotation candidate**, and the oracle over the candidate set equals
  the coarse rotation. Its rotation is genuinely not recoverable from dominant planes, and the
  coarse estimate is 9.25° wrong. This is the one measured regime where an outside source of
  structural information — window rows, storey lines, verticals, i.e. things a model looking at the
  photographs can name — would have to supply what the geometry does not.

### The bound, and the one job the interactive loop can claim

`eval/rotation_basin.py`, 25 targets:

- **achievable rotation precision: median 1.83°, range 0.65°–3.75°** (ICP started at the truth)
- **basin half-width: median 5°, range 2°–13°**
- rotation refinement + one anchor: translation 2.18 m → **0.20 m**, under 0.5 m 6/25 → **20/25**

Three regimes fall out, and they need different things:

| regime | examples | what it needs |
|---|---|---|
| floor small, basin wide | office v3 (0.99°/12°), courtyard (1.83°/12°), delivery (1.11°/4°) | already solved by the front end |
| floor large | courtyard 0000 (2.54°), office v4 (3.61°), delivery 0057 (3.75°) | nothing helps; a data limit, report as such |
| **floor small, basin narrow** | **playground (0.65°/±2°), coarse is 5.7° off** | **needs a rotation initialisation good to ~2° — the interactive loop's actual job** |

The 0.5 m target needs ≤1.4° at a 20 m lever arm, and the data supports 1.83° median. So the target
is inside capability up to ~12 m levers and at/beyond it at 20 m. Facade's 20.5 m pairs are
**precision-limited, not method-limited** — that distinction should be stated wherever their results
appear.

This also replaces the earlier general hypothesis with a narrow, falsifiable one: the loop's value
is not "beats one-shot matching" (the data does not support that), it is **"rescues narrow-basin
cases where good precision is reachable but the front end cannot start close enough."** Playground
is the test, and it is falsifiable — if iterative rotation adjustment cannot bring playground inside
±2°, the claim fails.

### The sharpest form of the same result

The coarse rotation (1.65° median) is already **at** the measured floor (1.83° median), so chaining
refinements can only add noise — it improved rotation monotonically in **1 of 25** targets.
Comparing each target's coarse rotation to its own floor and its own basin:

- **16/25 have ≤1° headroom** — done; 0.20–0.23 m is their limit.
- **5/25 have headroom and are inside their basin** — recoverable locally (office v3: 5.84°→0.99°).
- **4/25 have headroom and are outside their basin**: playground ×3 (start 5.7°, basin 2°) and
  facade v3 eth3d_0209 (start 9.25°, basin 4°). **These are exactly the four failing targets**, and
  a rotation initialisation accurate to a couple of degrees is both necessary and sufficient.

Per view count, for the validated method (rotation from FPFH/ICP, translation from SIFT):

| subset | n | rotation median | < 2° | translation median | < 0.5 m |
|---|---|---|---|---|---|
| v2 | 6 | 1.49° | 4/6 | 0.22 m | **6/6** |
| v3 | 10 | 2.44° | 4/10 | 0.26 m | 7/10 |
| v4 | 9 | 1.49° | 6/9 | 0.20 m | 8/9 |

Not monotonic in view count: the subsets are nested, and v3 happens to contain the harder pairs.

### Correction carried into this round

`Cloud.as_o3d()` returns `transformed()`, i.e. points already at the injected start pose, not the
own frame. Two experiments were invalidated by treating it as own-frame; their conclusions are
withdrawn and the corrected versions are recorded in `FINDING_rotation.md`. Scripts now use
`cloud.points` explicitly.


### Historical record begins here

Everything from this point down predates the three dated sections above and is kept for audit. The
current conclusions are the three sections at the top of this file and the `FINDING_*.md` documents;
where they disagree, the Finding documents win.

> **CORRECTION (2026-11-23):** Earlier rotation/translation scores in this file and in the
> first handoff are invalid. The scorer compared cumulative M to incremental A, and the
> original COLMAP relative-pose formula also used the wrong direction. The current task,
> GT transforms, and rescore files have been regenerated. Use only
> `mvrecon/out/baseline_rescored.json`, `agent_rescored.json`, and the corrected handoff.
> Pre-fix files are preserved under `mvrecon/archive/pre_pose_fix/` for audit, not as valid
> results. Details and the office-v2 forensic reproduction are in
> `mvrecon/out/correction_audit.json`.

## What is built and verified

| component | state | evidence |
|---|---|---|
| Data isolation | done | `agent_data/` has RGB + K + own-frame cloud only; `eval_data/` holds COLMAP poses |
| Candidates | 7 scenes x 10 views | `bench/build_clouds.py`, all 287 ETH3D cached images available |
| Nested subsets v2/v3/v4 | done | `bench/subsets.json`, baselines 0.10-7.04 m, overlap 0.12-0.76 |
| GT transforms | done | `eval_data/gt_transforms.json`, cam-j -> cam-ref relative pose |
| Tool smoke test | PASS | roundtrip 1.7e-16, ICP recovery 2e-15, KDtree returns real points |
| Workbench | PASS | transform/undo (exact restore 0.0), region pick returns real points, ref protected |
| Baseline A (FPFH+ICP) | **0/6 on office** | median rot error 102 deg against a true 20 deg |

## Key finding so far

Traditional global registration fails badly on these mono-depth clouds. On office, where the
true relative rotation is only 20.3 deg and the baseline is 0.285 m, FPFH+RANSAC+ICP gives
102 deg. The clouds are shape-noisy enough that local features do not match. That is the
gap the interactive agent is supposed to close, and it is now measured.

## What is NOT done

- **The ablation (fixed vs active viewpoint selection)** — not run, and now of lower value: the
  narrow-basin targets it was meant to address were solved by the mask + correction route without
  any viewpoint change.
- **Baseline B as originally specified** (model one-shot correspondences + the same solver) — the
  direct test in `FINDING_vlm_direct.md` shows the model cannot establish the correspondences, so B
  would fail for that reason rather than for any property of the solver.
- **Visualisations** (before/after fused clouds per method).
- **No-overlap refusal tests** — the specification's explicit refusal cases were never constructed.
- A **learned** correspondence model (not a VLM) as a stronger front end.

## Honest note on scope

The specification is large (7 sections, three baselines, an ablation, a failure taxonomy). What has
been established by measurement is narrower than the specification and mostly negative:

1. the residual error is **rotation**, not translation (`FINDING_rotation.md`);
2. the achievable rotation precision on these clouds is **1.83° median**, and 21/25 targets are
   already at their limit;
3. classical SIFT supplies identity correspondences better than the oracle single anchor expected
   (`FINDING_identity.md`);
4. a **model cannot do the registration itself** — it fails at correspondence establishment
   (`FINDING_vlm_direct.md`);
5. but a model *supervising* the pipeline produced the largest single improvement measured:
   sky masking + a fitted rotation correction, 0.94 → 0.27 m median, 10/25 → 20/25 under 0.5 m
   (`FINDING_vlm_loop.md`).

Those five together are the result. The remaining work is the visualisations, the refusal cases, and
a learned matcher — not another attempt at the interactive loop, which has now been tried both ways.

## CRITICAL METHODOLOGICAL PROBLEM found

Measuring pairwise overlap at IDENTITY (no transform applied at all):

| scene | pair | overlap@identity | true baseline | true rotation |
|---|---|---|---|---|
| courtyard | 0008\|0004 | **0.986** | 2.635 m | 1.7 deg |
| delivery_area | 0052\|0048 | 0.821 | 3.054 m | 38.0 deg |
| electro | 0111\|0106 | 0.573 | 0.199 m | 2.1 deg |
| facade | 0177\|0169 | 0.603 | 6.072 m | 14.4 deg |
| office | 0135\|0130 | **1.000** | 0.285 m | 20.3 deg |
| playground | 0274\|0270 | 0.560 | 1.452 m | 37.4 deg |
| terrace | 0236\|0233 | 0.881 | 0.607 m | 32.4 deg |

Every pair overlaps 56-100% **before any alignment**. A monocular depth model predicts
depth *relative to its own camera*, so each view's cloud sits in a similar frustum-shaped
volume around its own optical axis. Two such volumes overlap heavily even when the cameras
are metres apart and rotated tens of degrees.

### Why this invalidates the current setup

- "overlap 0.65 at identity" means a registration method that does nothing already scores
  well, and my baseline-C run showed exactly that: initial overlap 0.65, and ICP *reduced*
  it to 0.60.
- The real transform (0.2 m, 2.1 deg for electro) is smaller than the cloud noise, so
  there is almost no signal for any method to recover.
- Any experiment run on this data would report numbers that mean nothing.

### What is actually needed

The clouds must be placed in a **common frame with a known, large displacement** before the
task exists. Options:

1. **Apply the true camera poses to the clouds at build time** and then *remove* the agent's
   knowledge of them. The agent then sees clouds that are genuinely misaligned, and the
   task is to recover the relative pose. The eval side keeps the transform it applied.
2. **Use a multi-view model** to produce clouds in a shared frame — but the spec forbids
   using multi-view models to pre-align (each image must be independently inferred).

Option 1 preserves the spec's independence requirement: inference is still per-image, and
the misalignment is introduced deliberately by the harness rather than by a joint model.
The agent never learns the applied transform.

This must be fixed before any further baseline or agent runs.

## FIX: the task is now a real registration problem

Two errors were found and fixed:

1. **Raw mono clouds overlap 56-100% at identity.** Each view is predicted relative to
   its own camera, so frustums coincide. No registration signal exists in the raw setup.
   Fix: the harness places each cloud in the reference frame using its TRUE pose and then
   injects a deliberate unknown displacement D (25-70 deg, baseline-scaled translation).
   Per-image inference is untouched — only the harness's presentation changes.

2. **The recovery transform was `D @ inv(T)`; it must be `T @ inv(D)`.** Derived and
   checked numerically (a point transformed by T must equal the same point transformed by
   A after D: A D = T  =>  A = T D^-1). With the wrong form the "true" transform made
   overlap worse; with the right form it improves it.

### Verified after the fix

| scene | tag | overlap at start | overlap at true transform |
|---|---|---|---|
| electro | v2 | 0.169 | **0.579** |
| electro | v3 | 0.212 | 0.534 |
| playground | v2 | 0.036 | **0.347** |
| playground | v3 | 0.043 | 0.327 |
| delivery_area | v2 | 0.060 | 0.353 |
| terrace | v2 | 0.376 | 0.639 |
| office | v4 | 0.689 | 0.931 |
| courtyard | v3 | 0.193 | **0.086** (worse) |

courtyard is the exception: its true relative transform is nearly identity (1.7 deg), so
injecting D and undoing it leaves the cloud in a state the noise cannot distinguish.
courtyard should be excluded or reported separately as a near-identity case.

### What is ready to run

- The task is real: start overlap 0.02-0.69, inject 12-71 deg, recoverable to 0.09-0.93.
- Workbench, agent tools, budget, undo: all verified.
- Baseline A measured on the OLD (invalid) setup only; must be re-run on the task.
- Baselines B and C, the ablation, and the failure taxonomy are not done.

## First valid baseline comparison (electro, on the task)

| subset | method | rot err (deg) | post-overlap |
|---|---|---|---|
| v2 | identity | 97.1 | 0.169 |
| v2 | **baseline A** | **46.5** | **0.677** |
| v3 | identity | 65.6 / 100.4 | 0.212 / 0.007 |
| v3 | **baseline A** | **32.7 / 41.1** | **0.666 / 0.646** |
| v4 | identity | 107.9 / 87.4 / 102.9 | 0.169 / 0.005 / 0.392 |
| v4 | **baseline A** | **55.1 / 51.1 / 54.3** | **0.649 / 0.619 / 0.460** |

Baseline A works but does not converge: rotation error stays at 33-55 deg where the
recoverable answer is near 0, and translation error remains metres-scale (1.3-8.6 m).
Overlap improves from near-zero to ~0.65, i.e. it finds *a* alignment but not *the*
alignment.

That is the headroom the interactive agent is supposed to exploit, and it is now
quantified on a real task.

## Remaining work, honestly

- Baselines B (VLM one-shot correspondences) and C (VLM closed loop) are not implemented.
  Running them requires a VLM in the loop; this session has a vision model available
  (me), so C can be run as a manual closed loop and recorded, but B requires an
  independent one-shot answer.
- The ablation (fixed vs active viewpoint) is not done.
- Only one scene (electro) has been evaluated end to end; the other six need the same run.
- Failure taxonomy is not populated.

## Time estimate

Substrate, GT, tooling, workbench, protocol, and baseline A: done in this session.
B/C + all scenes + ablation: comparable effort again, and B needs a VLM call path that
is not wired yet.

## Baseline table (complete, 5 scenes x 3 subsets x 2 methods)

| method | n pairs | median rot err | median overlap | fraction < 20 deg |
|---|---|---|---|---|
| identity (do nothing) | 26 | 85.8 deg | 0.178 | **0.0%** |
| baseline A (FPFH+RANSAC+ICP) | 27 | 66.4 deg | 0.619 | **0.0%** |

Per-scene medians (rot err / overlap):

| scene | identity | baseline A |
|---|---|---|
| electro | 97.1 / 0.169 | 46.5 / 0.677 |
| playground | 92.0 / 0.036 | 79.2 / 0.505 |
| office | 83.4 / 0.689 | 59.6 / 0.188 |
| terrace | 84.2 / 0.452 | 48.8 / 0.732 |
| delivery_area | 117.2 / 0.060 | 143.0 / 0.506 |

**No method ever gets inside 20 degrees.** Baseline A improves nearest-neighbour overlap
substantially (0.178 -> 0.619 on average) while rotation stays at 66 degrees, i.e. it
finds *an* alignment, not *the* alignment. On delivery_area A is worse than identity
(143 vs 117 deg), which is a plain registration failure.

This is the measured headroom the interactive agent is meant to close, and it is
established on a task with known ground truth rather than on a setup where "do nothing"
already scores 0.6-1.0 overlap.

## Artifacts

```
mvrecon/bench/       build_clouds, select_subsets2, build_task, verify_task, identity_overlap
mvrecon/workbench/   wb.py (state machine), agent_tools.py (budgeted tool surface)
mvrecon/eval/        protocol.py (task GT, scoring, baseline A)
mvrecon/out/         tools_smoke.json, baseline_A.json, baseline_table.json
mvrecon/eval_data/   task.json (T_true, D_injected), gt_transforms.json  [eval only]
mvrecon/agent_data/  10 views per scene: RGB + K + own-frame depth        [agent only]
```

## Closed-loop agent: implemented, and it does NOT beat baseline A

`workbench/run_agent.py` implements a real closed loop over the budgeted tool surface:
observe stats -> coarse centroid -> region correspondences -> ICP refine -> check/undo ->
stop. It is scored by the same protocol as baseline A through `eval/score_agent.py`.

| method | n | median rot err | median overlap | frac < 20 deg |
|---|---|---|---|---|
| identity | 26 | 85.8 | 0.178 | 0% |
| baseline A (FPGFR+ICP) | 27 | 66.4 | 0.619 | 0% |
| **closed loop (scripted policy)** | 3 | **65.6** | **0.293** | **0%** |

On electro: v2 56.9 deg / 0.566 overlap; v3 65.6 / 0.096; v4 107.9 / 0.293.

So the loop runs, spends the budget, and produces an alignment — but it is no better than
the classical baseline and on v3/v4 it is worse. The policy uses no semantics at all: it
picks the same central uv box in both images and solves Kabsch on those points, which is
exactly the "fixed correspondences then ICP" pattern the spec warns against.

### What this establishes

1. The harness is real: budget, undo, trace, state, scoring all work.
2. The task is genuinely hard: no method tested reaches 20 deg.
3. A scripted non-semantic loop is **not** sufficient. Any claim that interactivity helps
   must come from a policy that actually uses image content to propose correspondences
   between specific structures — not from the same uv box in both views.

### Not done

- Baseline B (VLM one-shot correspondences) and a VLM-driven C. The scripted policy is a
  harness test, not the semantic agent the study is about.
- The other five scenes under the closed loop.
- The active-vs-fixed viewpoint ablation.
- Failure taxonomy.
