# The bottleneck is ROTATION, not translation (2026-11-23)

This supersedes the framing of `FINDING_identity.md` and the interpretations in
`FINDING_translation.md`. The measurements in those documents are correct; what they were
measuring is not what I said they were.

## The measurement

`eval/rotation_leak.py`. For each of the 25 rotation-successful targets, take a **perfect
correspondence** — a reference point that the moving view genuinely observes, so the anchor is
exact by construction — and solve translation twice, once with the baseline's estimated rotation
and once with the true rotation:

| rotation used | median lateral error | under 0.5 m |
|---|---|---|
| `R_est` (what any method can compute) | 0.17 m | 21/25 |
| `R_true` (rotation leakage removed) | **0.00 m** | **25/25** |

With correct rotation the translation is recovered **exactly** — not approximately, to machine
precision — from a single correspondence. There is no depth bias, no scale bias, no degeneracy
penalty left. And the residual with `R_est` tracks the rotation error almost perfectly:
`corr(log lateral, log rotation error) = +0.780`.

## What this means for everything measured this round

**1. The "lateral translation failure" was rotation error wearing a translation costume.**
A rotation error θ about an axis perpendicular to the lever arm moves the far end of the lever by
`|p|·sin θ`, which is indistinguishable from a translation error in the same direction. At the
levers in this data (3–20 m) the numbers line up exactly:

| target | rotation error | lever | predicted | measured lateral (SIFT / VLM anchor) |
|---|---|---|---|---|
| facade v3 eth3d_0209 | 9.2° | 11.0 m | 1.76 m | 1.19 / 1.14 m |
| facade v3 eth3d_0169 | 1.6° | 20.5 m | 0.57 m | 1.06 / — |
| playground v2 | 5.7° | 6.0 m | 0.60 m | 0.42 / — |

**2. This explains why every translation-specific remedy failed.** Similarity ICP, anchor
precision sweeps, anchor-placement rules, co-visibility fixed-point iteration, depth-agreement
search, footprint-coverage search — all were trying to correct a DOF that was already exactly
determined, while the actual error sat in a variable none of them touched. Their failure is
evidence about the wrong hypothesis, not about those methods.

**3. It unifies the earlier degeneracy result rather than contradicting it.** Planar overlap means
the rotation *about the plane normal* is poorly constrained. That rotation error then appears as
lateral translation. So the 13× lateral-error gap between planar and non-planar overlaps
(2.727 m vs 0.209 m) and the +0.780 rotation correlation are the same phenomenon described in two
coordinate systems. The normal-diversity measure was, without my realising it, measuring rotation
observability.

**4. The self-consistency negatives survive, but for a sharper reason.** A depth-agreement test
compares surfaces; a surface at a small rotation error is *still the same surface*, locally, over a
wide range. So the objective's insensitivity to rotation error is the same insensitivity measured
as "width at 90% of peak = 1.25 m" and "18/25 wrong poses score above the true one". Those
experiments are still valid; what they demonstrate is that agreement objectives cannot see
rotation error, which is now known to be the only error that matters.

## The corrected problem statement

> In sparse-view registration from frozen monocular depth, translation is not the hard part — one
> correct correspondence determines it exactly. **Rotation is the hard part**, it is limited to
> ~5° median (up to 9.2°) by feature-based coarse registration, and at 6–20 m lever arms that
> costs 0.5–1.8 m of apparent translation. The accuracy target (0.5 m at 20 m) therefore requires
> **rotation error below ~1.4°**, an order of magnitude better than the current front end.

## What this makes worth doing

The whole method's value now concentrates on precise rotation estimation. Two concrete routes:

- **Plane-based / Manhattan-frame alignment.** Architectural and indoor scenes are dominated by
  large planes whose normals are estimated from thousands of points and are therefore very
  accurate — far more accurate than the rotation implied by sparse feature matches. Aligning the
  dominant orthogonal frames of the two clouds, using the coarse feature rotation only to resolve
  the sign/permutation ambiguity, directly attacks the failing variable with the most reliable
  signal available.
- **Semantic structure.** Window rows, storey lines, edges and verticals give parallel-line
  constraints on rotation that are precise and are exactly what a VLM can identify from the
  photographs.

Both are measurable against the same 25 targets with the same protocol, and both are now
motivated by a decomposition rather than a guess.

---

# Addendum: a convention bug invalidated two experiments, and the corrected answer

## The bug

`Cloud.as_o3d()` returns `self.transformed()` — the points *already displaced by the injected
start pose D*, not the cloud's own frame. Two scripts treated that return value as own-frame
points and then applied `T_true` on top of `D`. (`eval/rotation_profile.py`, `eval/rotation_basin.py`)
`eval/plane_rotation.py` had the milder form: it estimated the moving cloud's dominant axes in the
D-rotated frame.

Consequence: my conclusion that "ICP drifts 78° even when started at the exact transform, so the
geometry is unstable" was **wrong**. ICP was handed a cloud in the wrong frame. The correct
statement is that point-to-plane ICP is merely fragile here; point-to-point at a 0.2 m threshold is
well behaved.

The affected scripts now use `cloud.points` (own frame) explicitly. The scripts from earlier in the
round were checked and were not affected — they used only `depth`, `K` and reference-side points.

## Corrected: does the geometry actually constrain rotation?

`eval/rotation_profile.py`, fixed. Hold translation at its exact value, scan rotation about each
axis, and evaluate the overlap (fraction of moving points within 0.20 m of the reference). No
optimiser involved.

- overlap at the true rotation: **median 0.38** (0.08–0.77) — the clouds genuinely overlap
- **the true rotation is the minimum of the cost within 2° in 22–23 of 25 targets on every axis**
- a 2° rotation error costs about **2.4 percentage points** of overlap

So rotation *is* observable from these clouds, and the truth is a minimum of a real objective. The
minimum is shallow — this is a precision problem, not an identifiability problem.

## Corrected: the dominant-frame method, properly framed

`eval/plane_rotation.py`, fixed. Aligning each cloud's dominant orthogonal frame, resolving the
sign/permutation ambiguity by closeness to the coarse rotation:

| | rotation | translation | under 0.5 m |
|---|---|---|---|
| coarse (FPFH/ICP) | 1.65° median | 0.19 m | 20/25 |
| frame alignment | 1.86° median | 0.22 m | 19/25 |
| | improved 9, worsened 16 | improved 6 | |

Aggregate is a wash, but the distribution is bimodal and informative:

- **large wins**: delivery_area v3 4.47° → **0.27°**, translation 0.51 → **0.03 m**;
  electro v2/v3 1.28° → **0.36/0.42°**, 0.10 → **0.03/0.04 m**; courtyard v3 2.82° → 1.47°.
- **catastrophic losses**: terrace v3 2.06° → **29.7°**; facade v2/v3/v4 3.44/9.25/3.44° →
  **35/34/37°**.

The losses are all in *ambiguity resolution*, not in the frame signal: choosing the alignment
nearest to a coarse rotation that is itself several degrees off picks the wrong one entirely. In
the cases where the correct alignment was picked, the frame gives **sub-half-degree** rotation —
well inside the 1.4° needed for a 0.5 m target at a 20 m lever arm.

That makes selection, not signal, the thing to fix: generate all 24 alignments plus the coarse
rotation, refine each locally, and keep the one that actually fits (`eval/frame_candidates.py`).
Including the coarse rotation as a candidate makes it a strict improvement search — it can only do
better than the baseline, never worse.

---

# Addendum 2: plane normals — the information is there, the precision is not

Four attempts to turn the most accurate orientation primitive available (a plane normal fitted to
thousands of points) into a better rotation. All four are recorded because the pattern they show is
the result.

## The information exists

`eval/plane_match_rotation.py`, diagnostic column. Segment up to 8 planes per cloud by iterative
RANSAC, then ask how many plane pairs agree under the **true** rotation:

- median **7** agreeing plane pairs per target (angle < 5°)
- **0 of 14** targets where no plane pair agrees
- best agreement per target: **0.5–1.3°**
- facade, the scene nothing else fixes: **8–18** agreeing pairs, best 0.3–1.1°

So the planes correspond, they are shared between the views, and individual normals are accurate to
about a degree. The information needed to correct facade's 9.25° error is present in the data.

## Four ways of failing to use it

| approach | script | result |
|---|---|---|
| dominant-frame alignment, nearest to coarse | `plane_rotation.py` | bimodal: 0.27–0.42° when right, 23–37° when wrong; 9 improved / 16 worsened |
| 24 candidates selected by ICP fitness | `frame_candidates.py` | worse: 1.65→2.36° median, under-2° 14→8 |
| 24 candidates selected by translation agreement | `frame_plus_identity.py` | best of the three: 1.49° median, 16/25 under 2°, translation 0.26 m — but oracle is 17/25, so selection still leaves gain |
| global RANSAC over sign-free normals | `plane_match_rotation.py` | 0/14 improved; degenerates to 180° flips about a matched normal, which normals cannot distinguish |
| local IRLS from the coarse rotation | `plane_refine.py` | 1.65→3.43° median, 7 improved / 18 worsened |

`plane_refine.py` is the informative one. Local refinement from a start already within 9° is
well-conditioned and cannot wander far, yet it lands at a **2–4° precision floor**: it improves
precisely the targets whose coarse rotation was worse than that floor (playground 5.7→2.8°,
5.8→3.4°, 5.6→4.3°; facade 9.25→5.45°) and degrades every target that started better.

## The conclusion this supports

Individual plane normals are accurate to ~1°, but a rotation *fitted across* a set of matched
planes lands at 2–4°. The gap is the cost of assignment errors and of systematic normal biases
that differ between grazing-angle surfaces — least squares trades one against the other. The
accuracy target derived at the top of this document is **1.4°** (0.5 m at a 20 m lever arm).

So the precision of rotation recoverable from these clouds is the binding constraint, and it sits
right at the edge of what the task needs. That is a statement about the data, not about any
particular front end, and it bounds what any registration method — geometric, learned, or an
interactive agent — can achieve on these view pairs.

---

# Addendum 3: the floor, the basin, and what they say the interactive loop is for

`eval/rotation_basin.py` (frames fixed; point-to-point ICP, 0.20 m threshold, translation seeded
from one correspondence consistent with each starting rotation so only rotation is varied).
25 targets.

| quantity | value |
|---|---|
| achievable rotation precision (ICP started at the true rotation) | **median 1.83°**, range 0.65–3.75° |
| basin half-width (largest offset that still converges) | **median 5°**, range 2–13° |
| rotation after refinement from the coarse estimate | 1.65° → 2.07° median, improved 13/25 |
| translation with that rotation + one anchor | 2.18 m → **0.20 m**, under 0.5 m 6/25 → **20/25** |

## The three regimes, and the one the interactive loop owns

Reading the per-target rows together, the targets fall into three groups that need different things:

1. **Floor small, basin wide** — office v3 (floor 0.99°, basin 12°), courtyard (1.83°, 12°),
   delivery_area (1.11°, 4°). ICP started from the coarse estimate reaches the floor, and
   translation lands at 0.10–0.25 m. **Solvable by the existing front end.**
2. **Floor large** — courtyard eth3d_0000 (2.54°), office v4 (3.61°), delivery_area 0057 (3.75°).
   At a 10–20 m lever arm that is 0.5–1.3 m of translation no matter what. **A data limit; nothing
   helps, and results here should be reported as such rather than tuned.**
3. **Floor small, basin narrow** — playground: floor **0.65°** (the best of any scene) but basin
   only **±2°**, while the coarse estimate is 5.7° off. The precision exists; the initialisation
   cannot reach it. **This is the case where an interactive loop has a job that is both necessary
   and sufficient: deliver a rotation good to a couple of degrees, and local refinement does the
   rest.**

That third regime is the specific, measurable role the project's interactive 3D loop can claim —
not "the loop beats one-shot matching" in general, which the data does not support, but
**"the loop rescues the narrow-basin cases, where good precision is reachable but the front end
cannot start close enough."** Playground is the test case, and it is falsifiable: if iterative
human/VLM rotation adjustment cannot get playground inside ±2°, the claim fails.

## The bound to report

The requirement derived at the top of this document is **≤1.4° rotation for 0.5 m at a 20 m lever
arm**. Local geometric refinement achieves **1.83° median** on these clouds. Therefore:

- at lever arms up to ~12 m the accuracy target is inside the measured capability;
- at 20 m it is at or beyond it;
- every local refinement tried — ICP, plane-normal IRLS, frame alignment — lands at or above this
  floor, so it is the *practical* precision limit of this data. It is an upper bound on
  achievable precision rather than a proof of impossibility: a fundamentally better estimator
  could in principle beat it, but nothing tried here does.

Any paper claim on these scenes should be stated with the lever arm attached, and facade's 20.5 m
pairs should be reported as precision-limited rather than method-limited.

---

# Addendum 4: the stack is already at the floor, and exactly four targets are not

`eval/stage_chain.py` chained the refinements (coarse → plane-normal IRLS → point-to-point ICP),
with translation always re-solved from the same correspondences so only rotation varied:

| stage | rotation median | < 2° | translation median | < 0.5 m |
|---|---|---|---|---|
| coarse | 1.65° | 14/25 | 0.23 m | 21/25 |
| + plane refine | 4.18° | 5/25 | 0.47 m | 13/25 |
| + ICP | 2.71° | 3/25 | 0.29 m | 18/25 |

Rotation improved monotonically in **1 of 25** targets. Chaining refinements makes things worse, and
the reason is now clear from the floor measurement: **the coarse rotation (1.65° median) is already
at the practical precision floor (1.83° median)**. There is nothing to gain on most targets, so any
refinement can only add noise.

## Where any headroom exists at all

Comparing each target's coarse rotation with its own floor, and its own basin:

| target | coarse | floor | headroom | inside basin? |
|---|---|---|---|---|
| facade v3 eth3d_0209 | 9.25° | 1.92° | 7.33° | **no** (basin 4°) |
| playground v2 / v3 / v4 | 5.7° | 0.65° | ~5.0° | **no** (basin 2°) |
| office v3 | 5.84° | 0.99° | 4.85° | yes (basin 12°) |
| facade v2 / v4 0209 | 3.44° | 1.92° | 1.52° | yes (basin 4°) |
| facade 0169 pairs | 1.65° | 0.64° | 1.02° | yes (basin 5°) |

16 of 25 targets have ≤1° of headroom — they are done, and the measured 0.20–0.23 m translation is
their limit.

Of the 9 with more, 5 are inside their basin and are recoverable by local refinement (office v3
demonstrates it: 5.84° → 0.99°). **Four are outside their basin: playground ×3 (need ≤2°) and
facade v3 0209 (needs ≤4°).** Those four are precisely the targets that still fail, and for them a
better rotation *initialisation* is both necessary and sufficient — the precision is reachable, the
starting point is not.

That is the complete, measured statement of what remains. It is not "the method is bad"; it is
"twenty-one targets are solved to about 0.2 m, and four need a rotation guess accurate to a couple
of degrees that no geometric front end here produced."

## A ground-truth-free signal for the loop to use

Across the 75 stage observations, the spread of per-correspondence translations (median deviation
from the median translation) correlates with rotation error at **+0.426**; above the median spread
the rotation error is 4.25°, below it 2.03°. So the agent can tell a good rotation adjustment from
a bad one without any reference geometry — which is exactly the accept/reject feedback §4 of the
specification requires, and it is what the workbench's check-and-undo step should be built on.

## What this makes the interactive loop responsible for

A narrow, falsifiable claim, replacing the original general hypothesis:

> The 3D interactive loop earns its place only on targets where the achievable rotation precision is
> high but the basin of local refinement is narrow and the front end cannot start inside it —
> measurably, playground (floor 0.65°, basin 2°, start 5.7°) and facade v3 0209 (floor 1.92°, basin
> 4°, start 9.25°). Its job is to bring rotation inside the basin, not to the floor. Success is
> measurable: those four targets drop below 0.5 m, or they do not.

Everything else — including twenty-one targets already at 0.2 m — is not evidence for or against it.

---

# Addendum 5: the narrow-basin case resists every geometric route tried

Before attributing the narrow-basin job to interaction or semantics, the cheap geometric
alternative had to be tested: systematic multi-start. `eval/multistart_rotation.py`, 33 starting
rotations on an 8° grid within a 16° ball around the coarse estimate, ICP from each, selection by
translation spread (the best GT-free signal available, correlation +0.43).

| target | coarse | selected | oracle over the same 33 starts | floor |
|---|---|---|---|---|
| playground v2 | 5.69° | 4.09° | 4.09° | 0.65° |
| playground v3 | 5.78° | 5.48° | 3.52° | 0.65° |
| playground v4 | 5.58° | 5.46° | 4.64° | 0.65° |
| facade v3 0209 | 9.25° | 7.75° | 5.24° | 1.92° |
| facade v2 0209 | 3.44° | **8.59°** | 4.37° | 1.92° |
| office v3 | 5.84° | 3.89° | 2.81° | 0.99° |
| courtyard 0000 | 2.82° | **0.98°** | 0.98° | 2.54° |
| delivery 0057 | 4.47° | 4.81° | 3.25° | 3.75° |

Aggregate: coarse 5.64° / 6-of-8 under 0.5 m; selected 5.13° / **5-of-8**; oracle 3.80° / 5-of-8.

Two things are settled:

1. **Multi-start ICP does not reach the narrow-basin solutions.** Playground needs ≤2° and the best
   of 33 starts gives 3.5–4.6°; facade v3 needs ≤4° and the best gives 5.24°. An 8° grid cannot
   compensate for a 2° basin by starting ICP outside it.
2. **Selection by spread is unreliable and can be harmful** — facade v2 is made 5° worse than doing
   nothing (3.44° → 8.59°), i.e. the selector chose a candidate *below* the baseline.

Together with the earlier results, four different ground-truth-free signals have now been tested as
rotation selectors and all fail: ICP overlap fitness (`frame_candidates.py`), translation spread
(here and `frame_plus_identity.py`), depth agreement (`lateral_search.py`, 18/25 spurious maxima),
and footprint coverage (`coverage_refine.py`).

**That is the result.** The correct rotation is a minimum of these objectives when one is already
close to it, but none of them ranks the correct rotation highest when choosing among distant
candidates. Geometric self-consistency cannot tell the agent *which* rotation is right — only
whether a rotation near the right one is slightly better than its neighbours. Selection therefore
has to come from a signal that is not derived from agreement between the clouds at all, and the
only such signals available are the ones the images carry: named structure, parallel edges, window
rows, horizontals and verticals. That is what the interactive 3D loop is for, and this is now a
measured argument rather than an assumption.






## Correction to `FINDING_identity.md`

Its central claim — that the disambiguating signal must be *instance identity* — is **not
supported**. Identity is necessary (a wrong correspondence is fatal, since it directly sets
translation), but it is not the limiting factor: a single exact correspondence still leaves
0.17 m median because the rotation is wrong, and no amount of correspondence quality improves
that. The claim that self-consistency objectives fail stands, with the corrected explanation above.
