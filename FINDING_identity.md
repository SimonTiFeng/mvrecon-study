# What the lateral-translation failure actually requires (2026-11-23)

This supersedes §6–§7 of `FINDING_translation.md`, one part of which was wrong. Corrected below,
with the script that decides each claim.

## Correction to my earlier claim

I previously wrote that "anchor visibility is the binding constraint" because 10/25 targets had
no usable anchor. That was an artefact of the *selection* procedure: `anchor_rules.py` picked the
single globally-flattest point and then asked whether it projected into the other view. When the
globally-flattest point happened to lie behind the other camera, the target was recorded as `nan`.

`eval/anchor_visibility.py` restricts the candidate pool to co-observed points *before* applying
the rule. Result: **0/25 targets lack a visible anchor.** The co-observed fraction is 4%–82% of
the cloud. Visibility is not scarce; only the selection was bad.

## The ceiling: one true identity correspondence

With rotation held at the baseline estimate, one 3D correspondence determines translation exactly
(`t = p_ref − R p_target`). Using an oracle-visible anchor:

| policy | median lateral | under 0.5 m |
|---|---|---|
| baseline (FPFH+RANSAC+ICP) | 0.94 m | 10/25 |
| oracle-visible, flattest | **0.13 m** | **20/25** |
| oracle-visible, **random** | 0.16 m | 20/25 |
| all in-frame, flattest | 0.14 m | 19/25 |

Two things follow.

1. **The information exists and one correspondence is enough.** A 7× reduction in median error,
   with the under-0.5 m count doubling.
2. **Which co-observed point you pick barely matters** (0.13 vs 0.16 m). The earlier apparent
   advantage of the `flat` rule over `corner` was really *avoid silhouettes*, which is a
   co-visibility effect. Playground makes this explicit: picking the flattest patch gives
   **4.07 m** while a random co-observed point gives **0.47 m**, because the flattest large patch
   in a playground is the ground plane at grazing incidence, where monocular depth is biased.

So the requirement is **identity** ("this is the same physical point in both images"), not
surface shape, not precision (`anchor_precision.py`: 0 px vs 20 px changes the answer by 0.02 m).

## Three self-consistency objectives all fail

If identity is what is missing, can any *geometric self-consistency* signal substitute for it?
Three were tested and all fail. These are the attempts that rule out the obvious alternatives.

### 1. Fixed-point iteration on co-visibility (`eval/anchor_est_loop.py`)

Project reference points through the *estimated* pose, keep those landing in-frame whose predicted
depth matches the moving view's own depth within τ, solve, repeat. 12 variants (τ ∈ {0.05, 0.10,
0.20}, 1 or 16 anchors, flat or random). Every variant: median 0.94–2.10 m, 7–12/25 under 0.5 m —
**no better than the baseline it started from**, and it barely moves over 4 rounds:
courtyard 3.31 → 3.28, electro 2.33 → 2.35, playground 3.11 → 3.12, facade 6.30 → 6.90.
It converges to a self-consistent wrong pose.

### 2. Why it cannot work: the agreement profile is not discriminative (`eval/selfconsistency_width.py`)

Sweep the translation along the direction of the true error and measure co-visibility agreement:

- **16/25 targets: the WRONG pose agrees at least as well as the true pose.**
- Median width at 90% of peak: **1.25 m**; 8/25 stay wide beyond 2 m.
- courtyard: 0.459 agreement at the true pose, 0.604 at the estimate — the wrong pose is
  *strictly more* consistent.

A large plane at a lateral offset is locally identical to itself. Depth agreement therefore cannot
tell the correct offset from a shifted one. This is the aperture problem re-entering through the
correspondence channel.

### 3. Searching the agreement objective instead of iterating it (`eval/lateral_search.py`)

Global coarse-to-fine 2D scan maximising agreement. Median 0.94 → **1.21 m**; 11 improved,
14 worsened; under 0.5 m unchanged at 10/25. Worse than doing nothing.

The reason it fails is the sharpest single number in this investigation: the objective has
**spurious maxima that beat the truth**. Comparing the search's best score against the score at
the true pose (from the width profiles) on all 25 targets:

- **18/25 targets: the search found a pose that scores above the true pose.**
- Median excess of that spurious peak: **+0.104** agreement.
- courtyard: 0.722 at a pose 2.74 m from the truth, versus 0.459 at the truth.
  electro v3: 0.847 at 2.75 m off, versus 0.766 at the truth.

The global maximum of the objective is not the correct pose, so no optimiser — local, global, or
learned — applied to this objective can recover it.


(Earlier, `eval/coverage_refine.py` showed the same for footprint overlap: median 2.33 → 2.73 m.)

## What this establishes

A precise, falsifiable statement, replacing the general hope that "interactive 3D helps":

> Under planar degeneracy, every signal derived from agreement between the two predicted clouds
> (depth consistency, footprint overlap, nearest-neighbour residual) admits poses that score
> above the correct one. Such signals cannot localise lateral translation regardless of the
> search or iteration applied. The disambiguating information must be **instance identity**:
> a statement that two observations are of the same physical thing, which is not a property of
> local geometry.

~~That is exactly the information a VLM can supply and a geometric pipeline cannot, and the oracle
ceiling quantifies the payoff (0.94 → 0.13 m, 10 → 20 of 25).~~

> **WRONG — corrected later in this document and in `FINDING_vlm_direct.md`.** Classical SIFT
> supplies this identity information at 0.27 m / 21-of-25, better than the single oracle anchor's
> coverage. And when the model was asked to supply correspondences itself, it could not
> (`FINDING_vlm_direct.md`). The oracle ceiling is a bound on *any* correspondence source, not
> evidence that a model is the source that can reach it.

It also fixes the VLM's task
precisely, and it is *not* the task one would have guessed:

- **not** "find the flattest / most corner-like region" (shape is the wrong criterion),
- **not** "check your alignment by depth agreement" (that check is the thing that fails),
- **but** "name a specific physical object or mark, decide whether both cameras see it, and point
  to it in both images".

## The control that decides whether semantics is needed — IT FAILS THE PREMISE

Identity is not inherently semantic — SIFT matches points by appearance. `eval/sift_identity.py`
ran exactly that control on the same 25 targets with the same predicted clouds and full-resolution
photographs: SIFT (4000 features, ratio 0.80, mutual check), backprojected with each view's own
depth, median 306 matches / 306 valid 3D pairs.

| policy | median lateral | under 0.5 m | improved vs baseline |
|---|---|---|---|
| baseline (FPFH+RANSAC+ICP) | 0.94 m | 10/25 | — |
| **SIFT, rotation from baseline** | **0.27 m** | **21/25** | 17/25 |
| SIFT, full rigid RANSAC | 0.27 m | 17/25 | 16/25 |
| SIFT, full similarity RANSAC | 0.22 m | 17/25 | 18/25 |
| oracle single anchor (ceiling) | 0.13 m | 20/25 | — |

**Classical appearance matching recovers essentially all of the available gain, and reaches more
targets under 0.5 m than the single oracle anchor.** The premise "semantic understanding supplies
identity that geometry cannot" is false on this data, at this scale, in this form. It has to be
dropped.

Two further facts from the table, both useful:

- **Rotation from SIFT is worse than rotation from the geometric baseline.** delivery_area:
  translation-only 0.28 m, but full rigid 1.27 m. So the two methods fail in complementary ways,
  and the best configuration measured anywhere in this investigation is *rotation from
  FPFH/ICP + translation from SIFT identity*: 0.27 m median, 21/25.
- **SIFT's failures are concentrated in one scene.** 3 of its 4 remaining failures are facade
  (0.49 / 1.19 / 1.06 m at v2 / v3 / v4). A facade is repeating windows: appearance matching
  aliases between identical instances, which is precisely the case where instance semantics
  ("the second window left of the door") should beat appearance.

## What survives, and the baseline gap this exposes

The claim that survives is much narrower and is now located in a specific regime:

> Under **repeated structure**, appearance-based identity is ambiguous between structurally
> identical instances, and the disambiguating information is instance semantics. Everywhere else,
> one-shot classical correspondence already matches the oracle ceiling.

This also exposes a hole in the evaluation design: `A_icp` (FPFH/RANSAC on the point clouds) is
not the strong conventional baseline — the conventional *image* pipeline (SIFT + known intrinsics
+ per-view monocular depth) is, at 0.27 m. It must be **added to baseline A**, not left to the
proposed method. Reporting 0.94 m as "conventional" would have been a strawman.

The VLM's honest task, therefore, is not "name a landmark". It is:

> name a landmark **that appearance cannot disambiguate**, or declare that none exists.

That is a much harder and much more interesting requirement, and it is exactly the case the
specification already asked for and the earlier experiments never tested: repeated structure,
and no-overlap refusal. Both are now the whole experiment.

