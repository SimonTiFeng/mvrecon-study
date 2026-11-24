# Where the registration error actually is (2026-11-23)

Sequence of measurements after the coordinate/scoring fix. Each step is a checkable claim,
each with the script that produced it.

## 1. Rotation is largely solved; translation is not

`out/baseline_rescored.json`, 40 targets, FPFH+RANSAC+ICP:
median rotation error **5.0 deg** (25/40 inside 20 deg), median translation error
**2.67 m** (6/40 inside 0.5 m).

Restricting to the 25 targets whose rotation is already inside 20 deg: median translation
error **2.18 m**, only 6 of 25 inside 0.5 m. courtyard reaches 1.5 deg rotation with
3.31 m translation error. So translation is a separate failure, not a by-product of
rotation.

## 2. The error is lateral, not axial

`out/degeneracy.json`, translation error decomposed in the reference camera frame:
median axial 0.47 m vs median lateral 0.94 m; axial exceeds lateral in only 36% of
targets. This refutes the scale-coupling hypothesis (a per-view scale bias would push
the error along the optical axis).

## 3. Cause: planar degeneracy

Over the overlapping region, compute the smallest eigenvalue of the normal covariance
(normal diversity). Correlation with lateral error:
`corr(log lateral, log normal diversity) = -0.559`.

| group | median lateral error |
|---|---|
| normal diversity < 0.09 (plane-dominated) | **2.727 m** (n=12) |
| normal diversity >= 0.09 | **0.209 m** (n=13) |

A 13x difference. ICP's mean-distance cost is nearly flat along a dominant plane, so the
in-plane translation simply has no gradient.

## 4. Coverage does not rescue it

`out/coverage_refine.json`: maximising the overlap footprint instead of the mean distance,
coarse-to-fine over the plane, on the 15 degenerate targets: median lateral
2.33 -> **2.73 m**, improved 8/15. Not a reliable gain — the footprint objective is also
degenerate (one can maximise coverage by sliding into a dense region).

## 5. One anchor determines the DOF, and precision is not the issue

With rotation fixed, one 3D correspondence fixes the translation exactly:
`t = p_ref - R p_target`.

`out/anchor_precision.json`: perturbing the corresponding pixel from 0 to 20 px changes the
resulting median lateral error from **0.479 to 0.500 m** — essentially nothing. So
correspondence precision is not the bottleneck, and a VLM does not need sub-pixel accuracy.

But even a *perfect* correspondence left 6/13 unfixed, and some got much worse. So the
anchor's own quality matters more than the matching.

## 6. Anchor placement is the binding constraint

`out/anchor_rules.json`, three automatic selection rules, one anchor per target, applied
only where the anchor projects inside the moving view:

| rule | anchors available | median lateral | inside 0.5 m |
|---|---|---|---|
| corner (max normal variation) | 8/25 | 0.94 m | 10/25 |
| interior (curvature floor + flat) | 12/25 | 0.79 m | 11/25 |
| **flat (min local depth gradient)** | **15/25** | **0.29 m** | **14/25** |

`flat` is clearly best: median lateral error over the 25 degenerate-eligible targets drops
**0.94 -> 0.29 m**, and targets inside 0.5 m go **10 -> 14 of 25**.

Individual results with the flat rule: electro 2.33 -> **0.13**, electro 4.75 -> **0.28**,
courtyard 3.13 -> **0.25** m.

## 7. What this says about the research question

> **CORRECTED 2026-11-23 — see `FINDING_identity.md`.** The claim below that "anchor visibility
> is the binding constraint" is wrong: the `nan` cases were the *globally* flattest point landing
> behind the other camera, an artefact of selecting before restricting to co-observed points.
> Restricting first, 0/25 targets lack a visible anchor. The real requirement is identity, not
> visibility. The measured table in §6 stands; the interpretation in §7 does not.


The residual failure is now localised and characterised:

- **15 of 25** targets can be fixed by ONE anchor placed on a depth-reliable surface.
- **10 of 25** had no usable anchor at all — the projection fell outside the other view.
  That is a *view and landmark selection* problem, not a numerical one.
- Anchor placement quality dominates: corner anchors (near silhouettes, where monocular
  depth is worst) leave 0.94 m median, flat anchors 0.29 m.
- Five targets got worse with an anchor (delivery_area v3 2.00 -> 4.58, playground
  1.83 -> 2.51 and 0.94 -> 2.32, terrace 0.27 -> 0.78, office 0.13 -> 0.27), so an anchor
  must be *validated* before being trusted — which is exactly what the workbench's
  check-and-undo step is for.

That converts "does interactive 3D help?" into a concrete, testable claim:

**A VLM can help iff it selects landmarks that (a) are visible in both views and (b) sit

> **SUPERSEDED.** Two later results replace this: classical SIFT selects such landmarks numerically
> at 0.27 m / 21-of-25 (`FINDING_identity.md`), and when the model was asked to select them itself
> it could not establish enough verified correspondences to solve at all (`FINDING_vlm_direct.md`).
> The condition stated here is correct as an information statement; the attribution of that
> capability to a VLM is not.


on depth-reliable surfaces; the numerical solve itself is trivial and precision-insensitive.**

The purely geometric baseline cannot make either choice — it has no notion of "this is the
same object in both pictures" nor of "this surface is where the depth model is trustworthy".
