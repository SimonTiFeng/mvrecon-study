# Can the model do the registration itself? Attempted, and it fails at correspondence (2026-11-23)

The question: not "can a model choose actions" (the earlier loop answered that), but "can the model
produce the registration" — its own correspondences, its own pose, nothing substituted.

## Protocol

`eval/vlm_direct.py`, built so the answer cannot be mistaken for something else:

- `answer` writes the model's correspondences (pixel positions plus a name for what the feature is)
  to a file and does nothing else
- `score` **refuses to run unless that file already exists**, computes the transform from the
  model's pairs alone (Kabsch; translation is exactly determined once rotation is), and reports the
  error
- Because the transform is computed only from the named pairs, a rotation cannot be smuggled in:
  every contribution is a claim of the form "this object here is that object there", checkable
  against the crops
- `mark` renders both crops with crosshairs at the model's proposed positions, so each claim can be
  falsified before it is used

## What happened on playground v2 (eth3d_0282 reference, eth3d_0286 moving)

**One feature was localised and verified in both images**: the rounded cap of the thickest wooden
post, reference ≈(3357, 272), moving ≈(4400, 600) region. The reference position is exact — the
crosshair sits on the cap apex.

**The second feature could not be established.** The obvious candidate was the white antenna mast on
the office roof. It is crisp and unique in the reference at ≈(3648, 1075). In the moving view it is
**not visible** — four separate zooms and crosshair checks placed it in empty sky, on the roofline,
or among branches. It is occluded by the parapet or genuinely absent at this viewpoint.

Localising the post cap in the moving view alone took five attempts, with errors of 50–170 px in the
reported position, each caught only by rendering the crosshair and looking. The failures were not
random: the grid labels on a downscaled panel are hard to read, and each zoom introduced its own
scale factor to get wrong.

## The two-pair solve, and why it is not merely inaccurate

Before verification, a two-pair answer was committed and scored: **174.4° rotation, 48.8 m
translation.** That number is not a measurement of the model's accuracy. Two point pairs cannot
determine a rotation at all: the centred cross-covariance has **rank 1** (verified numerically), so
the rotation about the axis joining the two points is completely free and SVD returns an arbitrary
element of that family — here, a flip.

The scorer now refuses fewer than three pairs, or collinear ones, rather than emitting a number
that looks like a result.

## The finding

**The model-direct route fails at correspondence establishment, not at pose estimation.** The model
can find and precisely locate a named structure in one image. It cannot reliably re-locate that same
structure in a second image of the same scene from a different viewpoint, even with iterative zoom
and crosshair verification. One verified pair in an hour of careful work, where the second feature
turned out not to exist in the other view at all.

That is exactly the task classical matching solves in milliseconds: SIFT produces a median of 306
correspondences per pair on these targets, and 21/25 register under 0.5 m from them.

## What this implies for the earlier positive result

It confirms the suspicion that prompted this test. The VLM loop that reached 0.27 m
(`FINDING_vlm_loop.md`) did **not** obtain its correspondences from the model — it ran SIFT and the
model chose *what to do with them* (discard sky, then fit a small rotation correction instead of
searching). The model's contribution was a decision about processing, not a correspondence.

So the honest summary of the whole project on this axis:

| role | who can do it |
|---|---|
| producing dense correspondences between two sparse views | classical matching; the VLM cannot |
| deciding that a third of the cloud is fabricated sky | the VLM (no geometric method tried noticed it) |
| deciding to fit a rotation correction rather than search | the VLM |
| the numeric solve | trivial algebra either way |

A model that cannot establish correspondences cannot do registration directly. Its value here is
supervision of the pipeline, and that value is real (0.94 → 0.27 m) but it is not what "LLM 配准"
usually means.

## What would make the model-direct route work

Not a better prompt. Either (a) a fine-tuned correspondence model — a learned matcher, which is not
a VLM and is already the classical baseline's job — or (b) a task where correspondence is
unambiguous by construction, e.g. scenes with a single named object and no repeated structure. The
targets in this benchmark are neither.

## Conclusion (accepted 2026-11-23)

This negative result is the project's final position on the question, and it was accepted rather
than pursued further through a region-box interface. The reasoning for stopping here, recorded so it
can be revisited rather than silently dropped:

- The failure is at correspondence, and a region-box interface would not remove that — it would move
  the same judgement one level up ("which box corresponds to which") while leaving the model to make
  the same visual re-identification that already failed. Tolerating my localisation error is not the
  same as supplying the missing capability.
- The information needed to do better is not absent; it is simply the classical matcher's job, and
  SIFT already does it at 0.27 m.
- Further attempts would be searching for a prompt or interface that makes one model succeed on one
  scene, which is not the question the project needs answered.

What stands, therefore:

1. A model cannot perform this registration directly on these scenes — it fails at correspondence.
2. A model *supervising* the pipeline is worth 0.94 → 0.27 m and 10/25 → 20/25 under 0.5 m, through
   two decisions a geometric pipeline never made: discard fabricated sky, and fit a small rotation
   correction rather than search for one.
3. The largest single lever on accuracy in the whole project is not the model at all — it is the
   rotation estimate, and the achievable precision of these clouds is 1.83° median
   (`FINDING_rotation.md`).

Boundary of claim 1: one model, one target, hand-built tooling. It establishes that a VLM cannot be
assumed to supply correspondences; it does not establish that no vision model can.

