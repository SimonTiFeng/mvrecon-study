# Correspondence sweep on HPC2 (job 14069627, 2026-11-23)

First experiment of this project to run on the cluster. CPU partition `i64m512u` (geometry work —
no GPU taken from the quota), 1 node, 4 cores, finished in ~7 minutes, 3752 records.

Setup: `mvrecon/hpc/exp_corr.py` reads the corrected handoff package directly, so the remote job
needs only numpy + OpenCV. Everything at 1/4 resolution so photograph and depth are pixel-aligned.
Environment: `$HOME/mvrenv` (Python 3.8, cv2 4.10, numpy 1.24.4, scipy 1.10.1) — pip had to be
bootstrapped with the 3.8 get-pip script because the venv module ships without ensurepip and no pip
exists anywhere on the cluster.

Swept: 4 matchers (SIFT, AKAZE, ORB, SIFT∪AKAZE) × 3 masks (none, sky, sky+adaptive distance) ×
2 solves (rotation fixed at the front end's estimate; or an iterated small-rotation correction) ×
4 injected-outlier rates (0/10/20/30%), over all 40 targets in the task set.

## Pareto front — accuracy against cost

On the 25 rotation-successful targets (comparable with every earlier result), no injected outliers:

| matcher | rot median | < 2° | lateral median | < 0.5 m | cost median |
|---|---|---|---|---|---|
| **ORB** | 1.65° | 14/25 | 0.31 m | 20/25 | **165 ms** |
| AKAZE | 1.65° | 14/25 | 0.27 m | **21/25** | 475 ms |
| SIFT | 1.65° | 14/25 | **0.25 m** | **21/25** | 819 ms |
| SIFT∪AKAZE | 1.65° | 14/25 | 0.27 m | 21/25 | 1284 ms |

**The interesting result is cost, not accuracy.** Accuracy is saturated at 0.25–0.31 m and 20–21 of
25 for every matcher; nothing moves it. But ORB reaches SIFT's accuracy band at **one fifth of the
time**, and AKAZE matches SIFT's exact count at 58%. SIFT is Pareto-dominated by AKAZE on this data:
same 21/25, same 0.27 m, 42% cheaper. The union of two matchers is strictly worse than either —
more time, no accuracy.

## What does not matter here

**Masks barely move the correspondence results.** `none`, `sky` and `sky+far` differ by ≤0.02 m and
≤1 target on this subset. This contradicts nothing but qualifies the earlier cloud-level finding: the
sky mask mattered when the *registration objective* was point-cloud overlap, because fabricated
far-field surfaces were participating in that objective. At the correspondence level the sky points
simply fail to match, so filtering them afterwards changes little.

**The rotation correction is conditional.** It improves rotation precision (SIFT: 14 → 16 of 25
under 2°) but costs a little lateral accuracy (0.25 → 0.27 m), because it re-trades the pose against
the same noisy matches. On the full 40-target set, where the front end fails badly on several
targets, the correction is clearly worse than leaving rotation alone. It helps when the start is
already good; it does not rescue a bad start. That is the same narrow-basin story measured on the
clouds earlier, now seen from the correspondence side.

## Robustness to gross outliers

Injected by replacing a fraction of moving-frame correspondences with random displacements at half
the scene's median range:

| outlier fraction | corrected: rot | corrected: lat | lat < 0.5 m |
|---|---|---|---|
| 0% | 3.38° | 0.59 m | 217 |
| 10% | 3.26° | 0.64 m | 224 |
| 20% | 3.38° | 0.63 m | 225 |
| 30% | 3.18° | 0.62 m | 210 |

(over the full 40-target set, all matchers pooled). **Essentially flat to 30% gross outliers** —
the IRLS weighting handles them. This is a genuine property worth having: the correction does not
depend on a clean correspondence set.

## Cross-scene transfer

For SIFT / sky+far / rotation-fixed, median lateral error per scene, against the front-end baseline:

| scene | n | rot median | lat median | < 0.5 m | baseline lat |
|---|---|---|---|---|---|
| courtyard | 5 | 1.49° | **0.10 m** | 5/5 | 3.31 m |
| delivery_area | 5 | 1.48° | 0.28 m | 3/5 | 1.60 m |
| office | 6 | 78.65° | 0.35 m | 3/6 | 1.37 m |
| playground | 6 | 16.45° | 1.04 m | 3/6 | 3.31 m |
| electro | 6 | 46.88° | 1.05 m | 3/6 | 10.81 m |
| facade | 6 | 3.44° | 1.32 m | 2/6 | 1.08 m |
| terrace | 6 | 46.34° | 1.65 m | 3/6 | 1.92 m |

The spread is the result: **one scene type (courtyard) is solved to 0.10 m on every target, and
others are not**. Since the same matcher and the same parameters are used everywhere, this is
domain shift within a single dataset, not tuning. It is the strongest argument that a single
recipe will not carry across domains, and it is measured rather than asserted.

## Reproduce

```bash
ssh hpc2 'cd ~/mvrecon_hpc && source /opt/slurm/profile.slurm && sbatch run_corr.sbatch'
scp hpc2:~/mvrecon_hpc/results_<jobid>.jsonl out/
python hpc/analyse_corr.py out/results_<jobid>.jsonl out/corr_sweep.md
```

---

# Estimator comparison: reprojection beats 3D-3D (`eval/reproj_refine.py`)

Every solve above fits in 3D. But the measurements that produced the correspondences live in the
image and err roughly isotropically in pixels, so minimising reprojection error is the better-posed
objective. Robust 6-DOF fit (rotation vector + translation), sky-masked correspondences, same 25
targets:

| correspondences + solve | rot median | < 2° | lateral median | < 0.5 m | improved |
|---|---|---|---|---|---|
| SIFT, rotation fixed (3D) | 1.65° | 14/25 | 0.25 m | **21/25** | — |
| ORB, rotation fixed (3D) | 1.65° | 14/25 | 0.31 m | 20/25 | — |
| SIFT + 3D small-rotation correction | 1.77° | **16/25** | 0.27 m | 20/25 | 10 |
| **SIFT + reprojection** | 2.06° | 12/25 | 0.29 m | **21/25** | **22 of 25** |
| **ORB + reprojection** | 2.15° | 12/25 | **0.23 m** | 17/25 | 21 of 25 |
| AKAZE + reprojection | 3.48° | 10/25 | 0.47 m | 15/25 | 17 |
| SIFT + reprojection **with free scale** | 2.06° | 12/25 | 0.32 m | 18/25 | 15 |

Four things:

1. **Reprojection refinement is the best single estimator** on median lateral error (ORB 0.23 m,
   SIFT 0.29 m) and improves 21–22 of 25 targets individually. The 3D objective was leaving
   accuracy on the table.
2. **It depends on correspondence quality.** With AKAZE it collapses to 0.47 m. SIFT or ORB
   correspondences are required; the estimator cannot compensate for weaker matches.
3. **Freeing scale hurts** (0.32 m, 18/25) and the estimated scale is **1.0033** — scale was never
   the problem, and the extra degree of freedom only overfits. This is the third independent
   confirmation of that, after the early similarity-ICP divergence and the exact-translation result.
4. **The tail and the median disagree.** ORB+reprojection has the best median but only 17/25 under
   0.5 m; SIFT with rotation fixed has a worse median (0.25) and the best tail (21/25) plus the
   best rotation. Neither dominates, and which to use depends on whether the metric is typical or
   worst-case accuracy.

## Final Pareto set on the 25 comparable targets

| configuration | rot med | < 2° | lat med | < 0.5 m | cost |
|---|---|---|---|---|---|
| ORB, rotation fixed | 1.65° | 14 | 0.31 m | 20/25 | **165 ms** |
| AKAZE, rotation fixed | 1.65° | 14 | 0.27 m | **21/25** | 475 ms |
| SIFT, rotation fixed | 1.65° | 14 | 0.25 m | **21/25** | 819 ms |
| ORB + reprojection | 2.15° | 12 | **0.23 m** | 17/25 | ~250 ms |
| SIFT + reprojection | 2.06° | 12 | 0.29 m | **21/25** | ~900 ms |

No configuration is dominated on all four axes. If cost dominates, ORB with rotation fixed. If
worst-case accuracy dominates, SIFT with rotation fixed. If typical error dominates, ORB with
reprojection. The 0.23–0.31 m band is where this data stops, for every combination tried.

---

# Selection between the two candidates is still unsolved (`eval/select_by_reproj.py`)

Every ground-truth-free selector tried earlier failed: ICP fitness, translation spread, depth
agreement, footprint coverage. The sweep showed the correction helps a good start and hurts a bad
one, so choosing between them should be worth something — if a signal could tell them apart.

The candidate that looked promising was **spatial coherence of the reprojection residual**: fit each
candidate on half the correspondences, then measure on the held-out half how much residual energy a
low-order field in image coordinates explains. A rotation error displaces the projection
systematically, so its residual varies smoothly; depth-model noise does not. RMS cannot see that
difference because it is swamped by the hundreds-of-pixels depth error.

32 targets, split-half, candidates A (rotation from the front end) and B (iterated correction):

| rule | rot median | < 2° | lateral median | < 0.5 m |
|---|---|---|---|---|
| A, always | 2.82° | 14/32 | 0.29 m | 22/32 |
| B, always | 2.09° | 13/32 | 0.36 m | 20/32 |
| select by reprojection RMS | 2.07° | 15/32 | 0.31 m | 22/32 |
| select by coherence | 1.83° | 17/32 | 0.29 m | 19/32 |
| oracle over the two | 1.48° | 19/32 | 0.22 m | 23/32 |

**I first read the coherence row as a win, and it is not.** Checking it directly:

- coherence agrees with the better-rotation candidate in **17 of 32** targets — a coin flip
- the two candidates' coherence values differ by only **0.012** at the median, even when their
  rotation errors differ by more than 0.5° (n=29): the statistic has essentially no separation
- when it picks A it is right about as often as wrong (rotA − rotB median **+0.06°**); the
  improved rotation figure comes from picking B often (22/32) on targets where B happens to be
  better, not from discriminating between them

So the apparent 1.83° is an artefact of how the choices happened to fall, not a signal. With 32
targets a handful of flips moves the median, which is exactly the trap.

**Conclusion: no selector tried in this project reliably chooses between pose candidates.** That is
now five failures — fitness, spread, depth agreement, footprint, coherence — and it is consistent
with the earlier finding that these objectives do not rank the correct pose first. The honest
current practice is to pick one candidate by policy (A for the best worst-case translation, B for
the best rotation) and report both, rather than to claim a selection rule that does not exist.

---

# Multi-view fusion makes it much worse (`eval/multiview_fuse.py`)

The nested v3/v4 subsets contain 3–4 images, but every result registers each view only to the
reference. For k views there are k(k-1)/2 pairs, so chaining through an intermediate view
(pref · via-k) and combining robustly (chordal mean of rotations, median translation) ought to
average down the error. It does the opposite, badly:

| | rotation median | lateral median | < 0.5 m |
|---|---|---|---|
| direct (each view against the reference) | **2.43°** | **0.28 m** | **27/40** |
| fused with chains through other views | 18.93° | 1.87 m | 13/40 |

improved 7 targets, **worsened 26**. Representative failures: courtyard v3 1.70° → 38.05°,
playground v3 0.34° → 42.35°, office v3 0.26° → 6.77°.

The reason is structural, not a bug: a chain through view k is only as good as the *worst* of its
two links, and the pairwise front end between non-reference views is much weaker than against the
reference (different overlap, worse initialisation, and Open3D repeatedly warning that too few
feature correspondences survived the mutual filter). Composing two noisy transforms compounds their
errors; averaging cannot recover that unless the estimates are independent *and* each is good.

Fusion would only pay if the chains could be filtered for quality — which is the same selection
problem this project has now failed to solve five times. The number of constraints is not the
bottleneck; the inability to tell a good estimate from a bad one is.

## The number worth keeping from this experiment

The *direct* column is the best aggregate measured anywhere in the project: **front-end
initialisation → sky-masked SIFT → robust reprojection refinement gives 27 of 40 targets under
0.5 m, median 0.28 m**, against 6 of 40 for the geometric front end alone. That is the current
method, and it is what the fusion experiment accidentally confirmed at full scale.
