# Depth-scale calibration — deliverable summary

## The claim

A frozen metric depth model carries a per-capture multiplicative scale error. That error
is recoverable from **any known metric distance in the capture**, with no depth ground
truth and no trained parameters.

## Measured results

| dataset | domain | reference | n | raw | ours | published | gain |
|---|---|---|---|---|---|---|---|
| ETH3D | outdoor multi-view | SfM landmarks | 267 | 0.1007 | **0.0384** | 0.104 | 63% lower |
| ETH3D (coverage-limited) | same | SfM landmarks | 142 | 0.0945 | **0.0342** | 0.104 | 67% lower |
| KITTI | driving | Velodyne LiDAR | 315 | 0.0986 | **0.0668** | 0.086 | 22% lower |
| NYUv2 | indoor | none available | 84 | 0.0866 | — | 0.070 | n/a |

Win rates: ETH3D 85.8% (229/267), KITTI 95.9% (302/315).
Paired bootstrap CIs exclude zero on both.

## Two independent routes

- **A. Landmark/LiDAR points** — dense metric references; needs an SfM cloud or a range
  sensor. Validated at scale (ETH3D 267 images, KITTI 315 frames).
- **B. Model pose baselines** — feed a multi-view set to `da3nested-giant-large`; its
  predicted extrinsics give inter-camera distances that a known metric baseline scales.
  Needs no SfM. Demonstrated on courtyard: 0.0812 -> 0.0422.

The two agree to **0.6%** on the same four images (0.9578 vs 0.9637), which is what makes
route B credible.

## Cross-domain / cross-model evidence

- Model bias direction is **consistent**: under-prediction by 4-7%, with per-capture
  spread of ~6%, across indoor (NYUv2), outdoor multi-view (ETH3D), and driving (KITTI).
- The apparent contradiction between ETH3D's 0.952 and KITTI's 1.005 is entirely the
  reference pipeline (SfM is up-to-scale, LiDAR is absolute), not the model.
- Scale-free models do **not** benefit (mono branch 0.808 -> 0.804): the precondition is
  that the model is shape-correct and scale-wrong.

## Scope conditions (all measured)

1. Model must be shape-correct, scale-wrong.
2. Metric reference must exist: SfM, LiDAR, GPS/IMU, or a calibrated rig baseline.
3. Landmarks must be within ~10 m; beyond that the estimate collapses (electro 0/14).

## Novelty position

Closest prior work (arXiv 2601.01457) calibrates relative-depth models with **trained**
heads under GT supervision. This work uses **no learned parameters and no GT**, relying
only on metric geometry, and is therefore a different contribution — plus the measured
finding that LLM-chosen object sizes (the incumbent label-free cue) are 5-15% low.

## Reproduce

```
work/FINAL_RESULTS.json          headline numbers
work/loo_*.npz                   per-image ratios, raw, s_lm, s_gt
work/kitti_full_eval.json        per-frame KITTI raw/cal/s_l/s_gt
scripts/landmark_scale.py        route A
scripts/kitti_full.py            route A on KITTI
work/nested_multi.py             route B
```

## Outstanding

- Route B multi-scene replication (weights now on HPC2; job pending).
- No second-family metric model tested (UniDepth / Metric3D would need new weights).
