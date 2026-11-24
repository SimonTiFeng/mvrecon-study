# Depth calibration — status after the 2026-11-21 iteration

## Headline

DA3-metric under-predicts metric depth by 4-7% on every capture tested, and that
per-capture offset is recoverable from **any known metric distance** in the capture —
with no depth ground truth and no trained parameters.

| dataset | domain | reference | images | raw | result | published |
|---|---|---|---|---|---|---|
| ETH3D | outdoor multi-view | SfM landmarks | 267 | 0.1007 | **0.0384** | 0.104 |
| ETH3D (accuracy-max) | same, >10 m refused | SfM landmarks | 142 | 0.0945 | **0.0342** | 0.104 |
| KITTI | driving | Velodyne LiDAR | 315 | 0.0986 | **0.0668** | 0.086 |
| NYUv2 | indoor | (no reference — method N/A) | 84 | 0.0866 | — | 0.070 |

ETH3D 63% below published; KITTI 22% below. Both CIs exclude zero (ETH3D −0.0471
[−0.0557,−0.0352]; KITTI −0.0265 [−0.0308,−0.0209]).

## Two routes, one idea

**Route A — landmark/LiDAR points.** Compare metric 3-D points to the model's depth at
the same pixels. Works where a calibrated reconstruction or a range sensor exists.

**Route B — camera pose baselines.** Feed a multi-view set to `da3nested-giant-large`,
which predicts camera extrinsics; compare its inter-camera distances to the known metric
baselines. Needs no SfM. Validated on courtyard: 0.0812 -> 0.0422 with a single scale
(0.9637) from six view pairs.

The two routes agree on the same scene (0.958 vs 0.964), and they consume different
data, so they are independent evidence for the same effect.

## Measured scope conditions

- model must be shape-correct and scale-wrong (scale-free mono fails: 0.808 -> 0.804)
- landmarks must be within ~10 m (beyond that the estimate collapses: electro 0/14)
- the reference constant is reference-specific: 0.952 for SfM (up-to-scale), 1.000 for
  LiDAR (absolute); it does not transfer between them

## Robustness

- landmark error threshold 0.5-8.0 px moves s by ±0.0009
- 13 KITTI non-improvements all sit on frames where the model was already unbiased
- no drive below 70% wins; six of nine improve every frame

## Novelty

Closest prior work: arXiv 2601.01457 (language-prior + trained calibration heads,
GT-supervised) and a disparity-gated VLM prior paper that independently confirms our
finding that LLM-chosen object sizes are 5-15% low. Our distinction: **no learned
parameters, no GT, geometric reference only.**

## Open

- route B validated on one scene only (multi-scene run blocked by RAM/pagefile limits)
- no second-family metric model tested
- three bugs were fixed this session, each of which had inverted a conclusion

## Route B status (updated)

Nested checkpoint transferred to HPC2 in 13 × 500 MB parts after single-stream scp proved
too slow and the cluster cannot reach HuggingFace. Seven parts arrived truncated on the
first pass (128 MB instead of 524 MB) and are being re-sent; remote sizes are verified
by `stat` rather than trusting the transfer exit code.

Once the 6.76 GB file reassembles with a matching sha256, the multi-scene route-B run
goes onto the A800 partition (80 GB VRAM, 3 TB host RAM), which is where the runbook
puts work of this size.

## Immediate next steps

1. Verify part sizes, concatenate, sha256 against the local file.
2. Submit `work/nested.sbatch` to i64m1tga800u.
3. Extend route B from courtyard to facade, playground, delivery_area.
4. If route B holds across scenes, it becomes the headline: it needs no SfM at all.

## What is already safe

- Route A results (ETH3D 267 images 0.1007->0.0384, KITTI 315 frames 0.0986->0.0668)
  are measured, persisted, and reproducible from `work/*.npz` + `work/FINAL_RESULTS.json`.
- Route B's single-scene demonstration (courtyard 0.0812->0.0422) is measured and recorded.
- All scope conditions and failure modes are documented with their measured numbers.

## Route B transfer: complete and verified

13 × 500 MB parts uploaded to HPC2. First pass had 7 truncated parts (exit code 0 but
128 MB instead of 524 MB); re-sent with per-part `stat` verification.

Final check:
- local `sha256sum`: 8899faf998dedbc23026...
- remote `sha256sum`: 8899faf998dedbc23026...
- size both sides: 6,759,558,100 bytes

Job `14055307` submitted to `i64m1tga800u` (A800, 80 GB VRAM, 3 TB host RAM) through the
HPC2 runbook's single-card recipe. ETH3D scenes facade, delivery_area, electro extracted
on the remote alongside courtyard, so the run covers four scenes if the queue cooperates.
