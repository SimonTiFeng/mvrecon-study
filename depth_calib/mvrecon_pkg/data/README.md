# MV-Recon handoff data

## What this is

Per-image point clouds produced by a frozen metric depth model (DA3-metric), plus the
sources, the registration task built on them, and its ground truth.

7 ETH3D scenes, 10 candidate views each. Each view is inferred INDEPENDENTLY — no
multi-view model was used to pre-align anything.

## Layout

```
data/<scene>/
  images/<sid>.jpg        1/4-scale RGB      (originals are 6048x4032)
  depth16/<sid>.npy       float16 depth, 1/4 scale, METRES, z along optical axis
  clouds/<sid>.ply        point cloud in its OWN camera frame, metric, with RGB colour
  meta.json               original-resolution intrinsics per view
  task.json               v2/v3/v4 subsets + injected displacement D per view
  gt_transforms.json      T_true per view (cam_j -> cam_ref), and subset composition
```

## Coordinate conventions

- Cloud points: `X = (u-cx)/fx * z`, `Y = (v-cy)/fy * z`, `Z = z`, in the ORIGINAL pixel grid.
  Intrinsics in `meta.json` are for the original resolution.
- Depth is **z along the optical axis**, not ray distance, not inverse depth.
- Depth already has the model's focal/300 conversion applied, so it is in metres
  (to within the model's own bias).

## The registration task

Each subset in `task.json` has:
- `ids`: the views in the subset, `[0]` is the reference
- `T_true[sid]`: the true transform taking view `sid`'s own-frame points into the
  reference frame:  `p_ref = T_true[sid] @ p_sid`
- `D_injected[sid]`: an unknown displacement the harness applied on top, so the starting
  state is genuinely misaligned. The agent should NOT be told this.

A method must recover

```
A = T_true[sid] @ inv(D_injected[sid])
```

so that `A @ (D @ p) = T @ p`.

### Why D exists

Two raw single-image clouds already overlap 56-100% at identity (measured), because each
is predicted relative to its own camera and the frustums coincide. Without an injected
displacement the task has no signal. Inference is still per-image; only presentation
changes.

## Scenes and difficulty (measured)

| scene | v2 baseline | overlap at start | recoverable overlap |
|---|---|---|---|
| electro | 0.19 m | 0.169 | 0.579 |
| playground | 1.45 m | 0.036 | 0.347 |
| delivery_area | 3.05 m | 0.060 | 0.353 |
| office | 0.29 m | 0.689 | 0.931 |
| terrace | 0.61 m | 0.376 | 0.639 |
| courtyard | 2.64 m | 0.088 | 0.110 (near-identity true transform; treat separately) |
| facade | 6.07 m | 0.138 | 0.379 |

## Baselines already measured

| method | pairs | median rot err | median overlap | frac < 20 deg |
|---|---|---|---|---|
| identity | 26 | 85.8 deg | 0.178 | 0% |
| FPFH+RANSAC+ICP | 27 | 66.4 deg | 0.619 | 0% |
| scripted closed loop | 3 | 65.6 deg | 0.293 | 0% |

No method tested reaches 20 degrees.

## Caveat on the reference

Poses come from ETH3D's COLMAP SfM, metric by calibration targets. Treat as "reference
reconstruction", not error-free truth: SfM scale is fixed by targets, and the mono depth
model itself carries a 4-7% scale bias.
