# MV-Recon handoff package

Few-view 3D registration task built on frozen monocular metric depth.

## Contents

```
data/                 the actual data (7 ETH3D scenes, 10 views each)
  <scene>/images/     1/4-scale RGB
  <scene>/depth16/    float16 metric depth, 1/4 scale
  <scene>/clouds/     PLY point clouds, own camera frame, metric, RGB
  <scene>/meta.json   original-resolution intrinsics
  <scene>/task.json   v2/v3/v4 subsets + injected displacement
  <scene>/gt_transforms.json   ground-truth relative poses
  README.md           conventions, task definition, measured baselines
  baseline_results.json
preview/              start-vs-aligned renders for 4 scenes (v3 subsets)
wb.py                 the 3D workbench (state, transform, undo, pick, residual)
agent_tools.py        budgeted tool surface the agent gets
protocol.py           task ground truth + scoring + baseline A
score_agent.py        scores a closed-loop run
render_preview.py     regenerates the previews
export.py             regenerates the package from the raw project
```

## Quick start

```python
import json, numpy as np, open3d as o3d
scene = 'office'
task = json.load(open(f'data/{scene}/task.json'))['subsets']['v2']
ref, tgt = task['ids'][0], task['ids'][1]

def load(sid):
    return o3d.io.read_point_cloud(f'data/{scene}/clouds/{sid}.ply')

# the state a method actually sees: true placement + unknown displacement
src = load(tgt); src.transform(np.asarray(task['D_injected'][tgt]))
dst = load(ref)

# what a method must recover:
A = np.asarray(task['T_true'][tgt]) @ np.linalg.inv(np.asarray(task['D_injected'][tgt]))
```

## Two things worth knowing before using this

1. **The clouds already overlap 56-100% at identity** if you do not apply `D_injected`.
   That is why the task injects it: without a deliberate misalignment there is no
   registration signal, because every single-image cloud is predicted relative to its own
   camera and the frustums coincide.
2. **No tested method reaches 20 degrees.** Identity: 85.8 deg median. FPFH+RANSAC+ICP:
   66.4 deg. A scripted closed loop: 65.6 deg. The classical baseline improves overlap
   (0.18 -> 0.62) without finding the right rotation, so overlap alone is not a success
   criterion.

## Provenance

Depth: DA3-metric (frozen), each image inferred independently.
Poses and scene geometry: ETH3D COLMAP SfM, metric by calibration targets. Reference
reconstruction, not error-free truth.
Images: ETH3D DSLR (CC BY-NC-SA per the dataset terms).
