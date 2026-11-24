"""Ceiling of the anchor route: does visibility, not surface choice, cap it?

anchor_rules.py showed the `flat` rule takes median lateral error 0.94 -> 0.29 m but only
15/25 targets had any anchor at all (the chosen point projected outside the moving view).

This separates the two causes:
  A) the ANCHOR IS NOT VISIBLE in the moving view   -> no correspondence exists
  B) the anchor is visible but sits on a bad surface -> correspondence exists but is noisy

Visibility is decided on the evaluation side only (this is an ORACLE upper bound, not an
agent capability): a reference point is visible in the moving view iff projecting it through
T_true lands in the image and the moving view's own depth model agrees with that surface
within 5% of depth. A point that projects in-frame but onto a different surface is OCCLUDED,
not visible -- that distinction is what a naive in-frame test gets wrong.

For each target we then solve translation from ONE anchor with rotation fixed at the
baseline's estimate, under four selection policies:
  in_frame   - any candidate that projects inside the image (the previous rule's pool)
  visible    - oracle-visible pool
  visible_flat   - oracle-visible AND flat rule
  visible_random - oracle-visible, random pick (control: is the surface rule needed?)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(7)


def project(pts, K):
    z = pts[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pts[:, 0] / np.where(np.abs(z) < 1e-9, np.nan, z) + cx
    v = fy * pts[:, 1] / np.where(np.abs(z) < 1e-9, np.nan, z) + cy
    return u, v, z


def local_flatness(pts, k=14):
    """sqrt of the smallest covariance eigenvalue over k nearest neighbours.

    Near zero on a locally planar patch: the surface has no thickness, so a point placed
    there survives monocular depth noise. Large near silhouettes and thin structure.
    """
    if len(pts) < k + 1:
        return np.full(len(pts), np.inf)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    P = pts[idx] - pts[idx].mean(1, keepdims=True)
    C = np.einsum('nki,nkj->nij', P, P) / k
    return np.sqrt(np.clip(np.linalg.eigvalsh(C)[:, 0], 0.0, None))


def main():
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        s = x['score_corrected']
        if s['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_est = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        t_true = T[:3, 3]
        K = mov_c.K
        H, W = mov_c.depth.shape

        p_w = np.asarray(ref_c.as_o3d().points)
        # true position of each reference point in the moving camera's own frame
        p_m = (p_w - t_true) @ T[:3, :3]
        u, v, z = project(p_m, K)
        in_frame = (np.isfinite(u) & np.isfinite(v) & (z > 0.2)
                    & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2))

        # visibility = in frame AND the moving view's depth agrees with the true surface
        vis = np.zeros(len(p_w), bool)
        idx = np.where(in_frame)[0]
        if len(idx):
            iu = np.clip(np.round(u[idx]).astype(int), 1, W - 2)
            iv = np.clip(np.round(v[idx]).astype(int), 1, H - 2)
            z_model = mov_c.depth[iv, iu].astype(float)
            ok = np.isfinite(z_model) & (z_model > 0)
            rel = np.abs(z_model - z[idx]) / np.maximum(z[idx], 1e-6)
            vis[idx[ok & (rel < 0.05)]] = True

        row = {'scene': scene, 'tag': tag, 'sid': sid,
               'before': float(np.linalg.norm((M[:3, 3] - t_true)[:2])),
               'n_frame': int(in_frame.sum()), 'n_visible': int(vis.sum()),
               'frac_visible': float(vis.sum() / max(len(p_w), 1))}

        sub = np.where(in_frame)[0]
        if len(sub) > 3000:
            sub = RNG.choice(sub, 3000, replace=False)
        pts_sub = p_w[sub]
        flat = local_flatness(pts_sub)

        def solve(i):
            p_ref = p_w[i]
            uu, vv = np.round(u[i]).astype(int), np.round(v[i]).astype(int)
            if not (1 <= uu < W - 1 and 1 <= vv < H - 1):
                return np.nan
            zm = float(np.median(mov_c.depth[vv - 1:vv + 2, uu - 1:uu + 2]))
            if not np.isfinite(zm) or zm <= 0:
                return np.nan
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            p2 = np.array([(uu - cx) / fx * zm, (vv - cy) / fy * zm, zm])
            t = p_ref - R_est @ p2
            return float(np.linalg.norm((t - t_true)[:2]))

        vis_sub = np.array([i for i in sub if vis[i]], int)
        row['in_frame'] = solve(sub[int(np.argmin(flat))]) if len(sub) else np.nan
        if len(vis_sub):
            fv = local_flatness(p_w[vis_sub])
            row['visible'] = solve(vis_sub[int(np.argmin(fv))])
            row['visible_random'] = float(np.median(
                [solve(vis_sub[RNG.integers(len(vis_sub))]) for _ in range(8)]))
        else:
            row['visible'] = np.nan
            row['visible_random'] = np.nan
        if len(sub):
            row['flat_any'] = solve(sub[int(np.argmin(flat))])
        rows.append(row)
        print('%-14s%-4s%-12s before %6.2f  frame %5d vis %5d (%.0f%%) | flat_any %6.2f  visible %6.2f  rand %6.2f'
              % (scene, tag, sid, row['before'], row['n_frame'], row['n_visible'],
                 100 * row['frac_visible'], row['flat_any'], row['visible'],
                 row['visible_random']), flush=True)

    p = ROOT / 'out' / 'anchor_visibility.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\nn=%d' % len(rows))
    N = len(rows)
    print('%-16s %8s %8s %10s' % ('policy', 'median', '<0.5m', 'applicable'))
    for k in ('before', 'flat_any', 'visible', 'visible_random'):
        v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
        if len(v):
            print('%-16s %7.2fm %5d/%d %8d/%d'
                  % (k, np.median(v), int((v < 0.5).sum()), len(v), len(v), N))
    print('coverage: no visible anchor in %d/%d targets'
          % (int(sum(1 for r in rows if r['n_visible'] == 0)), N))
    print('wrote', p)


if __name__ == '__main__':
    main()
