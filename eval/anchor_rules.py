"""Anchor quality, not correspondence precision, is the limiting factor.

The previous test showed that a perfect pixel correspondence still leaves large lateral
error on some targets. Hypothesis: the automatically chosen anchors sit on silhouettes,
where monocular depth is least reliable.

This compares, for the same targets, three anchor-selection rules and reports the lateral
error obtained from ONE anchor (rotation fixed from the geometric baseline):
  corner     - highest local normal variation (previous rule, likely near silhouettes)
  interior   - lowest local depth gradient above a curvature floor
  flat       - lowest local depth gradient, no curvature requirement
and, as a diagnostic only, the local depth reliability at the chosen pixel.
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


def local_stats(pts, nor, tree, i, radius=0.6):
    nb = tree.query_ball_point(pts[i], radius)
    if len(nb) < 6:
        return None, None
    C = nor[nb].T @ nor[nb] / len(nb)
    ev = np.sort(np.linalg.eigvalsh(C))
    curv = 1.0 - ev[0] / max(ev.sum(), 1e-12)
    z = pts[nb][:, 2]
    return curv, float(z.std())


def pick(pc, rule):
    pts = np.asarray(pc.points)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    nor = np.asarray(pc.normals)
    tree = cKDTree(pts)
    idx = np.arange(0, len(pts), max(1, len(pts) // 2500))
    cands = []
    for i in idx:
        c, zs = local_stats(pts, nor, tree, i)
        if c is None:
            continue
        cands.append((i, c, zs))
    if not cands:
        return None
    A = np.array([(c[1], c[2]) for c in cands])
    if rule == 'corner':
        score = A[:, 0]
    elif rule == 'interior':
        ok = A[:, 0] > np.quantile(A[:, 0], 0.6)
        score = np.where(ok, -A[:, 1], -1e9)
    else:  # flat
        score = -A[:, 1]
    k = int(np.argmax(score))
    return cands[k][0]


def project(pt, K):
    if pt[2] <= 1e-6:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([fx * pt[0] / pt[2] + cx, fy * pt[1] / pt[2] + cy])


def backproject(uv, depth, K):
    u, v = uv
    iu, iv = int(round(u)), int(round(v))
    H, W = depth.shape
    if not (2 <= iu < W - 2 and 2 <= iv < H - 2):
        return None
    z = float(np.median(depth[iv-1:iv+2, iu-1:iu+2]))
    if not np.isfinite(z) or z <= 0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])


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
        Ttrue = np.asarray(x['matrix_gt'], float)
        ref_pc = ref_c.as_o3d()
        pts = np.asarray(ref_pc.points)
        e = {'scene': scene, 'tag': tag, 'sid': sid,
             'before': float(np.linalg.norm((M[:3, 3] - Ttrue[:3, 3])[:2]))}
        for rule in ('corner', 'interior', 'flat'):
            i = pick(ref_pc, rule)
            if i is None:
                e[rule] = float('nan')
                continue
            p_ref = pts[i]
            p_mov = (p_ref - Ttrue[:3, 3]) @ Ttrue[:3, :3]
            uv = project(p_mov, mov_c.K)
            if uv is None:
                e[rule] = float('nan')
                continue
            p2 = backproject(uv, mov_c.depth, mov_c.K)
            if p2 is None:
                e[rule] = float('nan')
                continue
            t = p_ref - R_est @ p2
            e[rule] = float(np.linalg.norm((t - Ttrue[:3, 3])[:2]))
        rows.append(e)
        print('%-14s%-4s%-12s before %6.2f | corner %7.2f  interior %7.2f  flat %7.2f'
              % (scene, tag, sid, e['before'], e['corner'], e['interior'], e['flat']),
              flush=True)
    p = ROOT / 'out' / 'anchor_rules.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\nn=%d' % len(rows))
    for rule in ('corner', 'interior', 'flat'):
        v = np.array([r[rule] for r in rows if np.isfinite(r[rule])])
        b = np.array([r['before'] for r in rows if np.isfinite(r[rule])])
        print('  %-9s median lateral %.3f m  under 0.5 m %2d/%d  better than before %2d/%d'
              % (rule, np.median(v), int((v < 0.5).sum()), len(v), int((v < b).sum()), len(v)))
    print('wrote', p)


if __name__ == '__main__':
    main()
