"""How precise must a semantic correspondence be to fix the lateral DOF?

Rotation is already recovered by the geometric baseline. With R fixed, ONE 3D
correspondence determines the translation completely:

    t = p_ref - R p_target

So the question "would a VLM anchor help?" reduces to a precision question: if the VLM
picks the corresponding pixel to within delta pixels, what lateral error remains?

This is computed exactly, without any vision model, by:
  1. taking an identifiable point (high local curvature) in the reference cloud,
  2. mapping it through T_true to the moving view to get the true match,
  3. perturbing that pixel by delta px, re-backprojecting with the moving view's own depth,
  4. solving translation from the single (rotation-fixed) correspondence,
  5. measuring error against T_true.

Note: no absolute-size prior is involved, only correspondence. That is why this bound can
be adequate even though earlier VLM size estimates were 5-15% off.
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
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
DEG = {r['sid'] + r['scene'] + r['tag']: r for r in
       json.loads((ROOT / 'out' / 'degeneracy.json').read_text())}
RNG = np.random.default_rng(11)


def curvature_pick(pc: o3d.geometry.PointCloud, n=1):
    """Indices of the most 'corner-like' points: largest normal variation in a neighbourhood."""
    pts = np.asarray(pc.points)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    nor = np.asarray(pc.normals)
    tree = cKDTree(pts)
    score = np.zeros(len(pts))
    step = max(1, len(pts) // 4000)
    idx = np.arange(0, len(pts), step)
    for i in idx:
        nb = tree.query_ball_point(pts[i], 0.6)
        if len(nb) < 6:
            continue
        C = nor[nb].T @ nor[nb] / len(nb)
        ev = np.sort(np.linalg.eigvalsh(C))
        score[i] = 1.0 - ev[0] / max(ev.sum(), 1e-12)
    order = np.argsort(score)[::-1]
    return order[:n], score[order[:n]]


def project(pt_cam: np.ndarray, K: np.ndarray):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if pt_cam[2] <= 1e-6:
        return None
    return np.array([fx * pt_cam[0] / pt_cam[2] + cx,
                     fy * pt_cam[1] / pt_cam[2] + cy])


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
    deltas = [0, 1, 2, 5, 10, 20]
    trials = 12
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        s = x['score_corrected']
        if s['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        nd = DEG.get(sid + scene + tag, {}).get('normal_div', 1.0)
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c = wb.clouds[wb.ref]
        mov_c = wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_est = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        Ttrue = np.asarray(x['matrix_gt'], float)
        K = mov_c.K
        depth = mov_c.depth

        # candidate anchor: corner-like point of the reference cloud, its 3D world position
        idxs, _ = curvature_pick(ref_c.as_o3d(), n=1)
        p_ref_w = np.asarray(ref_c.as_o3d().points)[idxs[0]]
        # true match in the moving cloud's own (untransformed) frame
        p_mov = (p_ref_w - Ttrue[:3, 3]) @ Ttrue[:3, :3]
        uv_true = project(p_mov, K)
        if uv_true is None:
            continue
        entry = {'scene': scene, 'tag': tag, 'sid': sid, 'normal_div': nd,
                 'lateral_before': float(np.linalg.norm((M[:3, 3] - Ttrue[:3, 3])[:2]))}
        ok = True
        for d in deltas:
            errs = []
            for _ in range(trials if d else 1):
                ang = RNG.uniform(0, 2 * np.pi)
                uv = uv_true + d * np.array([np.cos(ang), np.sin(ang)])
                p2 = backproject(uv, depth, K)
                if p2 is None:
                    continue
                t = p_ref_w - R_est @ p2
                errs.append(float(np.linalg.norm((t - Ttrue[:3, 3])[:2])))
            entry[f'lat@{d}px'] = float(np.median(errs)) if errs else float('nan')
            if not errs:
                ok = False
        if ok:
            rows.append(entry)
            print('%-14s%-4s%-12s nd %.3f  before %6.2f | ' % (scene, tag, sid, nd,
                                                              entry['lateral_before'])
                  + '  '.join('%dpx:%.2f' % (d, entry[f'lat@{d}px']) for d in deltas),
                  flush=True)
    p = ROOT / 'out' / 'anchor_precision.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        print('\nn=%d' % len(rows))
        for d in deltas:
            v = np.array([r[f'lat@{d}px'] for r in rows])
            print('  correspondence error %2d px -> median lateral %.3f m   under 0.5 m in %d/%d'
                  % (d, np.median(v), int((v < 0.5).sum()), len(v)))
    print('wrote', p)


if __name__ == '__main__':
    main()
