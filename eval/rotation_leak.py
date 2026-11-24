"""How much of the residual translation error is leaked rotation error?

Every translation experiment in this round held the rotation FIXED at the geometric baseline's
estimate. If that estimate is off by theta, then at a lever arm of |p| the translation is off by
about |p| * sin(theta) regardless of how good the correspondence is. At a 14 m facade a 9 deg
rotation error alone predicts ~1.7 m -- larger than the residuals being attributed to depth bias.

This isolates the two, using a PERFECT correspondence (the same physical point, i.e. the oracle
identity anchor). For each target:

  t_est  = p_ref - R_est  @ p_mov      (what the method can compute)
  t_true = p_ref - R_true @ p_mov      (rotation leakage removed, same anchor)

If t_true collapses to ~0 while t_est does not, the residual is rotation leakage, and no
correspondence method or depth correction can fix it -- the rotation estimate must improve.
If both stay large, the residual is in the depth itself (scale / distortion).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())


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
        R_true, t_true = T[:3, :3], T[:3, 3]
        p_w = np.asarray(ref_c.as_o3d().points)

        # oracle identity anchor: a reference point genuinely co-observed in the moving view
        p_m = (p_w - t_true) @ R_true
        z = p_m[:, 2]
        okz = z > 0.2
        fx, fy, cx, cy = mov_c.K[0, 0], mov_c.K[1, 1], mov_c.K[0, 2], mov_c.K[1, 2]
        H, W = mov_c.depth.shape
        u = np.full(len(p_w), np.nan)
        v = np.full(len(p_w), np.nan)
        u[okz] = fx * p_m[okz, 0] / z[okz] + cx
        v[okz] = fy * p_m[okz, 1] / z[okz] + cy
        frame = okz & np.isfinite(u) & np.isfinite(v) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        cand = np.where(frame)[0]
        if len(cand) == 0:
            continue
        iu = np.clip(np.round(u[cand]).astype(int), 1, W - 2)
        iv = np.clip(np.round(v[cand]).astype(int), 1, H - 2)
        zm = mov_c.depth[iv, iu].astype(float)
        good = np.isfinite(zm) & (zm > 0)
        rel = np.abs(zm - z[cand]) / np.maximum(z[cand], 1e-6)
        vis = cand[good & (rel < 0.05)]
        if len(vis) == 0:
            continue
        # spread the pick over the visible set so it is representative, not one lucky point
        pick = vis[np.linspace(0, len(vis) - 1, min(7, len(vis))).astype(int)]
        e_est, e_true = [], []
        for i in pick:
            pr = p_w[i]
            pm = p_m[i]
            t1 = pr - R_est @ pm
            t2 = pr - R_true @ pm
            e_est.append(np.linalg.norm((t1 - t_true)[:2]))
            e_true.append(np.linalg.norm((t2 - t_true)[:2]))
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'rot_err': s['rot_err_deg'],
                     'lat_est': float(np.median(e_est)),
                     'lat_true_rot': float(np.median(e_true)),
                     'lever': float(np.median(np.linalg.norm(p_m[pick], axis=1)))})
        r = rows[-1]
        print('%-14s%-4s%-12s rot %5.1f  lever %5.1f m | lat(R_est) %6.2f  lat(R_true) %6.2f  ratio %4.1f'
              % (scene, tag, sid, r['rot_err'], r['lever'], r['lat_est'], r['lat_true_rot'],
                 r['lat_est'] / max(r['lat_true_rot'], 1e-6)), flush=True)

    p = ROOT / 'out' / 'rotation_leak.json'
    p.write_text(json.dumps(rows, indent=1))
    a = np.array([r['lat_est'] for r in rows])
    b = np.array([r['lat_true_rot'] for r in rows])
    rot = np.array([r['rot_err'] for r in rows])
    print('\nn=%d' % len(rows))
    print('median lateral with R_est  %.2f m   (under 0.5 m %d/%d)'
          % (np.median(a), int((a < 0.5).sum()), len(a)))
    print('median lateral with R_true %.2f m   (under 0.5 m %d/%d)'
          % (np.median(b), int((b < 0.5).sum()), len(b)))
    print('corr(log lat_est, log rot) = %+.3f' % np.corrcoef(np.log1p(a), np.log1p(rot))[0, 1])
    if len(rows) > 3:
        big = rot >= np.median(rot)
        print('rotation error above median: lat_est %.2f -> lat_true %.2f m'
              % (np.median(a[big]), np.median(b[big])))
        print('rotation error below median: lat_est %.2f -> lat_true %.2f m'
              % (np.median(a[~big]), np.median(b[~big])))
    print('wrote', p)


if __name__ == '__main__':
    main()
