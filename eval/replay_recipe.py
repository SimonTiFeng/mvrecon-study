"""Replay the VLM agent's action sequence on every target (generalisation test).

The agent's decisions on playground v2 were:
  1. exclude sky-coloured points -- the monocular model fabricates far-field depth there, and it
     was ~35-40% of each cloud
  2. solve translation from correspondences restricted to that mask
  3. treat the residual as a small rotation error and fit the correction linearly

Steps 1 and 3 are the parts no geometric baseline in this project did. This replays exactly that
sequence, with no per-target tuning and no ground truth in the loop, to see whether it generalises
or was one lucky target. Mask = sky colour only (scale-free, so it means the same thing in every
scene); translation and correction are fitted from correspondences inside the mask.

Reported against the same 25 rotation-successful targets as every other experiment, with the
geometric baseline and the scripted loop alongside.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

_s = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
si = importlib.util.module_from_spec(_s)
_s.loader.exec_module(si)

TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
AGENT = json.loads((ROOT / 'out' / 'agent_rescored.json').read_text())


def sky_mask(scene, sid, cloud, stride=32):
    img = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                       cv2.COLOR_BGR2RGB)
    H, W = cloud.depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = cloud.depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs = ys[ok], xs[ok]
    col = img[ys, xs].astype(float)
    sky = (col[:, 2] > col[:, 0] + 25) & (col[:, 2] > 110) & (col[:, 1] > col[:, 0])
    return sky


def correction(Pr, Pm, R0, iters=8):
    """Small-rotation + translation fit, robust; returns (R_new, t_new)."""
    w = np.zeros(3)
    t = np.median(Pr - Pm @ R0.T, axis=0)
    for _ in range(iters):
        q = Pm @ R0.T
        A = np.zeros((3 * len(q), 6))
        b = (Pr - q).ravel()
        for i, (qx, qy, qz) in enumerate(q):
            A[3 * i:3 * i + 3, 0:3] = [[0, qz, -qy], [-qz, 0, qx], [qy, -qx, 0]]
            A[3 * i:3 * i + 3, 3:6] = np.eye(3)
        res0 = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
        sc = max(np.median(res0), 1e-3)
        wts = 1.0 / (1.0 + (res0 / (2.0 * sc)) ** 2)
        sol, *_ = np.linalg.lstsq(A * np.repeat(wts, 3)[:, None],
                                  b * np.repeat(wts, 3), rcond=None)
        w, t = sol[:3], sol[3:]
    th = np.linalg.norm(w)
    if th > 1e-12:
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        C = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    else:
        C = np.eye(3)
    return C @ R0, t


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp' or x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_base = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        gi, gj = si.load_rgb(scene, wb.ref), si.load_rgb(scene, sid)
        if gi is None or gj is None:
            continue
        k1, d1 = sift.detectAndCompute(gi, None)
        k2, d2 = sift.detectAndCompute(gj, None)
        if d1 is None or d2 is None:
            continue
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        m2 = bf.knnMatch(d2, d1, k=2)
        back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < 0.8 * b.distance}
        good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
        if len(good) < 12:
            continue
        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
        Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)
        keep = okr & okm
        # match-level sky exclusion, applied at the matched pixels in the colour photographs
        cr = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (wb.ref + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        cm = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        Hr, Wr = cr.shape[:2]
        Hm, Wm = cm.shape[:2]
        ur = np.clip(np.round(uv_r[:, 0]).astype(int), 0, Wr - 1)
        vr = np.clip(np.round(uv_r[:, 1]).astype(int), 0, Hr - 1)
        um = np.clip(np.round(uv_m[:, 0]).astype(int), 0, Wm - 1)
        vm = np.clip(np.round(uv_m[:, 1]).astype(int), 0, Hm - 1)
        pr_c = cr[vr, ur]
        pm_c = cm[vm, um]
        sky_r = (pr_c[:, 2] > pr_c[:, 0] + 25) & (pr_c[:, 2] > 110) & (pr_c[:, 1] > pr_c[:, 0])
        sky_m = (pm_c[:, 2] > pm_c[:, 0] + 25) & (pm_c[:, 2] > 110) & (pm_c[:, 1] > pm_c[:, 0])
        keep &= ~sky_r & ~sky_m
        if keep.sum() < 12:
            continue
        Pr_k, Pm_k = Pr[keep], Pm[keep]
        # the correction is a small-angle linearisation, so iterate it to convergence
        R_new, t_new = R_base, np.median(Pr_k - Pm_k @ R_base.T, axis=0)
        iters_used = 0
        for _ in range(4):
            Rn, tn = correction(Pr_k, Pm_k, R_new)
            step = rotdeg(Rn @ R_new.T)
            R_new, t_new = Rn, tn
            iters_used += 1
            if step < 0.05:
                break

        def spread(R, t):
            q = Pm_k @ R.T + t
            return float(np.median(np.linalg.norm(q - Pr_k, axis=1)))

        t_base = np.median(Pr_k - Pm_k @ R_base.T, axis=0)
        sp_base, sp_new = spread(R_base, t_base), spread(R_new, t_new)
        # accept the correction only if it makes the correspondences agree better (no GT used)
        if sp_new <= sp_base:
            R_sel, t_sel = R_new, t_new
        else:
            R_sel, t_sel = R_base, t_base
        row = {'scene': scene, 'tag': tag, 'sid': sid,
               'rot_base': float(rotdeg(R_base @ R_true.T)),
               'lat_base': float(np.linalg.norm((M[:3, 3] - t_true)[:2])),
               'rot_rep': float(rotdeg(R_new @ R_true.T)),
               'lat_rep': float(np.linalg.norm((t_new - t_true)[:2])),
               'rot_sel': float(rotdeg(R_sel @ R_true.T)),
               'lat_sel': float(np.linalg.norm((t_sel - t_true)[:2])),
               'spread_base': sp_base, 'spread_new': sp_new,
               'accepted': bool(sp_new <= sp_base), 'iters': iters_used,
               'n_pairs': int(keep.sum()),
               'skipped_sky': int((sky_r | sky_m).sum())}
        rows.append(row)
        print('%-14s%-4s%-12s rot %5.2f -> %5.2f -> %5.2f | lat %5.2f -> %5.2f -> %5.2f | %s'
              % (scene, tag, sid, row['rot_base'], row['rot_rep'], row['rot_sel'],
                 row['lat_base'], row['lat_rep'], row['lat_sel'],
                 'accept' if row['accepted'] else 'keep baseline'), flush=True)

    p = ROOT / 'out' / 'replay_recipe.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        for name, a, b in (('rotation', 'rot_base', 'rot_rep'), ('lateral', 'lat_base', 'lat_rep'),
                           ('rotation*', 'rot_base', 'rot_sel'), ('lateral*', 'lat_base', 'lat_sel')):
            A, B = g(a), g(b)
            print('%-9s median %5.2f -> %5.2f   improved %2d  worsened %2d'
                  % (name, np.median(A), np.median(B),
                     int((B < A - 0.05).sum()), int((B > A + 0.05).sum())))
        print('under 0.5 m lateral: %d -> %d (unconditional) -> %d (selected) of %d'
              % (int((g('lat_base') < 0.5).sum()), int((g('lat_rep') < 0.5).sum()),
                 int((g('lat_sel') < 0.5).sum()), len(rows)))
        print('selector accepted %d/%d corrections' % (int(g('accepted').sum()), len(rows)))
        print('rotation under 2 deg: %d -> %d of %d'
              % (int((g('rot_base') < 2).sum()), int((g('rot_rep') < 2).sum()), len(rows)))
        small = [r for r in rows if r['scene'] != 'playground']
        if small:
            print('excluding playground: lateral %d/%d under 0.5 m (was %d)'
                  % (int(sum(r['lat_rep'] < 0.5 for r in small)),
                     len(small), int(sum(r['lat_base'] < 0.5 for r in small))))
    print('wrote', p)


if __name__ == '__main__':
    main()
