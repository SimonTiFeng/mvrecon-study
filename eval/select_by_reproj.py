"""Choose between the two pose candidates with held-out correspondences, not with a guess.

The sweep showed the small-rotation correction helps a good start and hurts a bad one, and no
ground-truth-free signal tried so far could tell which case a target is in. But there is a
principled one that was not tried: split the correspondences, fit each candidate on one half, and
score it on the other half by reprojection error. That is ordinary cross-validated model selection,
it needs no reference geometry, and it uses the objective the estimator comparison just showed to be
the better-posed one.

Candidates per target, both fitted on half the matches:
  A  rotation from the front end, translation from the matches
  B  iterated small-rotation correction fitted on the same half
Selection: lower reprojection RMS on the held-out half.
Reported against always-A, always-B, and the oracle choice.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

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


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def fit_3d(Pr, Pm, R0, iters=6):
    w = np.zeros(3)
    t = np.median(Pr - Pm @ R0.T, axis=0)
    for _ in range(iters):
        q = Pm @ R0.T
        n = len(q)
        A = np.zeros((3 * n, 6))
        b = (Pr - q).ravel()
        for i in range(n):
            A[3 * i:3 * i + 3, 0:3] = [[0, q[i, 2], -q[i, 1]],
                                       [-q[i, 2], 0, q[i, 0]],
                                       [q[i, 1], -q[i, 0], 0]]
            A[3 * i:3 * i + 3, 3:6] = np.eye(3)
        res = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
        sc = max(np.median(res), 1e-3)
        wts = 1.0 / (1.0 + (res / (2.0 * sc)) ** 2)
        sol, *_ = np.linalg.lstsq(A * np.repeat(wts, 3)[:, None],
                                  b * np.repeat(wts, 3), rcond=None)
        w, t = sol[:3], sol[3:]
    return rodrigues(w) @ R0, t


def reproj_resid(R, t, Pm, uv, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    q = Pm @ R.T + t
    z = np.clip(q[:, 2], 1e-3, None)
    return np.stack([fx * q[:, 0] / z + cx - uv[:, 0],
                     fy * q[:, 1] / z + cy - uv[:, 1]], 1)


def coherence(R, t, Pm, uv, K):
    """Fraction of reprojection residual energy explained by a low-order field in image space.

    A rotation error displaces the projection systematically, so its residual varies smoothly with
    position. Depth noise does not. The explained fraction therefore measures how much of the
    residual is pose error rather than measurement error, and unlike RMS it is not swamped by the
    gross depth outliers that dominate the raw residual.
    """
    r = reproj_resid(R, t, Pm, uv, K)
    A = np.stack([np.ones(len(uv)), uv[:, 0] - uv[:, 0].mean(),
                  uv[:, 1] - uv[:, 1].mean()], 1)
    tot = float((r ** 2).sum())
    if tot <= 0:
        return 0.0
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    fit = A @ coef
    return float((fit ** 2).sum() / tot)


def reproj_rms(R, t, Pm, uv, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    q = Pm @ R.T + t
    z = np.clip(q[:, 2], 1e-3, None)
    u = fx * q[:, 0] / z + cx
    v = fy * q[:, 1] / z + cy
    return float(np.sqrt(np.mean((u - uv[:, 0]) ** 2 + (v - uv[:, 1]) ** 2)))


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R0 = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
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
        if len(good) < 40:
            continue
        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
        Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)
        cr = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (wb.ref + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        cm = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        ur = np.clip(np.round(uv_r[:, 0]).astype(int), 0, cr.shape[1] - 1)
        vr = np.clip(np.round(uv_r[:, 1]).astype(int), 0, cr.shape[0] - 1)
        um = np.clip(np.round(uv_m[:, 0]).astype(int), 0, cm.shape[1] - 1)
        vm = np.clip(np.round(uv_m[:, 1]).astype(int), 0, cm.shape[0] - 1)
        prc, pmc = cr[vr, ur], cm[vm, um]
        sky = (((prc[:, 2] > prc[:, 0] + 25) & (prc[:, 2] > 110) & (prc[:, 1] > prc[:, 0])) |
               ((pmc[:, 2] > pmc[:, 0] + 25) & (pmc[:, 2] > 110) & (pmc[:, 1] > pmc[:, 0])))
        keep = okr & okm & ~sky
        if keep.sum() < 40:
            continue
        Pr, Pm, uv = Pr[keep], Pm[keep], uv_r[keep]
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(Pr))
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]

        RA, tA = R0, np.median(Pr[a] - Pm[a] @ R0.T, axis=0)
        RB, tB = fit_3d(Pr[a], Pm[a], R0)
        rmsA = reproj_rms(RA, tA, Pm[b], uv[b], ref_c.K)
        rmsB = reproj_rms(RB, tB, Pm[b], uv[b], ref_c.K)
        cohA = coherence(RA, tA, Pm[b], uv[b], ref_c.K)
        cohB = coherence(RB, tB, Pm[b], uv[b], ref_c.K)
        pick = 'A' if rmsA <= rmsB else 'B'
        pick_c = 'A' if cohA <= cohB else 'B'
        Rsel, tsel = (RA, tA) if pick == 'A' else (RB, tB)
        Rselc, tselc = (RA, tA) if pick_c == 'A' else (RB, tB)

        def err(R, t):
            return float(rotdeg(R @ R_true.T)), float(np.linalg.norm((t - t_true)[:2]))

        eA, eB, eS = err(RA, tA), err(RB, tB), err(Rsel, tsel)
        eC = err(Rselc, tselc)
        oracle = eA if (eA[1] <= eB[1]) else eB
        rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'pick': pick, 'pick_coh': pick_c,
                     'rmsA': rmsA, 'rmsB': rmsB, 'cohA': cohA, 'cohB': cohB,
                     'rotC': eC[0], 'latC': eC[1],
                     'rotA': eA[0], 'latA': eA[1], 'rotB': eB[0], 'latB': eB[1],
                     'rotS': eS[0], 'latS': eS[1],
                     'rot_oracle': oracle[0], 'lat_oracle': oracle[1],
                     'base_rot': x['score_corrected']['rot_err_deg']})
        r = rows[-1]
        print('%-14s%-4s%-12s rmsA %7.1f rmsB %7.1f pick %s | lat A %5.2f B %5.2f sel %5.2f'
              % (scene, tag, sid, rmsA, rmsB, pick, eA[1], eB[1], eS[1]), flush=True)

    p = ROOT / 'out' / 'select_by_reproj.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d   picked B in %d' % (len(rows), int((g('pick') == 'B').sum())))
        print('picked B by coherence in %d' % int((g('pick_coh') == 'B').sum()))
        for name in ('A', 'B', 'S', 'C', 'oracle'):
            k_lat = 'lat_oracle' if name == 'oracle' else 'lat' + name
            k_rot = 'rot_oracle' if name == 'oracle' else 'rot' + name
            lat, rot = g(k_lat), g(k_rot)
            print('%-7s rot med %5.2f°  <2° %2d | lat med %5.2f m  <0.5 m %2d/%d'
                  % (name, np.median(rot), int((rot < 2).sum()),
                     np.median(lat), int((lat < 0.5).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
