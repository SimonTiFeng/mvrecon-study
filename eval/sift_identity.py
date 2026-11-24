"""Control experiment: can CLASSICAL appearance matching supply the identity correspondences?

Everything so far says the lateral failure needs a correspondence that identifies the same
physical point in both views -- one such anchor fixes translation to 0.13 m median, while
three self-consistency objectives (fixed-point, footprint, depth-agreement search) all fail.

But "identity" is not inherently semantic. SIFT + ratio test + RANSAC identifies points by
appearance, needs no learning, and is a standard geometric baseline. If it supplies good
correspondences here, then a VLM has no information advantage and the project's premise is
wrong. That has to be measured, not assumed.

Identical inputs to the agent: the two original RGB images, each view's own predicted depth
and intrinsics. Three solves from the same matches:
  T   - rotation held at the geometric baseline's estimate, translation from the matches
  RT  - full rigid, RANSAC over 3D correspondences
  RTs - full similarity (scale free), RANSAC over 3D correspondences
Scored against T_true on the same 25 rotation-successful targets.
"""
from __future__ import annotations

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

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
SCALE = 0.25
RATIO = 0.80


def load_rgb(scene, sid):
    img = cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png')),
                     cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def backproject(uv_full, depth, K):
    """uv in FULL-resolution pixels -> 3D in that camera's frame, using its own depth."""
    H, W = depth.shape
    u = np.round(uv_full[:, 0]).astype(int)
    v = np.round(uv_full[:, 1]).astype(int)
    ok = (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
    z = np.zeros(len(uv_full))
    z[ok] = depth[v[ok], u[ok]]
    ok = ok & np.isfinite(z) & (z > 0.1) & (z < 200)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.full((len(uv_full), 3), np.nan)
    P[ok, 0] = (uv_full[ok, 0] - cx) / fx * z[ok]
    P[ok, 1] = (uv_full[ok, 1] - cy) / fy * z[ok]
    P[ok, 2] = z[ok]
    return P, ok


def kabsch(A, B, with_scale=False):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float(S.sum() / max(((A - ca) ** 2).sum(), 1e-12)) if with_scale else 1.0
    return s, R, cb - s * R @ ca


def ransac_fit(Pm, Pr, with_scale, iters=3000, tol=0.12, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(Pm)
    if n < 4:
        return None
    best_k, best_idx = -1, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = kabsch(Pm[idx], Pr[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm((Pm * s) @ R.T + t - Pr, axis=1)
        k = int((r < tol).sum())
        if k > best_k:
            best_k, best_idx = k, np.where(r < tol)[0]
    if best_idx is None or len(best_idx) < 4:
        return None
    s, R, t = kabsch(Pm[best_idx], Pr[best_idx], with_scale)
    return {'s': s, 'R': R, 't': t, 'inliers': int(len(best_idx)), 'n': n}


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        if x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_sid = wb.ref
        ref_c, mov_c = wb.clouds[ref_sid], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_est = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)

        gi, gj = load_rgb(scene, ref_sid), load_rgb(scene, sid)
        row = {'scene': scene, 'tag': tag, 'sid': sid,
               'lateral_before': float(np.linalg.norm((M[:3, 3] - T[:3, 3])[:2]))}
        if gi is None or gj is None:
            row['error'] = 'no_image'
            rows.append(row)
            print('%-14s%-4s%-12s NO IMAGE' % (scene, tag, sid), flush=True)
            continue
        k1, d1 = sift.detectAndCompute(gi, None)
        k2, d2 = sift.detectAndCompute(gj, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            row['error'] = 'no_features'
            rows.append(row)
            print('%-14s%-4s%-12s NO FEATURES (%d/%d)'
                  % (scene, tag, sid, len(k1), len(k2)), flush=True)
            continue
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < RATIO * b.distance]
        m2 = bf.knnMatch(d2, d1, k=2)
        good2 = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < RATIO * b.distance}
        good = [a for a in good if good2.get(a.trainIdx) == a.queryIdx]

        if len(good) < 6:
            row.update({'error': 'too_few_matches', 'n_sift': [len(k1), len(k2)],
                        'n_match': len(good)})
            rows.append(row)
            print('%-14s%-4s%-12s only %d matches' % (scene, tag, sid, len(good)), flush=True)
            continue

        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / SCALE
        Pr, okr = backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = backproject(uv_m, mov_c.depth, mov_c.K)
        both = okr & okm
        Pr, Pm = Pr[both], Pm[both]
        row.update({'n_sift': [len(k1), len(k2)], 'n_match': len(good),
                    'n_3d': int(both.sum())})
        if both.sum() < 6:
            row['error'] = 'too_few_3d'
            rows.append(row)
            print('%-14s%-4s%-12s only %d 3D pairs'
                  % (scene, tag, sid, both.sum()), flush=True)
            continue

        d_t = Pr - Pm @ R_est.T
        med = np.median(d_t, axis=0)
        res = np.linalg.norm(d_t - med, axis=1)
        keep = res < max(0.15, np.quantile(res, 0.6))
        t_T = np.median(d_t[keep], axis=0) if keep.sum() >= 3 else med
        row['T'] = {'lateral': float(np.linalg.norm((t_T - T[:3, 3])[:2])),
                    'axial': float(abs(t_T[2] - T[:3, 3][2])), 'kept': int(keep.sum())}

        row['RTs'] = {'lateral': float('nan'), 'rot': float('nan'),
                      'scale': float('nan'), 'inliers': 0}
        row['RT'] = {'lateral': float('nan'), 'rot': float('nan'), 'inliers': 0}
        for name, ws in (('RT', False), ('RTs', True)):
            fit = ransac_fit(Pm, Pr, ws)
            if fit is None:
                continue
            R = fit['R']
            t = fit['t'] / max(fit['s'], 1e-12) if ws else fit['t']
            row[name] = {'lateral': float(np.linalg.norm((t - T[:3, 3])[:2])),
                         'rot': float(rotdeg(R @ T[:3, :3].T)),
                         'scale': fit['s'], 'inliers': fit['inliers'], 'n': fit['n']}
        rows.append(row)
        print('%-14s%-4s%-12s before %5.2f | SIFT %4d/%4d m=%3d 3d=%3d | T %5.2f  RT %5.2f/%4.1f  RTs %5.2f/%4.1f s=%.2f'
              % (scene, tag, sid, row['lateral_before'], len(k1), len(k2), len(good),
                 both.sum(), row['T']['lateral'], row['RT']['lateral'], row['RT']['rot'],
                 row['RTs']['lateral'], row['RTs']['rot'], row['RTs']['scale']), flush=True)

    p = ROOT / 'out' / 'sift_identity.json'
    p.write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if 'T' in r]
    print('\nscored %d/%d targets' % (len(ok), len(rows)))
    if ok:
        b = np.array([r['lateral_before'] for r in ok])
        print('%-14s %8s %8s %8s' % ('policy', 'median', '<0.5m', 'improved'))
        for k in ('before', 'T', 'RT', 'RTs'):
            v = b.copy() if k == 'before' else np.array([r[k]['lateral'] for r in ok])
            fin = np.isfinite(v)
            v = v[fin]
            print('%-14s %7.2fm %5d/%d %7d/%d'
                  % (k, np.median(v), int((v < 0.5).sum()), len(v),
                     int((v < b[fin]).sum()), len(v)))
        print('median SIFT matches %d, 3D pairs %d'
              % (np.median([r['n_match'] for r in ok]),
                 np.median([r['n_3d'] for r in ok])))
    print('wrote', p)


if __name__ == '__main__':
    main()
