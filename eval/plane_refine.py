"""Refine a coarse rotation locally using matched plane normals.

plane_match_rotation.py established that the planes DO correspond: under the true rotation a
median of 7 plane pairs agree, best agreement 0.5-1.3 deg, and every facade pair has 8-18 of them.
So the information needed to fix the 9.25 deg facade rotation is present in the data. Its
combinatorial matcher could not exploit it -- global RANSAC over sign-free normals is degenerate
(normals cannot distinguish a 180 deg flip about a matched normal, and many-to-one pairings inflate
support).

But a global search is not needed. The coarse rotation is already within ~9 deg, so this is a
local refinement problem: alternate one-to-one assignment and a least-squares rotation update,
which is well conditioned and cannot wander far. Starting point is the coarse rotation, so the
result is only trusted if it improves the objective.

Reported against the same targets, with translation always solved from SIFT correspondences so
that only rotation is being judged (the earlier version compared against the geometric baseline's
translation, which was not a fair comparison).
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

_spec = importlib.util.spec_from_file_location('pm', ROOT / 'eval' / 'plane_match_rotation.py')
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)
_spec2 = importlib.util.spec_from_file_location('fpi', ROOT / 'eval' / 'frame_plus_identity.py')
fpi = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(fpi)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def assign(R, pr_ref, pr_mov, tol=25.0):
    A = np.array([[np.degrees(np.arccos(np.clip(abs((R @ m['n']) @ r['n']), 0, 1)))
                   for m in pr_mov] for r in pr_ref])
    used_r, used_m, pairs = set(), set(), []
    while True:
        sub = A.copy()
        for i in used_r:
            sub[i, :] = 1e9
        for j in used_m:
            sub[:, j] = 1e9
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if sub[i, j] > tol:
            break
        used_r.add(int(i))
        used_m.add(int(j))
        pairs.append((int(i), int(j)))
    return pairs


def refine(R0, pr_ref, pr_mov, rounds=6):
    R = R0.copy()
    hist = []
    for _ in range(rounds):
        pairs = assign(R, pr_ref, pr_mov)
        if len(pairs) < 2:
            break
        tgt, src = [], []
        for i, j in pairs:
            nr, nm = pr_ref[i]['n'], pr_mov[j]['n']
            s = 1.0 if (R @ nm) @ nr >= 0 else -1.0
            tgt.append(nr)
            src.append(s * nm)
        tgt, src = np.array(tgt), np.array(src)

        def resid(w):
            Rr = rodrigues(w) @ R
            return (src @ Rr.T - tgt).ravel()

        sol = least_squares(resid, np.zeros(3), method='lm')
        R = rodrigues(sol.x) @ R
        hist.append(float(np.degrees(np.linalg.norm(sol.x))))
    return R, hist, len(assign(R, pr_ref, pr_mov))


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
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        pr_ref = pm.seg_planes(ref_c.points)
        pr_mov = pm.seg_planes(mov_c.points)
        if len(pr_ref) < 2 or len(pr_mov) < 2:
            print('%-14s%-4s%-12s too few planes' % (scene, tag, sid), flush=True)
            continue

        pair = fpi.sift_pairs(scene, wb.ref, sid, ref_c, mov_c, sift, bf)
        if pair is None:
            print('%-14s%-4s%-12s no correspondences' % (scene, tag, sid), flush=True)
            continue
        Pr, Pm = pair

        def lat(R):
            ti = Pr - Pm @ R.T
            return float(np.linalg.norm((np.median(ti, axis=0) - t_true)[:2]))

        R_ref, hist, n_after = refine(R_coarse, pr_ref, pr_mov)
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'rot_coarse': float(rotdeg(R_coarse @ R_true.T)),
                     'rot_refined': float(rotdeg(R_ref @ R_true.T)),
                     'lat_coarse': lat(R_coarse), 'lat_refined': lat(R_ref),
                     'n_planes': [len(pr_ref), len(pr_mov)], 'n_pairs_after': n_after,
                     'steps_deg': hist})
        r = rows[-1]
        print('%-14s%-4s%-12s planes %d/%d pairs %d | rot %5.2f -> %5.2f | lat %5.2f -> %5.2f'
              % (scene, tag, sid, len(pr_ref), len(pr_mov), n_after,
                 r['rot_coarse'], r['rot_refined'], r['lat_coarse'], r['lat_refined']),
              flush=True)

    p = ROOT / 'out' / 'plane_refine.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        print('rotation median %5.2f -> %5.2f deg  improved %d  worsened %d'
              % (np.median(g('rot_coarse')), np.median(g('rot_refined')),
                 int((g('rot_refined') < g('rot_coarse') - 0.05).sum()),
                 int((g('rot_refined') > g('rot_coarse') + 0.05).sum())))
        print('translation median %5.2f -> %5.2f m  under 0.5 m %d -> %d'
              % (np.median(g('lat_coarse')), np.median(g('lat_refined')),
                 int((g('lat_coarse') < 0.5).sum()), int((g('lat_refined') < 0.5).sum())))
        print('rotation under 2 deg: %d -> %d of %d'
              % (int((g('rot_coarse') < 2).sum()), int((g('rot_refined') < 2).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
