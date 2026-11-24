"""Rotation from plane frames, translation from identity correspondences, selected without GT.

The decomposition says each signal should do the job it is good at:
  rotation    - dominant-plane frames give sub-half-degree accuracy WHEN the right alignment is
                chosen (delivery_area 0.27 deg, electro 0.36 deg), and fail by 30+ deg when the
                wrong one is chosen
  translation - exactly determined by correspondences once rotation is right (0.00 m residual),
                and appearance matching (SIFT) supplies those correspondences reliably

So: enumerate the plane-frame alignments, and for each one solve translation as the median over
SIFT correspondences. The correct rotation makes all the per-pair translations agree; a wrong
rotation scatters them. That spread is a ground-truth-free selection criterion and it is the whole
point of the exercise:

    score(R) = median_i | t_i - median(t) |,   t_i = p_ref_i - R p_mov_i

Pick the R with the tightest cluster and take its median translation. No ICP, no GT, and each
signal used only for what it is accurate at. Reported against the coarse baseline and against
SIFT-with-coarse-rotation.
"""
from __future__ import annotations

import importlib.util
import itertools
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

_spec = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)
_spec2 = importlib.util.spec_from_file_location('pr', ROOT / 'eval' / 'plane_rotation.py')
pr = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(pr)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RATIO = 0.80


def frame_candidates(F_ref, F_mov):
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = np.zeros((3, 3))
            for j in range(3):
                P[perm[j], j] = signs[j]
            if np.linalg.det(P) < 0.5:
                continue
            R = F_ref @ P @ F_mov.T
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def sift_pairs(scene, ref_sid, sid, ref_c, mov_c, sift, bf):
    gi = si.load_rgb(scene, ref_sid)
    gj = si.load_rgb(scene, sid)
    if gi is None or gj is None:
        return None
    k1, d1 = sift.detectAndCompute(gi, None)
    k2, d2 = sift.detectAndCompute(gj, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < RATIO * b.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < RATIO * b.distance}
    good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
    if len(good) < 8:
        return None
    uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
    uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
    Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
    Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)
    both = okr & okm
    if both.sum() < 8:
        return None
    return Pr[both], Pm[both]


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
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        pair = sift_pairs(scene, ref_sid, sid, ref_c, mov_c, sift, bf)
        if pair is None:
            print('%-14s%-4s%-12s no correspondences' % (scene, tag, sid), flush=True)
            continue
        Pr, Pm = pair

        def solve(R):
            ti = Pr - Pm @ R.T
            med = np.median(ti, axis=0)
            spread = float(np.median(np.linalg.norm(ti - med, axis=1)))
            return med, spread

        t_coarse, sp_coarse = solve(R_coarse)
        try:
            F_ref = pr.dominant_axes(ref_c.points)
            F_mov = pr.dominant_axes(mov_c.points)
            cands = frame_candidates(F_ref, F_mov) + [R_coarse]
        except Exception:                               # noqa: BLE001
            cands = [R_coarse]
        scored = []
        for R in cands:
            t, sp = solve(R)
            scored.append((sp, R, t))
        scored.sort(key=lambda z: z[0])
        sp_best, R_best, t_best = scored[0]
        # oracle over the same candidate set, to show what selection costs
        oracle = min(cands, key=lambda R: rotdeg(R @ R_true.T))
        t_or, _ = solve(oracle)

        rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'n_pairs': int(len(Pr)),
                     'rot_coarse': float(rotdeg(R_coarse @ R_true.T)),
                     'lat_coarse': float(np.linalg.norm((t_coarse - t_true)[:2])),
                     'spread_coarse': sp_coarse,
                     'rot_sel': float(rotdeg(R_best @ R_true.T)),
                     'lat_sel': float(np.linalg.norm((t_best - t_true)[:2])),
                     'spread_sel': sp_best,
                     'rot_oracle': float(rotdeg(oracle @ R_true.T)),
                     'lat_oracle': float(np.linalg.norm((t_or - t_true)[:2]))})
        r = rows[-1]
        print('%-14s%-4s%-12s rot %5.2f -> %5.2f (oracle %5.2f) | lat %5.2f -> %5.2f (oracle %5.2f) | spread %.2f'
              % (scene, tag, sid, r['rot_coarse'], r['rot_sel'], r['rot_oracle'],
                 r['lat_coarse'], r['lat_sel'], r['lat_oracle'], r['spread_sel']), flush=True)

    p = ROOT / 'out' / 'frame_plus_identity.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        print('%-10s %8s %8s %8s' % ('', 'rot med', 'lat med', '<0.5m'))
        for k in ('coarse', 'sel', 'oracle'):
            print('%-10s %7.2f° %7.2f m %6d/%d'
                  % (k, np.median(g('rot_' + k)), np.median(g('lat_' + k)),
                     int((g('lat_' + k) < 0.5).sum()), len(rows)))
        print('rotation under 2 deg: coarse %d  selected %d  oracle %d'
              % (int((g('rot_coarse') < 2).sum()), int((g('rot_sel') < 2).sum()),
                 int((g('rot_oracle') < 2).sum())))
    print('wrote', p)


if __name__ == '__main__':
    main()
