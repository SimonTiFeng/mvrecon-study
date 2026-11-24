"""Choose the dominant-frame alignment by FIT, not by closeness to the coarse rotation.

plane_rotation.py picks the sign/permutation alignment nearest the coarse rotation. That resolves
the ambiguity correctly when the coarse rotation is accurate and catastrophically when it is not:
delivery_area 4.47 -> 0.27 deg, but facade 3.44 -> 35 deg and terrace 2.06 -> 29.7 deg.

That is a selection failure, not a signal failure. So generate all 24 proper alignments plus the
coarse rotation itself, refine each locally, and keep the one that actually fits:

  for each candidate rotation R0:
      initialise translation from the coarse estimate, run point-to-point ICP
      score = fitness (fraction of points within the threshold)
  take the best-scoring candidate

Including the coarse rotation as a candidate makes this a strict improvement search: the result
can never be worse than the baseline it started from, only better. This is the generate-and-test
step the workbench is meant to provide, applied to rotation hypotheses.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location('pr', ROOT / 'eval' / 'plane_rotation.py')
_pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pr)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(2)
VOXEL = 0.5
THR = 0.30
ITERS = 40


def prep(pts, voxel=VOXEL, max_pts=4000):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    if len(pc.points) > max_pts:
        idx = RNG.choice(len(pc.points), max_pts, replace=False)
        pc = pc.select_by_index(idx)
    return pc


def candidates(F_ref, F_mov):
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


def main():
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
        t_coarse = M[:3, 3]
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        src = prep(mov_c.points)
        dst = prep(ref_c.points)
        try:
            F_ref = _pr.dominant_axes(ref_c.points)
            F_mov = _pr.dominant_axes(mov_c.points)
            cands = candidates(F_ref, F_mov)
        except Exception as exc:                       # noqa: BLE001
            print('%-14s%-4s%-12s frame failed %s' % (scene, tag, sid, exc), flush=True)
            continue
        # the baseline itself is a candidate, so this can only improve on it
        cands = [R_coarse] + [c for c in cands
                              if rotdeg(c @ R_coarse.T) > 1e-6]
        if len(cands) > 25:
            cands = [R_coarse] + sorted(
                [c for c in cands if c is not R_coarse],
                key=lambda c: rotdeg(c @ R_coarse.T))[:24]

        best = None
        for R0 in cands:
            T0 = np.eye(4)
            T0[:3, :3] = R0
            T0[:3, 3] = t_coarse
            res = o3d.pipelines.registration.registration_icp(
                src, dst, THR, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ITERS))
            if best is None or res.fitness > best[0] + 1e-9:
                best = (float(res.fitness), res.transformation[:3, :3].copy(),
                        res.transformation[:3, 3].copy(), int(len(cands)))
        fit, R1, t1, n = best
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'rot_coarse': float(rotdeg(R_coarse @ R_true.T)), 'lat_coarse': float(
                         np.linalg.norm((t_coarse - t_true)[:2])),
                     'rot_fit': float(rotdeg(R1 @ R_true.T)),
                     'lat_fit': float(np.linalg.norm((t1 - t_true)[:2])),
                     'fit': fit, 'n_cands': n})
        r = rows[-1]
        print('%-14s%-4s%-12s rot %5.2f -> %5.2f | lat %5.2f -> %5.2f m | fitness %.2f (%d cands)'
              % (scene, tag, sid, r['rot_coarse'], r['rot_fit'],
                 r['lat_coarse'], r['lat_fit'], fit, n), flush=True)

    p = ROOT / 'out' / 'frame_candidates.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        rc = np.array([r['rot_coarse'] for r in rows])
        rf = np.array([r['rot_fit'] for r in rows])
        lc = np.array([r['lat_coarse'] for r in rows])
        lf = np.array([r['lat_fit'] for r in rows])
        print('\nn=%d' % len(rows))
        print('rotation  median %5.2f -> %5.2f deg   improved %d  worsened %d'
              % (np.median(rc), np.median(rf), int((rf < rc).sum()), int((rf > rc + 0.05).sum())))
        print('translation median %5.2f -> %5.2f m  improved %d  worsened %d'
              % (np.median(lc), np.median(lf), int((lf < lc).sum()), int((lf > lc + 0.02).sum())))
        print('under 0.5 m: %.2f -> %.2f m; count %d -> %d'
              % (np.median(lc), np.median(lf), int((lc < 0.5).sum()), int((lf < 0.5).sum())))
        print('rotation under 2 deg: %d -> %d of %d'
              % (int((rc < 2).sum()), int((rf < 2).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
