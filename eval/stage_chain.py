"""Does a staged chain rescue the narrow-basin cases?

rotation_basin.py says: playground's achievable rotation precision is 0.65 deg (the best of any
scene) but its basin is only +-2 deg, while the coarse estimate is 5.7 deg off. So refinement
cannot start. The narrow-basin regime is the one place the interactive loop has a job.

This tests whether a chain of cheap stages gets there -- which is what an interactive loop does in
practice, one adjustment at a time:

  stage 0  FPFH/RANSAC/ICP coarse rotation                     (the front end)
  stage 1  local IRLS refinement on matched plane normals      (plane_refine.py)
  stage 2  point-to-point ICP from that rotation               (local geometric refinement)

Translation at every stage is solved from the SAME SIFT correspondences, so the columns differ only
in rotation. Reported per stage: rotation error, and the resulting translation error.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

_s = importlib.util.spec_from_file_location('pref', ROOT / 'eval' / 'plane_refine.py')
pref = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pref)
_s2 = importlib.util.spec_from_file_location('pm', ROOT / 'eval' / 'plane_match_rotation.py')
pm = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(pm)
_s3 = importlib.util.spec_from_file_location('fpi', ROOT / 'eval' / 'frame_plus_identity.py')
fpi = importlib.util.module_from_spec(_s3)
_s3.loader.exec_module(fpi)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(1)


def prep(pts, voxel=0.5, max_pts=4000):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    if len(pc.points) > max_pts:
        pc = pc.select_by_index(RNG.choice(len(pc.points), max_pts, replace=False))
    return pc


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
        R0 = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        pair = fpi.sift_pairs(scene, wb.ref, sid, ref_c, mov_c, sift, bf)
        if pair is None:
            continue
        Pr, Pm = pair

        def score(R):
            ti = Pr - Pm @ R.T
            t = np.median(ti, axis=0)
            spread = float(np.median(np.linalg.norm(ti - t, axis=1)))
            return (float(rotdeg(R @ R_true.T)),
                    float(np.linalg.norm((t - t_true)[:2])), spread, t)

        pr_ref = pm.seg_planes(ref_c.points)
        pr_mov = pm.seg_planes(mov_c.points)
        R1 = R0
        if len(pr_ref) >= 2 and len(pr_mov) >= 2:
            R1, _, _ = pref.refine(R0, pr_ref, pr_mov)

        src, dst = prep(mov_c.points), prep(ref_c.points)
        _, _, _, t_seed = score(R1)
        T0 = np.eye(4)
        T0[:3, :3] = R1
        T0[:3, 3] = t_seed
        res = o3d.pipelines.registration.registration_icp(
            src, dst, 0.20, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        R2 = res.transformation[:3, :3].copy()
        # keep the rotation only; translation always re-solved from the correspondences
        r0, l0, s0, _ = score(R0)
        r1, l1, s1, _ = score(R1)
        r2, l2, s2, _ = score(R2)
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'rot': [r0, r1, r2], 'lat': [l0, l1, l2],
                     'spread': [s0, s1, s2], 'fitness': float(res.fitness)})
        print('%-14s%-4s%-12s rot %5.2f -> %5.2f -> %5.2f | lat %5.2f -> %5.2f -> %5.2f | spread %.2f/%.2f/%.2f'
              % (scene, tag, sid, r0, r1, r2, l0, l1, l2, s0, s1, s2), flush=True)

    p = ROOT / 'out' / 'stage_chain.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        rot = np.array([r['rot'] for r in rows])
        lat = np.array([r['lat'] for r in rows])
        print('\nn=%d' % len(rows))
        for i, name in enumerate(('coarse', 'plane refine', 'then ICP')):
            print('%-14s rot median %5.2f  <2deg %2d/%d | lat median %5.2f  <0.5m %2d/%d'
                  % (name, np.median(rot[:, i]), int((rot[:, i] < 2).sum()), len(rows),
                     np.median(lat[:, i]), int((lat[:, i] < 0.5).sum()), len(rows)))
        mono = int((np.diff(rot, axis=1) < 0).all(axis=1).sum())
        print('monotonically improving rotation: %d/%d' % (mono, len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
